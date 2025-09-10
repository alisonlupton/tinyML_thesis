#metrics.py

import math
import torch.nn as nn
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset

#---------- helpers

def conv1d_out_len(Lin: int, k: int, stride: int = 1, pad: int = 0, dilation: int = 1) -> int:
    #PyTorch formula
    return math.floor((Lin + 2*pad - dilation*(k - 1) - 1)/stride + 1)

def sizeof_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

def nonzero_fraction_mask_or_weight(head) -> float:
    """
    Prefer a binary mask (head.mask) if present; otherwise estimate from weights.
    """
    if hasattr(head, "mask") and head.mask is not None:
        m = head.mask
        return float((m != 0).float().mean().item())
    W = head.linear.weight.data
    #treat near-zeros as zero to avoid tiny fp noise
    return float((W.abs() > 0).float().mean().item())

def format_bytes(n: float) -> str:
    for unit in ["B","KB","MB","GB"]:
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}"
        n /= 1024.0

#FLOPs/MACs for 1D conv & linear (per-sample, forward)
def conv1d_macs_per_sample(in_c: int, out_c: int, k: int, Lout: int) -> int:
    #MACs = out_c * Lout * in_c * k
    return int(out_c) * int(Lout) * int(in_c) * int(k)

def linear_macs_per_sample(in_dim: int, out_dim: int, nonzero_frac: float = 1.0) -> int:
    #dense MACs; scale by nonzero fraction to reflect head sparsity at inference
    return int(in_dim * out_dim * nonzero_frac)

@dataclass
class CILResourceReport:
    #parameter counts
    params_backbone: int
    params_head_dense: int
    params_total_dense: int
    head_nonzero_frac: float

    #deployment (flash) bytes (assuming int8 for both backbone and head)
    flash_backbone_int8: int
    flash_head_int8_dense: int
    flash_head_int8_sparse: int
    flash_total_int8_dense: int
    flash_total_int8_sparse: int

    #training SRAM estimates (bytes) head-only training
    train_weights_head_fp32: int
    train_grads_head_fp32: int
    train_opt_states_head_fp32: int
    train_activations_head_fp32: int
    replay_buffer_bytes: int
    replay_buffer_bytes_per_example: int
    train_sram_total_bytes: int

    #inference SRAM (peak activation buffer, bytes)
    infer_peak_activation_bytes_b1: int

    #compute (MACs/FLOPs)
    forward_macs_backbone_per_sample: int
    forward_macs_head_per_sample_dense: int
    forward_macs_head_per_sample_sparse: int
    forward_macs_total_dense_per_sample: int
    forward_macs_total_sparse_per_sample: int
    train_macs_per_batch_dense: int
    train_macs_per_batch_sparse: int
    forward_flops_total_sparse_per_sample: int
    train_flops_per_batch_sparse: int

    #shapes
    conv_lengths: Tuple[int, ...]   #all conv output lengths, e.g., (L1, L2)
    head_in_dim: int
    head_out_dim: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        #d_pretty = {
        #**d,
        #"flash_backbone_int8_pretty": format_bytes(self.flash_backbone_int8),
        #"flash_head_int8_dense_pretty": format_bytes(self.flash_head_int8_dense),
        #"flash_head_int8_sparse_pretty": format_bytes(self.flash_head_int8_sparse),
        #"flash_total_int8_dense_pretty": format_bytes(self.flash_total_int8_dense),
        #"flash_total_int8_sparse_pretty": format_bytes(self.flash_total_int8_sparse),
        #"train_weights_head_fp32_pretty": format_bytes(self.train_weights_head_fp32),
        #"train_grads_head_fp32_pretty": format_bytes(self.train_grads_head_fp32),
        #"train_opt_states_head_fp32_pretty": format_bytes(self.train_opt_states_head_fp32),
        #"train_activations_head_fp32_pretty": format_bytes(self.train_activations_head_fp32),
        #"replay_buffer_bytes_pretty": format_bytes(self.replay_buffer_bytes),
        #"train_sram_total_bytes_pretty": format_bytes(self.train_sram_total_bytes),
        #"infer_peak_activation_bytes_b1_pretty": format_bytes(self.infer_peak_activation_bytes_b1),
        #"replay_buffer_bytes_per_example_pretty": format_bytes(self.replay_buffer_bytes_per_example),
        #}
        return d


def _sequential_conv1d_lengths(backbone: nn.Module, C: int, L: int) -> Tuple[List[Tuple[int,int,int,int,int]], Tuple[int, ...]]:
    """
    Walk backbone, collect Conv1d layers (in_c, out_c, k, stride, pad, dilation) and compute
    the sequence of output lengths starting from input length L.
    Returns:
      conv_specs: list of tuples (in_c, out_c, k, stride, pad, dilation)
      Louts: tuple of conv output lengths (L1, L2, ...)
    """
    conv_specs = []
    for m in backbone.modules():
        if isinstance(m, nn.Conv1d):
            k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
            s = m.stride if isinstance(m.stride, int) else m.stride[0]
            p = m.padding if isinstance(m.padding, int) else m.padding[0]
            d = m.dilation if isinstance(m.dilation, int) else m.dilation[0]
            conv_specs.append((m.in_channels, m.out_channels, k, s, p, d))

    Louts: List[int] = []
    curL = int(L)
    for (_, _, k, s, p, d) in conv_specs:
        curL = conv1d_out_len(curL, k=k, stride=s, pad=p, dilation=d)
        Louts.append(curL)
    return conv_specs, tuple(Louts)


def profile_cil_resources(
    cfg, 
    model: nn.Module,
    C: int, L: int,                     #C = sensor channels, L = window_len (NOT num classes)
    batch_size_train: int,
    batch_size_infer: int,
    replay_size: int,
    optimizer_kind: str,
    deployed_bits_backbone: int,
    deployed_bits_classifier: int,
    training_bits_classifier: int,
    replay_bits: int,               #INT8 latent replay
    activation_bits_train: int,
    activation_bits_infer: int,
    replay_batch_size: int ,        #number of replay samples per step
    kd_prev_rows: int,              #prev_num (teacher rows); 0 if no KD this task
    kd_enabled: bool,               #whether KD path is active
    autograd_grad_for_cwi: bool     #True = use autograd.grad(live) [+grad(replay if any)]
) -> CILResourceReport:
    """
    Accurate sizing for your pipeline:
      - Frozen backbone (no backward through backbone).
      - Linear head trains in FP32 with SparCL mask + CWI buffers.
      - Replay buffer stores INT8 latent + FP32 scale (NO label bytes).
      - Replay assumed to live in SRAM (conservative over-estimate).
      - KD teacher snapshot (prev_rows  in_dim + bias) counted when enabled.
      - BN running stats included in Flash.
      - Transient per-step device bytes reported separately (not folded into persistent).
    """
    if L <= 0:
        raise ValueError("L must be the temporal window length (>0).")
    if L <= 8:
        print("[metrics] Warning: L is very small. Make sure you're passing the window length, not num classes.")

    assert hasattr(model, "backbone") and hasattr(model, "head"), "Model must expose .backbone and .head"
    head = model.head
    in_dim = int(getattr(head, "in_dim"))
    out_dim = int(getattr(head, "out_dim"))

    #---------- sparsity
    head_nonzero = nonzero_fraction_mask_or_weight(head)

    #---------- params
    params_backbone = sizeof_params(model.backbone)
    numel_w = head.linear.weight.numel()
    numel_b = head.linear.bias.numel() if head.linear.bias is not None else 0
    params_head_dense = numel_w + numel_b
    params_total_dense = params_backbone + params_head_dense

    #---------- flash (include BN running stats; generic "deploy" naming internally)
    bytes_per_w_backbone = deployed_bits_backbone // 8
    bytes_per_w_head = deployed_bits_classifier // 8

    flash_backbone_params = params_backbone * bytes_per_w_backbone
    bn_running = 0
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm1d):
            bn_running += 2 * m.num_features  #running mean + running var
    flash_backbone_deploy = flash_backbone_params + bn_running * bytes_per_w_backbone

    flash_head_deploy_dense  = (numel_w + numel_b) * bytes_per_w_head
    flash_head_deploy_sparse = int(numel_w * head_nonzero * bytes_per_w_head) + numel_b * bytes_per_w_head
    flash_total_deploy_dense  = flash_backbone_deploy + flash_head_deploy_dense
    flash_total_deploy_sparse = flash_backbone_deploy + flash_head_deploy_sparse

    #---------- conv MACs
    conv_specs, Louts = _sequential_conv1d_lengths(model.backbone, C, L)
    macs_backbone_fwd = 0
    for (in_c, out_c, k, s, p, d), Lout in zip(conv_specs, Louts):
        macs_backbone_fwd += conv1d_macs_per_sample(in_c, out_c, k, Lout)

    #head MACs per sample
    macs_head_fwd_dense  = linear_macs_per_sample(in_dim, out_dim, nonzero_frac=1.0)
    macs_head_fwd_sparse = linear_macs_per_sample(in_dim, out_dim, nonzero_frac=head_nonzero)

    macs_total_fwd_dense  = macs_backbone_fwd + macs_head_fwd_dense
    macs_total_fwd_sparse = macs_backbone_fwd + macs_head_fwd_sparse

    #---------- training MACs
    B = int(batch_size_train)
    R = int(max(0, replay_batch_size))
    use_replay_in_step = R > 0

    #live forward (backbone + head)
    macs_live_fwd   = B * (macs_backbone_fwd + macs_head_fwd_sparse)
    #replay forward (head only)
    macs_replay_fwd = (R if use_replay_in_step else 0) * macs_head_fwd_sparse

    #backward equivalents on head:
    head_bwd_equiv = 1  #final loss.backward
    if autograd_grad_for_cwi:
        head_bwd_equiv += 1           #grad(live_loss)
        if use_replay_in_step:
            head_bwd_equiv += 1       #grad(replay_loss)

    #backward cost scales with (B + R)
    macs_head_backward = head_bwd_equiv * ((B + (R if use_replay_in_step else 0)) * macs_head_fwd_sparse)

    #KD/DDR compute
    macs_kd = 0
    if kd_enabled and kd_prev_rows > 0:
        macs_kd += B * macs_backbone_fwd
        macs_kd += B * linear_macs_per_sample(in_dim, kd_prev_rows, nonzero_frac=head_nonzero)

    macs_train_per_batch_sparse = macs_live_fwd + macs_replay_fwd + macs_head_backward + macs_kd
    macs_train_per_batch_dense  = (
        macs_live_fwd + macs_replay_fwd +
        head_bwd_equiv * ((B + (R if use_replay_in_step else 0)) * macs_head_fwd_dense) +
        (B * macs_backbone_fwd + B * linear_macs_per_sample(in_dim, kd_prev_rows, nonzero_frac=1.0)
         if (kd_enabled and kd_prev_rows > 0) else 0)
    )

    flops_fwd_total_sparse_per_sample = 2 * macs_total_fwd_sparse
    flops_train_per_batch_sparse      = 2 * macs_train_per_batch_sparse

    #---------- training SRAM (persistent; replay assumed to live in SRAM per request)
    bytes_per_fp_train = training_bits_classifier // 8
    w_bytes = params_head_dense * bytes_per_fp_train
    g_bytes = params_head_dense * bytes_per_fp_train

    ok = optimizer_kind.lower()
    if ok == "adam":
        opt_bytes = 2 * params_head_dense * bytes_per_fp_train   #m & v
    elif ok in ("sgd", "sgd_momentum", "momentum"):
        opt_bytes = 1 * params_head_dense * bytes_per_fp_train
    else:
        opt_bytes = 0

    #head activations required for backward (live + replay)
    act_head = (B + (R if use_replay_in_step else 0)) * in_dim * (activation_bits_train // 8)

    #SparCL persistent buffers
    mask_bytes = numel_w * bytes_per_fp_train                    #mask kept as float
    cwi_bytes  = 3 * numel_w * bytes_per_fp_train               #cwi, cwi_curr, cwi_mem

    #teacher snapshot (if used this task)
    teacher_bytes = 0
    if kd_enabled and kd_prev_rows > 0:
        teacher_bytes = (kd_prev_rows * in_dim + kd_prev_rows) * bytes_per_fp_train

    #replay buffer (SRAM, conservative)
    replay_feat_bytes_per_ex  = in_dim * (replay_bits // 8)  #int8 features
    replay_scale_bytes_per_ex = 4                            #fp32 scale
    replay_per_ex = replay_feat_bytes_per_ex + replay_scale_bytes_per_ex
    replay_bytes  = replay_size * replay_per_ex

    train_persistent_bytes = w_bytes + g_bytes + opt_bytes + act_head + mask_bytes + cwi_bytes + teacher_bytes
    train_sram_total = train_persistent_bytes + replay_bytes

    #---------- transient per-step bytes (device, not folded into persistent)
    #live input + peak backbone activations (eval forward)
    peak_conv_elems = 0
    for (in_c, out_c, k, s, p, d), Lout in zip(conv_specs, Louts):
        peak_conv_elems = max(peak_conv_elems, out_c * Lout)

    live_input_bytes     = B * C * L * 4
    backbone_peak_bytes  = B * peak_conv_elems * 4

    #replay staging on device + dequant temp
    replay_q_stage_bytes = (R if use_replay_in_step else 0) * in_dim * 1
    replay_s_stage_bytes = (R if use_replay_in_step else 0) * 4
    replay_dequant_bytes = (R if use_replay_in_step else 0) * in_dim * 4

    train_transient_peak_step_bytes = (
        live_input_bytes + backbone_peak_bytes +
        replay_q_stage_bytes + replay_s_stage_bytes + replay_dequant_bytes
    )

    #---------- inference SRAM (peak activations)
    peak_elems_candidates = [
        batch_size_infer * C * L,
        *[batch_size_infer * spec[1] * Lout for spec, Lout in zip(conv_specs, Louts)],
        batch_size_infer * in_dim,
        batch_size_infer * out_dim,
    ]
    peak_elems = max(peak_elems_candidates) if peak_elems_candidates else batch_size_infer * max(in_dim, out_dim)
    infer_peak_activation_bytes_b1 = peak_elems * (activation_bits_infer // 8)

    #---------- pack report (keep existing field names for compatibility)
    report = CILResourceReport(
        params_backbone=params_backbone,
        params_head_dense=params_head_dense,
        params_total_dense=params_total_dense,
        head_nonzero_frac=head_nonzero,

        #these names say "int8" in dataclass; we fill them with deploy bytes (could be 32-bit if passed)
        flash_backbone_int8=flash_backbone_deploy,
        flash_head_int8_dense=flash_head_deploy_dense,
        flash_head_int8_sparse=flash_head_deploy_sparse,
        flash_total_int8_dense=flash_total_deploy_dense,
        flash_total_int8_sparse=flash_total_deploy_sparse,

        train_weights_head_fp32=w_bytes,
        train_grads_head_fp32=g_bytes,
        train_opt_states_head_fp32=opt_bytes,
        train_activations_head_fp32=act_head,
        replay_buffer_bytes=replay_bytes,                          #SRAM per assumption
        replay_buffer_bytes_per_example=replay_per_ex,
        train_sram_total_bytes=train_sram_total,

        infer_peak_activation_bytes_b1=infer_peak_activation_bytes_b1,

        forward_macs_backbone_per_sample=macs_backbone_fwd,
        forward_macs_head_per_sample_dense=macs_head_fwd_dense,
        forward_macs_head_per_sample_sparse=macs_head_fwd_sparse,
        forward_macs_total_dense_per_sample=macs_total_fwd_dense,
        forward_macs_total_sparse_per_sample=macs_total_fwd_sparse,
        train_macs_per_batch_dense=macs_train_per_batch_dense,
        train_macs_per_batch_sparse=macs_train_per_batch_sparse,
        forward_flops_total_sparse_per_sample=flops_fwd_total_sparse_per_sample,
        train_flops_per_batch_sparse=flops_train_per_batch_sparse,

        conv_lengths=Louts,
        head_in_dim=in_dim,
        head_out_dim=out_dim,
    )

    #---------- write metrics (add helpful breakdown fields)
    d = report.to_dict()
    d.update({
        "train_persistent_bytes": train_persistent_bytes,
        "train_transient_peak_step_bytes": train_transient_peak_step_bytes,
        "train_head_mask_bytes": mask_bytes,
        "train_cwi_bytes": cwi_bytes,
        "teacher_snapshot_bytes": teacher_bytes,
        "replay_batch_size": R,
        "kd_prev_rows": kd_prev_rows,
        "head_backward_equiv_per_batch": head_bwd_equiv,
        "macs_live_forward_per_batch": macs_live_fwd,
        "macs_replay_forward_per_batch": macs_replay_fwd,
        "macs_kd_per_batch": macs_kd,
        "bn_running_params_count": bn_running,
    })

    save_path = cfg['metrics_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        for k, v in d.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved CIL metrics to {save_path}")

    return report
