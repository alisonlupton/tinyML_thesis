# metrics.py

import math
import torch.nn as nn
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List

# ---------- helpers

def conv1d_out_len(Lin: int, k: int, stride: int = 1, pad: int = 0, dilation: int = 1) -> int:
    # PyTorch formula
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
    # treat near-zeros as zero to avoid tiny fp noise
    return float((W.abs() > 0).float().mean().item())

def format_bytes(n: float) -> str:
    for unit in ["B","KB","MB","GB"]:
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}"
        n /= 1024.0

# FLOPs/MACs for 1D conv & linear (per-sample, forward)
def conv1d_macs_per_sample(in_c: int, out_c: int, k: int, Lout: int) -> int:
    # MACs = out_c * Lout * in_c * k
    return int(out_c) * int(Lout) * int(in_c) * int(k)

def linear_macs_per_sample(in_dim: int, out_dim: int, nonzero_frac: float = 1.0) -> int:
    # dense MACs; scale by nonzero fraction to reflect head sparsity at inference
    return int(in_dim * out_dim * nonzero_frac)

@dataclass
class CILResourceReport:
    # parameter counts
    params_backbone: int
    params_head_dense: int
    params_total_dense: int
    head_nonzero_frac: float

    # deployment (flash) bytes (assuming int8 for both backbone and head)
    flash_backbone_int8: int
    flash_head_int8_dense: int
    flash_head_int8_sparse: int
    flash_total_int8_dense: int
    flash_total_int8_sparse: int

    # training SRAM estimates (bytes) – head-only training
    train_weights_head_fp32: int
    train_grads_head_fp32: int
    train_opt_states_head_fp32: int
    train_activations_head_fp32: int
    replay_buffer_bytes: int
    replay_buffer_bytes_per_example: int
    train_sram_total_bytes: int

    # inference SRAM (peak activation buffer, bytes)
    infer_peak_activation_bytes_b1: int

    # compute (MACs/FLOPs)
    forward_macs_backbone_per_sample: int
    forward_macs_head_per_sample_dense: int
    forward_macs_head_per_sample_sparse: int
    forward_macs_total_dense_per_sample: int
    forward_macs_total_sparse_per_sample: int
    train_macs_per_batch_dense: int
    train_macs_per_batch_sparse: int
    forward_flops_total_sparse_per_sample: int
    train_flops_per_batch_sparse: int

    # shapes
    conv_lengths: Tuple[int, ...]   # all conv output lengths, e.g., (L1, L2)
    head_in_dim: int
    head_out_dim: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # d_pretty = {
        #     **d,
        #     "flash_backbone_int8_pretty": format_bytes(self.flash_backbone_int8),
        #     "flash_head_int8_dense_pretty": format_bytes(self.flash_head_int8_dense),
        #     "flash_head_int8_sparse_pretty": format_bytes(self.flash_head_int8_sparse),
        #     "flash_total_int8_dense_pretty": format_bytes(self.flash_total_int8_dense),
        #     "flash_total_int8_sparse_pretty": format_bytes(self.flash_total_int8_sparse),
        #     "train_weights_head_fp32_pretty": format_bytes(self.train_weights_head_fp32),
        #     "train_grads_head_fp32_pretty": format_bytes(self.train_grads_head_fp32),
        #     "train_opt_states_head_fp32_pretty": format_bytes(self.train_opt_states_head_fp32),
        #     "train_activations_head_fp32_pretty": format_bytes(self.train_activations_head_fp32),
        #     "replay_buffer_bytes_pretty": format_bytes(self.replay_buffer_bytes),
        #     "train_sram_total_bytes_pretty": format_bytes(self.train_sram_total_bytes),
        #     "infer_peak_activation_bytes_b1_pretty": format_bytes(self.infer_peak_activation_bytes_b1),
        #     "replay_buffer_bytes_per_example_pretty": format_bytes(self.replay_buffer_bytes_per_example),
        # }
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
    C: int, L: int,                     # C = sensor channels, L = window_len (NOT num classes)
    batch_size_train: int = 16,
    batch_size_infer: int = 1,
    replay_size: int = 5000,
    optimizer_kind: str = "adam",       # "adam" or "sgd"
    deployed_bits_backbone: int = 8,
    deployed_bits_classifier: int = 8,
    training_bits_classifier: int = 32,
    replay_bits: int = 8,               # INT8 latent replay
    activation_bits_train: int = 32,
    activation_bits_infer: int = 8,
) -> CILResourceReport:
    """
    Profiles memory (SRAM/flash), MACs/FLOPs, and params for the **CIL phase** with your setup:
      - Backbone is **frozen** (no backward through backbone).
      - Only the **linear head** trains (FP32 weights/grads/optimizer state).
      - Head uses a **binary mask** (if present) for SparCL-style sparsity.
      - Replay buffer stores **INT8 latent features + 1 FP32 scale per vector + int16 label**.
    """

    # ---------- sanity
    if L <= 0:
        raise ValueError("L must be the temporal window length (>0). You passed L <= 0.")
    if L <= 8:
        print("[metrics] Warning: L is very small. Make sure you're passing the window length, not num classes.")

    assert hasattr(model, "backbone") and hasattr(model, "head"), "Model must expose .backbone and .head"
    head = model.head
    in_dim = int(getattr(head, "in_dim"))
    out_dim = int(getattr(head, "out_dim"))

    # ---------- sparsity (prefer mask if present)
    head_nonzero = nonzero_fraction_mask_or_weight(head)

    # ---------- params
    params_backbone = sizeof_params(model.backbone)
    params_head_dense = head.linear.weight.numel() + head.linear.bias.numel()
    params_total_dense = params_backbone + params_head_dense

    # ---------- deployment flash (assume INT8 weights for both backbone + head)
    bytes_per_w_backbone = deployed_bits_backbone // 8
    bytes_per_w_head = deployed_bits_classifier // 8

    flash_backbone_int8 = params_backbone * bytes_per_w_backbone
    numel_w = head.linear.weight.numel()
    numel_b = head.linear.bias.numel() if head.linear.bias is not None else 0
    flash_head_int8_dense = (numel_w + numel_b) * bytes_per_w_head
    flash_total_int8_dense = flash_backbone_int8 + flash_head_int8_dense
    
    # "Sparse" head: approximate by scaling dense params with nonzero fraction
    flash_head_int8_sparse = int(numel_w * head_nonzero * (deployed_bits_classifier // 8)) + numel_b * (deployed_bits_classifier // 8)
    flash_total_int8_sparse = flash_backbone_int8 + flash_head_int8_sparse
    
    # ---------- conv shapes / MACs
    conv_specs, Louts = _sequential_conv1d_lengths(model.backbone, C, L)

    # per-layer conv MACs
    macs_backbone_fwd = 0
    cur_in_c = C
    for (in_c, out_c, k, s, p, d), Lout in zip(conv_specs, Louts):
        # (in_c in the layer may match cur_in_c; we trust the module values)
        macs_backbone_fwd += conv1d_macs_per_sample(in_c, out_c, k, Lout)
        cur_in_c = out_c

    # head MACs
    macs_head_fwd_dense  = linear_macs_per_sample(in_dim, out_dim, nonzero_frac=1.0)
    macs_head_fwd_sparse = linear_macs_per_sample(in_dim, out_dim, nonzero_frac=head_nonzero)

    macs_total_fwd_dense  = macs_backbone_fwd + macs_head_fwd_dense
    macs_total_fwd_sparse = macs_backbone_fwd + macs_head_fwd_sparse

    # ---------- training MACs/batch (backbone: fwd only; head: fwd + 2x for backward)
    macs_train_per_batch_dense  = batch_size_train * (macs_total_fwd_dense  + 2 * macs_head_fwd_dense)
    macs_train_per_batch_sparse = batch_size_train * (macs_total_fwd_sparse + 2 * macs_head_fwd_sparse)

    # FLOPs: 1 MAC ~= 2 FLOPs
    flops_fwd_total_sparse_per_sample = 2 * macs_total_fwd_sparse
    flops_train_per_batch_sparse      = 2 * macs_train_per_batch_sparse

    # ---------- training SRAM (head-only)
    # FP32 weights + grads
    bytes_per_fp_train = training_bits_classifier // 8
    w_bytes = params_head_dense * bytes_per_fp_train
    g_bytes = params_head_dense * bytes_per_fp_train

    # optimizer state
    opt_bytes = 0
    ok = optimizer_kind.lower()
    if ok == "adam":
        opt_bytes = 2 * params_head_dense * bytes_per_fp_train   # m & v
    elif ok in ("sgd", "sgd_momentum", "momentum"):
        opt_bytes = 1 * params_head_dense * bytes_per_fp_train   # momentum
    # else: keep 0 if unknown

    # activations for head backward: (B, in_dim) FP32
    act_head = batch_size_train * in_dim * (activation_bits_train // 8)

    # replay buffer (INT8 latent + 1 FP32 scale + int16 label) PER EXAMPLE
    replay_feat_bytes_per_ex = in_dim * (replay_bits // 8)   # INT8 vector
    replay_scale_bytes_per_ex = 4                            # float32 scale
    replay_label_bytes_per_ex = 2                            # int16 label
    replay_per_ex = replay_feat_bytes_per_ex + replay_scale_bytes_per_ex + replay_label_bytes_per_ex
    replay_bytes = replay_size * replay_per_ex

    train_sram_total = w_bytes + g_bytes + opt_bytes + act_head + replay_bytes

    # ---------- inference SRAM (peak activations, assume int8 activations if you deploy that way)
    # peak over the pipeline: (B*C*L), then each conv output, then GAP (B*in_dim), then logits (B*out_dim)
    peak_elems_candidates = [
        batch_size_infer * C * L,
        *[batch_size_infer * spec[1] * Lout for spec, Lout in zip(conv_specs, Louts)],  # out_c * Lout
        batch_size_infer * in_dim,
        batch_size_infer * out_dim,
    ]
    peak_elems = max(peak_elems_candidates) if peak_elems_candidates else batch_size_infer * max(in_dim, out_dim)
    infer_peak_activation_bytes_b1 = peak_elems * (activation_bits_infer // 8)

    # ---------- pack
    report = CILResourceReport(
        params_backbone=params_backbone,
        params_head_dense=params_head_dense,
        params_total_dense=params_total_dense,
        head_nonzero_frac=head_nonzero,

        flash_backbone_int8=flash_backbone_int8,
        flash_head_int8_dense=flash_head_int8_dense,
        flash_head_int8_sparse=flash_head_int8_sparse,
        flash_total_int8_dense=flash_total_int8_dense,
        flash_total_int8_sparse=flash_total_int8_sparse,

        train_weights_head_fp32=w_bytes,
        train_grads_head_fp32=g_bytes,
        train_opt_states_head_fp32=opt_bytes,
        train_activations_head_fp32=act_head,
        replay_buffer_bytes=replay_bytes,
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

    # ---------- nice print
    d = report.to_dict()
    # print("\n=== CIL Resource Profile (Frozen Backbone • Head-only Training • INT8 Latent Replay) ===")
    # print(f"Input: C={C}, L={L}")
    # print(f"Head dims: in={in_dim}, out={out_dim}, mask nonzero={100.0*report.head_nonzero_frac:.1f}%")
    # if len(report.conv_lengths) > 0:
    #     print(f"Conv out lengths: " + " → ".join([f"L{i+1}={v}" for i, v in enumerate(report.conv_lengths)]))
    # print(f"Params (dense): backbone={report.params_backbone:,}, head={report.params_head_dense:,}, total={report.params_total_dense:,}")
    # print(f"Flash int8 (dense): {d['flash_total_int8_dense_pretty']}  "
    #       f"(backbone={d['flash_backbone_int8_pretty']}, head={d['flash_head_int8_dense_pretty']})")
    # print(f"Flash int8 (sparse head): {d['flash_total_int8_sparse_pretty']}  "
    #       f"(head={d['flash_head_int8_sparse_pretty']})")
    # print(f"Train SRAM (head): weights={d['train_weights_head_fp32_pretty']}, grads={d['train_grads_head_fp32_pretty']}, "
    #       f"opt={d['train_opt_states_head_fp32_pretty']}, act={d['train_activations_head_fp32_pretty']}")
    # print(f"Replay buffer: total={d['replay_buffer_bytes_pretty']} "
    #       f"(per-example={d['replay_buffer_bytes_per_example_pretty']})")
    # print(f"Train SRAM total (approx): {d['train_sram_total_bytes_pretty']}")
    # print(f"Infer peak activation (B={batch_size_infer}): {d['infer_peak_activation_bytes_b1_pretty']}")
    # print(f"Forward MACs / sample (sparse): {report.forward_macs_total_sparse_per_sample:,}  "
    #       f"({report.forward_flops_total_sparse_per_sample:,} FLOPs)")
    # print(f"Train MACs / batch (sparse): {report.train_macs_per_batch_sparse:,}  "
    #       f"({report.train_flops_per_batch_sparse:,} FLOPs)")
    # print("===========================================================================================\n")
    
    save_path = cfg['metrics_path']
    with open(save_path, "w") as f:
        for k,v in d.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved CIL metrics to {save_path}")
        
    return report