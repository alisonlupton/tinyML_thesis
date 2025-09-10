import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, Tuple, Optional, Any
import pandas as pd

#---------- Core hook-based profiler ----------
class _FrozenFeatExtractor(nn.Module):
    """
    Expose z = proj(_features(x)) for profiling.
    Keeps ref to full model so hooks can see quantized modules too.
    """
    def __init__(self, cil_model: nn.Module):
        super().__init__()
        self.model = cil_model
        self.proj = cil_model.proj
    def forward(self, x):
        feats = self.model._features(x)
        z = self.proj(feats)
        return z

def _is_quant_conv(m: nn.Module) -> bool:
    qmod = getattr(torch.nn, "quantized", None)
    return qmod is not None and isinstance(m, getattr(qmod, "Conv2d", tuple()))

@torch.no_grad()
def _profile_forward(model: nn.Module,
                     input_size: Tuple[int,int,int,int],
                     activation_bits: int = 8,
                     include_elementwise: bool = False):
    """
    Returns:
      macs_total, peak_act_bytes, per-module dicts {name->value}
    Counts Conv2d/Linear MACs; optionally 1 op/elem for BN/ReLU/Pool.
    Peak activation  largest single output tensor (per *sample* forward).
    """
    #If proj is on CPU and has params, device() access can fail; default to cpu
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    model.eval()
    macs_per_module = defaultdict(float)
    params_per_module = defaultdict(int)
    peak_act_bytes_per_module = defaultdict(int)
    peak_activation_bytes = 0

    def num_params_local(m):
        return sum(p.numel() for p in m.parameters(recurse=False))

    def hook_fn(name, m):
        def fn(mod, inp, out):
            nonlocal peak_activation_bytes
            y = out
            params_per_module[name] += num_params_local(mod)

            macs = 0.0
            if isinstance(mod, nn.Conv2d) or _is_quant_conv(mod):
                n, cout, h, w = y.shape
                x = inp[0]
                cin = getattr(mod, "in_channels", x.shape[1])
                kh, kw = getattr(mod, "kernel_size", (1, 1))
                groups = getattr(mod, "groups", 1)
                macs = n * h * w * cout * (cin // groups) * kh * kw
            elif isinstance(mod, nn.Linear):
                n = y.shape[0]
                macs = n * mod.in_features * mod.out_features
            elif include_elementwise and isinstance(mod, (nn.BatchNorm2d, nn.ReLU, nn.ReLU6, nn.SiLU,
                                                          nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.AvgPool2d)):
                macs = y.numel()

            macs_per_module[name] += macs
            out_bytes = y.numel() * (activation_bits // 8)
            peak_activation_bytes = max(peak_activation_bytes, out_bytes)
            peak_act_bytes_per_module[name] = max(peak_act_bytes_per_module[name], out_bytes)
        return fn

    hooks = []
    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(hook_fn(name, m)))

    x = torch.zeros(*input_size, device=device)
    _ = model(x)

    for h in hooks: h.remove()

    macs_total = sum(macs_per_module.values())
    return macs_total, peak_activation_bytes, dict(macs_per_module), dict(params_per_module), dict(peak_act_bytes_per_module)

#---------- Head / buffers / replay sizing ----------
def _head_resources(head_linear: nn.Linear,
                    feat_dim: int,
                    num_classes: int,
                    train_dtype_bits: int = 32,
                    infer_dtype_bits: int = 8,
                    optimizer: str = "adam") -> Dict:
    params = feat_dim * num_classes + num_classes
    macs_infer = feat_dim * num_classes
    flash_kB_int8 = params / 1024
    flash_kB_fp32 = (params * 4) / 1024

    if optimizer.lower() == "adam":
        #Training state ~= W + m + v (all kept in train precision)
        train_state_bytes = params * 3 * (train_dtype_bits // 8)
    elif optimizer.lower() in ("sgd", "momentum", "sgd_momentum", "sgdm"):
        #W + momentum
        train_state_bytes = params * 2 * (train_dtype_bits // 8)
    else:
        #Fallback: only W
        train_state_bytes = params * (train_dtype_bits // 8)

    return {
        "params": params,
        "flash_kB_INT8": flash_kB_int8,
        "flash_kB_FP32": flash_kB_fp32,
        "infer_macs_per_sample": macs_infer,
        "infer_macs_per_sample_M": macs_infer / 1e6,
        "train_state_kB_FP32_est": train_state_bytes / 1024,
        "act_bytes_per_sample_INFER": feat_dim * (infer_dtype_bits // 8),
        "act_bytes_per_sample_TRAIN": feat_dim * (train_dtype_bits // 8),
        "param_bytes_FP32": params * 4,
    }

def _tdm_buffers_bytes(head_module) -> int:
    """mask (float32) + cwi, cwi_curr, cwi_mem (float32 each)."""
    W = head_module.linear.weight
    elems = W.numel()
    bytes_mask = elems * 4
    bytes_cwi_pack = elems * 4 * 3
    return bytes_mask + bytes_cwi_pack

#---------- Helpers to read config safely ----------
def _cfg_get(cfg: Optional[Dict[str, Any]], key: str, default):
    if cfg is None:
        return default
    return cfg.get(key, default)

#---------- Public API ----------
@torch.no_grad()
def profile_tinyml_cil(model: nn.Module,
                       input_size: Tuple[int,int,int,int] = (1,3,160,160),
                       activation_bits_infer: int = 8,
                       activation_bits_train: int = 32,
                       #deployed weight precisions
                       backbone_weight_bits_deployed: int = 8,
                       proj_weight_bits_deployed: int = 32,
                       head_weight_bits_deployed: int = 32,
                       #training batch sizes
                       batch_size_live: int = 1,
                       batch_size_replay: int = 4,
                       #replay buffer
                       replay_enabled: bool = True,
                       replay_size: int = 0,
                       replay_bits: int = 8,
                       include_replay_scores: bool = True,
                       include_perchannel_scales: bool = True,
                       num_seen_classes: Optional[int] = None,
                       optimizer: str = "adam",
                       include_elementwise: bool = False,
                       #optional config (to auto-fill args)
                       config: Optional[Dict[str, Any]] = None) -> Dict:
    """
    If `config` is provided, the following keys are consumed (with fallback to explicit args):
      - img_size -> input_size (assumes (1,3,img_size,img_size))
      - CIL_batch_size_train -> batch_size_live
      - replay_batch_size -> batch_size_replay
      - replay -> replay_enabled
      - buffer_size -> replay_size
      - feat_dim (not required; taken from model.head)
      - optimizer (string)
    """

    #----- hydrate from config if provided -----
    if config is not None:
        img_size_cfg = _cfg_get(config, "img_size", None)
        if img_size_cfg is not None:
            input_size = (1, 3, int(img_size_cfg), int(img_size_cfg))
        batch_size_live = int(_cfg_get(config, "CIL_batch_size_train", batch_size_live))
        batch_size_replay = int(_cfg_get(config, "replay_batch_size", batch_size_replay))
        replay_enabled = bool(_cfg_get(config, "replay", replay_enabled))
        replay_size = int(_cfg_get(config, "buffer_size", replay_size))
        optimizer = str(_cfg_get(config, "optimizer", optimizer))

        #If replay is disabled, zero-out replay batch size for SRAM/compute
        if not replay_enabled:
            batch_size_replay = 0

    #----- Inference path MACs / peak activations for (backbone->proj) -----
    frozen = _FrozenFeatExtractor(model).eval()
    macs_bbp_per_sample, peak_act_bytes_per_sample, per_macs, per_params_local, per_peak_act = _profile_forward(
        frozen, input_size=input_size, activation_bits=activation_bits_infer, include_elementwise=include_elementwise
    )

    #----- Split params: backbone (pre-quant) vs proj (in proj) -----
    float_backbone_params = getattr(model, "_float_backbone_params", None)
    backbone_params = int(float_backbone_params) if float_backbone_params is not None else 0
    proj_params = sum(p.numel() for p in frozen.proj.parameters())

    #----- Flash sizes using configured bit-widths -----
    def params_to_kb(n_params: int, bits: int) -> float:
        return (n_params * (bits / 8)) / 1024.0

    flash_backbone_kB = params_to_kb(backbone_params, backbone_weight_bits_deployed)
    flash_proj_kB     = params_to_kb(proj_params,     proj_weight_bits_deployed)

    #----- Head stats -----
    head_lin: nn.Linear = model.head.linear
    feat_dim = model.head.in_dim
    num_classes = model.head.out_dim
    head_stat = _head_resources(
        head_lin, feat_dim, num_classes,
        train_dtype_bits=activation_bits_train, infer_dtype_bits=activation_bits_infer, optimizer=optimizer
    )
    head_flash_kB = head_stat["flash_kB_FP32"] if head_weight_bits_deployed == 32 else head_stat["flash_kB_INT8"]

    #----- Deployed totals (per-sample) -----
    macs_total_per_sample = macs_bbp_per_sample + head_stat["infer_macs_per_sample"]
    macs_total_per_sample_M = macs_total_per_sample / 1e6
    flash_total_kB = flash_backbone_kB + flash_proj_kB + head_flash_kB

    #----- Replay SRAM footprint (stored buffer on-device) -----
    replay_per_sample_bytes = feat_dim * (replay_bits // 8)
    replay_latent_total = (replay_size * replay_per_sample_bytes) if replay_enabled else 0
    replay_score_total = ((replay_size * 4) if include_replay_scores else 0) if replay_enabled else 0
    scales_bytes = (feat_dim * 4) if include_perchannel_scales else 0
    cls_stats_bytes = (num_seen_classes * (4 + 4 + 4)) if (num_seen_classes and num_seen_classes > 0) else 0
    replay_total_bytes = replay_latent_total + replay_score_total + scales_bytes + cls_stats_bytes

    #----- Training SRAM (head only) -----
    #We assume backbone is frozen: no backbone grads/acts are kept.
    #Live samples: go through backbone+proj, but only head activations are retained for training.
    #Replay samples: already latent -> only head activations.
    bytes_per_act_head_train = feat_dim * (activation_bits_train // 8)
    train_acts_bytes_live   = batch_size_live   * bytes_per_act_head_train
    train_acts_bytes_replay = batch_size_replay * bytes_per_act_head_train
    train_acts_bytes_total  = train_acts_bytes_live + train_acts_bytes_replay

    #(Optional) gradient memory for head output layer (params) ~ same order as params FP32
    #Not double-counting optimizer state which is reported separately.
    head_param_grad_bytes = head_stat["param_bytes_FP32"]

    tdm_buffers_bytes = _tdm_buffers_bytes(model.head)

    #----- Per-step MACs (training) -----
    #Backbone+proj computed only for live samples.
    #Head is computed for both live and replay.
    macs_bbp_per_step      = macs_bbp_per_sample + 0  #per-sample number; multiply by live batch below
    macs_head_per_sample   = head_stat["infer_macs_per_sample"]
    macs_train_per_step    = (batch_size_live * macs_bbp_per_step) + ((batch_size_live + batch_size_replay) * macs_head_per_sample)
    macs_train_per_step_M  = macs_train_per_step / 1e6

    #----- Convenience unit converters -----
    kb = lambda x: x / 1024
    mb = lambda x: x / (1024*1024)

    report = {
        "config": {
            "input_size": input_size,
            "activation_bits_infer": activation_bits_infer,
            "activation_bits_train": activation_bits_train,
            "backbone_weight_bits_deployed": backbone_weight_bits_deployed,
            "proj_weight_bits_deployed": proj_weight_bits_deployed,
            "head_weight_bits_deployed": head_weight_bits_deployed,
            "batch_size_live": batch_size_live,
            "batch_size_replay": batch_size_replay,
            "replay_enabled": replay_enabled,
            "replay_size": replay_size,
            "replay_bits": replay_bits,
            "include_replay_scores": include_replay_scores,
            "include_perchannel_scales": include_perchannel_scales,
            "num_seen_classes": num_seen_classes,
            "optimizer": optimizer,
        },
        "backbone": {
            "params": backbone_params,
            "flash_kB": flash_backbone_kB,
        },
        "proj": {
            "params": proj_params,
            "flash_kB": flash_proj_kB,
        },
        "backbone_proj_runtime": {
            "macs_per_sample": macs_bbp_per_sample,
            "macs_per_sample_M": macs_bbp_per_sample / 1e6,
            "peak_activation_bytes": peak_act_bytes_per_sample,
            "peak_activation_kB": kb(peak_act_bytes_per_sample),
            "peak_activation_MB": mb(peak_act_bytes_per_sample),
            "per_module_macs": dict(sorted(per_macs.items(), key=lambda kv: kv[1], reverse=True)),
            "per_module_peak_act_bytes": dict(sorted(per_peak_act.items(), key=lambda kv: kv[1], reverse=True)),
        },
        "head": head_stat,
        "deployed_total": {
            "flash_kB": flash_total_kB,
            "macs_per_sample_M": macs_total_per_sample_M,
        },
        "training_overheads": {
            "head_train_state_kB_FP32_est": head_stat["train_state_kB_FP32_est"],  #independent of batch size
            "head_param_grad_bytes_est": head_param_grad_bytes,  #rough lower bound
            "train_activations_bytes_head_live": train_acts_bytes_live,
            "train_activations_bytes_head_replay": train_acts_bytes_replay,
            "train_activations_bytes_head_total": train_acts_bytes_total,
            "train_activations_kB_head_total": kb(train_acts_bytes_total),
            "train_macs_per_step": macs_train_per_step,
            "train_macs_per_step_M": macs_train_per_step_M,
        },
        "replay_storage": {
            "feat_dim": feat_dim,
            "latent_per_sample_bytes": replay_per_sample_bytes,
            "latent_total_bytes": replay_latent_total,
            "latent_total_kB": kb(replay_latent_total),
            "scores_total_bytes": replay_score_total,
            "scales_bytes": scales_bytes,
            "per_class_stats_bytes": cls_stats_bytes,
            "replay_total_bytes": replay_total_bytes,
            "replay_total_kB": kb(replay_total_bytes),
        },
        "tdm_buffers": {
            "bytes": tdm_buffers_bytes,
            "kB": kb(tdm_buffers_bytes),
        }
    }
    return report

def pretty_print_tinyml_report(report: Dict):
    def fmt_kb(x): return f"{x:.1f} kB"
    def fmt_mb(x): return f"{x:.3f} MB"
    cfg = report["config"]
    bb = report["backbone"]
    pj = report["proj"]
    rt = report["backbone_proj_runtime"]
    head = report["head"]
    tot = report["deployed_total"]
    tr = report["training_overheads"]
    rp = report["replay_storage"]
    tdm = report["tdm_buffers"]

    print("\n=== TinyML Profile ===")
    print(f"Input: {cfg['input_size']}, infer_act_bits={cfg['activation_bits_infer']}, "
          f"deployed_w_bits=(bb:{cfg['backbone_weight_bits_deployed']}, proj:{cfg['proj_weight_bits_deployed']}, head:{cfg['head_weight_bits_deployed']})")
    print(f"Backbone params: {bb['params']:,} | Flash: {fmt_kb(bb['flash_kB'])}")
    print(f"Proj params:     {pj['params']:,} | Flash: {fmt_kb(pj['flash_kB'])}")
    print(f"Backbone+Proj MACs / sample: {rt['macs_per_sample_M']:.2f} M")
    print(f"Peak activation (single tensor): {fmt_kb(rt['peak_activation_kB'])} ({fmt_mb(rt['peak_activation_MB'])})")
    print(f"Head params: {head['params']:,} | Flash INT8: {head['flash_kB_INT8']:.1f} kB | Flash FP32: {head['flash_kB_FP32']:.1f} kB | MACs/sample: {head['infer_macs_per_sample_M']:.3f} M")
    print(f"--- Deployed total ---")
    print(f"Flash total: {fmt_kb(tot['flash_kB'])}")
    print(f"MACs total / sample (infer): {tot['macs_per_sample_M']:.2f} M")
    print(f"--- Training (head only) ---  optimizer={cfg['optimizer']}")
    print(f"Batch sizes: live={cfg['batch_size_live']}, replay={cfg['batch_size_replay']} (replay_enabled={cfg['replay_enabled']})")
    print(f"Head train state (FP32 est): {head['train_state_kB_FP32_est']:.1f} kB | Param grad bytes (est): {tr['head_param_grad_bytes_est']:,}")
    print(f"Head activations / step: live={tr['train_activations_bytes_head_live']:,} B, replay={tr['train_activations_bytes_head_replay']:,} B, total={tr['train_activations_bytes_head_total']:,} B ({tr['train_activations_kB_head_total']:.2f} kB)")
    print(f"Train MACs / step: {tr['train_macs_per_step_M']:.2f} M")
    print(f"--- Replay storage ---  feat_dim={rp['feat_dim']}, buffer={cfg['replay_size']}, bits/sample={cfg['replay_bits']}")
    print(f"Latent per sample: {rp['latent_per_sample_bytes']} B | Latent total: {rp['latent_total_kB']:.1f} kB")
    if cfg['include_replay_scores']:
        print(f"Scores total: {fmt_kb(rp['scores_total_bytes']/1024)}")
    if cfg['include_perchannel_scales']:
        print(f"Per-channel scales: {fmt_kb(rp['scales_bytes']/1024)}")
    if cfg['num_seen_classes']:
        print(f"Per-class running stats: {fmt_kb(rp['per_class_stats_bytes']/1024)}")
    print(f"Replay total: {fmt_kb(rp['replay_total_kB'])}")
    print(f"TDM buffers (mask + cwi*): {tdm['kB']:.1f} kB")

def save_tinyml_report_csv(report: dict, out_csv: str):
    rows = []
    for section, vals in report.items():
        if isinstance(vals, dict):
            for k, v in vals.items():
                rows.append({"Section": section, "Metric": k, "Value": v})
        else:
            rows.append({"Section": "root", "Metric": section, "Value": vals})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[metrics] TinyML profile saved to {out_csv}")