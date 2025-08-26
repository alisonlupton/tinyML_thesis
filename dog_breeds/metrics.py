# metrics.py
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, Tuple, Optional

# ---------- Core hook-based profiler over backbone + proj ----------
class _FrozenFeatExtractor(nn.Module):
    """Expose z = proj(_features(x)) for profiling."""
    def __init__(self, cil_model: nn.Module):
        super().__init__()
        self.model = cil_model      # keep full model (uses quant path inside)
        self.proj = cil_model.proj  # same proj module you train
    def forward(self, x):
        feats = self.model._features(x)   # (N, D_backbone) e.g., 160
        z = self.proj(feats)              # (N, feat_dim)   e.g., 128
        return z

@torch.no_grad()
def _profile_forward(model: nn.Module,
                     input_size: Tuple[int,int,int,int],
                     activation_bits: int = 8,
                     include_elementwise: bool = False):
    """
    Returns:
      macs_total, peak_act_bytes, per-module dicts {name->value}
    Counts Conv2d/Linear MACs; optional 1 op/elem for BN/ReLU/Pool.
    Peak activation = largest single output tensor (assuming single-buffer).
    """
    device = next(model.parameters()).device
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
            # params (module-local)
            params_per_module[name] += num_params_local(mod)

            macs = 0.0
            if isinstance(mod, nn.Conv2d):
                n, cout, h, w = y.shape
                cin = mod.in_channels
                kh, kw = mod.kernel_size
                groups = mod.groups
                macs = n * h * w * cout * (cin // groups) * kh * kw
            elif isinstance(mod, nn.Linear):
                n = y.shape[0]
                macs = n * mod.in_features * mod.out_features
            elif include_elementwise and isinstance(mod, (nn.BatchNorm2d, nn.ReLU, nn.ReLU6, nn.SiLU, nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.AvgPool2d)):
                macs = y.numel()

            macs_per_module[name] += macs

            # activation size (approx: single output tensor)
            out_bytes = y.numel() * (activation_bits // 8)
            peak_activation_bytes = max(peak_activation_bytes, out_bytes)
            peak_act_bytes_per_module[name] = max(peak_act_bytes_per_module[name], out_bytes)
        return fn

    hooks = []
    # only leaf modules
    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(hook_fn(name, m)))

    x = torch.zeros(*input_size, device=device)
    _ = model(x)

    for h in hooks:
        h.remove()

    macs_total = sum(macs_per_module.values())
    return macs_total, peak_activation_bytes, dict(macs_per_module), dict(params_per_module), dict(peak_act_bytes_per_module)

# ---------- Head / buffers / replay sizing ----------
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

    # Optimizer state (Adam: weights + m + v), SGD(mom): weights + mom
    if optimizer.lower() == "adam":
        train_state_bytes = params * 3 * (train_dtype_bits // 8)
    elif optimizer.lower() in ("sgd", "momentum", "sgd_momentum", "sgdm"):
        train_state_bytes = params * 2 * (train_dtype_bits // 8)
    else:
        train_state_bytes = params * (train_dtype_bits // 8)  # conservative

    return {
        "params": params,
        "flash_kB_INT8": flash_kB_int8,
        "flash_kB_FP32": flash_kB_fp32,
        "infer_macs_per_sample": macs_infer,
        "infer_macs_per_sample_M": macs_infer / 1e6,
        "train_state_kB_FP32_est": train_state_bytes / 1024,
        "act_bytes_per_sample_INFER": feat_dim * (infer_dtype_bits // 8),
        "act_bytes_per_sample_TRAIN": feat_dim * (train_dtype_bits // 8),
    }

def _tdm_buffers_bytes(head_module) -> int:
    """
    Accounts for mask, cwi, cwi_curr, cwi_mem (same shape as weight).
    """
    W = head_module.linear.weight
    elems = W.numel()
    # mask is float tensor in your code; if you later store as bool, divide by 4.
    bytes_mask = elems * 4
    bytes_cwi_pack = elems * 4 * 3  # cwi, cwi_curr, cwi_mem
    return bytes_mask + bytes_cwi_pack

# ---------- Public API ----------
@torch.no_grad()
def profile_tinyml_cil(model: nn.Module,
                       input_size: Tuple[int,int,int,int] = (1,3,160,160),
                       activation_bits_infer: int = 8,
                       activation_bits_train: int = 32,
                       weight_bits_deployed: int = 8,
                       batch_size_train: int = 32,
                       replay_size: int = 0,
                       replay_bits: int = 8,
                       optimizer: str = "adam",
                       include_elementwise: bool = False) -> Dict:
    """
    Profiles a CIL model with frozen backbone:
      - backbone+gap+proj MACs & peak activation (inference path)
      - params & flash (backbone+proj and head)
      - head training memory (params + optimizer states)
      - replay buffer memory (latent-based)
      - TDM buffers (mask + cwi*)
    Expects model to have attributes: backbone, gap, proj, head (nn.Linear inside).
    """
    device = next(model.parameters()).device
    model.eval()

    # expose frozen feature extractor (backbone->gap->proj)
    feat_dim = model.head.in_dim
    num_classes = model.head.out_dim

    frozen = _FrozenFeatExtractor(model).to(device).eval()

    # Inference path profile (backbone+proj)
    macs_total, peak_act_bytes, per_macs, per_params_local, per_peak_act = _profile_forward(
        frozen, input_size=input_size, activation_bits=activation_bits_infer, include_elementwise=include_elementwise
    )

    # Params/flash for backbone+proj only (use module tree)
    params_backbone_proj = sum(p.numel() for n, p in frozen.named_parameters())
    flash_backbone_proj_int8_kB = params_backbone_proj / 1024
    flash_backbone_proj_fp32_kB = (params_backbone_proj * 4) / 1024

    # Head stats (linear only; TDM head wraps nn.Linear)
    head_lin: nn.Linear = model.head.linear
    head_stat = _head_resources(
        head_lin, feat_dim, num_classes,
        train_dtype_bits=32, infer_dtype_bits=activation_bits_infer, optimizer=optimizer
    )

    # Deployed flash total
    flash_total_int8_kB = flash_backbone_proj_int8_kB + head_stat["flash_kB_INT8"]
    flash_total_fp32_kB = flash_backbone_proj_fp32_kB + head_stat["flash_kB_FP32"]

    # Replay buffer footprint (latent only; add a small per-sample header if you like)
    # If you store INT8 latents length=feat_dim, bytes/sample = feat_dim * (replay_bits/8).
    replay_per_sample_bytes = feat_dim * (replay_bits // 8)
    replay_total_bytes = replay_size * replay_per_sample_bytes

    # Training activations (head-only train): batch_size * feat_dim * 4 bytes (FP32)
    train_acts_bytes = batch_size_train * feat_dim * 4

    # TDM buffers (mask + cwi + cwi_curr + cwi_mem)
    tdm_buffers_bytes = _tdm_buffers_bytes(model.head)

    # Compose report
    def kb(x): return x / 1024
    def mb(x): return x / (1024*1024)

    report = {
        "config": {
            "input_size": input_size,
            "activation_bits_infer": activation_bits_infer,
            "activation_bits_train": activation_bits_train,
            "weight_bits_deployed": weight_bits_deployed,
            "batch_size_train": batch_size_train,
            "replay_size": replay_size,
            "replay_bits": replay_bits,
            "optimizer": optimizer,
        },
        "backbone_proj": {
            "params": params_backbone_proj,
            "flash_INT8_kB": flash_backbone_proj_int8_kB,
            "flash_FP32_kB": flash_backbone_proj_fp32_kB,
            "macs_per_sample": macs_total,
            "macs_per_sample_M": macs_total / 1e6,
            "peak_activation_bytes": peak_act_bytes,
            "peak_activation_kB": kb(peak_act_bytes),
            "peak_activation_MB": mb(peak_act_bytes),
            "per_module_macs": dict(sorted(per_macs.items(), key=lambda kv: kv[1], reverse=True)),
            "per_module_peak_act_bytes": dict(sorted(per_peak_act.items(), key=lambda kv: kv[1], reverse=True)),
        },
        "head": head_stat,
        "deployed_total": {
            "flash_INT8_kB": flash_total_int8_kB,
            "flash_FP32_kB": flash_total_fp32_kB,
            "macs_per_sample_M": (macs_total / 1e6) + head_stat["infer_macs_per_sample_M"],
        },
        "training_overheads": {
            "head_train_state_kB_FP32_est": head_stat["train_state_kB_FP32_est"],
            "train_activations_bytes_head_only": train_acts_bytes,
            "train_activations_kB_head_only": kb(train_acts_bytes),
        },
        "replay": {
            "feat_dim": feat_dim,
            "replay_per_sample_bytes": replay_per_sample_bytes,
            "replay_per_sample_kB": kb(replay_per_sample_bytes),
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
    tot = report["deployed_total"]
    bb = report["backbone_proj"]
    head = report["head"]
    tr = report["training_overheads"]
    rp = report["replay"]
    tdm = report["tdm_buffers"]

    print("\n=== TinyML Profile ===")
    print(f"Input: {report['config']['input_size']}, infer_act_bits={report['config']['activation_bits_infer']}, deployed_w_bits={report['config']['weight_bits_deployed']}")
    print(f"Backbone+Proj params: {bb['params']:,} | Flash INT8: {fmt_kb(bb['flash_INT8_kB'])} | Flash FP32: {fmt_kb(bb['flash_FP32_kB'])}")
    print(f"Backbone+Proj MACs / sample: {bb['macs_per_sample_M']:.2f} M")
    print(f"Peak activation (single tensor): {fmt_kb(bb['peak_activation_kB'])} ({fmt_mb(bb['peak_activation_MB'])})")
    print(f"Head params: {head['params']:,} | Flash INT8: {head['flash_kB_INT8']:.1f} kB | MACs/sample: {head['infer_macs_per_sample_M']:.3f} M")
    print(f"--- Deployed total ---")
    print(f"Flash INT8 total: {fmt_kb(tot['flash_INT8_kB'])} | FP32 total: {fmt_kb(tot['flash_FP32_kB'])}")
    print(f"MACs total / sample: {tot['macs_per_sample_M']:.2f} M")
    print(f"--- Training (head only) ---  optimizer={report['config']['optimizer']}")
    print(f"Head train state (FP32 est): {tr['head_train_state_kB_FP32_est']:.1f} kB | Train activations/head/batch: {tr['train_activations_kB_head_only']:.2f} kB")
    print(f"--- Replay ---  feat_dim={rp['feat_dim']}, buffer={report['config']['replay_size']}, bits/sample={report['config']['replay_bits']}")
    print(f"Replay per sample: {rp['replay_per_sample_kB']:.3f} kB | Total: {rp['replay_total_kB']:.1f} kB")
    print(f"TDM buffers (mask + cwi*): {tdm['kB']:.1f} kB")