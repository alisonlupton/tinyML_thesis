# metrics.py
import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List, Optional
import torch
import torch.nn as nn

# ---------- helpers ----------
def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

def _nz_frac_from_head(head) -> float:
    # prefer binary pruning mask if present
    if hasattr(head, "mask") and head.mask is not None:
        m = head.mask
        return float((m != 0).float().mean().item())
    W = head.linear.weight.data
    return float((W.abs() > 0).float().mean().item())

def _fmt_bytes(n: float) -> str:
    u = ["B","KB","MB","GB","TB"]
    i = 0
    while n >= 1024 and i < len(u)-1:
        n /= 1024.0; i += 1
    return f"{n:.2f} {u[i]}"

def _conv2d_out_hw(H: int, W: int, k: int, s: int, p: int, d: int) -> Tuple[int,int]:
    Hout = math.floor((H + 2*p - d*(k-1) - 1)/s + 1)
    Wout = math.floor((W + 2*p - d*(k-1) - 1)/s + 1)
    return Hout, Wout

def _scan_conv2d_specs(backbone: nn.Module):
    """Return list of (Cin, Cout, k, s, p, d) in backbone order for Conv2d only."""
    specs = []
    for m in backbone.modules():
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
            s = m.stride if isinstance(m.stride, int) else m.stride[0]
            p = m.padding if isinstance(m.padding, int) else m.padding[0]
            d = m.dilation if isinstance(m.dilation, int) else m.dilation[0]
            specs.append((m.in_channels, m.out_channels, int(k), int(s), int(p), int(d)))
    return specs

@dataclass
class CILProfile2D:
    # params
    backbone_params: int
    head_params: int
    total_params: int
    head_nonzero_frac: float

    # deployment memory (int8 weights, unless cfg overrides)
    deployed_backbone_MB: float
    deployed_head_dense_MB: float
    deployed_head_sparse_MB: float
    deployed_total_dense_MB: float
    deployed_total_sparse_MB: float

    # training SRAM (head-only training, fp32 unless cfg overrides)
    head_weights_MB: float
    head_grads_MB: float
    head_opt_MB: float
    cwi_buffers_MB: float
    mask_MB: float
    teacher_snapshot_MB: float
    replay_total_MB: float
    replay_per_sample_B: int
    activations_train_MB: float  # head-only training → features buffer

    # inference SRAM (peak working set for B=1)
    activations_infer_MB: float

    # compute
    macs_backbone_per_sample_M: float
    macs_proj_per_sample_M: float
    macs_head_per_sample_dense_M: float
    macs_head_per_sample_sparse_M: float
    macs_total_fwd_sparse_per_sample_M: float
    macs_train_per_batch_sparse_M: float

    # shapes
    final_feat_hw: Tuple[int,int]
    proj_in_channels: int
    feat_dim: int
    head_out_dim: int

    def pretty(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update({
            "deployed_total_sparse_pretty": _fmt_bytes(self.deployed_total_sparse_MB * 1024**2),
            "replay_total_pretty": _fmt_bytes(self.replay_total_MB * 1024**2),
            "activations_train_pretty": _fmt_bytes(self.activations_train_MB * 1024**2),
            "activations_infer_pretty": _fmt_bytes(self.activations_infer_MB * 1024**2),
        })
        return d

def profile_cil_resources_2d(cfg: Dict[str, Any], model: nn.Module) -> CILProfile2D:
    """
    Profiles your EXACT 2D pipeline (frozen backbone + linear head) using ONLY values from cfg.
    Call this before tasks (and optionally inside each task after you update cfg['kd_prev_rows']).
    """
    assert hasattr(model, "backbone") and hasattr(model, "gap") and hasattr(model, "proj") and hasattr(model, "head")

    # -------- read knobs from cfg (with fallbacks) --------
    img_size                 = int(cfg['img_size'])
    batch_size_train         = int(cfg['CIL_batch_size_train'])
    batch_size_infer         = int(cfg.get("CIL_batch_size_test", 32))
    replay_size              = int(cfg.get("buffer_size", 3000))
    optimizer_kind           = str(cfg.get("optimizer", "adam"))
    deployed_bits_backbone   = int(cfg.get("deployed_bits_backbone", 32))
    deployed_bits_classifier = int(cfg.get("deployed_bits_classifier", 32))
    training_bits_classifier = int(cfg.get("training_bits_classifier", 32))
    replay_bits              = int(cfg.get("replay_bits", 8))
    activation_bits_train    = int(cfg.get("activation_bits_train", 32))  # you are training in PyTorch; 32 is safe
    activation_bits_infer    = int(cfg.get("activation_bits_infer", 32))
    replay_batch_size        = int(cfg.get("replay_batch_size", 1))
    kd_prev_rows             = int(cfg.get("kd_prev_rows", 0))     # set this in-loop if KD active
    kd_enabled               = bool(cfg.get("kd_enabled", True))
    autograd_grad_for_cwi    = bool(cfg.get("autograd_grad_for_cwi", True))
    metrics_path             = cfg.get("metrics_path", None)

    # ----- params -----
    bb_params   = _count_params(model.backbone) + _count_params(model.gap) + _count_params(model.proj)
    head_params = _count_params(model.head)
    total_params = bb_params + head_params

    in_dim  = int(model.head.in_dim)   # feat_dim
    out_dim = int(model.head.out_dim)
    nz_frac = _nz_frac_from_head(model.head)

    # ----- deployment memory (assume int8 for both unless overridden) -----
    bb_MB          = bb_params   * (deployed_bits_backbone   / 8) / 1_048_576
    head_MB_dense  = head_params * (deployed_bits_classifier / 8) / 1_048_576
    # sparse head: weight zeros removed, bias kept
    num_w = model.head.linear.weight.numel()
    num_b = model.head.linear.bias.numel() if model.head.linear.bias is not None else 0
    head_MB_sparse = ((num_w * nz_frac) + num_b) * (deployed_bits_classifier / 8) / 1_048_576
    tot_MB_dense   = bb_MB + head_MB_dense
    tot_MB_sparse  = bb_MB + head_MB_sparse

    # ----- compute MACs per sample (follow your real backbone) -----
    H = W = int(img_size)
    conv_specs = _scan_conv2d_specs(model.backbone)
    macs_bb = 0
    peak_act_elems_infer = 3 * H * W  # input as a candidate
    last_out_C = None

    for Cin, Cout, k, s, p, d in conv_specs:
        H, W = _conv2d_out_hw(H, W, k, s, p, d)
        macs_bb += H * W * Cout * (Cin * k * k)
        peak_act_elems_infer = max(peak_act_elems_infer, Cout * H * W)
        last_out_C = Cout

    final_hw = (H, W)
    # GAP → (C, 1, 1), then proj last_out_C → feat_dim
    proj_in = int(last_out_C if last_out_C is not None else 256)
    macs_proj = proj_in * in_dim
    macs_head_dense  = in_dim * out_dim
    macs_head_sparse = int(in_dim * out_dim * nz_frac)
    macs_total_sparse = macs_bb + macs_proj + macs_head_sparse

    # ----- training MACs per batch (head-only training) -----
    B = int(batch_size_train)
    R = int(max(0, replay_batch_size))
    # live forward: backbone+proj+head (sparse)
    macs_live  = B * (macs_bb + macs_proj + macs_head_sparse)
    # replay forward: head only
    macs_replay = R * macs_head_sparse
    # backward equivalents on head
    head_bwd_equiv = 1  # final loss.backward
    if autograd_grad_for_cwi:
        head_bwd_equiv += 1            # extra grad(live)
        if R > 0:
            head_bwd_equiv += 1        # extra grad(replay)
    macs_head_bwd = head_bwd_equiv * (B * macs_head_sparse)
    # KD/DDR: extra backbone+proj on live (feats recomputed) + partial head over prev rows
    macs_kd = 0
    if kd_enabled and kd_prev_rows > 0:
        macs_kd += B * (macs_bb + macs_proj)
        macs_kd += B * (in_dim * kd_prev_rows)
    macs_train_batch_sparse = macs_live + macs_replay + macs_head_bwd + macs_kd

    # ----- SRAM: head-only training persistent (fp32 unless overridden) -----
    bpp = training_bits_classifier / 8
    w_bytes   = head_params * bpp
    g_bytes   = head_params * bpp
    if optimizer_kind.lower() == "adam":
        opt_bytes = 2 * head_params * bpp
    elif optimizer_kind.lower() in ("sgd","sgd_momentum","momentum"):
        opt_bytes = 1 * head_params * bpp
    else:
        opt_bytes = 0

    # CWI buffers (cwi, cwi_curr, cwi_mem) + mask (kept fp32 in your code)
    cwi_bytes  = 3 * num_w * bpp
    mask_bytes = num_w * bpp

    # teacher snapshot (if KD active)
    teacher_bytes = 0
    if kd_enabled and kd_prev_rows > 0:
        teacher_bytes = (kd_prev_rows * in_dim + kd_prev_rows) * bpp

    # replay memory (INT8 + 4-byte scale + 2-byte label) per latent
    replay_per = int(in_dim * (replay_bits/8) + 4 + 2)
    replay_total = replay_size * replay_per

    # training activations (head-only): features buffer (B, in_dim)
    act_train_bytes = B * in_dim * (activation_bits_train / 8)

    # inference activations (B=1): peak across conv outputs (already tracked)
    act_infer_bytes = peak_act_elems_infer * (activation_bits_infer / 8)

    report = CILProfile2D(
        backbone_params=bb_params,
        head_params=head_params,
        total_params=total_params,
        head_nonzero_frac=nz_frac,

        deployed_backbone_MB=bb_MB,
        deployed_head_dense_MB=head_MB_dense,
        deployed_head_sparse_MB=head_MB_sparse,
        deployed_total_dense_MB=tot_MB_dense,
        deployed_total_sparse_MB=tot_MB_sparse,

        head_weights_MB=w_bytes/1_048_576,
        head_grads_MB=g_bytes/1_048_576,
        head_opt_MB=opt_bytes/1_048_576,
        cwi_buffers_MB=cwi_bytes/1_048_576,
        mask_MB=mask_bytes/1_048_576,
        teacher_snapshot_MB=teacher_bytes/1_048_576,
        replay_total_MB=replay_total/1_048_576,
        replay_per_sample_B=replay_per,
        activations_train_MB=act_train_bytes/1_048_576,

        activations_infer_MB=act_infer_bytes/1_048_576,

        macs_backbone_per_sample_M=macs_bb/1_000_000,
        macs_proj_per_sample_M=macs_proj/1_000_000,
        macs_head_per_sample_dense_M=macs_head_dense/1_000_000,
        macs_head_per_sample_sparse_M=macs_head_sparse/1_000_000,
        macs_total_fwd_sparse_per_sample_M=macs_total_sparse/1_000_000,
        macs_train_per_batch_sparse_M=macs_train_batch_sparse/1_000_000,

        final_feat_hw=final_hw,
        proj_in_channels=proj_in,
        feat_dim=in_dim,
        head_out_dim=out_dim,
    )

    # optional save
    if metrics_path:
        d = report.pretty()
        with open(metrics_path, "w") as f:
            for k, v in d.items():
                f.write(f"{k}: {v}\n")
        print(f"[metrics] Saved 2D CIL profile to {metrics_path}")
    return report