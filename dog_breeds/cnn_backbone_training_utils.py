# cnn_backbone_training_utils.py
import torch
import copy
import torch.nn as nn
from dataclasses import dataclass
from utils import build_tensors
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Optional
from torch.ao.quantization import get_default_qconfig, fuse_modules, prepare, convert, QuantStub, DeQuantStub


@dataclass
class BackboneData:
    """Container for backbone training and validation data"""
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    gid_to_local: dict
    local_to_gid: list
    backbone_train_tf: list
    CIL_train_tf: list
    val_tf: list
    
    
def _make_transforms(cfg):
    deploy_tf = transforms.Compose([
    transforms.Resize((cfg['img_size'], cfg['img_size'])),   # match student + MCU
    transforms.ToTensor(),
    transforms.Normalize(cfg['mean_tf'], cfg['std_tf']),
])
    backbone_train_tf = deploy_tf
    CIL_train_tf = transforms.Compose([
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(cfg['mean_tf'], cfg['std_tf']),
    ])
    val_tf = deploy_tf
    return backbone_train_tf, CIL_train_tf, val_tf

def load_data_cnn_backbone(cfg, backbone_dogs, dog_data):
    
    backbone_df = dog_data[dog_data["gid"].isin(backbone_dogs)].copy()

    # Stable local label mapping (0..B-1)
    local_ids = sorted(backbone_dogs)
    gid_to_local = {g:i for i,g in enumerate(local_ids)}
    backbone_df["local"] = [gid_to_local[int(g)] for g in backbone_df["gid"].tolist()]

    df_train= backbone_df[backbone_df["split"]=="train"].copy()
    df_val  = backbone_df[backbone_df["split"]=="test"].copy()


    backbone_train_tf, CIL_train_tf,  val_tf = _make_transforms(cfg)
    
    X_train, y_train = build_tensors(df_train, backbone_train_tf)
    X_val, y_val = build_tensors(df_val, val_tf)
    
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("No backbone training data found!")

    return BackboneData(X_train, y_train, X_val, y_val, gid_to_local, local_ids, backbone_train_tf, CIL_train_tf, val_tf)

@torch.no_grad()
def eval_new_classes_on_backbone(model,
    train_loader: DataLoader,
    test_loader: DataLoader, 
    device: torch.device, 
    num_classes: Optional[int] = None,
) -> Tuple[float, Dict[int, float]]:
    """
    Evaluate a frozen backbone by building class prototypes from TRAIN features
    and classifying TEST features by cosine similarity to those prototypes.

    Assumes labels in loaders are *local* class indices [0..K-1] for the task.
    Returns: (overall_acc_percent, per_class_acc_percent_dict)
    """
    model.eval()

    # --------- 1) Build prototypes from TRAIN features ---------
    feats_sums = {}
    counts = {}
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        z = model._features(xb)                       # (N, D)
        z = torch.nn.functional.normalize(z, dim=1)   # cosine-ready
        for c in yb.unique().tolist():
            c = int(c)
            mask = (yb == c)
            feats_sums[c] = feats_sums.get(c, 0) + z[mask].sum(dim=0)
            counts[c] = counts.get(c, 0) + int(mask.sum().item())

    # If caller didn't pass num_classes, infer from what we saw
    if num_classes is None:
        num_classes = (max(counts.keys()) + 1) if counts else 0

    # Build prototype matrix [K, D], missing classes get zero vector (ignored at test)
    D = next(iter(feats_sums.values())).numel() if feats_sums else model.head.in_dim
    protos = torch.zeros(num_classes, D, device=device)
    seen_mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
    for c, s in feats_sums.items():
        if counts[c] > 0:
            protos[c] = s / counts[c]
            seen_mask[c] = True
    protos = torch.nn.functional.normalize(protos, dim=1)  # [K, D] unit vectors

    # --------- 2) Classify TEST features by cosine to prototypes ---------
    total = 0
    correct = 0
    cls_tot = [0] * num_classes
    cls_cor = [0] * num_classes

    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        z = model._features(xb)                        # [N, D]
        z = torch.nn.functional.normalize(z, dim=1)    # unit
        # cosine sim = dot product for normalized vectors
        sims = z @ protos.t()                          # [N, K]
        # For classes without a prototype, make them unselectable
        if (~seen_mask).any():
            sims[:, ~seen_mask] = -1e9
        pred = sims.argmax(dim=1)

        total += yb.size(0)
        correct += (pred == yb).sum().item()
        for c in range(num_classes):
            m = (yb == c)
            n = int(m.sum().item())
            if n > 0:
                cls_tot[c] += n
                cls_cor[c] += int((pred[m] == yb[m]).sum().item())

    overall = 100.0 * correct / max(1, total)
    per_class = {c: (100.0 * cls_cor[c] / cls_tot[c] if cls_tot[c] > 0 else 0.0)
                 for c in range(num_classes)}
    return overall, per_class

# Quantisation Helpers

def _fuse_cbr_blocks_inplace(backbone: nn.Sequential):
    for _, m in backbone.named_children():
        if isinstance(m, nn.Sequential) and len(m) >= 3:
            if isinstance(m[0], nn.Conv2d) and isinstance(m[1], nn.BatchNorm2d) and isinstance(m[2], nn.ReLU):
                fuse_modules(m, ['0','1','2'], inplace=True)

class QuantBackboneWrapper(nn.Module):
    def __init__(self, backbone: nn.Sequential):
        super().__init__()
        self.quant = QuantStub()
        self.backbone = copy.deepcopy(backbone)
        self.dequant = DeQuantStub()
        self.gap = nn.AdaptiveAvgPool2d(1)  # do GAP in float for portability
    def forward(self, x):
        x = self.quant(x)
        x = self.backbone(x)
        x = self.dequant(x)   # back to float
        x = self.gap(x)
        return x

@torch.no_grad()
def quantise_frozen_backbone(student_backbone: nn.Sequential, calib_loader):
    qb = QuantBackboneWrapper(student_backbone).to('cpu').eval()
    _fuse_cbr_blocks_inplace(qb.backbone)
    qb.qconfig = get_default_qconfig('qnnpack')
    qb_prepared = prepare(qb, inplace=False)
    for i, (images, _) in enumerate(calib_loader):
        _ = qb_prepared(images.to('cpu'))
        if i >= 20: break
    qb_int8 = convert(qb_prepared, inplace=False).eval()
    return qb_int8

def features_quantized(self, x):
    h = self.quant_backbone(x)  # float, shape (N,160,1,1)
    return h.flatten(1)         # (N,160)

@torch.no_grad()
def compute_perchannel_scales(calib_loader, cil_model, pct=99.9, eps=1e-8):
    feats_all = []
    for i, (images, _) in enumerate(calib_loader):
        f = cil_model._features(images)     # [B, C] FP32 (after dequant + GAP)
        feats_all.append(f)
        if i >= 20: break                   # ~20 batches is plenty
    F = torch.cat(feats_all, dim=0)         # [N, C]
    hi = torch.quantile(F.abs(), q=pct/100.0, dim=0)   # [C]
    max_abs = torch.clamp(hi, min=eps)
    scales = (max_abs / 127.0).contiguous()           # symmetric, zp=0
    return scales

