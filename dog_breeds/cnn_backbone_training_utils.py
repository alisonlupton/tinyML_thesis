#cnn_backbone_training_utils.py
import torch
import copy
import torch.nn as nn
from dataclasses import dataclass
from utils import build_tensors, ClassRegistry
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.ao.quantization import get_default_qconfig, fuse_modules, prepare, convert, QuantStub, DeQuantStub
from torchvision.transforms import InterpolationMode as I


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
    img = cfg['img_size']
    mean, std = cfg['mean_tf'], cfg['std_tf']  #from teacher meta

    #=== Inference / Validation (match offline "deploy_tf_strong") ===
    deploy_tf = transforms.Compose([
        transforms.Resize(int(round(1.10*img)), interpolation=I.BILINEAR),  #e.g., 176 for 160
        transforms.CenterCrop(img),                 #160x160
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    #=== Backbone/distill/replay (no aug) ===
    backbone_train_tf = deploy_tf
    
    
    #=== CIL on-device training (lite aug) ===
    #Cheap: tiny translate via RandomCrop + optional flip
    CIL_train_tf = transforms.Compose([
        transforms.Resize(int(round(1.05*img)), interpolation=I.BILINEAR),  #smaller upsample than deploy to save a bit
        transforms.RandomCrop(img),                 #cheap spatial jitter
        transforms.RandomHorizontalFlip(p=0.5),     #essentially free
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return backbone_train_tf, CIL_train_tf, deploy_tf

def load_data_cnn_backbone(cfg, backbone_dogs, dog_data):
    
    backbone_df = dog_data[dog_data["gid"].isin(backbone_dogs)].copy()

    #Stable local label mapping (0..B-1)
    local_ids = sorted(backbone_dogs)
    gid_to_local = {g:i for i,g in enumerate(local_ids)}
    backbone_df["local"] = [gid_to_local[int(g)] for g in backbone_df["gid"].tolist()]

    df_train= backbone_df[backbone_df["split"]=="train"].copy()
    df_val  = backbone_df[backbone_df["split"]=="test"].copy()


    backbone_train_tf, CIL_train_tf, val_tf = _make_transforms(cfg)
    
    X_train, y_train = build_tensors(df_train, backbone_train_tf)
    X_val, y_val = build_tensors(df_val, val_tf)
    
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("No backbone training data found!")

    return BackboneData(X_train, y_train, X_val, y_val, gid_to_local, local_ids, backbone_train_tf, CIL_train_tf, val_tf)

@torch.no_grad()
def eval_new_classes_on_backbone(
    model,
    train_loader,       #use deploy_tf here for fair comparison
    test_loader,        #use deploy_tf here too
    device,
    num_classes=None,
    use_proj=True,
) -> tuple[float, dict[int, float]]:
    """
    Prototype (NCM) eval using the same feature stage as deployment.
    For fair apples-to-apples, pass your quantized CIL model here and
    use deterministic deploy transforms for both TRAIN (prototype build)
    and TEST (classification).
    """
    model.eval()

    def feat_fn(x):
        f = model._features(x)                  #(N, 160) will be quantized path if attached it
        if use_proj and hasattr(model, "proj"):
            f = model.proj(f)                   #(N, feat_dim); Dropout off in eval()
        return torch.nn.functional.normalize(f, dim=1)

    #--- Build prototypes on TRAIN (deterministic transforms) ---
    sums, counts, D = {}, {}, None
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        z = feat_fn(xb)
        if D is None: D = z.shape[1]
        for c in yb.unique().tolist():
            c = int(c)
            m = (yb == c)
            sums[c]   = sums.get(c, torch.zeros(D, device=device)) + z[m].sum(0)
            counts[c] = counts.get(c, 0) + int(m.sum().item())

    if num_classes is None:
        num_classes = (max(counts.keys()) + 1) if counts else 0
    if D is None:
        return 0.0, {c: 0.0 for c in range(num_classes)}

    protos = torch.zeros(num_classes, D, device=device)
    seen = torch.zeros(num_classes, dtype=torch.bool, device=device)
    for c, s in sums.items():
        if counts[c] > 0:
            protos[c] = s / counts[c]
            seen[c] = True
    protos = torch.nn.functional.normalize(protos, dim=1)

    #--- Classify TEST by cosine to prototypes ---
    total = correct = 0
    cls_tot = [0] * num_classes
    cls_cor = [0] * num_classes

    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        z = feat_fn(xb)
        sims = z @ protos.t()
        if (~seen).any():
            sims[:, ~seen] = -1e9
        pred = sims.argmax(1)

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

@torch.no_grad()

def eval_on_fixed_gids(
    model,
    loader,                 #yields (xb, y_local) where y_local [0..len(fixed_gids)-1]
    registry,               #ClassRegistry that *may* know only a subset of fixed_gids
    fixed_gids,             #list[int] global IDs in the same order used to build y_local
    device,
):
    """
    Evaluate with the TDM head on a fixed set of global IDs (gids), even if some
    of them are unseen. For unseen gids, logits are set to -inf so they cant win.
    Returns: overall %, {gid: acc%}
    """
    model.eval()

    #Map fixed gids -> current head rows (or -1 if unseen)
    rows = []
    known_cols = []   #columns (in fixed_gids order) that are known
    for col, g in enumerate(fixed_gids):
        if int(g) in registry.row_for_gid:
            rows.append(registry.row_for_gid[int(g)])
            known_cols.append(col)
        else:
            rows.append(-1)

    total, correct = 0, 0
    per_tot = {int(g): 0 for g in fixed_gids}
    per_cor = {int(g): 0 for g in fixed_gids}

    for xb, y_local in loader:
        xb = xb.to(device); y_local = y_local.to(device)

        #features proj (match training path)
        model.backbone.eval(); model.proj.eval()
        feats = model._features(xb)
        z = model.proj(feats)

        #Allocate full logits [B, K] where K=len(fixed_gids)
        B, K = z.size(0), len(fixed_gids)
        logits_full = torch.full((B, K), -1e9, device=device)  #-inf for unseen columns

        #If we have any known rows, compute them and place into their columns
        if len(known_cols) > 0:
            known_rows = [r for r in rows if r >= 0]
            logits_known = model.head.forward_rows(z, known_rows)  #[B, |known|]
            logits_full[:, known_cols] = logits_known

        pred_local = logits_full.argmax(dim=1)  #predictions in *local-to-fixed* index space

        total += y_local.size(0)
        correct += (pred_local == y_local).sum().item()

        #per-class (indexed by gid)
        for col, gid in enumerate(fixed_gids):
            m = (y_local == col)
            n = int(m.sum().item())
            if n > 0:
                per_tot[int(gid)] += n
                per_cor[int(gid)] += int((pred_local[m] == y_local[m]).sum().item())

    overall = 100.0 * correct / max(1, total)
    per_class = {
        int(g): (100.0 * per_cor[int(g)] / per_tot[int(g)] if per_tot[int(g)] > 0 else 0.0)
        for g in fixed_gids
    }
    return overall, per_class

#Quantisation Helpers

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
        self.gap = nn.AdaptiveAvgPool2d(1)  #do GAP in float for portability
    def forward(self, x):
        x = self.quant(x)
        x = self.backbone(x)
        x = self.dequant(x)   #back to float
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
    h = self.quant_backbone(x)  #float, shape (N,160,1,1)
    return h.flatten(1)         #(N,160)

@torch.no_grad()
def compute_perchannel_scales(calib_loader, cil_model, pct=99.9, eps=1e-8):
    feats_all = []
    for i, (images, _) in enumerate(calib_loader):
        f = cil_model._features(images)     #[B, C] FP32 (after dequant + GAP)
        feats_all.append(f)
        if i >= 20: break                   #~20 batches is plenty
    F = torch.cat(feats_all, dim=0)         #[N, C]
    hi = torch.quantile(F.abs(), q=pct/100.0, dim=0)   #[C]
    max_abs = torch.clamp(hi, min=eps)
    scales = (max_abs / 127.0).contiguous()           #symmetric, zp=0
    return scales

