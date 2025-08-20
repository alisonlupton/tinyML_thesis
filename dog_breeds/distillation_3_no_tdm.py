# train_student_from_cache_images.py
# Distill teacher → tiny CNN *from images* using only cached targets (no teacher forward).
# 1) Build per-image feature targets using the projector you already trained from caches.
# 2) Train your tiny CNN on images with KD (cached logits) + CE + feature MSE.

import os, json, random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---- import your tiny CNN ----
# expects: class SimplifiedTDMModelCNN(nn.Module) with ._features() and .proj
from models.cnn import SimplifiedTDMModelCNN


from pathlib import Path

def canon_key(p: str) -> str:
    """Make a stable key relative to Images/ directory."""
    parts = Path(p).parts
    if "Images" in parts:
        i = parts.index("Images")
        rel = Path(*parts[i+1:])
    else:
        rel = Path(*parts[-2:])  # fallback: breed/filename
    return rel.as_posix()

# --- helpers -------------------------------------------------
from pathlib import Path

def canon_key_from_path(p: str) -> str:
    """
    Turn a CSV img_path into the canonical relative key used in the cache.
    Tries to strip up to and including 'Images/' if present; otherwise
    falls back to the last 2 path components (breed/file.jpg).
    """
    p = Path(p)
    parts = [*p.parts]
    if "Images" in parts:
        i = parts.index("Images")
        rel = parts[i+1:]
    else:
        rel = parts[-2:]
    return "/".join(rel).replace("\\", "/")

def build_cache_maps(cache: dict):
    keys   = [k.replace("\\","/") for k in cache["keys"]]
    logits = cache["logits"]            # [N, B]
    y      = cache["y_local"]           # [N]
    fts    = cache["feat_targets"]      # [N, feat_dim]  <-- use precomputed targets
    map_logits = {k: logits[i]     for i,k in enumerate(keys)}
    map_y      = {k: int(y[i])     for i,k in enumerate(keys)}
    map_ft     = {k: fts[i]        for i,k in enumerate(keys)}  # feat targets, not 1280-D penults!
    return map_ft, map_logits, map_y
# --- dataset that uses cached teacher targets ----------------
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class DogWithCacheTargets(Dataset):
    def __init__(self, df, split, tfm, gid_to_local,
                 map_logits, map_y, map_ft):
        self.df  = df[df["split"] == split].reset_index(drop=True)
        self.tfm = tfm
        self.g2l = gid_to_local
        self.map_logits = map_logits
        self.map_y      = map_y
        self.map_ft     = map_ft

        # precompute keys & validate coverage
        self.keys = [canon_key_from_path(r.img_path) for _, r in self.df.iterrows()]
        missing = [k for k in self.keys if k not in self.map_logits]
        if missing:
            raise RuntimeError(f"{len(missing)} images missing in cache for split={split}. "
                               f"Example: {missing[0]}")

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r   = self.df.iloc[i]
        img = Image.open(r.img_path).convert("RGB")
        x   = self.tfm(img)
        y   = self.g2l[int(r.gid)]              # local int label
        key = self.keys[i]
        tlog= self.map_logits[key]              # Tensor on CPU
        tft = self.map_ft[key]                  # Tensor on CPU
        return x, y, tlog, tft
# -------------------------
# Config
# -------------------------
CFG = {
    # dataset index you already created
    "index_csv": "../data/dog_breed_data/stanford_dogs_index.csv",
    "index_json": "../data/dog_breed_data/stanford_dogs_index.json",   # not strictly required here
    # teacher meta (for normalization + class order sanity if needed)
    "teacher_meta": "../data/backbone_meta.json",

    # caches from Colab step
    "cache_dir": "../data",
    "train_cache": "train_cache.pt",
    "val_cache":   "val_cache.pt",

    # projector you trained from cached teacher features
    "proj_ckpt": "../data/student_from_cache.pth",

    # output student weights (to plug into your CIL)
    "out_dir": "../data/student_from_images",

    # student model hyperparams (must match your on-device backbone)
    "in_channels": 3,
    "feat_dim": 128,             # same as in your CIL tiny model
    "sparsity_ratio": 0.30,      # head uses this, but we only save backbone+proj
    "drop_backbone": (0.1,0.1,0.2,0.2),
    "drop_proj": 0.3,

    # training
    "seed": 42,
    "img_size": 224,
    "batch_size_train": 128,
    "batch_size_val": 128,
    "epochs": 100,
    "patience": 7,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "label_smoothing": 0.05,

    # distillation losses
    "kd_T": 4.0,
    "alpha_kd": 1.0,     # KL(s||t)
    "alpha_ce": 0.0,     # CE(y_true)
    "mu_feat": 0.5,      # MSE(student_feat, target_feat)
    "use_feat_mse": True,
    "alpha_mse": 0.1,

    # workers
    "num_workers": 0,
}

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CFG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load meta, caches
# -------------------------
meta = json.loads(Path(CFG["teacher_meta"]).read_text())
mean = meta["normalization"]["mean"]; std = meta["normalization"]["std"]
img_size = int(meta["img_size"])
backbone_gids = list(map(int, meta["backbone_gids"]))  # fixed class order B

cache_dir = Path(CFG["cache_dir"])
train_cache = torch.load(cache_dir/CFG["train_cache"], map_location="cpu")
val_cache   = torch.load(cache_dir/CFG["val_cache"],   map_location="cpu")

train_cache["keys"] = [canon_key(k) for k in train_cache["keys"]]
val_cache["keys"]   = [canon_key(k) for k in val_cache["keys"]]


print(f"Train cache: feats={tuple(train_cache['features'].shape)}, logits={tuple(train_cache['logits'].shape)}")
print(f"Val   cache: feats={tuple(val_cache['features'].shape)},   logits={tuple(val_cache['logits'].shape)}")

# -------------------------
# 1) Build feature targets using your projector (dropout OFF)
# -------------------------
saved   = torch.load(CFG["proj_ckpt"], map_location="cpu")  # ../data/student_from_cache.pth
proj_sd = saved["proj_from_teacher_penult"]                 # state_dict of Sequential(Dropout, Linear)

class LinearOnly(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.fc(x)

P = LinearOnly(in_dim=train_cache["features"].shape[1],  # 1280
               out_dim=proj_sd["1.weight"].shape[0])     # feat_dim, e.g. 128
P.fc.weight.data.copy_(proj_sd["1.weight"])
P.fc.bias.data.copy_(proj_sd["1.bias"])
P = P.cpu().eval()

@torch.no_grad()
def add_feat_targets(cache):
    feats = cache["features"].float()          # CPU
    cache["feat_targets"] = P(feats).float()   # CPU -> stays CPU for dataset
    return cache


train_cache = add_feat_targets(train_cache)
val_cache   = add_feat_targets(val_cache)

# -------------------------
# 2) Build image dataset that looks up cached targets by path key
# -------------------------


# RIGHT order: (feats, logits, y)
tr_map_ft, tr_map_logits, tr_map_y = build_cache_maps(train_cache)
va_map_ft, va_map_logits, va_map_y = build_cache_maps(val_cache)

# Build dataframes filtered to backbone_gids (consistent with caches)
df = pd.read_csv(CFG["index_csv"])
df = df[df["gid"].isin(backbone_gids)].copy()

# Stable local mapping following backbone_gids order
gid_to_local = {g:i for i,g in enumerate(backbone_gids)}

# Weak/deploy transforms (match normalization used for caches)
deploy_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# DataLoaders
pin = (device.type == "cuda")
print("Example cache key:", train_cache["keys"][0])
ex = df.query("split=='train'").iloc[0]
print("CSV raw:", ex.img_path)
print("CSV canon:", canon_key(ex.img_path))
ds_tr = DogWithCacheTargets(df, "train", deploy_tf, gid_to_local,
                            tr_map_logits, tr_map_y, tr_map_ft)
ds_va = DogWithCacheTargets(df, "test",  deploy_tf, gid_to_local,
                            va_map_logits, va_map_y, va_map_ft)
dl_tr = DataLoader(ds_tr, batch_size=CFG["batch_size_train"], shuffle=True,
                   num_workers=CFG["num_workers"], pin_memory=pin)
dl_va = DataLoader(ds_va, batch_size=CFG["batch_size_val"], shuffle=False,
                   num_workers=CFG["num_workers"], pin_memory=pin)

# -------------------------
# 3) Build student tiny CNN
# -------------------------
# ---- build student ----
num_classes = len(backbone_gids)
student = SimplifiedTDMModelCNN(
    in_channels=CFG["in_channels"],
    init_num_classes=num_classes,     # ok, but we'll ignore its TDM head
    device=device,
    sparsity_ratio=CFG["sparsity_ratio"],
    feat_dim=CFG["feat_dim"],
).to(device)
# right after building student
def _zero_out_dropout(m):
    if isinstance(m, (nn.Dropout, nn.Dropout2d)):
        m.p = 0.0
student.apply(_zero_out_dropout)

# # after you build `student` and `P` (and move student to device)
# with torch.no_grad():
#     # student.proj = nn.Sequential(Dropout, Linear(128 -> feat_dim))
#     student.proj[1].weight.copy_(P.fc.weight.to(device))
#     student.proj[1].bias.copy_(P.fc.bias.to(device))


# >>> NEW: dense KD head for distillation
kd_head = torch.nn.Linear(CFG["feat_dim"], num_classes).to(device)

# (optional) freeze the SPARCL head so it doesn’t get touched
for p in student.head.parameters():
    p.requires_grad = False
    

def kd_loss(s_logits, t_logits, T=2.0):
    return (T*T) * F.kl_div(
        F.log_softmax(s_logits/T, dim=1),
        F.softmax(t_logits/T, dim=1),
        reduction="batchmean"
    )



best_state, best_val, stall = None, 0.0, 0

# -------------------------
# 4) Train loop
# -------------------------
print("\n[Distill from caches] KD + CE + Feature MSE (weak/deploy tfms)")

# SANITY TEST: Can student learn basic patterns?
print("\n=== SANITY TEST ===")
student.train(); kd_head.train()
test_batch = next(iter(dl_tr))
xb, yb, t_logits_cpu, t_feats_cpu = test_batch
xb = xb.to(device, non_blocking=True)
yb = yb.to(device, non_blocking=True)

print(f"Input shape: {xb.shape}")
print(f"Labels: {yb[:10].tolist()}")
print(f"Teacher logits shape: {t_logits_cpu.shape}")
print(f"Teacher logits range: [{t_logits_cpu.min():.3f}, {t_logits_cpu.max():.3f}]")

# Test forward pass
s_feats = student._features(xb)
s_logits = kd_head(s_feats)
print(f"Student features shape: {s_feats.shape}")
print(f"Student logits shape: {s_logits.shape}")
print(f"Student logits range: [{s_logits.min():.3f}, {s_logits.max():.3f}]")

# Test loss computation
ce = torch.nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])

kd_loss_val = CFG["alpha_kd"]*kd_loss(s_logits, t_logits_cpu.to(device), CFG["kd_T"])
ce_loss_val = CFG["alpha_ce"]*ce(s_logits, yb)
print(f"KD loss: {kd_loss_val:.3f}")
print(f"CE loss: {ce_loss_val:.3f}")
print("=== END SANITY TEST ===\n")
# ---- Stronger feature-only warmup ----
warmup_epochs = 20          # was 8
opt_wu = torch.optim.AdamW(
    list(student.backbone.parameters()) + list(student.proj.parameters()),
    lr=5e-3, weight_decay=0.0           # higher LR, no WD for warmup
)

def cos_loss(a,b):  # a,b: (N, feat_dim)
    return 1.0 - F.cosine_similarity(a,b,dim=1).mean()

best_cos = 0.0; patience = 5; stall = 0
for ep in range(warmup_epochs):
    student.train()
    run = 0.0
    for xb, _, _, tft_cpu in dl_tr:
        xb  = xb.to(device, non_blocking=True)
        tft = tft_cpu.to(device, non_blocking=True)
        sft = student._features(xb)
        loss = cos_loss(sft, tft)       # cosine only
        opt_wu.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(student.backbone.parameters())+list(student.proj.parameters()), 1.0)
        opt_wu.step()
        run += float(loss.item())

    # monitor cosine on val
    student.eval()
    with torch.no_grad():
        cs, n = 0.0, 0
        for xb, _, _, tft_cpu in dl_va:
            xb  = xb.to(device); tft = tft_cpu.to(device)
            sft = student._features(xb)
            cs += F.cosine_similarity(sft, tft, dim=1).sum().item()
            n  += xb.size(0)
        val_cos = cs/max(1,n)
    print(f"[WU] ep {ep:02d} | cos_loss={run/len(dl_tr):.3f} | val_cos={val_cos:.3f}")

    # early stop warmup when alignment is good
    if val_cos > best_cos + 1e-3:
        best_cos = val_cos; stall = 0
    else:
        stall += 1
        if val_cos >= 0.90 or stall >= patience:
            print(f"[WU] stop at ep {ep} (val_cos={val_cos:.3f})")
            break
# ---- Closed-form init for kd_head: feat_targets -> teacher logits ----
Z = train_cache["feat_targets"].float()   # (N, D=feat_dim), CPU
Y = train_cache["logits"].float()         # (N, C), CPU


# optimizer: backbone + proj + dense KD head
opt = torch.optim.AdamW(
    list(student.backbone.parameters()) +
    list(student.proj.parameters()) +
    list(kd_head.parameters()),
    lr=CFG["lr"], weight_decay=CFG["weight_decay"]
)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30, eta_min=1e-4)
grad_clip = 1.0

# ---- closed-form KD head init: student_feats -> teacher logits ----
student.eval()
S_list, T_list = [], []
with torch.no_grad():
    for xb, _, tlog_cpu, _ in dl_tr:
        xb = xb.to(device)
        S_list.append(student._features(xb).cpu())   # (N, D)
        T_list.append(tlog_cpu.float())              # (N, C)

S = torch.cat(S_list,0)     # (N,D)
T = torch.cat(T_list,0)     # (N,C)

# ridge regression: S_aug @ Wb ≈ T
N, D = S.shape; C = T.shape[1]
ones = torch.ones(N,1)
S_aug = torch.cat([S, ones], dim=1)        # (N, D+1)
lam = 1e-3
A = S_aug.T @ S_aug + lam * torch.eye(D+1)
B = S_aug.T @ T
Wb = torch.linalg.solve(A, B)              # (D+1, C)
W, b = Wb[:-1].T, Wb[-1]                   # (C,D), (C,)

with torch.no_grad():
    kd_head.weight.copy_(W.to(device))
    kd_head.bias.copy_(b.to(device))
    
for ep in range(CFG["epochs"]):
    student.train(); kd_head.train()
    run_loss = 0.0
    if ep == 5:       # after the student starts tracking the teacher
        alpha_ce = 0.1

    for xb, yb, t_logits_cpu, t_feats_cpu in dl_tr:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        t_logits = t_logits_cpu.to(device, non_blocking=True)
        t_feats  = t_feats_cpu.to(device, non_blocking=True)

        # student features (feat_dim) then dense KD head
        s_feats  = student._features(xb)   # [N, feat_dim]
        s_logits = kd_head(s_feats)        # [N, B]

        loss = CFG["alpha_kd"]*kd_loss(s_logits, t_logits, CFG["kd_T"]) \
             + CFG["alpha_ce"]*ce(s_logits, yb)

        if CFG["use_feat_mse"]:
            loss += CFG["alpha_mse"] * F.mse_loss(s_feats, t_feats)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((
            list(student.backbone.parameters()) +
            list(student.proj.parameters()) +
            list(kd_head.parameters())), grad_clip)
        opt.step()
        run_loss += float(loss.item())
    # debugging! 
    with torch.no_grad():
        m = 0; n = 0
        for _, yb, tlog_cpu, _ in dl_va:
            pred = tlog_cpu.argmax(1)
            m += (pred == yb).sum().item()
            n += yb.numel()
        print("!!!cache teacher top-1 vs labels:", 100*m/n, "%")
        
        # DATA ALIGNMENT CHECK
        print("\n=== DATA ALIGNMENT CHECK ===")
        test_batch = next(iter(dl_va))
        xb_test, yb_test, tlog_test, tfeat_test = test_batch
        print(f"Validation batch - Labels: {yb_test[:10].tolist()}")
        print(f"Validation batch - Teacher predictions: {tlog_test.argmax(1)[:10].tolist()}")
        print(f"Validation batch - Teacher correct: {(tlog_test.argmax(1) == yb_test)[:10].tolist()}")
        
        # Test student on same batch
        xb_test = xb_test.to(device)
        s_feats_test = student._features(xb_test)
        s_logits_test = kd_head(s_feats_test)
        s_pred_test = s_logits_test.argmax(1).cpu()
        print(f"Validation batch - Student predictions: {s_pred_test[:10].tolist()}")
        print(f"Validation batch - Student correct: {(s_pred_test == yb_test)[:10].tolist()}")
        print("=== END DATA ALIGNMENT CHECK ===\n")
    # validation
    student.eval(); kd_head.eval()
    v_tot = v_cor = 0
    with torch.no_grad():
        
        for xb, yb, _, _ in dl_va:
            # s_logits)≈", s_logits.std().item(), "std(t_logits)≈", t_logits.std().item())
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            # xb, yb, tlog, _ = next(iter(dl_va))
            # print("yb[:16]      =", yb[:16].tolist())
            # print("tlog.argmax =", tlog.argmax(1)[:16].tolist())
            s_logits = kd_head(student._features(xb))
            v_tot += yb.size(0); v_cor += (s_logits.argmax(1) == yb).sum().item()
    v_acc = 100.0*v_cor/max(1, v_tot)
    print(f"ep {ep:02d} | train_loss={run_loss/len(dl_tr):.3f} | val_acc={v_acc:.2f}%")

    if v_acc > best_val:
        best_val = v_acc
        best_state = {k: v.detach().cpu().clone() for k,v in student.state_dict().items()}
        stall = 0
    else:
        stall += 1
        if stall >= CFG["patience"]:
            print(f"Early stopping at ep {ep} (best {best_val:.2f}%)")
            break
    sched.step()

if best_state is not None:
    student.load_state_dict(best_state, strict=True)    

# -------------------------
# 5) Save backbone+proj for CIL
# -------------------------
out = Path(CFG["out_dir"]); out.mkdir(parents=True, exist_ok=True)

save = {
    "backbone": {k.replace("backbone.", ""): v for k,v in student.state_dict().items() if k.startswith("backbone.")},
    "proj":     {k.replace("proj.", ""):     v for k,v in student.state_dict().items() if k.startswith("proj.")},
}
torch.save(save, out/"student_backbone_proj.pth")

meta_out = {
    "student_arch": "tiny_cnn_v1",
    "feature_dim": CFG["feat_dim"],
    "img_size": img_size,
    "normalization": {"mean": mean, "std": std},
    "backbone_gids": backbone_gids,        # class order for the distilled head
    "best_val_acc": float(best_val),
}
(out/"student_meta.json").write_text(json.dumps(meta_out, indent=2))

print("\nSaved student artifacts:")
print(f"  • {out/'student_backbone_proj.pth'}")
print(f"  • {out/'student_meta.json'}")
print("Done ✅  — plug these into your CIL model and freeze the backbone there.")