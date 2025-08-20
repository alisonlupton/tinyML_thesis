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
    """
    Build dicts for fast lookup by canonical key.
    """
    keys   = [k.replace("\\","/") for k in cache["keys"]]
    feats  = cache["features"]          # Tensor [N, 1280]
    logits = cache["logits"]            # Tensor [N, B]
    y      = cache["y_local"]           # Tensor [N]
    map_ft     = {k: feats[i]  for i,k in enumerate(keys)}
    map_logits = {k: logits[i] for i,k in enumerate(keys)}
    map_y      = {k: int(y[i]) for i,k in enumerate(keys)}
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
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "label_smoothing": 0.05,

    # distillation losses
    "kd_T": 2.0,
    "alpha_kd": 1.0,     # KL(s||t)
    "alpha_ce": 0.5,     # CE(y_true)
    "mu_feat": 0.5,      # MSE(student_feat, target_feat)

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
saved = torch.load(CFG["proj_ckpt"], map_location="cpu")
if "proj_from_teacher_penult" not in saved:
    raise RuntimeError("Expected key 'proj_from_teacher_penult' in projector checkpoint.")

proj_sd = saved["proj_from_teacher_penult"]  # assumed Sequential(Dropout, Linear)

class LinearOnly(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.fc(x)

in_dim  = train_cache["features"].shape[1]                # 1280
out_dim = proj_sd["1.weight"].shape[0]                    # feat_dim (e.g., 128)

P = LinearOnly(in_dim, out_dim)
P.fc.weight.data.copy_(proj_sd["1.weight"])
P.fc.bias.data.copy_(proj_sd["1.bias"])
P.eval()

@torch.no_grad()
def add_feat_targets(cache):
    feats = cache["features"].float()          # [N, 1280]
    cache["feat_targets"] = P(feats).float()   # [N, feat_dim]
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
num_classes = len(backbone_gids)
student = SimplifiedTDMModelCNN(
    in_channels=CFG["in_channels"],
    init_num_classes=num_classes,     # temp: head size = |backbone classes|
    device=device,
    sparsity_ratio=CFG["sparsity_ratio"],
    feat_dim=CFG["feat_dim"],
).to(device)

# We'll train with the student's own full forward (backbone→proj→head)
# but compute additional feature loss using student._features() and student.proj

# -------------------------
# Losses & Optimizer
# -------------------------
# losses
ce = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
def kd_loss(s_logits, t_logits, T=2.0):
    return (T*T) * torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(s_logits/T, dim=1),
        torch.nn.functional.softmax(t_logits/T, dim=1),
        reduction="batchmean"
    )

alpha_kd = 1.0
alpha_ce = 0.5
alpha_mse = 0.1   # set 0 to disable feature-MSE
T = 2.0

opt = torch.optim.AdamW(student.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])

best_state, best_val, stall = None, 0.0, 0

# -------------------------
# 4) Train loop
# -------------------------
print("\n[Distill from caches] KD + CE + Feature MSE (weak/deploy tfms)")
for ep in range(CFG["epochs"]):
    student.train()
    run_loss = 0.0
    student.train()
    for xb, yb, t_logits_cpu, t_feats_cpu in dl_tr:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        t_logits = t_logits_cpu.to(device, non_blocking=True)  # [N, B]
        t_feats  = t_feats_cpu.to(device, non_blocking=True)   # [N, 1280]  (feature loss)

        s_logits = student(xb)            # [N, B]
        # optional: if you want student features, expose a method
        # s_feats = student.features(xb)  # [N, D]

        loss = alpha_kd * kd_loss(s_logits, t_logits, T) + alpha_ce * ce(s_logits, yb)
        # + alpha_mse * torch.nn.functional.mse_loss(s_feats, t_feats)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        run_loss += float(loss.item())

    # validation (accuracy over backbone classes)
    student.eval()
    v_tot = v_cor = 0
    with torch.no_grad():
        for xb, yb, t_logits_cpu, t_feats_cpu in dl_va:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            s_logits = student(xb)
            v_tot += yb.size(0)
            v_cor += (s_logits.argmax(1) == yb).sum().item()
    v_acc = 100.0 * v_cor / max(1, v_tot)
    print(f"val_acc={v_acc:.2f}%")

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