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
###UTILS
def kd_loss(s_logits, t_logits, T=4.0):
    return (T*T) * F.kl_div(
        F.log_softmax(s_logits/T, dim=1),
        F.softmax(t_logits/T, dim=1),
        reduction="batchmean"
    )

# Contrastive (InfoNCE) between student feats and teacher feats (both 128-D)
# Positives: (i,i). Negatives: (i,j!=i) within batch.
def crd_loss(s_feat, t_feat, tau=0.07):
    s = F.normalize(s_feat, dim=1)
    t = F.normalize(t_feat, dim=1)
    logits = s @ t.t() / tau                 # (N,N)
    labels = torch.arange(s.size(0), device=s.device)
    return F.cross_entropy(logits, labels)

# Relational (Affinity/Gram) — match pairwise cosine sims within the batch
def gram_rel_loss(s_feat, t_feat):
    s = F.normalize(s_feat, dim=1)
    t = F.normalize(t_feat, dim=1)
    # Gram: G = X X^T (cosine since normalized)
    Gs = s @ s.t()
    Gt = t @ t.t()
    # stop gradients through teacher gram
    return F.mse_loss(Gs, Gt)

# Prototype pull — pull student feats to teacher class centroid
def proto_loss(s_feat, y, centroids_device):
    target = centroids_device[y]             # (N, D)
    # cosine distance is scale-invariant and works better across nets
    s = F.normalize(s_feat, dim=1)
    t = F.normalize(target,   dim=1)
    return 1.0 - F.cosine_similarity(s, t, dim=1).mean()
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


# teacher 128-D features you've already computed: train_cache["feat_targets"] (N, D)
# y_local are the local labels 0..B-1
tfeat_tr = train_cache["feat_targets"].float()     # CPU
y_tr     = train_cache["y_local"].long()           # CPU
num_classes = len(backbone_gids)
feat_dim = tfeat_tr.shape[1]

# compute teacher class centroids in 128-D
centroids = torch.stack([
    tfeat_tr[y_tr == c].mean(0) for c in range(num_classes)
])
# guard NaNs if any class is empty (shouldn’t happen with your 10 classes)
centroids[torch.isnan(centroids)] = 0
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

norm = nn.LayerNorm(CFG["feat_dim"], elementwise_affine=False).to(device)
def _zero_out_dropout(m):
    if isinstance(m, (nn.Dropout, nn.Dropout2d)):
        m.p = 0.0
student.apply(_zero_out_dropout)

# # after you build `student` and `P` (and move student to device)
# with torch.no_grad():
#     # student.proj = nn.Sequential(Dropout, Linear(128 -> feat_dim))
#     student.proj[1].weight.copy_(P.fc.weight.to(device))
#     student.proj[1].bias.copy_(P.fc.bias.to(device))

# (optional) freeze the SPARCL head so it doesn’t get touched
for p in student.head.parameters():
    p.requires_grad = False
    




best_state, best_val, stall = None, 0.0, 0

# -------------------------
# 4) Train loop
# -------------------------
print("\n[Distill from caches] KD + CE + Feature MSE (weak/deploy tfms)")


def lerp(a,b,t): return a + t*(b-a)


# move centroids to device once
centroids_device = centroids.to(device)

warmup_epochs = 50
opt_wu = torch.optim.AdamW(
    list(student.backbone.parameters()) + list(student.proj.parameters()),
    lr=5e-3, weight_decay=0.0
)
grad_clip = 1.0

for ep in range(warmup_epochs):
    student.train()
    run = 0.0
    for xb, yb, _, tfeat_cpu in dl_tr:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        tfeat = tfeat_cpu.to(device, non_blocking=True)        # teacher 128-D targets

        sfeat = norm(student._features(xb))                          # (N,128)

        # weights: tune lightly; keep CRD modest (negatives are in-batch)
        loss = (0.6 * (1.0 - F.cosine_similarity(
                    F.normalize(sfeat, dim=1),
                    F.normalize(tfeat, dim=1), dim=1).mean())
               +0.3 * crd_loss(sfeat, tfeat, tau=0.07)
               +0.1 * proto_loss(sfeat, yb, centroids_device))

        opt_wu.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(student.backbone.parameters()) + list(student.proj.parameters()), grad_clip)
        opt_wu.step()
        run += float(loss.item())

    # monitor alignment (cosine)
    student.eval()
    with torch.no_grad():
        cs, n = 0.0, 0
        for xb, _, _, tfeat_cpu in dl_va:
            xb = xb.to(device); tfeat = tfeat_cpu.to(device)
            sfeat = norm(student._features(xb))
            cs += F.cosine_similarity(F.normalize(sfeat,dim=1), F.normalize(tfeat,dim=1), dim=1).sum().item()
            n  += xb.size(0)
        val_cos = cs/max(1,n)
    print(f"[WU] ep {ep:02d} | loss={run/len(dl_tr):.3f} | val_cos={val_cos:.3f}")

    # soft early-stop if alignment stalls
    if val_cos >= 0.70:  # you were ~0.37 before; aim for ≥0.70
        print(f"[WU] stop early at ep {ep} (val_cos={val_cos:.3f})")
        break



# ---- closed-form KD head init: student_feats -> teacher logits ----
# collect student features on the train set (no grad)
student.eval()
S_list, Tlog_list, Tfeat_list = [], [], []
with torch.no_grad():
    for xb, _, tlog_cpu, tfeat_cpu in dl_tr:
        xb = xb.to(device)
        S_list.append(norm(student._features(xb)).cpu())  # (N,D)
        Tlog_list.append(tlog_cpu.float())          # (N,C)
        Tfeat_list.append(tfeat_cpu.float())

S = torch.cat(S_list, 0)      # (N,D)
Tlog = torch.cat(Tlog_list,0) # (N,C)

# ridge regression S_aug -> Tlog
N, D = S.shape; C = Tlog.shape[1]
S_aug = torch.cat([S, torch.ones(N,1)], dim=1)
lam = 1e-3
A = S_aug.T @ S_aug + lam * torch.eye(D+1)
B = S_aug.T @ Tlog
Wb = torch.linalg.solve(A, B)  # (D+1, C)
W, b = Wb[:-1].T, Wb[-1]

kd_head = nn.Linear(D, C).to(device)
with torch.no_grad():
    kd_head.weight.copy_((W * 0.1).to(device))  # Scale weights directly
    kd_head.bias.copy_((b * 0.1).to(device))    # Scale bias directly
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
s_feats = norm(student._features(xb))
s_logits = kd_head(s_feats)  # No scaling needed - weights are already scaled
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

opt = torch.optim.AdamW(
    list(student.backbone.parameters()) +
    list(student.proj.parameters()) +
    list(kd_head.parameters()),
    lr=1e-3, weight_decay=1e-4
)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40, eta_min=1e-4)
T = 4.0
def logit_scale_loss(slogits, tlogits):
    # per-sample std then mean
    s_std = slogits.std(dim=1)
    t_std = tlogits.std(dim=1)
    return F.l1_loss(s_std, t_std)
w_scale = 0.05
fixed = next(iter(dl_va))
fixed_x, _, fixed_tlog, _ = fixed
fixed_x = fixed_x.to(device); fixed_tlog = fixed_tlog.to(device)

w_kd   = 1.0
w_crd  = 0.05   # was 0.2
w_rel  = 0.05   # was 0.1
w_proto= 0.05
w_ce   = 0.1
# once before training
import collections
cnt = collections.Counter()
for _, _, tlog_cpu, _ in dl_tr:
    cnt.update(tlog_cpu.argmax(1).tolist())
freeze_k = 5
for p in kd_head.parameters():
    p.requires_grad = False
    
opt = torch.optim.AdamW(
    list(student.backbone.parameters()) + list(student.proj.parameters()),
    lr=1e-4, weight_decay=1e-4  # was 2e-4 - even lower LR for stability
)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40, eta_min=1e-4)

print("[dbg] teacher-pred train hist:", dict(cnt))
for ep in range(60):
    student.train(); kd_head.train()
    run = 0.0
    for xb, yb, tlog_cpu, tfeat_cpu in dl_tr:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        tlog  = tlog_cpu.to(device, non_blocking=True)   # (N,C)
        tfeat = tfeat_cpu.to(device, non_blocking=True)  # (N,D)

        sfeat  = norm(student._features(xb))
        slogit = kd_head(sfeat)
        if ep < 10:
            w_ce_now = 0.0
        elif ep < 20:
            w_ce_now = lerp(0.0, 0.2, (ep-10)/10)
        else:
            w_ce_now = 0.2

        loss = (w_kd   * kd_loss(slogit, tlog, T)
              + w_crd  * crd_loss(sfeat, tfeat, tau=0.07)
              + w_rel  * gram_rel_loss(sfeat, tfeat)
              + w_proto* proto_loss(sfeat, yb, centroids_device)
              + w_ce_now   * F.cross_entropy(slogit, yb, label_smoothing=0.05)
              + w_scale * logit_scale_loss(slogit, tlog))
        if ep == freeze_k:
            for p in kd_head.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(
                list(student.backbone.parameters()) +
                list(student.proj.parameters()) +
                list(kd_head.parameters()),
                lr=5e-4, weight_decay=1e-4
            )
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40, eta_min=1e-4)

        opt.zero_grad(set_to_none=True)
        loss.backward()

        total_params = (
            list(student.backbone.parameters()) +
            list(student.proj.parameters()) +
            list(kd_head.parameters())
        )
        gn = torch.nn.utils.clip_grad_norm_(total_params, 1.0)
        if ep % 5 == 0: print(f"[dbg] grad_norm≈{float(gn):.3f}")
        opt.step()
        run += float(loss.item())
    with torch.no_grad():
        s_log = kd_head(norm(student._features(fixed_x)))
        kl_fixed = kd_loss(s_log, fixed_tlog, T)
    print(f"[dbg] KL(val-fixed): {kl_fixed.item():.3f}")
            
    if ep % 3 == 0:
        with torch.no_grad():
            s_sample = kd_head(norm(student._features(fixed_x))).std(dim=1).mean().item()
            t_sample = fixed_tlog.std(dim=1).mean().item()
        print(f"[dbg] logits std (student≈{s_sample:.3f}, teacher≈{t_sample:.3f})")

    # val accuracy on backbone classes
    # --- Validation block (safe shapes/devices) ---
    student.eval(); kd_head.eval()
    v_tot = v_cor = 0
    val_cos_sum = 0.0
    val_num = 0

    with torch.no_grad():
        for xb_val, yb_val, _, tfeat_cpu_val in dl_va:
            xb_val = xb_val.to(device)
            yb_val = yb_val.to(device)
            tfeat_val = tfeat_cpu_val.to(device)

            sfeat_val = norm(student._features(xb_val))

            # cosine only when shapes match (they should)
            if sfeat_val.shape == tfeat_val.shape:
                val_cos_sum += F.cosine_similarity(
                    F.normalize(sfeat_val, dim=1),
                    F.normalize(tfeat_val, dim=1),
                    dim=1
                ).sum().item()
                val_num += sfeat_val.size(0)

            logits_val = kd_head(sfeat_val)
            v_tot += yb_val.size(0)
            v_cor += (logits_val.argmax(1) == yb_val).sum().item()

    val_cos_kd = val_cos_sum / max(1, val_num)
    v_acc = 100.0 * v_cor / max(1, v_tot)
    print(f"[dbg] feat cosine (val): {val_cos_kd:.3f}")
    if v_acc > best_val:
        best_val = v_acc
        best_state = {k: v.detach().cpu().clone() for k,v in student.state_dict().items()}
        stall = 0
    else:
        stall += 1
        if stall >= CFG["patience"]:
            print(f"Early stopping at ep {ep} (best {best_val:.2f}%)")
            break
        
    with torch.no_grad():
        st_agree = tot = 0
        for xb, _, tlog_cpu, _ in dl_va:
            xb = xb.to(device)
            s_pred = kd_head(norm(student._features(xb))).argmax(1).cpu()
            t_pred = tlog_cpu.argmax(1)
            st_agree += (s_pred == t_pred).sum().item()
            tot += s_pred.numel()
        agree_pct = 100.0 * st_agree / max(1, tot)
    print(f"[dbg] student-teacher agreement (val): {agree_pct:.2f}%")
    print(f"[KD] ep {ep:02d} | train_loss={run/len(dl_tr):.3f} | val_acc={v_acc:.2f}%")
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