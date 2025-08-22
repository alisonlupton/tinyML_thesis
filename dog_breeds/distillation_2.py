# distillation_2.py
# (image KD) CNN training 
# Distill teacher → tiny CNN *from images* using only cached targets (no teacher forward).
# 1) Build per-image feature targets using the projector (trained in distillaion_1.py from caches)
# 2) Train my deplyment tiny CNN on images with KD (cached logits) and optionally CE + feature MSE.

import json, random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.cnn import SimplifiedTDMModelCNN
from pathlib import Path


###UTILS

def set_bn_eval(module: nn.Module):
    # Put BN layers in eval mode (use running stats), keep gamma/beta trainable
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        module.eval()

def keep_bn_frozen(model: nn.Module):
    # Call this right after model.train() to re-put BN in eval
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
            
@torch.no_grad()
def log_logit_std(student, kd_head, loader, norm, device, prefix="train", max_batches=2):
    was_training = student.training
    student.eval(); kd_head.eval()

    s_std_list, t_std_list = [], []
    batches = 0
    for xb, _, tlog, _ in loader:   # <--- 4-tuple
        xb   = xb.to(device)
        tlog = tlog.to(device)

        sfeat = norm(student.proj(student._features(xb)))
        slogits = kd_head(sfeat)

        s_std_list.append(slogits.std(dim=1).mean().item())
        t_std_list.append(tlog.std(dim=1).mean().item())

        batches += 1
        if batches >= max_batches:
            break

    if was_training:
        student.train()
        keep_bn_frozen(student)

    s_mean = sum(s_std_list) / max(1, len(s_std_list))
    t_mean = sum(t_std_list) / max(1, len(t_std_list))
    ratio  = s_mean / max(1e-8, t_mean)
    print(f"[{prefix}] avg per-sample logit std: student≈{s_mean:.3f}, teacher≈{t_mean:.3f} (ratio≈{ratio:.2f})")
    
def kd_loss_znorm(s_logits, t_logits, T=4.0, eps=1e-3):
    # per-sample mean/var normalize both s and t
    s = s_logits - s_logits.mean(dim=1, keepdim=True)
    t = t_logits - t_logits.mean(dim=1, keepdim=True)
    s = s / (s.std(dim=1, keepdim=True) + eps)
    t = t / (t.std(dim=1, keepdim=True) + eps)
    return (T*T) * F.kl_div(
        F.log_softmax(s/T, dim=1),
        F.softmax(t/T, dim=1),
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


@torch.no_grad()
def refit_kd_head(student, kd_head, loader, norm, device, lam=1e-1, max_batches=8):
    student.eval()
    keep_bn_frozen(student)

    S_list, Tlog_list = [], []
    batches = 0
    for xb, _, tlog_cpu, _ in loader:
        xb = xb.to(device)
        sfeat = norm(student.proj(student._features(xb)))
        S_list.append(sfeat.cpu())
        Tlog_list.append(tlog_cpu.float())
        batches += 1
        if batches >= max_batches:
            break

    S = torch.cat(S_list, 0)         # (N,D)
    Tlog = torch.cat(Tlog_list, 0)   # (N,C)
    N, D = S.shape; C = Tlog.shape[1]
    S_aug = torch.cat([S, torch.ones(N,1)], dim=1)
    A = S_aug.T @ S_aug + lam * torch.eye(D+1)
    B = S_aug.T @ Tlog
    Wb = torch.linalg.solve(A, B)    # (D+1, C)
    W, b = Wb[:-1].T, Wb[-1]

    # optional: keep scales in the same ballpark as teacher
    s_raw = (S @ W.T) + b
    s_std = s_raw.std(dim=1).mean().item()
    t_std = Tlog.std(dim=1).mean().item()
    scale = (s_std / max(1e-6, t_std)) if s_std > 0 else 1.0

    kd_head.weight.copy_((W/scale).to(device))
    kd_head.bias.copy_((b/scale).to(device))
    kd_head.eval()
    
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


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
@torch.no_grad()
def add_feat_targets(cache, P):
    feats = cache["features"].float()          # CPU
    cache["feat_targets"] = P(feats).float()   # CPU -> stays CPU for dataset
    return cache
def _zero_out_dropout(m):
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.p = 0.0
class LinearOnly(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.fc(x)

def logit_scale_loss(slogits, tlogits):
    # per-sample std then mean
    s_std = slogits.std(dim=1)
    t_std = tlogits.std(dim=1)
    return F.l1_loss(s_std, t_std)
# -------------------------
# Temp Config
# -------------------------
CFG = {
    # dataset index you already created
    "index_csv": "../data/dog_breed_data/stanford_dogs_index.csv",
    "index_json": "../data/dog_breed_data/stanford_dogs_index.json",   # not strictly required here
    # teacher meta (for normalization + class order sanity if needed)
    "teacher_meta": "../data/dog_breed_data/offline/backbone_meta.json",

    # caches from Colab step
    "cache_dir": "../data/dog_breed_data/offline",
    "train_cache": "train_cache.pt",
    "val_cache":   "val_cache.pt",

    # projector you trained from cached teacher features
    "proj_ckpt": "../data/dog_breed_data/distillation_1/student_from_cache.pth",

    # output student weights (to plug into your CIL)
    "out_dir": "../data/dog_breed_data/distillation_2/",

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
    "student_img_size": 224,   # <--  ( MCU size)

    # distillation losses
    "kd_T": 6.0,
    "alpha_kd": 1.0,     # KL(s||t)
    "alpha_ce": 0.0,     # CE(y_true)
    "mu_feat": 0.5,      # MSE(student_feat, target_feat)
    "use_feat_mse": True,
    "alpha_mse": 0.1,
    "w_kd": 1.0,
    "w_feat": 0.2,
    "w_crd":0.0, 
    "w_rel": 0.0, 
    "w_proto": 0.0,
    "w_ce": 0.0,
    "w_scale": 0.0,
    # workers
    "num_workers": 0,
}


#### MAIN FUNCTION !!! 
def distillation_2(CFG):
    set_seed(CFG["seed"])
    device = torch.device("cpu")
    

    # -------------------------
    # Load meta, caches
    # -------------------------
    meta = json.loads(Path(CFG["teacher_meta"]).read_text())
    mean = meta["normalization"]["mean"]; std = meta["normalization"]["std"]
    img_size = int(meta["img_size"])
    backbone_gids = list(map(int, meta["backbone_gids"]))  # fixed class order B

    cache_dir = Path(CFG["cache_dir"])
    train_cache = torch.load(cache_dir/CFG["train_cache"], map_location="cpu", weights_only= True)
    val_cache   = torch.load(cache_dir/CFG["val_cache"],   map_location="cpu", weights_only= True)

    train_cache["keys"] = [canon_key(k) for k in train_cache["keys"]]
    val_cache["keys"]   = [canon_key(k) for k in val_cache["keys"]]

    print(f"Train cache: feats={tuple(train_cache['features'].shape)}, logits={tuple(train_cache['logits'].shape)}")
    print(f"Val   cache: feats={tuple(val_cache['features'].shape)},   logits={tuple(val_cache['logits'].shape)}")

    # -------------------------
    # 1) Build feature targets using projector 
    # -------------------------
    saved   = torch.load(CFG["proj_ckpt"], map_location="cpu", weights_only= True)  # ../data/student_from_cache.pth
    proj_sd = saved["proj_from_teacher_penult"]                 # state_dict of Sequential(Dropout, Linear)


    P = LinearOnly(in_dim=train_cache["features"].shape[1],  # 1280
                out_dim=proj_sd["1.weight"].shape[0])     # feat_dim, e.g. 128
    P.fc.weight.data.copy_(proj_sd["1.weight"])
    P.fc.bias.data.copy_(proj_sd["1.bias"])
    P = P.cpu().eval()


    train_cache = add_feat_targets(train_cache, P)
    val_cache   = add_feat_targets(val_cache, P)


    # teacher 128-D features, already computed: train_cache["feat_targets"] (N, D)
    # y_local are the local labels 0..B-1
    tfeat_tr = train_cache["feat_targets"].float()     # CPU
    y_tr     = train_cache["y_local"].long()           # CPU
    num_classes = len(backbone_gids)
    feat_dim = tfeat_tr.shape[1]

    # compute teacher class centroids in 128-D
    centroids = torch.stack([
        tfeat_tr[y_tr == c].mean(0) for c in range(num_classes)
    ])
    # guard NaNs if any class is empty
    # centroids[torch.isnan(centroids)] = 0
    
    
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
    # student_img_size = CFG["student_img_size"]    
    # try 224
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
    # 3) Build student tiny CNN from simplified TDM model
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
    student.apply(set_bn_eval)   # lock BN running stats
    for p in student.proj.parameters():
        p.requires_grad = False   # freeze Linear AND BN affine
    norm = nn.LayerNorm(CFG["feat_dim"], elementwise_affine=False).to(device)
    
    student.apply(_zero_out_dropout)

    best_state, best_val, stall = None, 0.0, 0

    # -------------------------
    # 4) Train loop
    # -------------------------
    print("\n[Distill from caches] KD + CE + Feature MSE (weak/deploy tfms)")



    # ---- closed-form KD head init: student_feats -> teacher logits ----
    # collect student features on the train set (no grad)
    student.eval()
    student.apply(set_bn_eval)   # BN eval for calibration
    S_list, Tlog_list = [], []
    with torch.no_grad():
        for xb, _, tlog_cpu, _ in dl_tr:
            xb = xb.to(device)
            sfeat = norm(student.proj(student._features(xb)))  # <-- exact path
            S_list.append(sfeat.cpu())
            Tlog_list.append(tlog_cpu.float())
    S    = torch.cat(S_list, 0)      # (N, D)
    Tlog = torch.cat(Tlog_list, 0)   # (N, C)


    # ridge regression S_aug -> Tlog
    # Ridge
    N, D = S.shape; C = Tlog.shape[1]
    S_aug = torch.cat([S, torch.ones(N,1)], dim=1)
    lam = 1e-1  # was 1e-3 -> stronger regularization
    A = S_aug.T @ S_aug + lam * torch.eye(D+1)
    B = S_aug.T @ Tlog
    Wb = torch.linalg.solve(A, B)    # (D+1, C)
    W, b = Wb[:-1].T, Wb[-1]         # W: (C,D), b: (C,)

    # Calibrate scale on train set
    with torch.no_grad():
        s_raw = (S @ W.T) + b         # (N, C)
        s_std = s_raw.std(dim=1).mean().item()
        t_std = Tlog.std(dim=1).mean().item()
        scale = (s_std / max(1e-6, t_std)) if s_std > 0 else 1.0

    kd_head = nn.Linear(D, C).to(device)
    logit_gain = torch.nn.Parameter(torch.tensor(0.0, device=device))  # logits *= exp(gain)

    with torch.no_grad():
        kd_head.weight.copy_((W/scale).to(device))
        kd_head.bias.copy_((b/scale).to(device))
            
    for p in kd_head.parameters(): p.requires_grad = False
    kd_head.eval()
    # SANITY TEST  
    with torch.no_grad():
        student.eval(); student.apply(keep_bn_frozen)
        xb,_,tlog,_ = next(iter(dl_tr))
        xb   = xb.to(device); tlog = tlog.to(device)
        sfeat   = norm(student.proj(student._features(xb)))
        slogits = kd_head(sfeat)
        print(f"[post-fit] std mean: student≈{slogits.std(dim=1).mean().item():.3f}, "
            f"teacher≈{tlog.std(dim=1).mean().item():.3f}")
    # END SANITY TEST  
    # SANITY TEST: Can student learn basic patterns?
    print("\n=== SANITY TEST ===")
    student.train(); student.apply(keep_bn_frozen)
    student.apply(keep_bn_frozen)   # <- keep BN fixed
    test_batch = next(iter(dl_tr))
    xb, yb, t_logits_cpu, t_feats_cpu = test_batch
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)

    print(f"Input shape: {xb.shape}")
    print(f"Labels: {yb[:10].tolist()}")
    print(f"Teacher logits shape: {t_logits_cpu.shape}")
    print(f"Teacher logits range: [{t_logits_cpu.min():.3f}, {t_logits_cpu.max():.3f}]")

    # Test forward pass
    s_feats = norm(student.proj(student._features(xb)))
    s_logits = kd_head(s_feats)

    print(f"Student features shape: {s_feats.shape}")
    print(f"Student logits shape: {s_logits.shape}")
    print(f"Student logits range: [{s_logits.min():.3f}, {s_logits.max():.3f}]")

    # Test loss computation
    ce = torch.nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])

    kd_loss_val = CFG["alpha_kd"]*kd_loss_znorm(s_logits, t_logits_cpu.to(device), CFG["kd_T"])
    ce_loss_val = CFG["alpha_ce"]*ce(s_logits, yb)
    print(f"KD loss: {kd_loss_val:.3f}")
    print(f"CE loss: {ce_loss_val:.3f}")
    print("=== END SANITY TEST ===\n")

    fixed = next(iter(dl_va))
    fixed_x, _, fixed_tlog, _ = fixed
    fixed_x = fixed_x.to(device); fixed_tlog = fixed_tlog.to(device)


    # once before training
    import collections
    cnt = collections.Counter()
    for _, _, tlog_cpu, _ in dl_tr:
        cnt.update(tlog_cpu.argmax(1).tolist())
    # before training
    for p in student.proj.parameters(): 
        p.requires_grad = False

    BACKBONE_LR = 1e-4
    PROJ_LR     = 5e-5   # for after unfreezing

    opt = torch.optim.AdamW(
        [p for p in student.backbone.parameters() if p.requires_grad],
        lr=BACKBONE_LR, weight_decay=1e-4
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40, eta_min=1e-4)
    refit_kd_head(student, kd_head, dl_tr, norm, device, lam=1e-1, max_batches=8)
    print("[dbg] teacher-pred train hist:", dict(cnt))
    
    
    # SANITY CHECK!!!
    tot = cor = 0
    for _, yb, tlog, _ in dl_tr:
        cor += (tlog.argmax(1) == yb).sum().item()
        tot += yb.numel()
    print("Teacher top1 on TRAIN via cache vs y:", 100*cor/tot, "%")
    
    for i in [0, 123, 456]:
        r = ds_tr.df.iloc[i]
        print("csv:", r.img_path, "canon:", canon_key_from_path(r.img_path),
            "in_cache:", canon_key_from_path(r.img_path) in tr_map_logits)
        
    # END SANITY CHECK!!!

    for ep in range(60):
        
        student.train(); keep_bn_frozen(student)  # <-- re-freeze BN running stats each epoch
        run = 0.0
        # periodic refit at the START of the epoch 
        # if ep > 0 and (ep % 2 == 0):
        #     refit_kd_head(...)
       
        # if ep == 5:
        #     for p in student.proj.parameters():
        #         p.requires_grad = True
        #     opt = torch.optim.AdamW(
        #         [
        #             {"params": [p for p in student.backbone.parameters() if p.requires_grad], "lr": BACKBONE_LR},
        #             {"params": [p for p in student.proj.parameters() if p.requires_grad],     "lr": PROJ_LR},
        #         ],
        #         weight_decay=1e-4
        #     )
        for xb, yb, tlog_cpu, tfeat_cpu in dl_tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            tlog  = tlog_cpu.to(device, non_blocking=True)   # (N,C)
            tfeat = tfeat_cpu.to(device, non_blocking=True)  # (N,D)
            
        

            sfeat = norm(student.proj(student._features(xb)))
            slogit = kd_head(sfeat)
            slogit = torch.exp(logit_gain) * slogit

            kd   = kd_loss_znorm(slogit, tlog, CFG['kd_T'])
            feat = 1.0 - F.cosine_similarity(
                    F.normalize(sfeat, dim=1),
                    F.normalize(tfeat, dim=1), dim=1).mean()
            loss = CFG['w_kd']*kd + CFG['w_feat']*feat + 0.05*logit_scale_loss(slogit, tlog)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            total_params = (
                list(student.backbone.parameters()) +
                list(student.proj.parameters()) +
                list(kd_head.parameters())
            )
            trainable = [p for p in student.backbone.parameters() if p.requires_grad] \
            + [p for p in student.proj.parameters() if p.requires_grad]   
            gn = torch.nn.utils.clip_grad_norm_(trainable, 0.05)
            if ep % 5 == 0: print(f"[dbg] grad_norm≈{float(gn):.3f}")
            opt.step()
            run += float(loss.item())
        with torch.no_grad():
            sfeat = norm(student.proj(student._features(fixed_x)))
            s_logit = kd_head(sfeat)
            kl_fixed = kd_loss_znorm(s_logit, fixed_tlog, CFG['kd_T'])
        print(f"[dbg] KL(val-fixed): {kl_fixed.item():.3f}")
                
        if ep % 3 == 0:
            with torch.no_grad():
                sfeat = norm(student.proj(student._features(fixed_x)))
                slogit = kd_head(sfeat)
                s_sample = slogit.std(dim=1).mean().item()
                t_sample = fixed_tlog.std(dim=1).mean().item()
            print(f"[dbg] logits std (student≈{s_sample:.3f}, teacher≈{t_sample:.3f})")
        log_logit_std(student, kd_head, dl_tr, norm, device, prefix="train")
        log_logit_std(student, kd_head, dl_va, norm, device, prefix="val")
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

                sfeat_val = norm(student.proj(student._features(xb_val)))
                logits_val = kd_head(sfeat_val)

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
                sfeat = norm(student.proj(student._features(xb)))
                slogit = kd_head(sfeat)
                s_pred = slogit.argmax(1).cpu()
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
        "img_size": CFG["student_img_size"],   # <--- student size
        "normalization": {"mean": mean, "std": std},
        "backbone_gids": backbone_gids,
        "best_val_acc": float(best_val),
    }
    (out/"student_meta.json").write_text(json.dumps(meta_out, indent=2))

    print("\nSaved student artifacts:")
    print(f"  • {out/'student_backbone_proj.pth'}")
    print(f"  • {out/'student_meta.json'}")
    print("Done.")

if __name__ == "__main__":
    distillation_2(CFG)