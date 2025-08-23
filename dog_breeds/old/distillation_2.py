# distillation_mcu_friendly_tuned.py
# 
# 
# 
# Tuned version of MCU-friendly distillation with better hyperparameters
# Based on distillation_mcu_friendly.py but with improvements

import json, random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm

### UTILS

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def canon_key_from_path(p: str) -> str:
    p = Path(p)
    parts = [*p.parts]
    if "Images" in parts:
        i = parts.index("Images")
        rel = parts[i+1:]
    else:
        rel = parts[-2:]
    return "/".join(rel).replace("\\", "/")

def build_cache_maps(cache: dict):
    keys = [k.replace("\\","/") for k in cache["keys"]]
    logits = cache["logits"]
    y = cache["y_local"]
    fts = cache["feat_targets"]
    map_logits = {k: logits[i] for i,k in enumerate(keys)}
    map_y = {k: int(y[i]) for i,k in enumerate(keys)}
    map_ft = {k: fts[i] for i,k in enumerate(keys)}
    return map_ft, map_logits, map_y

@torch.no_grad()
def add_feat_targets(cache, P):
    feats = cache["features"].float()
    cache["feat_targets"] = P(feats).float()
    return cache

### IMPROVED MCU STUDENT ARCHITECTURE

class TunedMCUStudentCNN(nn.Module):
    """
    Tuned MCU student with better initialization and architecture
    - Same size as original but with improvements
    - Better weight initialization
    - Slightly improved architecture
    """
    def __init__(self, in_channels=3, feat_dim=64, num_classes=10, img_size=160):
        super().__init__()
        
        # Improved backbone with better channel progression
        self.backbone = nn.Sequential(
            # First conv block - more channels for better feature extraction
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Improved projection head with better dropout
        self.proj = nn.Sequential(
            nn.Dropout(0.25),  # Slightly higher dropout for regularization
            nn.Linear(128, feat_dim),
        )
        
        # Classification head
        self.classifier = nn.Linear(feat_dim, num_classes)
        
        # Better weight initialization
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _features(self, x):
        h = self.backbone(x)
        h = self.gap(h).flatten(1)
        return h
    
    def forward(self, x):
        feats = self._features(x)
        proj_feats = self.proj(feats)
        logits = self.classifier(proj_feats)
        return logits, proj_feats

### IMPROVED LOSS FUNCTIONS

def tuned_kd_loss(s_logits, t_logits, T=2.5, alpha=0.8):
    """Improved KD loss with better temperature and alpha balance"""
    # Soft targets
    soft_loss = F.kl_div(
        F.log_softmax(s_logits/T, dim=1),
        F.softmax(t_logits/T, dim=1),
        reduction="batchmean"
    ) * (T * T)
    
    # Hard targets (ground truth)
    hard_loss = F.cross_entropy(s_logits, t_logits.argmax(dim=1))
    
    return alpha * soft_loss + (1 - alpha) * hard_loss

def tuned_feature_loss(s_feat, t_feat):
    """Improved feature matching loss"""
    s_norm = F.normalize(s_feat, dim=1)
    t_norm = F.normalize(t_feat, dim=1)
    return 1.0 - F.cosine_similarity(s_norm, t_norm, dim=1).mean()

### DATASET

class DogWithCacheTargets(Dataset):
    def __init__(self, df, split, tfm, gid_to_local, map_logits, map_y, map_ft):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.tfm = tfm
        self.g2l = gid_to_local
        self.map_logits = map_logits
        self.map_y = map_y
        self.map_ft = map_ft
        
        self.keys = [canon_key_from_path(r.img_path) for _, r in self.df.iterrows()]
        missing = [k for k in self.keys if k not in self.map_logits]
        if missing:
            raise RuntimeError(f"{len(missing)} images missing in cache for split={split}")

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r.img_path).convert("RGB")
        x = self.tfm(img)
        y = self.g2l[int(r.gid)]
        key = self.keys[i]
        tlog = self.map_logits[key]
        tft = self.map_ft[key]
        return x, y, tlog, tft

### MAIN DISTILLATION FUNCTION

def tuned_mcu_distillation():
    """Tuned MCU-friendly distillation with improved hyperparameters"""
    
    # Improved configuration for better performance
    CFG = {
        "index_csv": "../data/dog_breed_data/stanford_dogs_index.csv",
        "teacher_meta": "../data/dog_breed_data/offline/backbone_meta.json",
        "cache_dir": "../data/dog_breed_data/offline",
        "train_cache": "train_cache.pt",
        "val_cache": "val_cache.pt",
        "proj_ckpt": "../data/dog_breed_data/distillation_1/student_from_cache.pth",
        "out_dir": "../data/dog_breed_data/distillation_mcu_tuned/",
        
        # MCU-friendly student model (same size)
        "feat_dim": 64,
        "num_classes": 10,
        "img_size": 160,
        
        # Improved training parameters
        "seed": 42,
        "batch_size_train": 32,  # Larger batch for better gradients
        "batch_size_val": 32,
        "epochs": 120,  # More epochs
        "patience": 25,  # More patience
        "lr": 5e-4,  # Lower learning rate for stability
        "weight_decay": 5e-5,  # Reduced weight decay
        
        # Improved distillation parameters
        "kd_T": 2.5,  # Better temperature
        "kd_alpha": 0.8,  # Better alpha balance
        "kd_weight": 1.0,
        "feat_weight": 0.5,  # Increased feature weight
        "ce_weight": 0.3,  # Increased CE weight
        
        "num_workers": 0,
    }
    
    set_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=== Tuned MCU-Friendly Knowledge Distillation ===")
    print(f"Student: {CFG['feat_dim']}D features, {CFG['num_classes']} classes, {CFG['img_size']}x{CFG['img_size']} images")
    print("Improved hyperparameters for better performance")
    
    # Load meta and caches
    meta = json.loads(Path(CFG["teacher_meta"]).read_text())
    mean = meta["normalization"]["mean"]
    std = meta["normalization"]["std"]
    backbone_gids = list(map(int, meta["backbone_gids"]))
    
    cache_dir = Path(CFG["cache_dir"])
    train_cache = torch.load(cache_dir/CFG["train_cache"], map_location="cpu", weights_only=True)
    val_cache = torch.load(cache_dir/CFG["val_cache"], map_location="cpu", weights_only=True)
    out_dir = Path(CFG["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
            
    train_cache["keys"] = [canon_key_from_path(k) for k in train_cache["keys"]]
    val_cache["keys"] = [canon_key_from_path(k) for k in val_cache["keys"]]
    
    print(f"Train cache: {len(train_cache['keys'])} samples")
    print(f"Val cache: {len(val_cache['keys'])} samples")
    
    # Build feature targets using projector
    saved = torch.load(CFG["proj_ckpt"], map_location="cpu", weights_only=True)
    proj_sd = saved["proj_from_teacher_penult"]
    
    # Create a new projector that maps to student feature dimension
    P = nn.Linear(train_cache["features"].shape[1], CFG["feat_dim"])
    P.weight.data.copy_(proj_sd["1.weight"][:CFG["feat_dim"], :])
    P.bias.data.copy_(proj_sd["1.bias"][:CFG["feat_dim"]])
    P = P.cpu().eval()
    
    train_cache = add_feat_targets(train_cache, P)
    val_cache = add_feat_targets(val_cache, P)
    
    # Build datasets
    tr_map_ft, tr_map_logits, tr_map_y = build_cache_maps(train_cache)
    va_map_ft, va_map_logits, va_map_y = build_cache_maps(val_cache)
    
    df = pd.read_csv(CFG["index_csv"])
    df = df[df["gid"].isin(backbone_gids)].copy()
    gid_to_local = {g:i for i,g in enumerate(backbone_gids)}
    
    # Simple transforms for on-device training
    train_tf = transforms.Compose([
        transforms.Resize((CFG["img_size"], CFG["img_size"])),
        transforms.RandomHorizontalFlip(p=0.5),  # Only basic flip
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize((CFG["img_size"], CFG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    ds_tr = DogWithCacheTargets(df, "train", train_tf, gid_to_local, tr_map_logits, tr_map_y, tr_map_ft)
    ds_va = DogWithCacheTargets(df, "test", val_tf, gid_to_local, va_map_logits, va_map_y, va_map_ft)
    
    dl_tr = DataLoader(ds_tr, batch_size=CFG["batch_size_train"], shuffle=True, num_workers=CFG["num_workers"])
    dl_va = DataLoader(ds_va, batch_size=CFG["batch_size_val"], shuffle=False, num_workers=CFG["num_workers"])
    
    # Build tuned MCU student model
    student = TunedMCUStudentCNN(
        in_channels=3,
        feat_dim=CFG["feat_dim"],
        num_classes=CFG["num_classes"],
        img_size=CFG["img_size"]
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in student.parameters())
    print(f"Student parameters: {total_params:,}")
    print(f"Estimated model size: ~{total_params * 4 / 1024:.1f}KB (FP32)")
    print(f"After quantization (INT8): ~{total_params / 1024:.1f}KB")
    print(f"After 50% pruning + INT8: ~{total_params / 2 / 1024:.1f}KB")
    
    # Improved optimizer and scheduler
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"]
    )
    
    # Better scheduler with warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CFG["lr"],
        epochs=CFG["epochs"],
        steps_per_epoch=len(dl_tr),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    
    # Training loop
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    
    print("\nStarting tuned MCU distillation training...")
    
    for epoch in range(CFG["epochs"]):
        # Training phase
        student.train()
        train_loss = 0.0
        train_kd_loss = 0.0
        train_feat_loss = 0.0
        train_ce_loss = 0.0
        batch_iter = tqdm(enumerate(dl_tr), total = len(dl_tr),  desc = f"On epoch: {epoch+1}/{CFG['epochs']}")
        for batch_idx, (xb, yb, tlog_cpu, tfeat_cpu) in batch_iter:
            xb = xb.to(device)
            yb = yb.to(device)
            tlog = tlog_cpu.to(device)
            tfeat = tfeat_cpu.to(device)
            
            # Forward pass
            slogits, sfeat = student(xb)
            
            # Compute losses
            kd_loss = tuned_kd_loss(slogits, tlog, CFG["kd_T"], CFG["kd_alpha"])
            feat_loss = tuned_feature_loss(sfeat, tfeat)
            ce_loss = F.cross_entropy(slogits, yb)
            
            # Combined loss
            total_loss = (CFG["kd_weight"] * kd_loss + 
                         CFG["feat_weight"] * feat_loss + 
                         CFG["ce_weight"] * ce_loss)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Log losses
            train_loss += total_loss.item()
            train_kd_loss += kd_loss.item()
            train_feat_loss += feat_loss.item()
            train_ce_loss += ce_loss.item()
        
        # Validation phase
        student.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for xb, yb, tlog_cpu, tfeat_cpu in dl_va:
                xb = xb.to(device)
                yb = yb.to(device)
                tlog = tlog_cpu.to(device)
                tfeat = tfeat_cpu.to(device)
                
                slogits, sfeat = student(xb)
                
                # Compute validation loss
                kd_loss = tuned_kd_loss(slogits, tlog, CFG["kd_T"], CFG["kd_alpha"])
                feat_loss = tuned_feature_loss(sfeat, tfeat)
                ce_loss = F.cross_entropy(slogits, yb)
                
                val_loss += (CFG["kd_weight"] * kd_loss + 
                           CFG["feat_weight"] * feat_loss + 
                           CFG["ce_weight"] * ce_loss).item()
                
                # Compute accuracy
                _, predicted = slogits.max(1)
                val_total += yb.size(0)
                val_correct += predicted.eq(yb).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        avg_train_loss = train_loss / len(dl_tr)
        avg_val_loss = val_loss / len(dl_va)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            patience_counter = 0
            
            #  CHECKPOINTING EVERY EPOCH!!! --> Load best model
            if best_state is not None:
                student.load_state_dict(best_state)
            
            print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
            
            # Save the tuned student

            save_dict = {
                "backbone": {k.replace("backbone.", ""): v for k, v in student.state_dict().items() if k.startswith("backbone.")},
                "proj": {k.replace("proj.", ""): v for k, v in student.state_dict().items() if k.startswith("proj.")},
                "classifier": {k.replace("classifier.", ""): v for k, v in student.state_dict().items() if k.startswith("classifier.")},
            }
            
            torch.save(save_dict, out_dir / "tuned_mcu_student.pth")
        else:
            patience_counter += 1
        
        # Print progress
        if epoch % 10 == 0 or epoch == CFG["epochs"] - 1:
            print(f"Epoch {epoch:3d}/{CFG['epochs']} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}% | "
                  f"Best: {best_val_acc:.2f}% | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if patience_counter >= CFG["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break
    

    
    meta_out = {
        "student_arch": "tuned_mcu_cnn",
        "feature_dim": CFG["feat_dim"],
        "img_size": CFG["img_size"],
        "normalization": {"mean": mean, "std": std},
        "backbone_gids": backbone_gids,
        "best_val_acc": float(best_val_acc),
        "total_params": total_params,
        "model_size_fp32_kb": total_params * 4 / 1024,
        "model_size_int8_kb": total_params / 1024,
        "model_size_pruned_int8_kb": total_params / 2 / 1024,  # 50% pruning estimate
    }
    
    (out_dir / "tuned_mcu_student_meta.json").write_text(json.dumps(meta_out, indent=2))
    
    print(f"\nSaved tuned MCU student to: {out_dir}")
    print(f"Model size: {total_params:,} parameters")
    print(f"FP32 size: ~{total_params * 4 / 1024:.1f}KB")
    print(f"INT8 size: ~{total_params / 1024:.1f}KB")
    print(f"After 50% pruning + INT8: ~{total_params / 2 / 1024:.1f}KB")
    print("Ready for pruning and quantization!")
    print("Done!")

if __name__ == "__main__":
    tuned_mcu_distillation()
