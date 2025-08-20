import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# -------------- config --------------
CFG = {
    "cache_dir": "../data",   # where train_cache.pt / val_cache.pt live
    "out_dir":   "../data",
    "feat_dim_student": 128,        # must match your CIL feat_dim
    "batch_size_train": 256,        # you can go bigger now (no teacher forward)
    "batch_size_val":   256,
    "epochs": 50,
    "patience": 8,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "label_smoothing": 0.05,
    "alpha_kd": 1.0,                # KL distillation weight (teacher logits)
    "kd_T": 2.0,                    # temperature
    "alpha_ce": 0.5,                # optional CE to ground truth
    "alpha_feat": 0.0,              # set >0.0 to also match penultimate features (MSE)
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------- load caches --------------
cache_dir = Path(CFG["cache_dir"])
train_cache = torch.load(cache_dir/"train_cache.pt", map_location="cpu")
val_cache   = torch.load(cache_dir/"val_cache.pt",   map_location="cpu")

# tensors
xtr = train_cache["features"]    # [Ntr, 1280]  teacher penult features
ytr = train_cache["y_local"]     # [Ntr]
tlog_tr = train_cache["logits"]  # [Ntr, B]

xva = val_cache["features"]      # [Nva, 1280]
yva = val_cache["y_local"]       # [Nva]
tlog_va = val_cache["logits"]    # [Nva, B]

meta = train_cache["meta"]
local_ids = meta["backbone_gids"]               # global IDs order used in head
num_classes = tlog_tr.shape[1]                  # B
feat_teacher = xtr.shape[1]                     # 1280

print(f"Train: feats={tuple(xtr.shape)} logits={tuple(tlog_tr.shape)}")
print(f" Val : feats={tuple(xva.shape)} logits={tuple(tlog_va.shape)}")
print(f"Classes: {num_classes} | teacher feat dim: {feat_teacher}")

# -------------- student on cached features --------------
# We’re not folding the whole CNN here; we learn a small projector (1280→feat_dim) + KD head.
# After training, we save only the projector weights as your “proj” (for your tiny CNN it’s the same shape).
class StudentOnCache(nn.Module):
    def __init__(self, in_dim=1280, feat_dim=128, num_classes=10, drop=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(in_dim, feat_dim))
        self.kd_head = nn.Linear(feat_dim, num_classes)

    def forward(self, z_teacher_penult):
        z = self.proj(z_teacher_penult)   # [N, feat_dim]
        logits = self.kd_head(z)
        return logits, z

student = StudentOnCache(in_dim=feat_teacher, feat_dim=CFG["feat_dim_student"],
                         num_classes=num_classes, drop=0.3).to(device)

# -------------- loaders --------------
tr_ds = TensorDataset(xtr, ytr, tlog_tr)
va_ds = TensorDataset(xva, yva, tlog_va)
tr_dl = DataLoader(tr_ds, batch_size=CFG["batch_size_train"], shuffle=True)
va_dl = DataLoader(va_ds, batch_size=CFG["batch_size_val"], shuffle=False)

# -------------- losses --------------
def kd_loss(student_logits, teacher_logits, T=2.0):
    return (T*T) * F.kl_div(
        F.log_softmax(student_logits/T, dim=1),
        F.softmax(teacher_logits/T, dim=1),
        reduction="batchmean",
    )

ce = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])
mse = nn.MSELoss()

opt = torch.optim.AdamW(student.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])

# -------------- train --------------
best_val, best_state, stall = 0.0, None, 0
for ep in range(CFG["epochs"]):
    student.train()
    loss_sum = 0.0
    for xb, yb, tb in tr_dl:
        xb = xb.to(device); yb = yb.to(device); tb = tb.to(device)
        slogits, z = student(xb)
        loss = CFG["alpha_kd"]*kd_loss(slogits, tb, CFG["kd_T"]) \
             + CFG["alpha_ce"]*ce(slogits, yb)
        if CFG["alpha_feat"] > 0:
            # optionally match teacher penultimate features (xb) to student features (z)
            loss = loss + CFG["alpha_feat"]*mse(
                F.normalize(z, dim=1),
                F.normalize(xb, dim=1)
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_sum += float(loss.item())

    # val
    student.eval()
    tot = cor = 0
    with torch.no_grad():
        for xb, yb, _ in va_dl:
            xb = xb.to(device); yb = yb.to(device)
            slogits, _ = student(xb)
            pred = slogits.argmax(1)
            tot += yb.size(0); cor += (pred == yb).sum().item()
    v_acc = 100.0 * cor / max(1, tot)

    print(f"ep {ep:02d} | train_loss={loss_sum/len(tr_dl):.3f} | val_acc={v_acc:.2f}%")
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

# -------------- per-class report --------------
@torch.no_grad()
def per_class(model, loader):
    tot = [0]*num_classes
    cor = [0]*num_classes
    model.eval()
    for xb, yb, _ in loader:
        xb = xb.to(device); yb = yb.to(device)
        slogits, _ = model(xb)
        pred = slogits.argmax(1)
        for c in range(num_classes):
            m = (yb == c)
            n = int(m.sum().item())
            if n > 0:
                tot[c] += n
                cor[c] += int((pred[m] == yb[m]).sum().item())
    return [100.0*cor[c]/tot[c] if tot[c]>0 else 0.0 for c in range(num_classes)]

cls_acc = per_class(student, va_dl)
print("\nPer-class (student on cached features):")
for i, acc in enumerate(cls_acc):
    print(f"  class {i:2d}: {acc:5.1f}%")
print(f"Overall val acc (student): {best_val:.2f}%")

# -------------- save artifacts for your CIL --------------
out = Path(CFG["out_dir"]); out.mkdir(parents=True, exist_ok=True)

# We only need the projector weights; your tiny CNN backbone outputs 128 then proj→feat_dim in your CIL.
# If your tiny CNN already projects 128→feat_dim, you can use this matrix to initialize that layer via name mapping.
save = {
    "proj_from_teacher_penult": {k.replace("proj.", ""): v for k, v in student.state_dict().items() if k.startswith("proj.")},
    "kd_head": {k.replace("kd_head.", ""): v for k, v in student.state_dict().items() if k.startswith("kd_head.")},
    "meta": {
        "teacher_penult_dim": feat_teacher,
        "student_feat_dim": CFG["feat_dim_student"],
        "backbone_gids": local_ids,
        "val_best": float(best_val),
    }
}
torch.save(save, out/"student_from_cache.pth")
(out/"student_from_cache_meta.json").write_text(json.dumps(save["meta"], indent=2))

print("\nSaved:")
print(f"  • {out/'student_from_cache.pth'}")
print(f"  • {out/'student_from_cache_meta.json'}")