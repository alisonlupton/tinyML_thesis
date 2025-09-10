#distillation_1.py
#(cache KD) linear projector
'''
 learn a projector that maps teacher penultimate space (1280-D)  student feature space
 can use both teacher logits and teacher features
 
 Outputs: 
 - proj_from_teacher_penult: weights of the learned projector (Linear 1280128).
    - This tells the student what good 128-D features should look like (in teachers sense)!!
 -kd_head: a small linear head (128B) trained in feature space.
	-Useful for monitoring / sanity ( on CIL I will use my own TDM head)
 '''



import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils import load_config

### distillation helpers (move to utils later)


class StudentOnCache(nn.Module):
    def __init__(self, in_dim, feat_dim, num_classes, drop):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, elementwise_affine=False)
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(in_dim, feat_dim))
        self.kd_head = nn.Linear(feat_dim, num_classes)
    def forward(self, z_teacher_penult):
        z_in = self.norm(z_teacher_penult)
        z = self.proj(z_in)
        logits = self.kd_head(z)
        return logits, z


def kd_loss(student_logits, teacher_logits, T=2.0):
    return (T*T) * F.kl_div(
        F.log_softmax(student_logits/T, dim=1),
        F.softmax(teacher_logits/T, dim=1),
        reduction="batchmean",
    )
@torch.no_grad()
def log_logit_std(model, loader, device, prefix="train", max_batches=2):
    """Logs avg per-sample std of logits for student vs teacher."""
    was_training = model.training
    model.eval()                       #disable dropout for clean stats

    s_std_list, t_std_list = [], []
    batches = 0
    for xb, _, tb in loader:
        xb = xb.to(device)
        tb = tb.to(device)

        slogits, _ = model(xb)         #[N, C]
        #per-sample std across classes
        s_std = slogits.std(dim=1)     #[N]
        t_std = tb.std(dim=1)          #[N]

        s_std_list.append(s_std.mean().item())
        t_std_list.append(t_std.mean().item())

        batches += 1
        if batches >= max_batches:
            break

    if was_training:
        model.train()

    s_mean = sum(s_std_list) / max(1, len(s_std_list))
    t_mean = sum(t_std_list) / max(1, len(t_std_list))
    ratio = s_mean / max(1e-8, t_mean)
    print(f"[{prefix}] avg per-sample logit std: student{s_mean:.3f}, teacher{t_mean:.3f} (ratio{ratio:.2f})")
#-------------- per-class report --------------
@torch.no_grad()
def eval_per_class(model, loader, num_classes, device):
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


#-------------- train --------------
def train(student, opt, distillation_epochs, device, tr_dl, va_dl, alpha_kd, alpha_ce, kd_T, ce, alpha_feat, mse, distillation_patience, num_classes):
    best_val, best_state, stall = 0.0, None, 0
    for ep in range(distillation_epochs):
        student.train()
        loss_sum = 0.0
        for batch_idx, (xb, yb, tb) in enumerate(tr_dl):
            xb = xb.to(device); yb = yb.to(device); tb = tb.to(device)
            slogits, z = student(xb)
            if ep==0 and batch_idx==0:
                with torch.no_grad():
                    s_std = slogits.std(dim=1).mean().item()
                    t_std = tb.std(dim=1).mean().item()
                    print(f"[sanity] avg per-sample logit std: student{s_std:.3f}, teacher{t_std:.3f}")
            loss = alpha_kd*kd_loss(slogits, tb, kd_T) \
                + alpha_ce*ce(slogits, yb)
            #use cosine instead of mse
            if alpha_feat > 0:
                loss += alpha_feat * (1.0 - F.cosine_similarity(
                    F.normalize(z, dim=1), F.normalize(xb, dim=1), dim=1
                ).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())

        #val
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
            if stall >= distillation_patience:
                print(f"Early stopping at ep {ep} (best {best_val:.2f}%)")
                break
        log_logit_std(student, tr_dl, device, prefix="train", max_batches=2)
        log_logit_std(student, va_dl, device, prefix="val",   max_batches=2)
        #val per class
        cls_acc = eval_per_class(student, va_dl, num_classes, device)
        print("\nPer-class (student on cached features):")
        for i, acc in enumerate(cls_acc):
            print(f"  class {i:2d}: {acc:5.1f}%")
        print(f"Overall val acc (student): {best_val:.2f}%")
                
    return best_state, best_val

################### MAIN FUNCTION!!!!!!
##################
def distillation_1():
    #-------------- config --------------
    cfg = load_config()
    cache_dir = cfg['cache_dir']    
    out_dir = cfg['out_dir']
    feat_dim_student = cfg['feat_dim_student']
    distillation_batch_size_train = cfg['distillation_batch_size_train']
    distillation_batch_size_val = cfg['distillation_batch_size_val'] 
    distillation_epochs = cfg['distillation_epochs']
    distillation_patience = cfg['distillation_patience']
    distillation_lr = cfg['distillation_lr']
    distillation_weight_decay = cfg['distillation_weight_decay']
    distillation_label_smoothing = cfg['distillation_label_smoothing']
    distillation_alpha_kd = cfg['distillation_alpha_kd']
    distillation_kd_T = cfg['distillation_kd_T']
    distillation_alpha_ce = cfg['distillation_alpha_ce']
    distillation_alpha_feat = cfg['distillation_alpha_feat']
    distillation_dropout = 0.3
    distillation_dropout = cfg['']

    #NOTE: if distillation_alpha_feat and distillation_alpha_ce > 0 , we are no longer using class "logit only distillation"
    #we have
    #- KD term (teacher soft labels) controlled by distillation_alpha_kd, KD_T
	#- CE term (ground truth) controlled by distillation_alpha_ce
	#- Feature term (rep alignment) controlled by distillation_alpha_feat
 
    #alpha_kd = how much I care about KD?
	#kd_T = how soft is the teachers signal in KD?

    device = torch.device("cpu")

    #-------------- load caches --------------
    cache_dir = Path(cache_dir)
    train_cache = torch.load(cache_dir/"train_cache.pt", map_location="cpu", weights_only= True)
    val_cache   = torch.load(cache_dir/"val_cache.pt",   map_location="cpu", weights_only= True)
    
    
    #----------------load tensors --------------
    xtr = train_cache["features"].float()    #[Ntr, 1280] teacher penult features
    ytr = train_cache["y_local"].long()     #[Ntr]
    tlog_tr = train_cache["logits"].float()  #[Ntr, B]

    xva = val_cache["features"].float()      #[Nva, 1280]
    yva = val_cache["y_local"].long()       #[Nva]
    tlog_va = val_cache["logits"].float()    #[Nva, B]

    meta = train_cache["meta"]
    local_ids = meta["backbone_gids"]               #global IDs order used in head
    num_classes = tlog_tr.shape[1]                  #B
    feat_teacher = xtr.shape[1]                     #1280

    print(f"Train: feats={tuple(xtr.shape)} logits={tuple(tlog_tr.shape)}")
    print(f" Val : feats={tuple(xva.shape)} logits={tuple(tlog_va.shape)}")
    print(f"Classes: {num_classes} | teacher feat dim: {feat_teacher}")
    
    
    print("[sanity] tlog mean/std:", tlog_tr.mean().item(), tlog_tr.std().item())

    
    
    #-------------- student on cached features --------------
    #Learn a small projector (1280feat_dim) + KD head.
    #After training, we save only the projector weights as proj (for tiny CNN its the same shape).
    
    student = StudentOnCache(in_dim=feat_teacher, feat_dim=feat_dim_student,
                         num_classes=num_classes, drop= distillation_dropout).to(device)

    #-------------- loaders --------------
    tr_ds = TensorDataset(xtr, ytr, tlog_tr)
    va_ds = TensorDataset(xva, yva, tlog_va)
    tr_dl = DataLoader(tr_ds, batch_size=distillation_batch_size_train, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=distillation_batch_size_val, shuffle=False)
    
    #-------------- losses --------------

    ce = nn.CrossEntropyLoss(label_smoothing=distillation_label_smoothing)
    mse = nn.MSELoss()

    opt = torch.optim.AdamW(student.parameters(), lr=distillation_lr, weight_decay=distillation_weight_decay)
    
    best_state, best_val = train(student, opt, 
                       distillation_epochs,
                       device, tr_dl, va_dl, distillation_alpha_kd, 
                       distillation_alpha_ce, distillation_kd_T, ce, distillation_alpha_feat, 
                       mse, distillation_patience, num_classes)
    
    if best_state is not None:
        student.load_state_dict(best_state, strict=True)
        
        
    

    #-------------- save artifacts for CIL --------------
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    #Only need projector weights; the CIL tiny CNN backbone outputs 128 then projfeat_dim in CIL.
    save = {
        "proj_from_teacher_penult": {k.replace("proj.", ""): v for k, v in student.state_dict().items() if k.startswith("proj.")},
        "kd_head": {k.replace("kd_head.", ""): v for k, v in student.state_dict().items() if k.startswith("kd_head.")},
        "meta": {
            "teacher_penult_dim": feat_teacher,
            "student_feat_dim": feat_dim_student,
            "backbone_gids": local_ids,
            "val_best": float(best_val),
        }
    }
    torch.save(save, out/"student_from_cache.pth")
    (out/"student_from_cache_meta.json").write_text(json.dumps(save["meta"], indent=2))

    print("\nSaved:")
    print(f"   {out/'student_from_cache.pth'}")
    print(f"   {out/'student_from_cache_meta.json'}")


if __name__ == "__main__":
    distillation_1()
