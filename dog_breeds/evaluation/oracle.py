#build_oracle_baselines.py
import os, glob, re, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

TASK_RE = re.compile(r'T(\d+)')

def _task_idx(p): 
    m = TASK_RE.search(os.path.basename(p)); 
    return int(m.group(1)) if m else None

@torch.no_grad()
def _acc(logits, y):
    return (logits.argmax(1) == y).float().mean().item() * 100.0

def _prep_feats(t):
    X = t["feats"]  #[N, C] or [N, C, T]
    if X.dim() == 3:         #e.g., [N, 32, 25]
        X = X.mean(dim=2)    #GAP to [N, 32]
    elif X.dim() > 2:
        X = X.view(X.size(0), -1)
    y_g = t["y_global"]
    m_all = t.get("m_all", None)
    return X, y_g, m_all

def main(train_path="train_feats.pt", test_path="test_feats.pt", mask_dir="task_masks",
         epochs=10, bs=256, lr=1e-2, wd=0.0, device="cpu", out_path="metrics_meta/oracle_baseline.json"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ttr = torch.load(train_path, map_location="cpu", weights_only=False)
    tte = torch.load(test_path,  map_location="cpu", weights_only=False)
    Xtr, ytr_g, m_all = _prep_feats(ttr)
    Xte, yte_g, _     = _prep_feats(tte)

    #collect task masks (use full masks, not _new masks for proper Intransigence calculation)
    train_masks = sorted(glob.glob(os.path.join(mask_dir, "mask_train_T*.npy")), key=_task_idx)
    test_masks  = sorted(glob.glob(os.path.join(mask_dir, "mask_T*.npy")),        key=_task_idx)
    
    #Filter out _new masks to use only full cumulative masks
    train_masks = [m for m in train_masks if '_new' not in m]
    test_masks  = [m for m in test_masks if '_new' not in m]
    assert len(train_masks) == len(test_masks) > 0

    #Load target_13_gids for class mapping
    with open("metrics_meta/run_meta.json", "r") as f:
        run_meta = json.load(f)
    target_13_gids = run_meta["target_13_gids"]
    
    oracle = {}
    for k, (mtr_p, mte_p) in enumerate(zip(train_masks, test_masks), start=1):
        class_mask_tr = np.load(mtr_p).astype(bool)
        class_mask_te = np.load(mte_p).astype(bool)
        
        #Filter samples by their class labels
        sample_mask_tr = torch.zeros(len(ytr_g), dtype=torch.bool)
        sample_mask_te = torch.zeros(len(yte_g), dtype=torch.bool)
        
        for i, gid in enumerate(target_13_gids):
            if class_mask_tr[i]:  #If this class is included in training
                sample_mask_tr |= (ytr_g == i)
            if class_mask_te[i]:  #If this class is included in testing
                sample_mask_te |= (yte_g == i)

        Xtr_k = Xtr[sample_mask_tr]
        ytr_g_k = ytr_g[sample_mask_tr]
        Xte_k = Xte[sample_mask_te]
        yte_g_k = yte_g[sample_mask_te]
        

        #build local labels 0..(Ck-1) for this task
        uniq = torch.unique(ytr_g_k)
        #map globals to local
        g2l = {int(g): i for i, g in enumerate(uniq.tolist())}
        ytr_loc = torch.tensor([g2l[int(g)] for g in ytr_g_k.tolist()], dtype=torch.long)
        yte_loc = torch.tensor([g2l[int(g)] for g in yte_g_k.tolist() if int(g) in g2l], dtype=torch.long)

        #filter Xte_k to keep only samples whose globals exist in train uniq
        keep = torch.tensor([int(g) in g2l for g in yte_g_k.tolist()], dtype=torch.bool)
        Xte_k, yte_loc = Xte_k[keep], yte_loc

            
        nclass = len(uniq)
        if nclass < 2:
            oracle[f"T{k}"] = float("nan")
            print(f"[oracle] T{k}: skipped (C={nclass})")
            continue

        in_dim  = Xtr_k.shape[1]
        nclass  = len(uniq)
        head = nn.Linear(in_dim, nclass, bias=True).to(device)
        head.train()

        ds_tr = TensorDataset(Xtr_k.to(device), ytr_loc.to(device))
        dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
        opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        crit = nn.CrossEntropyLoss()

        for _ in range(epochs):
            for xb, yb in dl_tr:
                opt.zero_grad()
                loss = crit(head(xb), yb)
                loss.backward()
                opt.step()

        head.eval()
        with torch.no_grad():
            acc = _acc(head(Xte_k.to(device)), yte_loc.to(device))

        oracle[f"T{k}"] = acc
        print(f"[oracle] T{k}: {acc:.2f}% (C={nclass}, in_dim={in_dim})")

    with open(out_path, "w") as f:
        json.dump(oracle, f, indent=2)
    print(f"Saved oracle baselines to {out_path}")

if __name__ == "__main__":
    main()