# build_oracle_baselines.py
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
    X = t["feats"]  # [N, C] or [N, C, T]
    if X.dim() == 3:         # e.g., [N, 32, 25]
        X = X.mean(dim=2)    # GAP to [N, 32]
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

    # collect task masks
    train_masks = sorted(glob.glob(os.path.join(mask_dir, "mask_train_T*.npy")), key=_task_idx)
    test_masks  = sorted(glob.glob(os.path.join(mask_dir, "mask_T*.npy")),        key=_task_idx)
    assert len(train_masks) == len(test_masks) > 0

    oracle = {}
    for k, (mtr_p, mte_p) in enumerate(zip(train_masks, test_masks), start=1):
        mtr = torch.tensor(np.load(mtr_p).astype(bool))
        mte = torch.tensor(np.load(mte_p).astype(bool))

        Xtr_k = Xtr[mtr]
        ytr_g_k = ytr_g[mtr]
        Xte_k = Xte[mte]
        yte_g_k = yte_g[mte]

        # build local labels 0..(Ck-1) for this task
        uniq = torch.unique(ytr_g_k)
        # map globals to local
        g2l = {int(g): i for i, g in enumerate(uniq.tolist())}
        ytr_loc = torch.tensor([g2l[int(g)] for g in ytr_g_k.tolist()], dtype=torch.long)
        yte_loc = torch.tensor([g2l[int(g)] for g in yte_g_k.tolist() if int(g) in g2l], dtype=torch.long)

        # filter Xte_k to keep only samples whose globals exist in train uniq
        keep = torch.tensor([int(g) in g2l for g in yte_g_k.tolist()], dtype=torch.bool)
        Xte_k, yte_loc = Xte_k[keep], yte_loc

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