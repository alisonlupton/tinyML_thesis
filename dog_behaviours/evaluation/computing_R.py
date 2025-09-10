#computing_R.py
import os, re, glob, json
import numpy as np
import torch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.cnn_head import SimplifiedTDMHead
from utils import ClassRegistry

TASK_RE = re.compile(r'T(\d+)')  #matches ...T1, T2, ...

def _extract_task_idx(path):
    m = TASK_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None

def load_registry(snapshot):
    reg = ClassRegistry(snapshot["registry"]["all_behaviors"])
    reg.row_for_global = snapshot["registry"]["row_for_global"]
    reg.global_for_row = snapshot["registry"]["global_for_row"]
    return reg

@torch.no_grad()
def eval_task_offline(head, registry, X_task_f32, y_task_global_0to6):
    head.eval()
    logits = head.forward_eval7(X_task_f32, registry)
    preds = logits.argmax(dim=1).cpu().numpy()
    return (preds == y_task_global_0to6.cpu().numpy()).mean() * 100.0

def eval_backbone_offline(head, X_task_f32, y_task_global_0to6):
    """Evaluate backbone (T0) checkpoint using simple forward pass for 4 classes only"""
    head.eval()
    with torch.no_grad():
        logits = head.linear(X_task_f32)  #Direct linear layer for 4 classes
        preds = logits.argmax(dim=1).cpu().numpy()
        return (preds == y_task_global_0to6.cpu().numpy()).mean() * 100.0

def build_R_offline(
    test_feats_path="test_feats.pt",
    ckpt_dir="checkpoints_cil",
    masks_dir="task_masks",
    include_backbone_row=False
):
    #1) Load frozen test features + mapping
    t = torch.load(test_feats_path, map_location="cpu", weights_only=False)
    X_all = t["feats"]          #[N, 32, 25] - need to flatten to [N, 800]
    
    #Process features to match the expected input dimension
    if X_all.dim() == 3:  #[N, 32, 25] - need to apply GAP to get [N, 32]
        #Apply Global Average Pooling like the backbone does
        X_all = X_all.mean(dim=2)  #[N, 32]
    elif X_all.dim() > 2:
        X_all = X_all.view(X_all.size(0), -1)   #fallback: flatten
    y_global_raw = t["y_global"]  #raw global labels
    m_test = t.get("m_test", None)
    if m_test is None:
        raise RuntimeError("m_test not found in test_feats.pt; save it during training.")
    y_0to6 = m_test[y_global_raw]  #map to 0..6 consistent with forward_eval7 order

    #2) Collect and align masks ckpts by task index

    cum_mask_paths = glob.glob(os.path.join(masks_dir, "mask_T*.npy"))
    cum_mask_paths = [p for p in cum_mask_paths if "_new" not in p]  #exclude _new files
    new_mask_paths = glob.glob(os.path.join(masks_dir, "mask_T*_new.npy"))
    ckpt_paths     = glob.glob(os.path.join(ckpt_dir,  "head_after_T*.pt"))

    def _keep_and_sort(paths):
        pairs = [(idx, p) for p in paths if (idx := _extract_task_idx(p)) is not None]
        pairs.sort(key=lambda x: x[0])
        return pairs

    cum_masks = _keep_and_sort(cum_mask_paths)     #(T, path) cumulative
    new_masks = _keep_and_sort(new_mask_paths)     #(T, path) new-class-only
    ckpts     = _keep_and_sort(ckpt_paths)         #(T, path)

    #Intersect on task indices so we only use consistent sets
    cum_ids = {i for i, _ in cum_masks}
    new_ids = {i for i, _ in new_masks}
    ckp_ids = {i for i, _ in ckpts}
    
    #Exclude T0 (backbone) - only use T1, T2, T3 for CIL evaluation
    common = sorted(cum_ids & new_ids & ckp_ids)
    if not common:
        raise AssertionError(
            f"No common task indices. "
            f"cumulative={sorted(cum_ids)}, new={sorted(new_ids)}, ckpts={sorted(ckp_ids)}"
        )

    cum_masks = [(i, p) for i, p in cum_masks if i in common]
    new_masks = [(i, p) for i, p in new_masks if i in common]
    ckpts     = [(i, p) for i, p in ckpts     if i in common]

    K = len(common)
    Rdim = K + (1 if include_backbone_row else 0)
    R       = np.zeros((Rdim, Rdim), dtype=float)  #cumulative
    R_new   = np.zeros((Rdim, Rdim), dtype=float)  #new-class-only
    

    #index offset if decide to include a backbone row later
    row0 = 1 if include_backbone_row else 0
    col0 = 1 if include_backbone_row else 0

    #--- replace double for-loop (ckpts masks) with this ---
    for ckpt_idx, (i_task, ckpt_path) in enumerate(ckpts):
        row_idx = ckpt_idx + row0
        snap = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        seen_names = snap["registry"]["seen_names"]
        checkpoint_in_dim = snap["head_state"]["linear.weight"].shape[1]

        head = SimplifiedTDMHead(in_dim=checkpoint_in_dim,
                                init_num_classes=len(seen_names),
                                device="cpu", sparsity_ratio=0.0)
        head.load_state_dict(snap["head_state"], strict=True)
        registry = load_registry(snap)

        #cumulative masks R
        for mask_idx, (j_task, mask_path) in enumerate(cum_masks):
            col_idx = mask_idx + col0
            m = np.load(mask_path).astype(bool)
            if not m.any():
                R[row_idx, col_idx] = 0.0
            else:
                Xj = X_all[m]; yj = y_0to6[m]
                if Xj.shape[1] != checkpoint_in_dim:
                    if Xj.shape[1] > checkpoint_in_dim:
                        Xj = Xj[:, :checkpoint_in_dim]
                    else:
                        Xj = torch.cat([Xj, torch.zeros(Xj.shape[0], checkpoint_in_dim - Xj.shape[1])], dim=1)
                R[row_idx, col_idx] = eval_task_offline(head, registry, Xj, yj)
        

        #new-class-only masks R_new (for "pure FWT" on superdiagonal later)
        #For each checkpoint, test only on the new class for that specific task
        for mask_idx, (j_task, mask_path) in enumerate(new_masks):
            col_idx = mask_idx + col0
            m = np.load(mask_path).astype(bool)
            if not m.any():
                R_new[row_idx, col_idx] = 0.0
            else:
                Xj = X_all[m]; yj = y_0to6[m]
                if Xj.shape[1] != checkpoint_in_dim:
                    if Xj.shape[1] > checkpoint_in_dim:
                        Xj = Xj[:, :checkpoint_in_dim]
                    else:
                        Xj = torch.cat([Xj, torch.zeros(Xj.shape[0], checkpoint_in_dim - Xj.shape[1])], dim=1)
                R_new[row_idx, col_idx] = eval_task_offline(head, registry, Xj, yj)
    
    return R, R_new


if __name__ == "__main__":
    R, R_new = build_R_offline(
        test_feats_path="test_feats.pt",
        ckpt_dir="checkpoints_cil",  #<- files are in current evaluation directory
        masks_dir="task_masks",      #<- files are in current evaluation directory
        include_backbone_row=False
    )
    np.save("R_offline.npy",     np.round(R,     2))
    np.save("R_offline_new.npy", np.round(R_new, 2))
    print("Saved R_offline.npy", R.shape, "and R_offline_new.npy", R_new.shape)