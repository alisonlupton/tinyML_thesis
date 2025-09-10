#computing_R.py
import os, re, glob, json
import numpy as np
import torch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.cnn import TunedMCUStudentCNN_CIL
from utils import ClassRegistry

TASK_RE = re.compile(r'T(\d+)')  #matches ...T1, T2, ...

def _extract_task_idx(path):
    m = TASK_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None

def load_registry(snapshot):
    #The registry is saved as a ClassRegistry object directly
    if isinstance(snapshot["registry"], ClassRegistry):
        return snapshot["registry"]
    else:
        #Fallback for dictionary format
        reg = ClassRegistry(gid2name=snapshot["registry"]["gid2name"])
        reg.row_for_gid = snapshot["registry"]["row_for_gid"]
        reg.gid_for_row = snapshot["registry"]["gid_for_row"]
        return reg

@torch.no_grad()
def eval_task_offline(head, registry, X_task_f32, y_task_local, target_13_gids):
    head.eval()
    
    #Map target GIDs to head rows
    rows = []
    for gid in target_13_gids:
        if int(gid) in registry.row_for_gid:
            rows.append(registry.row_for_gid[int(gid)])
        else:
            rows.append(-1)  #Unseen class
    
    #Get logits for known classes only
    known_rows = [r for r in rows if r >= 0]
    if len(known_rows) == 0:
        return 0.0  #No known classes
    
    logits_known = head.forward_rows(X_task_f32, known_rows)
    
    #Create full logits with -inf for unseen classes
    B, K = X_task_f32.size(0), len(target_13_gids)
    logits_full = torch.full((B, K), -1e9, device=X_task_f32.device)
    
    #Place known logits in correct positions
    known_cols = [i for i, r in enumerate(rows) if r >= 0]
    logits_full[:, known_cols] = logits_known
    
    preds = logits_full.argmax(dim=1).cpu().numpy()
    return (preds == y_task_local.cpu().numpy()).mean() * 100.0

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
    y_global_raw = t["y_global"]  #raw global labels (0..12 for 13 dogs)
    
    #Load run metadata to get target_13_gids
    with open("metrics_meta/run_meta.json", "r") as f:
        run_meta = json.load(f)
    target_13_gids = run_meta["target_13_gids"]
    
    #y_global_raw should already be in the correct 0..12 range for 13 dogs
    y_0to12 = y_global_raw

    #2) Collect and align masks, ckpts by task index

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
    

    row0 = 1 if include_backbone_row else 0
    col0 = 1 if include_backbone_row else 0

    for ckpt_idx, (i_task, ckpt_path) in enumerate(ckpts):
        row_idx = ckpt_idx + row0
        snap = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        checkpoint_in_dim = snap["checkpoint_in_dim"]
        
        #Get the actual number of classes from the head state dict
        head_state = snap["head_state_dict"]
        actual_num_classes = head_state["linear.weight"].shape[0]
        
        #Create a dummy CIL model with the correct number of classes
        cil_model = TunedMCUStudentCNN_CIL(3, actual_num_classes, "cpu", 0.0, checkpoint_in_dim)
        cil_model.head.load_state_dict(snap["head_state_dict"], strict=True)
        head = cil_model.head
        registry = load_registry(snap)

        #cumulative masks --> R
        for mask_idx, (j_task, mask_path) in enumerate(cum_masks):
            col_idx = mask_idx + col0
            class_mask = np.load(mask_path).astype(bool)
            if not class_mask.any():
                R[row_idx, col_idx] = 0.0
            else:
                #Filter samples by their class labels (which classes are included)
                sample_mask = torch.zeros(len(y_0to12), dtype=torch.bool)
                for i, gid in enumerate(target_13_gids):
                    if class_mask[i]:  #If this class is included
                        sample_mask |= (y_0to12 == i)  #Include all samples of this class
                
                if not sample_mask.any():
                    R[row_idx, col_idx] = 0.0
                else:
                    Xj = X_all[sample_mask]; yj = y_0to12[sample_mask]
                    if Xj.shape[1] != checkpoint_in_dim:
                        if Xj.shape[1] > checkpoint_in_dim:
                            Xj = Xj[:, :checkpoint_in_dim]
                        else:
                            Xj = torch.cat([Xj, torch.zeros(Xj.shape[0], checkpoint_in_dim - Xj.shape[1])], dim=1)
                    R[row_idx, col_idx] = eval_task_offline(head, registry, Xj, yj, target_13_gids)
        
        #For each checkpoint, test only on the new class for that specific task
        for mask_idx, (j_task, mask_path) in enumerate(new_masks):
            col_idx = mask_idx + col0
            class_mask = np.load(mask_path).astype(bool)
            if not class_mask.any():
                R_new[row_idx, col_idx] = 0.0
            else:
                #Filter samples by their class labels (which classes are included)
                sample_mask = torch.zeros(len(y_0to12), dtype=torch.bool)
                for i, gid in enumerate(target_13_gids):
                    if class_mask[i]:  #If this class is included
                        sample_mask |= (y_0to12 == i)  #Include all samples of this class
                
                if not sample_mask.any():
                    R_new[row_idx, col_idx] = 0.0
                else:
                    Xj = X_all[sample_mask]; yj = y_0to12[sample_mask]
                    if Xj.shape[1] != checkpoint_in_dim:
                        if Xj.shape[1] > checkpoint_in_dim:
                            Xj = Xj[:, :checkpoint_in_dim]
                        else:
                            Xj = torch.cat([Xj, torch.zeros(Xj.shape[0], checkpoint_in_dim - Xj.shape[1])], dim=1)
                    R_new[row_idx, col_idx] = eval_task_offline(head, registry, Xj, yj, target_13_gids)
    
    return R, R_new


if __name__ == "__main__":
    R, R_new = build_R_offline(
        test_feats_path="test_feats.pt",
        ckpt_dir="checkpoints_cil", 
        masks_dir="task_masks",      
        include_backbone_row=False
    )
    np.save("R_offline.npy",     np.round(R,     2))
    np.save("R_offline_new.npy", np.round(R_new, 2))
    print("Saved R_offline.npy", R.shape, "and R_offline_new.npy", R_new.shape)