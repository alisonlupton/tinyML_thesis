# utils.py

import torch
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import random
from PIL import Image

def load_config():
    """Load configuration."""
    with open("dog_breeds_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config
def build_tensors(sub_df, tf):
        Xs, ys = [], []
        for _, r in sub_df.iterrows():
            img = Image.open(r.img_path).convert("RGB")
            Xs.append(tf(img))
            ys.append(r.local)
        X = torch.stack(Xs, dim=0)   # (N, C, H, W)
        y = torch.tensor(ys, dtype=torch.long)
        return X, y

# def load_processed_dog_data():
#     """Load pre-processed dog data from intelligent sampling."""
#     print("Loading pre-processed dog data...")
    
#     # Check if processed data exists
#     processed_dir = Path("processed_data")
#     if not processed_dir.exists():
#         raise FileNotFoundError(
#             "Processed data not found! Please run process_dog_data.py first."
#         )
    
#     # Load metadata
#     metadata_file = processed_dir / "intelligent_sampling_metadata.yaml"
#     if not metadata_file.exists():
#         raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
#     with open(metadata_file, 'r') as f:
#         metadata = yaml.safe_load(f)
    
#     behavior_to_idx = metadata['behavior_to_idx']
#     behaviors = metadata['behaviors']
    
#     print(f"Loading data for {len(behaviors)} behaviors: {behaviors}")
    
#     # Load each dog's data
#     dog_data = {}
#     total_samples = 0
    
#     for dog_file in sorted(processed_dir.glob("dog_*_intelligent.npz")):
#         dog_id = int(dog_file.stem.split('_')[1])  # Extract dog ID from filename
        
#         # Load numpy data
#         data = np.load(dog_file)
#         X = data['X']  # Already (N, L, C) format
#         y = data['y']
        
#         sensor_cols = list(data['sensor_cols'])
#         L = int(data['window_len'])  # Time dimension
#         C = len(sensor_cols)         # Channel dimension
        
#         # Convert to (N, C, L) for Conv1d
#         X = np.transpose(X, (0, 2, 1)).astype(np.float32)  # (N, C, L)
        
#         # Load windowing data
#         seg_ids = data['segment_ids']
#         sess_ids = data.get('session_ids', None)
#         window_len = int(data.get('window_len', 0))
#         stride = int(data.get('stride', 0))
        
#         # Convert to tensors (raw data - no normalization yet)
#         dog_data[dog_id] = {
#             'X': torch.from_numpy(X),
#             'y': torch.from_numpy(y.astype(np.int64)),
#             'segment_ids': data['segment_ids'].astype(np.int32),
#             'session_ids': data['session_ids'].astype(np.int32) if 'session_ids' in data else None,
#             'behaviors': list(data['behaviors']),
#             'sensor_cols': sensor_cols,
#             'window_len': L,
#         }
        

#         total_samples += len(X)
    
#     # print(f"\nTotal samples loaded: {total_samples:,}")
    
#     # Analyze sample distribution
#     all_behaviors = []
#     for dog_id, data in dog_data.items():
#         all_behaviors.extend([behaviors[y] for y in data['y'].numpy()])
    
#     behavior_counts = pd.Series(all_behaviors).value_counts()
#     # print(f"\nSample behavior distribution:")
#     # for behavior, count in behavior_counts.items():
#     #     print(f"  {behavior}: {count:,} samples")
    
#     return dog_data, behavior_to_idx, behaviors


# def normalize_features(X, mean, std):
#      # X: (N, C, L) CNN version
#     return (X - mean) / std

# def compute_normalization_stats(X_train):
#     """Compute normalization statistics from training data only."""
#     mean = X_train.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
#     std  = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
#     return mean, std

def set_seed(seed):
    """Set random seeds for reproducibility across all libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---- Optional: KD / DDR loss between current & teacher on *previous* classes ----
def kd_loss_ce(current_logits, teacher_logits, T=2.0, weight=1.0):
    # logits expected over the same class rows
    pc = torch.log_softmax(current_logits / T, dim=1)
    pt = torch.softmax(teacher_logits / T, dim=1).detach()
    return weight * (T*T) * torch.nn.functional.kl_div(pc, pt, reduction='batchmean')


# evaluate 7-way logits (unseen -> -inf, effectively ignored if mask)
def evaluate_all_classes(model, loader, registry, all_behaviors, device):
    model.eval()
    correct = total = 0
    per_correct = {c: 0 for c in all_behaviors}
    per_total   = {c: 0 for c in all_behaviors}
    with torch.no_grad():
        for xb, yb7 in loader:
            xb, yb7 = xb.to(device), yb7.to(device)
            logits7 = model.forward_eval7(xb, registry)  # (N, 7)
            pred7   = logits7.argmax(dim=1)
            total  += yb7.size(0)
            correct += (pred7 == yb7).sum().item()
            for i, cname in enumerate(all_behaviors):
                m = (yb7 == i)
                per_total[cname] += m.sum().item()
                per_correct[cname] += (pred7[m] == i).sum().item()
    overall = 100.0 * correct / max(total, 1)
    per = {c: (100.0 * per_correct[c] / per_total[c]) if per_total[c] > 0 else 0.0 for c in all_behaviors}
    return overall, per

class ClassRegistry:
    """
    Maps global class IDs (gids, e.g., 0..119) to row indices in your expandable head.
    Optionally carries pretty names for logging.
    """
    def __init__(self, gid2name: dict | None = None):
        # gid -> row index (assigned in the order classes are added)
        self.row_for_gid: dict[int, int] = {}
        # row index -> gid (by position)
        self.gid_for_row: list[int] = []
        # optional name mapping for convenience/printing
        self.gid2name = {int(k): v for k, v in (gid2name or {}).items()}

    # ---- add / query ----
    def add_gids(self, gids: list[int]):
        """Append any unseen gids to the head layout."""
        for g in gids:
            g = int(g)
            if g in self.row_for_gid:
                continue
            self.row_for_gid[g] = len(self.gid_for_row)
            self.gid_for_row.append(g)

    def seen_gids(self) -> list[int]:
        """Return the set of all gids currently mapped to rows (in row order)."""
        return list(self.gid_for_row)

    def num_classes(self) -> int:
        return len(self.gid_for_row)

    # ---- row lookups ----
    def rows_for_gids(self, gids: list[int]) -> list[int]:
        """Return the row indices (in the same order as gids)."""
        return [self.row_for_gid[int(g)] for g in gids]

    def rows_for_all_known(self) -> list[int]:
        """Return [0..num_classes-1]."""
        return list(range(len(self.gid_for_row)))

    # ---- optional convenience for names ----
    def name_for_gid(self, gid: int) -> str:
        return self.gid2name.get(int(gid), f"gid_{int(gid)}")

# ----------------- Label mapping (gid -> local) -----------------
def make_global_to_local_map_gids(seen_gids: list[int], device: str | torch.device = "cpu"):
    """
    Create a vectorized map from global gid to local index [0..|seen|-1].
    Returns:
      m: tensor of shape (max_gid+1,), filled with -1 except for seen gids
      local_to_gid: LongTensor of shape (|seen|,) listing gids in ascending order
    """
    if len(seen_gids) == 0:
        return torch.empty((0,), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.long, device=device)

    seen_sorted = sorted(int(g) for g in seen_gids)
    max_gid = max(seen_sorted)
    m = torch.full((max_gid + 1,), -1, dtype=torch.long, device=device)
    local_to_gid = torch.tensor(seen_sorted, dtype=torch.long, device=device)
    m[local_to_gid] = torch.arange(len(seen_sorted), dtype=torch.long, device=device)
    return m, local_to_gid

# ----------------- (Optional) simple evaluation over seen classes -----------------
@torch.no_grad()
def eval_over_seen(model, loader, registry: ClassRegistry, local_to_gid: torch.LongTensor, device):
    """
    Evaluate accuracy over the SEEN classes currently in loader.
    loader yields (x, y_local) with local indices aligned to local_to_gid (gid order).
    """
    model.eval()
    rows_seen = torch.tensor(registry.rows_for_gids(local_to_gid.tolist()), device=device)

    total = 0
    correct = 0
    per_tot = [0] * len(local_to_gid)
    per_cor = [0] * len(local_to_gid)

    for xb, y_local in loader:
        xb, y_local = xb.to(device), y_local.to(device)
        feats = model._features(xb)
        logits = model.head.forward_rows(feats, rows_seen)
        pred = logits.argmax(1)

        total += y_local.size(0)
        correct += (pred == y_local).sum().item()

        for c in range(len(local_to_gid)):
            m = (y_local == c)
            if m.any():
                per_tot[c] += m.sum().item()
                per_cor[c] += (pred[m] == y_local[m]).sum().item()

    overall = 100.0 * correct / max(1, total)
    per_cls = [100.0 * per_cor[c] / per_tot[c] if per_tot[c] > 0 else 0.0 for c in range(len(local_to_gid))]
    return per_cls, overall