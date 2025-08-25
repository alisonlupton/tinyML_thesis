# utils.py

import torch
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import random

def load_config():
    """Load configuration."""
    with open("dog_behaviors_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def load_processed_dog_data():
    """Load pre-processed dog data from intelligent sampling."""
    print("Loading pre-processed dog data...")
    
    # Check if processed data exists
    processed_dir = Path("processed_data")
    if not processed_dir.exists():
        raise FileNotFoundError(
            "Processed data not found! Please run process_dog_data.py first."
        )
    
    # Load metadata
    metadata_file = processed_dir / "intelligent_sampling_metadata.yaml"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = yaml.safe_load(f)
    
    behavior_to_idx = metadata['behavior_to_idx']
    behaviors = metadata['behaviors']
    
    print(f"Loading data for {len(behaviors)} behaviors: {behaviors}")
    
    # Load each dog's data
    dog_data = {}
    total_samples = 0
    
    for dog_file in sorted(processed_dir.glob("dog_*_intelligent_new.npz")):
        dog_id = int(dog_file.stem.split('_')[1])  # Extract dog ID from filename
        
        # Load numpy data
        data = np.load(dog_file)
        X = data['X']  # Already (N, L, C) format
        y = data['y']
        
        sensor_cols = list(data['sensor_cols'])
        L = int(data['window_len'])  # Time dimension
        C = len(sensor_cols)         # Channel dimension
        
        # Convert to (N, C, L) for Conv1d
        X = np.transpose(X, (0, 2, 1)).astype(np.float32)  # (N, C, L)
        
        # Load windowing data
        seg_ids = data['segment_ids']
        sess_ids = data.get('session_ids', None)
        window_len = int(data.get('window_len', 0))
        stride = int(data.get('stride', 0))
        sessions = data['sessions']
        
        check = False
        if len(sessions) == 2:
            check = True
        
        # Convert to tensors (raw data - no normalization yet)
        dog_data[dog_id] = {
            'X': torch.from_numpy(X),
            'y': torch.from_numpy(y.astype(np.int64)),
            'segment_ids': data['segment_ids'].astype(np.int32),
            'session_ids': data['session_ids'].astype(np.int32) if 'session_ids' in data else None,
            'behaviors': list(data['behaviors']),
            'sensor_cols': sensor_cols,
            'window_len': L,
            'multiple_sessions': check
            }
        

        total_samples += len(X)
    
    # print(f"\nTotal samples loaded: {total_samples:,}")
    
    # Analyze sample distribution
    all_behaviors = []
    for dog_id, data in dog_data.items():
        all_behaviors.extend([behaviors[y] for y in data['y'].numpy()])
    
    behavior_counts = pd.Series(all_behaviors).value_counts()
    # print(f"\nSample behavior distribution:")
    # for behavior, count in behavior_counts.items():
    #     print(f"  {behavior}: {count:,} samples")
    
    return dog_data, behavior_to_idx, behaviors


def normalize_features(X, mean, std):
     # X: (N, C, L) CNN version
    return (X - mean) / std

def compute_normalization_stats(X_train):
    """Compute normalization statistics from training data only."""
    mean = X_train.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
    std  = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    return mean, std

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
    def __init__(self, all_behaviors):
        self.all_behaviors = list(all_behaviors)  # fixed universe (7 for now)
        self.row_for_global = {}   # e.g., {'Standing': 0, 'Walking': 1, ...} for SEEN classes
        self.global_for_row = []   # inverse mapping for current head rows

    def seen_classes(self):
        return list(self.row_for_global.keys())

    def add_classes(self, new_class_names):
        for cname in new_class_names:
            if cname in self.row_for_global:
                continue
            self.row_for_global[cname] = len(self.global_for_row)
            self.global_for_row.append(cname)

    def rows_for_task(self, task_classes):
        return [self.row_for_global[c] for c in task_classes]

    def rows_for_all_known(self):
        return list(range(len(self.global_for_row)))

    def gather_rows_for_eval7(self):
        """Return indices into current classifier rows in the order of all_behaviors.
        Unseen classes will be marked as -1."""
        rows = []
        for cname in self.all_behaviors:
            rows.append(self.row_for_global.get(cname, -1))
        return rows
    def num_classes(self):
        return len(self.global_for_row)

    def rows_for(self, class_names):
        return [self.row_for_global[c] for c in class_names]
    
    def make_global_to_local_map(self, class_names, behavior_to_idx, device=None):
        """
        Given a list of class names for the current task,
        return a 1D tensor mapping from global label IDs (in behavior_to_idx space)
        to local label IDs (0..len(class_names)-1), or -1 if not in class_names.
        """
        n_classes = max(behavior_to_idx.values()) + 1
        mapping = torch.full((n_classes,), -1, dtype=torch.long, device=device)
        for local_id, cname in enumerate(class_names):
            global_id = behavior_to_idx[cname]
            mapping[global_id] = local_id
        return mapping

def make_global_to_local_map(class_names, behavior_to_idx, device="cpu"):
    max_gid = max(behavior_to_idx.values())
    m = torch.full((max_gid + 1,), -1, dtype=torch.long, device=device)
    gids = torch.tensor([behavior_to_idx[c] for c in class_names], dtype=torch.long, device=device)
    m[gids] = torch.arange(len(class_names), dtype=torch.long, device=device)
    return m, gids
