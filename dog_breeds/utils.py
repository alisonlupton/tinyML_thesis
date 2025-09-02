# utils.py

import torch
import yaml
import numpy as np
import random
from PIL import Image
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
torch.set_num_threads(1)  # single-thread CPU
from tqdm import tqdm

#####
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


def set_seed(seed):
    """Set random seeds for reproducibility across all libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---- KD / DDR loss between current & teacher on *previous* classes ----
def kd_loss_ce(current_logits, teacher_logits, T=2.0, weight=1.0):
    # logits expected over the same class rows
    pc = torch.log_softmax(current_logits / T, dim=1)
    pt = torch.softmax(teacher_logits / T, dim=1).detach()
    return weight * (T*T) * torch.nn.functional.kl_div(pc, pt, reduction='batchmean')
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
    def gather_rows_for_eval_fixed(self, fixed_gids: list[int]) -> list[int]:
        """
        For a fixed ordered list of global IDs (e.g., your 13-dog eval set),
        return a same-length list of row indices into the current head.
        Unseen classes return -1.
        """
        rows = []
        for g in fixed_gids:
            g = int(g)
            rows.append(self.row_for_gid.get(g, -1))
        return rows

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
        z = model.proj(feats)
        logits = model.head.forward_rows(z, rows_seen)
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