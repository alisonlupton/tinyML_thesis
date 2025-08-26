# replay.py
import random
from collections import defaultdict
import torch

@torch.no_grad()
def quantize_latents_perchannel_sym(feats, scales):
    q = torch.round(feats / scales)                  # [B, C] / [C]
    q = torch.clamp(q, -127, 127).to(torch.int8)
    return q

@torch.no_grad()
def dequantize_latents_perchannel_sym(q_feats, scales):
    return q_feats.to(torch.float32) * scales        # [B, C] * [C]
class BalancedQuantReplayDynamic:
    """
    Class-balanced INT8 latent replay with global buffer size.
    Uses shared per-channel symmetric scales (one vector) for all samples.
    """
    def __init__(self, buffer_size: int, feat_dim: int):
        self.buffer_size = int(buffer_size)
        self.feat_dim = int(feat_dim)
        self.q = defaultdict(list)      # class_id -> [int8 (C,), ...]
        self.labels = defaultdict(list) # mirrors q
        self.seen = []
        self._cap = None
        # shared scales
        self.scales = torch.ones(self.feat_dim, dtype=torch.float32)

    def set_scales(self, scales: torch.Tensor):
        assert scales.numel() == self.feat_dim
        self.scales = scales.detach().cpu().to(torch.float32).contiguous()

    def __len__(self):
        return sum(len(v) for v in self.q.values())

    def counts_per_class(self):
        return {cid: len(v) for cid, v in self.q.items()}

    def _recompute_cap(self):
        S = max(1, len(self.seen))
        self._cap = self.buffer_size // S

    def on_new_task(self, seen_class_ids):
        seen_set = set(self.seen)
        for cid in seen_class_ids:
            cid = int(cid)
            if cid not in seen_set:
                self.seen.append(cid)
        self._recompute_cap()
        self._rebalance_to_cap()

    def _rebalance_to_cap(self):
        if self._cap is None:
            self._recompute_cap()
        for cid in list(self.q.keys()):
            pool = self.q[cid]
            if len(pool) > self._cap:
                keep = set(random.sample(range(len(pool)), self._cap))
                self.q[cid] = [pool[i] for i in range(len(pool)) if i in keep]
                self.labels[cid] = [self.labels[cid][i] for i in range(len(self.labels[cid])) if i in keep]
        # global trim if rounding overflow
        while len(self) > self.buffer_size:
            cand = None; best = -1
            for cid in self.seen:
                L = len(self.q.get(cid, []))
                if L > best:
                    best, cand = L, cid
            if cand is None or best <= 0:
                break
            j = random.randrange(best)
            del self.q[cand][j]
            del self.labels[cand][j]

    @torch.no_grad()
    def add_batch(self, feats_f32: torch.Tensor, class_ids: torch.Tensor):
        feats_f32 = feats_f32.detach().cpu()                # [B, C]
        class_ids = class_ids.detach().cpu().tolist()
        q = quantize_latents_perchannel_sym(feats_f32, self.scales)  # [B, C] int8
        for f_q, y in zip(q, class_ids):
            y = int(y)
            self.q[y].append(f_q)
            self.labels[y].append(y)
        self._rebalance_to_cap()

    @torch.no_grad()
    def sample_q(self, batch_size: int, seen_class_ids, device="cpu"):
        seen = list(seen_class_ids)
        if len(seen) == 0:
            return None, None
        k = max(1, batch_size // len(seen))
        xs_q, ys = [], []

        # up to k per class
        for cid in seen:
            pool = self.q.get(int(cid), [])
            if len(pool) == 0: continue
            take = min(k, len(pool))
            idxs = random.sample(range(len(pool)), take)
            for j in idxs:
                xs_q.append(pool[j])
                ys.append(int(cid))

        # top-up
        if len(xs_q) < batch_size:
            need = batch_size - len(xs_q)
            flat = []
            for cid in seen:
                for j in range(len(self.q.get(int(cid), []))):
                    flat.append((int(cid), j))
            if flat:
                extra = random.sample(flat, min(need, len(flat)))
                for cid, j in extra:
                    xs_q.append(self.q[cid][j])
                    ys.append(int(cid))

        if not xs_q:
            return None, None
        if len(xs_q) > batch_size:
            idx = random.sample(range(len(xs_q)), batch_size)
            xs_q = [xs_q[i] for i in idx]
            ys   = [ys[i]   for i in idx]

        q_batch = torch.stack(xs_q).to(device)         # [B, C] int8
        feats = dequantize_latents_perchannel_sym(q_batch, self.scales.to(device))  # [B, C] float
        y = torch.tensor(ys, dtype=torch.long, device=device)
        return feats, y
    
    
# This is non quantized replay  (not currently using it)
class BalancedReplay:
    def __init__(self, per_class_cap):
        self.feats = {}   # cname -> [tensor, ...]
        self.labels = {}  # cname -> int
        self.cap = per_class_cap

    def add_batch(self, feats, labels, idx2cname):
        for f, y in zip(feats, labels):
            cname = idx2cname[int(y)]
            if cname not in self.feats: self.feats[cname] = []
            if len(self.feats[cname]) < self.cap:
                self.feats[cname].append(f.cpu())
            else:
                j = random.randint(0, len(self.feats[cname]) - 1)
                self.feats[cname][j] = f.cpu()

    def sample(self, batch_size, seen_class_list):
        if len(seen_class_list) == 0: return None, None
        k = max(1, batch_size // len(seen_class_list))
        xs, ys = [], []
        for cname in seen_class_list:
            pool = self.feats.get(cname, [])
            if len(pool) == 0: continue
            take = min(k, len(pool))
            idxs = random.sample(range(len(pool)), take)
            for j in idxs:
                xs.append(pool[j])
                ys.append(cname)
        if len(xs) == 0: return None, None
        X = torch.stack(xs)
        return X, ys
    

def update_replay_reservoir(replay_buffer, replay_labels, features, targets, buffer_size):
    """Update replay buffer using reservoir sampling."""
    for feat, target in zip(features, targets):
        if len(replay_buffer) < buffer_size:
            replay_buffer.append(feat)
            replay_labels.append(target.item())
        else:
            j = random.randint(0, len(replay_buffer) - 1)
            if j < buffer_size:
                replay_buffer[j] = feat
                replay_labels[j] = target.item()

def sample_replay(replay_buffer, replay_labels, batch_size=32):
    """Sample from replay buffer."""
    if len(replay_buffer) == 0:
        return None, None
    
    indices = random.sample(range(len(replay_buffer)), min(batch_size, len(replay_buffer)))
    features = torch.stack([replay_buffer[i] for i in indices])
    targets = torch.tensor([replay_labels[i] for i in indices], dtype=torch.long)
    return features, targets