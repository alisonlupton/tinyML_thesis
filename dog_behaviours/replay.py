#replay.py
import random
from collections import defaultdict
import torch

@torch.no_grad()
def quantize_tensor_sym(x: torch.Tensor):
    amax = x.abs().max()
    if amax == 0:
        scale = torch.tensor(1.0, dtype=torch.float32)
        q = torch.zeros_like(x, dtype=torch.int8)
        return q, scale
    scale = (amax / 127.0).to(torch.float32)
    q = torch.clamp((x / scale).round(), -128, 127).to(torch.int8)
    return q, scale

@torch.no_grad()
def dequantize_tensor_sym(q: torch.Tensor, scale: torch.Tensor):
    return q.to(torch.float32) * scale

class BalancedQuantReplayDynamic:
    """
    Class-balanced INT8 latent replay with a *global* buffer budget.
    - Uses all memory early; rebalances when new classes appear.
    - Evicts uniformly-at-random from classes that exceed their current cap.
    """
    def __init__(self, buffer_size: int):
        self.buffer_size = int(buffer_size)
        self.q = defaultdict(list)       #class_id -> [int8 (D,)]
        self.s = defaultdict(list)       #class_id -> [float32 scale]
        self.seen = []                   #list of seen class_ids (dedup, order stable)
        self._cap = None                 #current per-class cap

    def __len__(self):
        return sum(len(v) for v in self.q.values())

    def counts_per_class(self):
        return {cid: len(v) for cid, v in self.q.items()}

    def _recompute_cap(self):
        S = max(1, len(self.seen))
        self._cap = self.buffer_size // S

    def on_new_task(self, seen_class_ids):
        """Call once per task (after registry update)."""
        #Keep order stable but dedup
        seen_set = set(self.seen)
        for cid in seen_class_ids:
            if cid not in seen_set:
                self.seen.append(int(cid))
                seen_set.add(int(cid))
        self._recompute_cap()
        self._rebalance_to_cap()

    def _rebalance_to_cap(self):
        """Trim classes that exceed the per-class cap; if still over budget, trim largest pools."""
        if self._cap is None:
            self._recompute_cap()

        #First, cap each class individually
        for cid in list(self.q.keys()):
            pool_q = self.q[cid]
            pool_s = self.s[cid]
            if len(pool_q) > self._cap:
                #randomly keep 'cap' items
                keep = set(random.sample(range(len(pool_q)), self._cap))
                self.q[cid] = [pool_q[i] for i in range(len(pool_q)) if i in keep]
                self.s[cid] = [pool_s[i] for i in range(len(pool_s)) if i in keep]

        #Then, if total still exceeds buffer_size (can happen due to rounding), trim from largest.
        while len(self) > self.buffer_size:
            #pick class with the largest pool among seen classes
            cand = None
            best_len = -1
            for cid in self.seen:
                L = len(self.q[cid])
                if L > best_len:
                    best_len = L
                    cand = cid
            if cand is None or best_len <= 0:
                break
            j = random.randrange(best_len)
            del self.q[cand][j]
            del self.s[cand][j]

    @torch.no_grad()
    def add_batch(self, feats_f32: torch.Tensor, class_ids: torch.Tensor):
        """
        Add features; if we overflow, evict from over-cap classes first, then largest pools.
        Features arrive in FP32, but this func. quantizes for replay storage (saves 4x memory for buffer)
        """
        feats_f32 = feats_f32.detach().cpu()
        class_ids = class_ids.detach().cpu()

        #Insert
        for f, y in zip(feats_f32, class_ids):
            cid = int(y)
            q, sc = quantize_tensor_sym(f)
            self.q[cid].append(q)
            self.s[cid].append(sc)

        #Enforce budget: cap per class, then global
        self._rebalance_to_cap()

    @torch.no_grad()
    def sample_q(self, batch_size: int, seen_class_ids, device="cpu"):
        """
        Balanced sample across *current seen* classes.
        Samples are returned in FP32
        """
        seen = list(seen_class_ids)
        if len(seen) == 0:
            return None, None

        k = max(1, batch_size // len(seen))
        xs_q, xs_s, ys = [], [], []

        #1) take up to k per class
        for cid in seen:
            pool_q = self.q.get(int(cid), [])
            pool_s = self.s.get(int(cid), [])
            if len(pool_q) == 0:
                continue
            take = min(k, len(pool_q))
            idxs = random.sample(range(len(pool_q)), take)
            for j in idxs:
                xs_q.append(pool_q[j])
                xs_s.append(pool_s[j])
                ys.append(int(cid))

        #2) top-up if short
        if len(xs_q) < batch_size:
            need = batch_size - len(xs_q)
            flat = []
            for cid in seen:
                for j in range(len(self.q.get(int(cid), []))):
                    flat.append((int(cid), j))
            if len(flat) > 0:
                extra = random.sample(flat, min(need, len(flat)))
                for (cid, j) in extra:
                    xs_q.append(self.q[cid][j])
                    xs_s.append(self.s[cid][j])
                    ys.append(int(cid))

        if len(xs_q) == 0:
            return None, None
        
        #Double check that replay never samples more than the batch size
        if len(xs_q) > batch_size:
            idx = random.sample(range(len(xs_q)), batch_size)
            xs_q = [xs_q[i] for i in idx]
            xs_s = [xs_s[i] for i in idx]
            ys   = [ys[i]   for i in idx]


        q_batch = torch.stack(xs_q).to(device)
        s_batch = torch.stack(xs_s).to(device)
        X = dequantize_tensor_sym(q_batch, s_batch.view(-1, 1))
        y = torch.tensor(ys, dtype=torch.long, device=device)
        return X, y
    
    
  