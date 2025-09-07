# replay.py
import random
from collections import defaultdict
import torch
import torch.nn.functional as F
import heapq

@torch.no_grad()
def quantize_latents_perchannel_sym(feats, scales):
    q = torch.round(feats / scales)                  # [B, C] / [C]
    q = torch.clamp(q, -127, 127).to(torch.int8)
    return q

@torch.no_grad()
def dequantize_latents_perchannel_sym(q_feats, scales):
    return q_feats.to(torch.float32) * scales        # [B, C] * [C]
class BalancedQuantReplayDynamic:
    def __init__(self, buffer_size: int, feat_dim: int):
        self.buffer_size = int(buffer_size)
        self.feat_dim = int(feat_dim)
        self.seen = []
        self._cap = None

        # shared scales
        self.scales = torch.ones(self.feat_dim, dtype=torch.float32)

        # per-class storage is now split:
        # hard pool (top-K by score) and anchor pool (diversity)
        self.hard = defaultdict(list)    # cid -> [(score, q_feat), ...]  (min-heap semantics)
        self.anchor = defaultdict(list)  # cid -> [q_feat, ...]
        self.anchor_scores = defaultdict(list)  # optional: scores for debug

        # scores only used for debug aggregates (hard+anchor combined)
        self.scores_all = defaultdict(list)     # cid -> [float score, ...] (append-only for stats)

        # DRIP-ish scoring control
        self.hard_frac = 0.3      # percent of class buffer allocated to 'hard' samples
        self.eps = 0.1           # replace the weakest hard example if the new one is >= eps better
        self.rho = 0.2    # chance to admit to anchors even if reservoir replacement wouldn’t (extra rho percent sample stays), stops resevoir from going stale

        # running stats for z-score (per class)
        self.cls_mu = defaultdict(float)
        self.cls_m2 = defaultdict(float)
        self.cls_n  = defaultdict(int)

        # counters / debug
        self.evictions = defaultdict(int)
        self.added = defaultdict(int)
        self.seen_since_rebal = defaultdict(int)
        self.kept_since_rebal = defaultdict(int)
        self.last_seen_window = defaultdict(int)
        self.last_kept_window = defaultdict(int)
        self.seen_lifetime = defaultdict(int)
        self.kept_lifetime = defaultdict(int)

        # reservoir bookkeeping for anchors
        self.anchor_seen = defaultdict(int)  # how many anchor candidates seen per class

        # last min hard score for debug (“cut” print)
        self.last_min_hard = {}
        
    def _safe_stats(self, vals):
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "p50": float("nan"), "p90": float("nan")}
        import numpy as np
        arr = np.array(vals, dtype=float)
        return {
            "mean": float(arr.mean()),
            "std":  float(arr.std(ddof=0)),
            "p50":  float(np.percentile(arr, 50)),
            "p90":  float(np.percentile(arr, 90)),
        }

    def class_summary(self, cid: int):
        n = len(self.hard[cid]) + len(self.anchor[cid])

        # stats for display: combine hard+anchor scores if you want
        vals = [s for s,_ in self.hard[cid]] + list(self.anchor_scores[cid])
        score_stats = self._safe_stats(vals)

        cutoff = self.last_min_hard.get(cid, float("nan"))

        # window snapshots for acceptance rate
        w_seen = self.last_seen_window.get(cid, 0) or self.seen_since_rebal.get(cid, 0)
        w_kept = self.last_kept_window.get(cid, 0) or self.kept_since_rebal.get(cid, 0)
        acc = (100.0 * w_kept / w_seen) if w_seen > 0 else float("nan")

        tot_seen = self.seen_lifetime.get(cid, 0)
        eff_q = (n / tot_seen) if tot_seen > 0 else float("nan")
        n_h = len(self.hard[cid])
        n_a = len(self.anchor[cid])

        return {
            "count": n,
            "score_mean": score_stats["mean"],
            "score_std":  score_stats["std"],
            "score_p50":  score_stats["p50"],
            "score_p90":  score_stats["p90"],
            "last_cutoff": cutoff,
            "added": int(self.added.get(cid, 0)),
            "evicted": int(self.evictions.get(cid, 0)),
            "cap": int(self._cap if self._cap is not None else -1),
            "acc_rate": acc,
            "eff_q": eff_q,
            "w_kept": w_kept, "w_seen": w_seen,
            "tot_seen": tot_seen,
            "anchor_samples": n_a,
            "hard_samples": n_h
            
        }
    def debug_summary(self, title: str = ""):
        if title:
            print(f"\n=== Replay Debug Summary: {title} ===")
        counts = self.counts_per_class()
        if not counts:
            print("  (empty buffer)")
            return
        print(f"cap={self._cap} | total={len(self)}")
        for cid in sorted(counts.keys()):
            s = self.class_summary(cid)
            print(
            f"  cid={cid:>4} | n={s['count']:>4}/{s['cap']:>4} "
            f"(hard={s['hard_samples']}, anchor={s['anchor_samples']}) | " 
            f"mean={s['score_mean']:.3f} std={s['score_std']:.3f} "
            f"p50={s['score_p50']:.3f} p90={s['score_p90']:.3f} | "
            f"cut={s['last_cutoff']:.3f} | "
            f"acc={(s['acc_rate'] if s['w_seen']>0 else float('nan')):.1f}% "
            f"({s['w_kept']}/{s['w_seen']}) | eff_q={(s['eff_q'] if s['tot_seen']>0 else float('nan')):.3f} | "
            f"added={s['added']:>4} evicted={s['evicted']:>4}"
        )

    def snapshot_rows(self):
        rows = []
        for cid in sorted(self.seen):
            s = self.class_summary(cid)
            rows.append({"cid": cid, **s})
        return rows
    
    def _update_running_stats(self, cid, s):
        n = self.cls_n[cid] + 1
        delta = s - self.cls_mu[cid]
        mu = self.cls_mu[cid] + delta / n
        m2 = self.cls_m2[cid] + delta * (s - mu)
        self.cls_mu[cid], self.cls_m2[cid], self.cls_n[cid] = mu, m2, n

    def _zscore(self, cid, s):
        n = self.cls_n[cid]
        if n < 10: return s  # warm-up: raw score
        var = max(self.cls_m2[cid] / (n - 1), 1e-6)
        return (s - self.cls_mu[cid]) / (var ** 0.5)


    def set_scales(self, scales: torch.Tensor):
        assert scales.numel() == self.feat_dim
        self.scales = scales.detach().cpu().to(torch.float32).contiguous()

    def __len__(self):
        return sum(len(self.hard[cid]) + len(self.anchor[cid]) for cid in self.seen)

    def counts_per_class(self):
        return {cid: (len(self.hard[cid]) + len(self.anchor[cid])) for cid in self.seen}

    def _recompute_cap(self):
        S = max(1, len(self.seen))
        self._cap = self.buffer_size // S
    def _cap_parts(self):
        cap = self._cap if self._cap is not None else 1
        K = max(1, int(round(self.hard_frac * cap)))
        A = max(0, cap - K)
        return cap, K, A

    def _hard_min(self, cid):
        # hard is maintained as a min-heap by score (tuple (score, q_feat))
        if not self.hard[cid]:
            return float("-inf")
        return self.hard[cid][0][0]  # smallest score at index 0

    def on_new_task(self, seen_class_ids):
        seen_set = set(self.seen)
        for cid in seen_class_ids:
            c = int(cid)
            if c not in seen_set:
                self.seen.append(c)
        self._recompute_cap()
        self.rebalance_to_cap()

    def rebalance_to_cap(self):
        if self._cap is None:
            self._recompute_cap()

        _, K, A = self._cap_parts()

        for cid in list(self.seen):
            # 1) Trim HARD to top-K by score
            if len(self.hard[cid]) > 0:
                # heapify if not already (safe idempotent)
                heapq.heapify(self.hard[cid])
                while len(self.hard[cid]) > K:
                    _, q_feat = heapq.heappop(self.hard[cid])
                    self.evictions[cid] += 1

            # record min-hard (“cut”) for debug
            self.last_min_hard[cid] = self._hard_min(cid) if len(self.hard[cid]) else float("nan")

            # 2) Trim ANCHOR to A (random drop)
            while len(self.anchor[cid]) > A:
                j = random.randrange(len(self.anchor[cid]))
                del self.anchor[cid][j]
                del self.anchor_scores[cid][j]
                self.evictions[cid] += 1

        # 3) Global overflow guard
        while len(self) > self.buffer_size:
            # evict from the class with largest total
            cand, best = None, -1
            for cid in self.seen:
                L = len(self.hard[cid]) + len(self.anchor[cid])
                if L > best:
                    best, cand = L, cid
            if cand is None or best <= 0:
                break
            # prefer to evict from anchor; if empty, evict the min hard
            if len(self.anchor[cand]) > 0:
                j = random.randrange(len(self.anchor[cand]))
                del self.anchor[cand][j]
                del self.anchor_scores[cand][j]
            elif len(self.hard[cand]) > 0:
                heapq.heapify(self.hard[cand])
                heapq.heappop(self.hard[cand])
            self.evictions[cand] += 1

        # snapshot acceptance window for debug & reset
        for cid in self.seen:
            self.last_seen_window[cid] = self.seen_since_rebal.get(cid, 0)
            self.last_kept_window[cid] = self.kept_since_rebal.get(cid, 0)
            self.seen_since_rebal[cid] = 0
            self.kept_since_rebal[cid] = 0

    @torch.no_grad()
    def add_batch(self, feats_f32, class_ids, raw_scores=None):
        feats_f32 = feats_f32.detach().cpu()
        class_ids = class_ids.detach().cpu().tolist()
        q = quantize_latents_perchannel_sym(feats_f32, self.scales)

        if raw_scores is None:
            raw_scores = feats_f32.norm(dim=1).tolist()
        else:
            raw_scores = raw_scores.detach().cpu().tolist()

        _, K, A = self._cap_parts()

        for f_q, y, s in zip(q, class_ids, raw_scores):
            y = int(y)

            # update running stats then z-score
            self._update_running_stats(y, s)
            s_norm = self._zscore(y, s)

            # counters
            self.seen_since_rebal[y] += 1
            self.seen_lifetime[y]    += 1

            # -------- HARD HEAP (top-K by s_norm) --------
            # store as (score, q_feat) in a MIN-heap
            heap = self.hard[y]
            if len(heap) < K:
                heapq.heappush(heap, (float(s_norm), f_q))
                self.added[y] += 1
                self.kept_since_rebal[y] += 1
                self.kept_lifetime[y]    += 1
                self.scores_all[y].append(float(s_norm))
                continue

            # heap full: compare with current minimum
            min_hard = heap[0][0]
            if float(s_norm) > (min_hard + self.eps):
                # replace the easiest hard exemplar
                heapq.heapreplace(heap, (float(s_norm), f_q))
                self.evictions[y] += 1
                self.added[y] += 1
                self.kept_since_rebal[y] += 1
                self.kept_lifetime[y]    += 1
                self.scores_all[y].append(float(s_norm))
                continue

            # -------- ANCHOR POOL (reservoir) --------
            self.anchor_seen[y] += 1
            anchors = self.anchor[y]
            a_scores = self.anchor_scores[y]
            if len(anchors) < A:
                anchors.append(f_q); a_scores.append(float(s_norm))
                self.added[y] += 1
                self.kept_since_rebal[y] += 1
                self.kept_lifetime[y]    += 1
                self.scores_all[y].append(float(s_norm))
            else:
                # reservoir: replace with probability A / seen
                m = self.anchor_seen[y]
                j = random.randint(0, m - 1)
                if j < A or random.random() < self.rho:  # small exploration bump
                    idx = j if j < A else random.randrange(A)  # safe index
                    anchors[idx] = f_q
                    a_scores[idx] = float(s_norm)
                    self.evictions[y] += 1
                    self.added[y] += 1
                    self.kept_since_rebal[y] += 1
                    self.kept_lifetime[y]    += 1
                    self.scores_all[y].append(float(s_norm))
                # else: drop

    @torch.no_grad()
    def sample_q(self, batch_size: int, seen_class_ids, device="cpu"):
        seen = list(seen_class_ids)
        if len(seen) == 0:
            return None, None

        xs_q, ys = [], []
        per_cls = max(1, batch_size // len(seen))

        hard_ratio = 0.5 # percentage of replay sampled from hard selection

        for cid in seen:
            cid = int(cid)
            H = self.hard[cid]
            A = self.anchor[cid]
            n_h = len(H); n_a = len(A)
            if n_h + n_a == 0:
                continue

            take_h = min(n_h, int(round(hard_ratio * per_cls)))
            take_a = min(n_a, per_cls - take_h)

            # if hard is tiny and anchor big, top-up from anchor, and vice versa
            if take_h == 0 and n_h > 0 and take_a < per_cls:
                take_h = min(n_h, per_cls - take_a)
            if take_a == 0 and n_a > 0 and take_h < per_cls:
                take_a = min(n_a, per_cls - take_h)

            # sample from HARD uniformly (or weighted if you want)
            if take_h > 0:
                idxs = random.sample(range(n_h), take_h)
                for j in idxs:
                    _, q_feat = H[j]
                    xs_q.append(q_feat)
                    ys.append(cid)

            # sample from ANCHOR uniformly
            if take_a > 0:
                idxs = random.sample(range(n_a), take_a)
                for j in idxs:
                    xs_q.append(A[j])
                    ys.append(cid)

        # top-up across classes if needed
        if len(xs_q) < batch_size:
            need = batch_size - len(xs_q)
            flat = []
            for cid in seen:
                cid = int(cid)
                for j in range(len(self.hard[cid])):
                    flat.append((cid, "h", j))
                for j in range(len(self.anchor[cid])):
                    flat.append((cid, "a", j))
            if flat:
                extra = random.sample(flat, min(need, len(flat)))
                for cid, kind, j in extra:
                    if kind == "h":
                        _, qf = self.hard[cid][j]
                    else:
                        qf = self.anchor[cid][j]
                    xs_q.append(qf); ys.append(cid)

        if not xs_q:
            return None, None
        if len(xs_q) > batch_size:
            idx = random.sample(range(len(xs_q)), batch_size)
            xs_q = [xs_q[i] for i in idx]
            ys   = [ys[i]   for i in idx]

        q_batch = torch.stack(xs_q).to(device)  # int8
        feats = dequantize_latents_perchannel_sym(q_batch, self.scales.to(device))
        y = torch.tensor(ys, dtype=torch.long, device=device)
        return feats, y