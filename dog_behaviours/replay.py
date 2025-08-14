import random
import torch
class BalancedReplay:
    def __init__(self, per_class_cap=500):
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
    

def update_replay_reservoir(replay_buffer, replay_labels, features, targets, buffer_size=5000):
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
