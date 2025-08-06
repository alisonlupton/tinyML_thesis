
import torch
torch.backends.quantized.engine = "qnnpack"
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from pathlib import Path
from PIL import Image
import random
from collections import deque, defaultdict


# -------------------------------------------------------- CL REPLAY HELPERS  -------------------------------------------------------- #
def tail_forward(tail, classifier, lr_maps):
    z = tail(lr_maps)      # ends with global_pool
    z = torch.flatten(z, 1)
    return classifier(z)

def replay_ratio_schedule(t, epoch, step, steps_per_epoch):
    """
    r0: base ratio at task 0 after warm-up
    r_step: increment per task
    warmup: per-epoch multipliers for epoch 0/1, then 1.0 onwards
    """
    
    if epoch == 0:
        frac = step / max(1, steps_per_epoch)
        if frac < 0.25: return 0.0
        elif frac < 0.50: return 0.5
        else: return 1.0
    return 1.0

def sample_replay(classwise_feats,         # dict[int] -> deque of int8 CPU tensors
                available_classes,       # list[int]
                total_k,                 # total number of samples to draw
                prev_task_classes=None,  # list[int] or None
                prev_frac=0.4,           # fraction of budget for prev-task classes
                rng=None):
    """
    Returns (selected_feats, selected_labels).
    - Draws up to `total_k` items.
    - Reserves prev_frac for previous-task classes (if provided).
    - Fills remainder from all other available classes.
    - Class-balanced as much as possible; falls back gracefully if deques are short.
    """
    if rng is None:
        rng = random

    if total_k <= 0 or not available_classes:
        return [], []

    # Split budget
    k_prev = 0
    prev = []
    other = available_classes
    if prev_task_classes:
        prev = [c for c in prev_task_classes if c in available_classes]
        other = [c for c in available_classes if c not in set(prev)]
        k_prev = min(total_k, int(round(total_k * prev_frac)))
    k_other = total_k - k_prev

    def draw_from(classes, k):
        sel_f, sel_y = [], []
        if not classes or k <= 0:
            return sel_f, sel_y
        # roughly balanced quota per class
        per_class = max(1, k // max(1, len(classes)))
        remaining = k
        # first pass: per-class quota
        for c in classes:
            pool = classwise_feats[c]
            take = min(per_class, len(pool), remaining)
            if take > 0:
                # random.sample works on Python lists; convert deque -> list
                picks = rng.sample(list(pool), k=take)
                sel_f.extend(picks)
                sel_y.extend([c] * take)
                remaining -= take
            if remaining <= 0:
                break
        # second pass: top-up if we still need more
        if remaining > 0:
            for c in classes:
                if remaining <= 0:
                    break
                pool = classwise_feats[c]
                # take more if available
                can_take = min(len(pool), remaining)
                if can_take > 0:
                    picks = rng.sample(list(pool), k=can_take)
                    sel_f.extend(picks)
                    sel_y.extend([c] * can_take)
                    remaining -= can_take
        return sel_f, sel_y
    
    
    f_prev, y_prev = draw_from(prev, k_prev)
    f_oth,  y_oth  = draw_from(other, k_other)

    # --- Reallocate any shortfall to the other side ---
    used_prev = len(f_prev)
    used_oth  = len(f_oth)
    short_prev = k_prev - used_prev
    short_oth  = k_other - used_oth

    if short_oth > 0 and prev:
        f2, y2 = draw_from(prev, short_oth)
        f_prev.extend(f2); y_prev.extend(y2)

    if short_prev > 0 and other:
        f2, y2 = draw_from(other, short_prev)
        f_oth.extend(f2); y_oth.extend(y2)

    return f_prev + f_oth, y_prev + y_oth
    
def init_replay(max_per_class):
    return defaultdict(lambda: deque(maxlen=max_per_class))

def update_replay_reservoir(classwise_feats, new_pairs, max_per_class, rng=random):
    """
    new_pairs: iterable of (feat_int8, label_int) collected from current experience.
    Per-class reservoir sampling to keep at most max_per_class items per class.
    """
    per_class_seen = defaultdict(int)
    for feat, y in new_pairs:
        c = int(y)
        per_class_seen[c] += 1
        seen = per_class_seen[c]
        buf = classwise_feats[c]
        if len(buf) < max_per_class:
            buf.append(feat)
        else:
            j = rng.randint(0, seen - 1)
            if j < max_per_class:
                buf[j] = feat
                
# -------------------------------------------------------- DATALOADERS  -------------------------------------------------------- #
class TinyValDataset(Dataset):
    def __init__(self, root, class_to_idx, transform=None):
        root = Path(root)
        ann_path = "../data/tiny-imagenet-200/val/val_annotations.txt"
        # Six columns: image, class, x_min, y_min, x_max, y_max
        ann = pd.read_csv(
            ann_path,
            sep="\t",
            header=None,
            names=["img","cls","xmin","ymin","xmax","ymax"]
        )
        # Build a class→index map
        classes = sorted(set(ann.cls))
        if class_to_idx is None:
            classes = sorted({entry[1] for entry in ann})
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx
        # Store image file paths and integer labels
        self.imgs = [root/"images"/img_name for img_name in ann.img]
        self.targets = [self.class_to_idx[c] for c in ann.cls]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]

def load_dataset_custom(dataset_cls, training, transform):
    # 1) Load full set
    full_ds = dataset_cls(
        "../data",
        train=training,
        download=True,
        transform=transform
    )

    return full_ds

def make_split_dataset_loaders(
    full_ds,
    n_splits=5,
    train: bool = False,
    batch_size: int = 1000,
):
    """
    Partition any torchvision dataset (e.g. KMNIST) into `n_splits` tasks,
    each containing consecutive labels.  E.g. with n_splits=5:
      split 0 -> labels {0,1}, split 1 -> {2,3}, …, split 4 -> {8,9}.
    Returns a list of DataLoaders over those subsets.
    """

    per_split = len(set(full_ds.targets))  // n_splits
    loaders = []
    for split_id in range(n_splits):
        lo = split_id * per_split
        hi = lo + per_split
        idxs = [i for i, lbl in enumerate(full_ds.targets) if lo <= lbl < hi]
        sub = Subset(full_ds, idxs)
        loaders.append(DataLoader(sub, batch_size=batch_size, shuffle=not train))
    return loaders

def train_on_loader(model, loader, optimizer, criterion, device, epochs):   
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=False)
        for x,y in pbar:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            

# ----- Evaluator for standard backbone model
def eval_model(model, loader, device):
    print("starting eval")
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc ="evaluating..", unit="batch", ):
            x, y = batch[0].to(device), batch[1].to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
            print(f"Current accuracy: {correct/total}")
    return correct/total      
          