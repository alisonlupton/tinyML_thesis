
import torch
torch.backends.quantized.engine = "qnnpack"
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from pathlib import Path
from PIL import Image

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
          