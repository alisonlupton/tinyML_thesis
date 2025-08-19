# cnn_backbone_training_utils.py
import torch 
from dataclasses import dataclass
from utils import build_tensors
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Optional

@dataclass
class BackboneData:
    """Container for backbone training and validation data"""
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    gid_to_local: dict
    local_to_gid: list
    backbone_train_tf: list
    CIL_train_tf: list
    val_tf: list
    
    
def _make_transforms(img_size):
    backbone_train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    CIL_train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return backbone_train_tf, CIL_train_tf, val_tf

def load_data_cnn_backbone(cfg, backbone_dogs, dog_data):
    
    img_size = cfg['img_size']
    backbone_df = dog_data[dog_data["gid"].isin(backbone_dogs)].copy()

    # Stable local label mapping (0..B-1)
    local_ids = sorted(backbone_dogs)
    gid_to_local = {g:i for i,g in enumerate(local_ids)}
    backbone_df["local"] = [gid_to_local[int(g)] for g in backbone_df["gid"].tolist()]

    df_train= backbone_df[backbone_df["split"]=="train"].copy()
    df_val  = backbone_df[backbone_df["split"]=="test"].copy()


    backbone_train_tf, CIL_train_tf,  val_tf = _make_transforms(img_size)
    
    X_train, y_train = build_tensors(df_train, backbone_train_tf)
    X_val, y_val = build_tensors(df_val, val_tf)
    
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("No backbone training data found!")

    return BackboneData(X_train, y_train, X_val, y_val, gid_to_local, local_ids, backbone_train_tf, CIL_train_tf, val_tf)


def train_cnn_backbone(backbone_model, backbone_data, backbone_optimizer, backbone_criterion, backbone_train_loader, backbone_val_loader, device, cfg, gid2breed):
    print("Training backbone model...")
    best_val_acc = 0.0
    patience = 15
    counter = 0
    best_state = None
    epochs = cfg['backbone_epochs']
    
    for epoch in range(epochs):
        # Training phase
        backbone_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in backbone_train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            backbone_optimizer.zero_grad()
            outputs = backbone_model(X_batch)
            loss = backbone_criterion(outputs, y_batch)
            loss.backward()
            backbone_optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        
        # Validation phase
        backbone_model.eval()
        val_loss = 0    
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in backbone_val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                outputs = backbone_model(X_batch)
                loss = backbone_criterion(outputs, y_batch)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        # Calculate accuracies
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        
        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            best_state = {k: v.cpu().clone() for k,v in backbone_model.state_dict().items()}
        else:
            counter +=1
        
        if counter == patience:
            print(f"Early stopping at epoch {epoch}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
            break
        if epoch % 10 == 0:
            print(f"[Backbone] epoch {epoch}: train_acc={train_acc:.1f}%, val_acc={val_acc:.1f}%")
            
    
    if best_state is not None:
        backbone_model.load_state_dict(best_state, strict=True)
        
    print(f"Backbone training complete! Best validation accuracy: {best_val_acc:.3f}")

    print(f"\n--- BACKBONE VALIDATION PER-CLASS ACCURACY ---")
    backbone_model.eval()
    # Print per-class validation accuracy for backbone
    num_local = len(backbone_data.local_to_gid)

    val_class_correct = [0] * num_local
    val_class_total = [0] * num_local
    
    with torch.no_grad():
        for X_batch, y_batch in backbone_val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = backbone_model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            
            for c in range(num_local):
                mask = (y_batch == c)
                if mask.any():
                    val_class_total[c] += mask.sum().item()
                    val_class_correct[c] += (predicted[mask] == y_batch[mask]).sum().item()

    for local_idx, gid in enumerate(backbone_data.local_to_gid):
        if val_class_total[local_idx] > 0:
            acc = 100 * val_class_correct[local_idx] / val_class_total[local_idx]
            name = gid2breed[str(gid)]
            print(f" {name:35s} : {acc:5.1f}% ({val_class_correct[local_idx]}/{val_class_total[local_idx]})")
        else:
            print(f"{local_idx}: No samples")
        
        
    return backbone_model, best_val_acc


@torch.no_grad()
def eval_new_classes_on_backbone(model,
    train_loader: DataLoader,
    test_loader: DataLoader, 
    device: torch.device, 
    num_classes: Optional[int] = None,
) -> Tuple[float, Dict[int, float]]:
    """
    Evaluate a frozen backbone by building class prototypes from TRAIN features
    and classifying TEST features by cosine similarity to those prototypes.

    Assumes labels in loaders are *local* class indices [0..K-1] for the task.
    Returns: (overall_acc_percent, per_class_acc_percent_dict)
    """
    model.eval()

    # --------- 1) Build prototypes from TRAIN features ---------
    feats_sums = {}
    counts = {}
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        z = model._features(xb)                       # (N, D)
        z = torch.nn.functional.normalize(z, dim=1)   # cosine-ready
        for c in yb.unique().tolist():
            c = int(c)
            mask = (yb == c)
            feats_sums[c] = feats_sums.get(c, 0) + z[mask].sum(dim=0)
            counts[c] = counts.get(c, 0) + int(mask.sum().item())

    # If caller didn't pass num_classes, infer from what we saw
    if num_classes is None:
        num_classes = (max(counts.keys()) + 1) if counts else 0

    # Build prototype matrix [K, D], missing classes get zero vector (ignored at test)
    D = next(iter(feats_sums.values())).numel() if feats_sums else model.head.in_dim
    protos = torch.zeros(num_classes, D, device=device)
    seen_mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
    for c, s in feats_sums.items():
        if counts[c] > 0:
            protos[c] = s / counts[c]
            seen_mask[c] = True
    protos = torch.nn.functional.normalize(protos, dim=1)  # [K, D] unit vectors

    # --------- 2) Classify TEST features by cosine to prototypes ---------
    total = 0
    correct = 0
    cls_tot = [0] * num_classes
    cls_cor = [0] * num_classes

    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        z = model._features(xb)                        # [N, D]
        z = torch.nn.functional.normalize(z, dim=1)    # unit
        # cosine sim = dot product for normalized vectors
        sims = z @ protos.t()                          # [N, K]
        # For classes without a prototype, make them unselectable
        if (~seen_mask).any():
            sims[:, ~seen_mask] = -1e9
        pred = sims.argmax(dim=1)

        total += yb.size(0)
        correct += (pred == yb).sum().item()
        for c in range(num_classes):
            m = (yb == c)
            n = int(m.sum().item())
            if n > 0:
                cls_tot[c] += n
                cls_cor[c] += int((pred[m] == yb[m]).sum().item())

    overall = 100.0 * correct / max(1, total)
    per_class = {c: (100.0 * cls_cor[c] / cls_tot[c] if cls_tot[c] > 0 else 0.0)
                 for c in range(num_classes)}
    return overall, per_class
  