# cnn_backbone_training_utils.py
import torch 
import numpy as np 
from dataclasses import dataclass
from utils import build_tensors
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Dict

@dataclass
class BackboneData:
    """Container for backbone training and validation data"""
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    gid_to_local: dict
    local_to_gid: list
    train_tf: list
    val_tf: list
    
    
def _make_transforms(img_size):

    train_tf = transforms.Compose([
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
    return train_tf, val_tf

def load_data_cnn_backbone(cfg, backbone_dogs, dog_data):
    
    img_size = cfg['img_size']
    backbone_df = dog_data[dog_data["gid"].isin(backbone_dogs)].copy()

    # Stable local label mapping (0..B-1)
    local_ids = sorted(backbone_dogs)
    gid_to_local = {g:i for i,g in enumerate(local_ids)}
    backbone_df["local"] = [gid_to_local[int(g)] for g in backbone_df["gid"].tolist()]

    df_train= backbone_df[backbone_df["split"]=="train"].copy()
    df_val  = backbone_df[backbone_df["split"]=="test"].copy()


    train_tf, val_tf = _make_transforms(img_size)
    
    X_train, y_train = build_tensors(df_train, train_tf)
    X_val, y_val = build_tensors(df_val, val_tf)
    
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("No backbone training data found!")

    return BackboneData(X_train, y_train, X_val, y_val, gid_to_local, local_ids, train_tf, val_tf)


def train_cnn_backbone(backbone_model, backbone_data, backbone_optimizer, backbone_criterion, backbone_train_loader, backbone_val_loader, device, cfg, gid2breed):
    print("Training backbone model...")
    best_val_acc = 0.0
    patience = 5
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
            print(f"[Backbone] epoch {epoch}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
            
    
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


def eval_new_classes_on_backbone(backbone_model, task_test_loader, task_local_to_gid, backbone_registry_eval, device):
    rows_for_task, cols_have_rows = [], []
    for i, gid in enumerate(task_local_to_gid.tolist()):
        if gid in backbone_registry_eval.row_for_gid:
            rows_for_task.append(backbone_registry_eval.row_for_gid[gid])
            cols_have_rows.append(i)
    rows_for_task = torch.tensor(rows_for_task, device=device, dtype=torch.long)
    cols_have_rows = torch.tensor(cols_have_rows, device=device, dtype=torch.long)

    backbone_model.eval()
    total = correct = 0
    num_local = len(task_local_to_gid)
    cls_tot = [0]*num_local
    cls_cor = [0]*num_local

    with torch.no_grad():
        for xb, y_local in task_test_loader:
            xb, y_local = xb.to(device), y_local.to(device)
            feats = backbone_model._features(xb)
            logits_full = torch.full((xb.size(0), num_local), -1e9, device=device)
            if len(rows_for_task) > 0:
                logits_seen = backbone_model.head.forward_rows(feats, rows_for_task)
                logits_full[:, cols_have_rows] = logits_seen

            pred = logits_full.argmax(dim=1)
            total += y_local.size(0)
            correct += (pred == y_local).sum().item()
            for c in range(num_local):
                m = (y_local == c)
                if m.any():
                    cls_tot[c] += int(m.sum().item())
                    cls_cor[c] += int((pred[m] == y_local[m]).sum().item())

    overall = 100.0 * correct / max(1, total)
    per_class = {c: (100.0 * cls_cor[c] / cls_tot[c] if cls_tot[c] > 0 else 0.0) for c in range(num_local)}
    return overall, per_class

  