import torch 
import numpy as np 
from dataclasses import dataclass
from utils import compute_normalization_stats, normalize_features

@dataclass
class BackboneData:
    """Container for backbone training and validation data"""
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    train_mean: np.ndarray
    train_std: np.ndarray

def load_data_cnn_backbone(backbone_dogs, dog_data, behavior_to_idx, backbone_behaviors, validation_dog):
    backbone_X_list = []
    backbone_y_list = []
    
    for dog_id in backbone_dogs:
        dog_X = dog_data[dog_id]['X']
        dog_y = dog_data[dog_id]['y']
        
        # Only include backbone behaviors
        backbone_indices = [behavior_to_idx[cls] for cls in backbone_behaviors]
        mask = torch.isin(dog_y, torch.tensor(backbone_indices))
        if mask.sum() > 0:
            backbone_X_list.append(dog_X[mask])
            # Remap labels to backbone classes
            remapped_y = torch.zeros(mask.sum(), dtype=torch.long)
            for i, cls in enumerate(backbone_behaviors):
                cls_idx = behavior_to_idx[cls]
                remapped_y[dog_y[mask] == cls_idx] = i
            backbone_y_list.append(remapped_y)
    
    if len(backbone_X_list) == 0:
        print("No backbone training data found!")
        return
    
    backbone_X = torch.cat(backbone_X_list, dim=0)     # (Ntr, C, L)
    backbone_y = torch.cat(backbone_y_list, dim=0)     # (Ntr,)
    
    
    # Compute std and mean from TRAIN only
    X_train_np = backbone_X.numpy()
    y_train_np = backbone_y.numpy()
    train_mean, train_std = compute_normalization_stats(X_train_np)
    
    # Normalize
    X_train_np = normalize_features(X_train_np, train_mean, train_std)
    X_train = torch.from_numpy(X_train_np).float()
    y_train = torch.from_numpy(y_train_np).long()

    
    # Get separate validation dog data
    if validation_dog not in dog_data:
        print(f"Validation dog {validation_dog} not found!")
        return
    
    val_X_np = dog_data[validation_dog]['X'].numpy()  # (N, C, L)
    val_y_np = dog_data[validation_dog]['y'].numpy()  # (N,)
    
    # filter to backbone classes
    val_mask = np.isin(val_y_np, [behavior_to_idx[c] for c in backbone_behaviors])
    val_X_np = val_X_np[val_mask]
    val_y_np = val_y_np[val_mask]

    
    # remap labels 0..3
    val_y_remapped = np.zeros(len(val_y_np), dtype=np.int64)
    for i, cls in enumerate(backbone_behaviors):
        val_y_remapped[val_y_np == behavior_to_idx[cls]] = i
        
      
    print(f"Validation dog {validation_dog} behavior distribution:")
    for i, cls in enumerate(backbone_behaviors):
        count = (val_y_remapped == i).sum()
        print(f"  {cls}: {count} samples")

    # normalize using *training* stats
    val_X_np = normalize_features(val_X_np, train_mean, train_std)

    X_val = torch.from_numpy(val_X_np).float()
    y_val = torch.from_numpy(val_y_remapped).long()
    
    return BackboneData(X_train, y_train, X_val, y_val, train_mean, train_std)


def train_cnn_backbone(backbone_model, backbone_behaviors, backbone_optimizer, backbone_criterion, backbone_train_loader, backbone_val_loader, device, cfg):
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
        
        for batch_idx, (X_batch, y_batch) in enumerate(backbone_train_loader):
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
    val_class_correct = {cls: 0 for cls in backbone_behaviors}
    val_class_total = {cls: 0 for cls in backbone_behaviors}
    
    with torch.no_grad():
        for X_batch, y_batch in backbone_val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = backbone_model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            
            for i, cls in enumerate(backbone_behaviors):
                mask = (y_batch == i)
                val_class_total[cls] += mask.sum().item()
                val_class_correct[cls] += (predicted[mask] == y_batch[mask]).sum().item()
    
    for cls in backbone_behaviors:
        if val_class_total[cls] > 0:
            acc = 100 * val_class_correct[cls] / val_class_total[cls]
            print(f"  {cls}: {acc:.1f}% ({val_class_correct[cls]}/{val_class_total[cls]} samples)")
        else:
            print(f"  {cls}: No samples")
        
        
    return backbone_model, best_val_acc