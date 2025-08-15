#!/usr/bin/env python3
"""
INT8 Quantized CNN Pipeline for Dog Behavior CIL
Reduces SRAM usage by quantizing weights and activations to 8-bit
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml
import random
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_config():
    """Load configuration."""
    with open("dog_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config['dogmove']

def load_processed_dog_data():
    """Load pre-processed dog data from intelligent sampling."""
    print("Loading pre-processed dog data...")
    
    # Check if processed data exists
    processed_dir = Path("processed_data")
    if not processed_dir.exists():
        raise FileNotFoundError(
            "Processed data not found! Please run process_dog_data.py first."
        )
    
    # Load metadata
    metadata_file = processed_dir / "intelligent_sampling_metadata.yaml"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = yaml.safe_load(f)
    
    behavior_to_idx = metadata['behavior_to_idx']
    behaviors = metadata['behaviors']
    
    print(f"Loading data for {len(behaviors)} behaviors: {behaviors}")
    
    # Load each dog's data
    dog_data = {}
    total_samples = 0
    
    for dog_file in sorted(processed_dir.glob("dog_*_intelligent.npz")):
        dog_id = int(dog_file.stem.split('_')[1])  # Extract dog ID from filename
        
        # Load numpy data
        data = np.load(dog_file)
        X = data['X']  # Already (N, L, C) format
        y = data['y']
        
        sensor_cols = list(data['sensor_cols'])
        L = int(data['window_len'])  # Time dimension
        C = len(sensor_cols)         # Channel dimension
        
        # Convert to (N, C, L) for Conv1d
        X = np.transpose(X, (0, 2, 1)).astype(np.float32)  # (N, C, L)
        
        # Load windowing data
        seg_ids = data['segment_ids']
        sess_ids = data.get('session_ids', None)
        window_len = int(data.get('window_len', 0))
        stride = int(data.get('stride', 0))
        
        # Convert to tensors (raw data - no normalization yet)
        dog_data[dog_id] = {
            'X': torch.from_numpy(X),
            'y': torch.from_numpy(y.astype(np.int64)),
            'segment_ids': data['segment_ids'].astype(np.int32),
            'session_ids': data['session_ids'].astype(np.int32) if 'session_ids' in data else None,
            'behaviors': list(data['behaviors']),
            'sensor_cols': sensor_cols,
            'window_len': L,
        }
        
        total_samples += len(X)
    
    return dog_data, behavior_to_idx, behaviors

def normalize_features(X, mean, std):
     # X: (N, C, L) CNN version
    return (X - mean) / std

def compute_normalization_stats(X_train):
    """Compute normalization statistics from training data only."""
    mean = X_train.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
    std  = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    return mean, std

class QuantizedConv1d(nn.Module):
    """INT8 quantized Conv1d layer."""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.scale = nn.Parameter(torch.ones(1))
        self.zero_point = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # Quantize input to INT8
        x_q = torch.quantize_per_tensor(x, scale=self.scale, zero_point=self.zero_point, dtype=torch.qint8)
        x_dq = x_q.dequantize()
        
        # Apply convolution
        return self.conv(x_dq)

class QuantizedLinear(nn.Module):
    """INT8 quantized Linear layer."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.ones(1))
        self.zero_point = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # Quantize input to INT8
        x_q = torch.quantize_per_tensor(x, scale=self.scale, zero_point=self.zero_point, dtype=torch.qint8)
        x_dq = x_q.dequantize()
        
        # Apply linear transformation
        return self.linear(x_dq)

class SimplifiedTDMModelCNN_INT8(nn.Module):
    """
    INT8 Quantized 1D-CNN backbone -> 64-D features -> linear classifier with TDM masking.
    Input: (N, C, L)
    """
    def __init__(self, in_channels, num_classes, device, sparsity_ratio=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.device = device
        self.sparsity_ratio = sparsity_ratio

        # --- Backbone (temporal convs) - INT8 quantized ---
        self.backbone = nn.Sequential(
            QuantizedConv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            QuantizedConv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # Light dropout
            nn.Dropout(p=0.2),
        )

        # Global average pool over time -> (N, 64)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # --- Classifier - INT8 quantized ---
        self.classifier = QuantizedLinear(64, num_classes)

        # TDM mask on classifier weights
        self.mask = torch.ones_like(self.classifier.linear.weight.data, device=device)
        self.apply_initial_sparsity()

        self.eval_classifier = None  # filled later

        active = self.mask.sum().item()
        total = self.mask.numel()
        print(f"INT8 Quantized TDM (CNN) initialized: {active}/{total} active ({100*active/total:.1f}% density)")

    # ----- TDM helpers (unchanged) -----
    def apply_initial_sparsity(self):
        with torch.no_grad():
            flat = self.mask.view(-1)
            k = int(flat.numel() * self.sparsity_ratio)
            idx = torch.randperm(flat.numel())[:k]
            flat[idx] = 0
            self.mask = flat.view_as(self.classifier.linear.weight.data)
            self.classifier.linear.weight.data *= self.mask

    def apply_mask(self):
        with torch.no_grad():
            self.classifier.linear.weight.data *= self.mask

    def update_mask_intra_task(self, p_intra=0.05):
        with torch.no_grad():
            imp = torch.abs(self.classifier.linear.weight.data)
            flat_imp = imp.view(-1)
            flat_mask = self.mask.view(-1)
            num = int(flat_mask.numel() * p_intra)

            active_idx = torch.where(flat_mask > 0)[0]
            if len(active_idx) > num:
                least = torch.argsort(flat_imp[active_idx])[:num]
                flat_mask[active_idx[least]] = 0

            inactive_idx = torch.where(flat_mask == 0)[0]
            if len(inactive_idx) > num:
                add = torch.randperm(len(inactive_idx))[:num]
                flat_mask[inactive_idx[add]] = 1

            self.mask = flat_mask.view_as(self.classifier.linear.weight.data)
            self.apply_mask()

    def update_mask_inter_task(self, p_inter=0.1):
        with torch.no_grad():
            flat_mask = self.mask.view(-1)
            inactive_idx = torch.where(flat_mask == 0)[0]
            num = int(flat_mask.numel() * p_inter)
            if len(inactive_idx) > num:
                add = torch.randperm(len(inactive_idx))[:num]
                flat_mask[inactive_idx[add]] = 1
            self.mask = flat_mask.view_as(self.classifier.linear.weight.data)
            self.apply_mask()
            print(f"  INT8 TDM Inter-task: expand-and-shrink (p_inter={p_inter})")

    # ----- forward -----
    def _features(self, x):        # x: (N, C, L)
        h = self.backbone(x)       # (N, 64, L')
        h = self.gap(h).squeeze(-1)  # (N, 64)
        return h

    def forward(self, x):
        feats = self._features(x)
        return self.classifier(feats)

    def forward_eval(self, x):
        feats = self._features(x)
        return self.eval_classifier(feats)
    
    def expand_classifier(self, new_classes):
        """Expand classifier for new classes."""
        old_classifier = self.classifier
        old_num_classes = old_classifier.linear.out_features
        new_num_classes = old_num_classes + new_classes
        
        # Create new quantized classifier
        self.classifier = QuantizedLinear(64, new_num_classes)
        
        # Copy old weights
        self.classifier.linear.weight.data[:old_num_classes] = old_classifier.linear.weight.data
        self.classifier.linear.bias.data[:old_num_classes] = old_classifier.linear.bias.data
        
        # Initialize new weights
        nn.init.xavier_uniform_(self.classifier.linear.weight.data[old_num_classes:])
        nn.init.zeros_(self.classifier.linear.bias.data[old_num_classes:])
        
        # Update mask
        old_mask = self.mask
        self.mask = torch.ones_like(self.classifier.linear.weight.data, device=self.device)
        self.mask[:old_num_classes] = old_mask
        
        print(f"Expanded classifier: {old_num_classes} -> {new_num_classes} classes")

    def update_eval_classifier(self):
        """Update evaluation classifier for all classes."""
        self.eval_classifier = self.classifier

def train_with_simplified_tdm(model, train_loader, optimizer, criterion, device, 
                             replay_buffer, replay_labels, task_idx, epoch, num_classes, buffer_size,
                             p_intra=0.05, p_inter=0.1, delta_k=5):
    """Training function with simplified TDM and latent replay."""
    total_loss = 0
    correct = 0
    total = 0
    
    # Replay ratio - how often to use replay
    replay_ratio = 0.5 if task_idx > 0 else 0.0  # No replay in first task
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Extract features from backbone
        features = model._features(data)
        
        # Sample replay data (latent features)
        replay_data = None
        if len(replay_buffer) > 0 and random.random() < replay_ratio:
            replay_features, replay_targets = sample_replay(replay_buffer, replay_labels, batch_size=len(data))
            if replay_features is not None:
                replay_data = (replay_features.to(device), replay_targets.to(device))
        
        # Forward pass through classifier
        output = model.classifier(features)
        loss = criterion(output, target)
        
        # Add replay loss if available
        if replay_data is not None:
            replay_features, replay_targets = replay_data
            replay_output = model.classifier(replay_features)
            replay_loss = criterion(replay_output, replay_targets)
            loss = loss + replay_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Apply TDM mask less frequently
        if batch_idx % 10 == 0:  # Only apply every 10 batches
            model.apply_mask()
        
        # Update replay buffer with latent features
        update_replay_reservoir(replay_buffer, replay_labels, features.detach().cpu(), target.detach().cpu(), buffer_size)
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    return total_loss / len(train_loader), 100 * correct / total

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

def evaluate_all_classes(model, test_loader, device, all_behaviors):
    """Evaluate on all classes."""
    model.eval()
    correct = 0
    total = 0
    class_correct = {cls: 0 for cls in all_behaviors}
    class_total = {cls: 0 for cls in all_behaviors}
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model.forward_eval(data)
            _, predicted = torch.max(output.data, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Per-class accuracy
            for i, cls in enumerate(all_behaviors):
                mask = (target == i)
                class_total[cls] += mask.sum().item()
                class_correct[cls] += (predicted[mask] == target[mask]).sum().item()
    
    overall_acc = 100 * correct / total
    class_acc = {cls: 100 * class_correct[cls] / max(class_total[cls], 1) for cls in all_behaviors}
    class_counts = {cls: class_total[cls] for cls in all_behaviors}
    
    return overall_acc, class_acc, class_counts

def estimate_int8_memory(model, input_shape, batch_size=32):
    """Estimate INT8 memory usage."""
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # INT8 quantization reduces memory by 4x
    param_memory = total_params * 1  # 1 byte per INT8 parameter
    gradient_memory = trainable_params * 4  # FP32 gradients
    optimizer_memory = trainable_params * 2 * 4  # Adam states (FP32)
    
    # Activations (INT8)
    activation_memory = batch_size * 64 * 1  # 64 features, INT8
    
    # Replay buffer (INT8 features)
    replay_memory = 5000 * 64 * 1  # 5000 samples, 64 features, INT8
    
    # TDM masks (binary)
    mask_memory = trainable_params * 1  # 1 bit per parameter
    
    # Workspace
    workspace_memory = batch_size * 1024 * 1  # 1KB per sample, INT8
    
    total_sram = param_memory + gradient_memory + optimizer_memory + activation_memory + replay_memory + mask_memory + workspace_memory
    total_flash = param_memory + mask_memory
    
    return {
        'total_sram_mb': total_sram / (1024 * 1024),
        'total_flash_mb': total_flash / (1024 * 1024),
        'param_memory_mb': param_memory / (1024 * 1024),
        'replay_memory_mb': replay_memory / (1024 * 1024),
        'total_params': total_params,
        'trainable_params': trainable_params
    }

def main():
    """Main function for INT8 quantized CNN pipeline."""
    # Set fixed random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Load configuration
    cfg = load_config()
    
    # Load intelligent sampling data
    dog_data, behavior_to_idx, behaviors = load_processed_dog_data()
    
    if len(dog_data) == 0:
        print("No dog data loaded!")
        return
    
    # Define behaviors
    backbone_behaviors = ['Standing', 'Walking', 'Sitting', 'Lying chest']
    cil_behaviors = ['Sniffing', 'Trotting', 'Galloping']
    all_behaviors = backbone_behaviors + cil_behaviors
    
    # CIL tasks
    cil_tasks = [
        ['Standing', 'Walking', 'Sitting', 'Lying chest', 'Sniffing'],
        ['Standing', 'Walking', 'Sitting', 'Lying chest', 'Sniffing', 'Trotting'],
        ['Standing', 'Walking', 'Sitting', 'Lying chest', 'Sniffing', 'Trotting', 'Galloping']
    ]
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get input dimension
    input_dim = len(dog_data[list(dog_data.keys())[0]]['sensor_cols'])
    
    # STEP 1: BACKBONE TRAINING
    print(f"\n{'='*50}")
    print("STEP 1: BACKBONE TRAINING (INT8 Quantized)")
    print(f"{'='*50}")
    
    # Create backbone model
    backbone_num_classes = len(backbone_behaviors)
    C = len(dog_data[list(dog_data.keys())[0]]['sensor_cols'])
    backbone_model = SimplifiedTDMModelCNN_INT8(C, backbone_num_classes, device, sparsity_ratio=0.3)
    backbone_model.to(device)
    
    # Estimate memory usage
    memory_estimate = estimate_int8_memory(backbone_model, input_shape=(32, C, 100))
    print(f"INT8 Memory Estimate:")
    print(f"  SRAM: {memory_estimate['total_sram_mb']:.2f} MB")
    print(f"  Flash: {memory_estimate['total_flash_mb']:.2f} MB")
    print(f"  Parameters: {memory_estimate['total_params']:,}")
    print(f"  Replay Buffer: {memory_estimate['replay_memory_mb']:.2f} MB")
    
    # Prepare backbone training data
    target_dog = 47
    validation_dog = 48
    
    # Exclude target dog and validation dog from backbone training
    all_dogs = list(dog_data.keys())
    available_dogs = [dog for dog in all_dogs if dog not in [target_dog, validation_dog]]
    backbone_dogs = available_dogs[:10]
    
    print(f"Backbone training dogs: {backbone_dogs}")
    print(f"Backbone validation dog: {validation_dog}")
    print(f"Target dog for CIL: {target_dog}")
    
    # Prepare training data
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
    
    backbone_X = torch.cat(backbone_X_list, dim=0)
    backbone_y = torch.cat(backbone_y_list, dim=0)
    
    # Use all backbone training data
    X_train = backbone_X.numpy()
    y_train = backbone_y.numpy()
    
    # Get separate validation dog data
    if validation_dog not in dog_data:
        print(f"Validation dog {validation_dog} not found!")
        return
    
    val_X_np = dog_data[validation_dog]['X'].numpy()
    val_y_np = dog_data[validation_dog]['y'].numpy()
    
    # Filter to backbone classes
    val_mask = np.isin(val_y_np, [behavior_to_idx[c] for c in backbone_behaviors])
    val_X_np = val_X_np[val_mask]
    val_y_np = val_y_np[val_mask]
    
    # Remap validation labels
    val_y_remapped = np.zeros(len(val_y_np), dtype=np.int64)
    for i, cls in enumerate(backbone_behaviors):
        val_y_remapped[val_y_np == behavior_to_idx[cls]] = i
    
    print(f"Validation dog {validation_dog} behavior distribution:")
    for i, cls in enumerate(backbone_behaviors):
        count = (val_y_remapped == i).sum()
        print(f"  {cls}: {count} samples")
    
    # Compute normalization statistics
    train_mean, train_std = compute_normalization_stats(X_train)
    
    # Normalize data
    X_train_norm = normalize_features(X_train, train_mean, train_std)
    X_val_norm = normalize_features(val_X_np, train_mean, train_std)
    
    # Convert to tensors
    X_train = torch.from_numpy(X_train_norm).float()
    y_train = torch.from_numpy(y_train).long()
    X_val = torch.from_numpy(X_val_norm).float()
    y_val = torch.from_numpy(val_y_remapped).long()
    
    # Create data loaders
    backbone_train_loader = DataLoader(TensorDataset(X_train, y_train), 
                                      batch_size=32, shuffle=True)
    backbone_val_loader = DataLoader(TensorDataset(X_val, y_val), 
                                    batch_size=32, shuffle=False)
    
    backbone_optimizer = torch.optim.Adam(backbone_model.parameters(), lr=0.001)
    backbone_criterion = nn.CrossEntropyLoss()
    
    print("Training INT8 quantized backbone model...")
    best_val_acc = 0.0
    patience = 5
    counter = 0
    best_state = None
    
    for epoch in range(50):
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
            counter += 1
        
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
            break
        
        if epoch % 10 == 0:
            print(f"[Backbone] epoch {epoch}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
    
    if best_state is not None:
        backbone_model.load_state_dict(best_state, strict=True)
    
    # Print per-class validation accuracy
    print(f"\n--- BACKBONE VALIDATION PER-CLASS ACCURACY ---")
    backbone_model.eval()
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
    
    print(f"Backbone training complete! Best validation accuracy: {best_val_acc:.3f}")
    
    # Freeze backbone
    for param in backbone_model.backbone.parameters():
        param.requires_grad = False
    
    # Create CIL model
    cil_model = SimplifiedTDMModelCNN_INT8(C, len(backbone_behaviors), device, sparsity_ratio=0.3).to(device)
    cil_model.backbone.load_state_dict(backbone_model.backbone.state_dict())
    
    # Freeze CIL backbone
    for param in cil_model.backbone.parameters():
        param.requires_grad = False
    
    # STEP 2: CIL TRAINING
    print(f"\n{'='*50}")
    print("STEP 2: CIL TRAINING (INT8 Quantized)")
    print(f"{'='*50}")
    
    # Split target dog data with segment-aware splitting
    if target_dog not in dog_data:
        print(f"Target dog {target_dog} not found!")
        return
    
    X_target = dog_data[target_dog]['X']
    y_target = dog_data[target_dog]['y']
    segment_ids = dog_data[target_dog]['segment_ids']
    
    # Use GroupShuffleSplit to ensure no segment is split across train/test
    from sklearn.model_selection import GroupShuffleSplit
    
    # Convert to numpy for sklearn
    X_np = X_target.numpy()
    y_np = y_target.numpy()
    
    # GroupShuffleSplit to prevent temporal leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X_np, y_np, groups=segment_ids))
    
    X_train, X_test = X_np[train_idx], X_np[test_idx]
    y_train, y_test = y_np[train_idx], y_np[test_idx]
    
    print(f"Segment-aware split: {len(X_train)} train samples, {len(X_test)} test samples")
    print(f"Train segments: {len(np.unique(segment_ids[train_idx]))}, Test segments: {len(np.unique(segment_ids[test_idx]))}")
    print(f"Ensuring no segment overlap: {len(set(segment_ids[train_idx]) & set(segment_ids[test_idx])) == 0}")
    
    # Print target dog class distribution
    print(f"\n--- TARGET DOG {target_dog} CLASS DISTRIBUTION ---")
    train_class_counts = {}
    test_class_counts = {}
    
    for i, cls in enumerate(all_behaviors):
        train_class_counts[cls] = (y_train == i).sum()
        test_class_counts[cls] = (y_test == i).sum()
        print(f"  {cls}: {train_class_counts[cls]} train, {test_class_counts[cls]} test")
    
    # Use SAME normalization stats from backbone training
    X_train_norm = normalize_features(X_train, train_mean, train_std)
    X_test_norm = normalize_features(X_test, train_mean, train_std)
    
    # Convert to tensors
    X_train = torch.from_numpy(X_train_norm).float()
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test_norm).float()
    y_test = torch.from_numpy(y_test).long()
    
    # Evaluate backbone on all 7 classes before CIL starts
    print(f"\n--- BACKBONE EVALUATION ON ALL 7 CLASSES ---")
    
    # Create evaluation classifier for all 7 classes
    backbone_model.update_eval_classifier()
    
    # Remap test labels to 0-6 range for evaluation
    y_test_remapped = torch.zeros(len(y_test), dtype=torch.long)
    for i, cls in enumerate(all_behaviors):
        cls_idx = behavior_to_idx[cls]
        y_test_remapped[y_test == cls_idx] = i
    
    test_loader = DataLoader(TensorDataset(X_test, y_test_remapped), 
                           batch_size=32, shuffle=False)
    
    overall_acc, class_acc, class_counts = evaluate_all_classes(
        backbone_model, test_loader, device, all_behaviors
    )
    
    print(f"Backbone overall accuracy on all 7 classes: {overall_acc:.1f}%")
    print(f"Backbone per-class accuracy:")
    for cls, acc in class_acc.items():
        print(f"  {cls}: {acc:.1f}%")
    
    # Initialize replay buffer
    replay_buffer = []
    replay_labels = []
    buffer_size = 5000
    
    # Store accuracy history
    accuracy_history = []
    
    # Simplified TDM parameters
    p_intra = 0.02
    p_inter = 0.05
    delta_k = 10
    
    # CIL Training loop
    for task_idx, task_classes in enumerate(cil_tasks):
        # Determine which new class is being added
        if task_idx == 0:
            new_class = "Sniffing"
        elif task_idx == 1:
            new_class = "Trotting"
        else:
            new_class = "Galloping"
        
        print(f"\n--- CIL TASK {task_idx + 1}: Added '{new_class}' ---")
        
        # Expand classifier if needed
        if task_idx == 0:
            new_classes = len(task_classes) - cil_model.num_classes
            cil_model.expand_classifier(new_classes)
        else:
            new_classes = len(task_classes) - len(cil_tasks[task_idx - 1])
            cil_model.expand_classifier(new_classes)
        
        # Inter-task adjustment (expand-and-shrink)
        if task_idx > 0:
            cil_model.update_mask_inter_task(p_inter=p_inter)
        
        # Create task-specific data loader
        task_indices = []
        for i, cls in enumerate(task_classes):
            cls_idx = behavior_to_idx[cls]
            task_indices.extend(torch.where(y_train == cls_idx)[0].tolist())
        
        task_X = X_train[task_indices]
        task_y = y_train[task_indices]
        
        # Remap labels to current task
        task_y_remapped = torch.zeros(len(task_y), dtype=torch.long)
        for i, cls in enumerate(task_classes):
            cls_idx = behavior_to_idx[cls]
            task_y_remapped[task_y == cls_idx] = i
        
        task_loader = DataLoader(TensorDataset(task_X, task_y_remapped), 
                               batch_size=32, shuffle=True)
        
        # Train the model
        optimizer = torch.optim.Adam(cil_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(15):
            cil_model.classifier.train()
            cil_model.backbone.eval()
            for m in cil_model.backbone.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.track_running_stats = False  # can repeat
            
            # Learning rate scheduling
            if epoch == 5:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
            
            train_loss, train_acc = train_with_simplified_tdm(
                cil_model, task_loader, optimizer, criterion, device,
                replay_buffer, replay_labels, task_idx, epoch, num_classes=len(task_classes), buffer_size=buffer_size,
                p_intra=p_intra, p_inter=p_inter, delta_k=delta_k
            )
        
        # Update evaluation classifier
        cil_model.update_eval_classifier()
        
        # Evaluate on all classes
        test_loader = DataLoader(TensorDataset(X_test, y_test_remapped), 
                               batch_size=32, shuffle=False)
        
        overall_acc, class_acc, class_counts = evaluate_all_classes(
            cil_model, test_loader, device, all_behaviors
        )
        
        print(f"Task {task_idx + 1} complete - Overall accuracy: {overall_acc:.1f}%")
        print(f"Per-class accuracy after task {task_idx + 1}:")
        for cls, acc in class_acc.items():
            print(f"  {cls}: {acc:.1f}%")
        
        accuracy_history.append(overall_acc)
    
    # Final evaluation
    print(f"\n{'='*50}")
    print("FINAL SUMMARY")
    print(f"{'='*50}")
    
    final_overall_acc, final_class_acc, final_class_counts = evaluate_all_classes(
        cil_model, test_loader, device, all_behaviors
    )
    
    print(f"Final overall accuracy: {final_overall_acc:.1f}%")
    print(f"Final per-class accuracy:")
    for cls, acc in final_class_acc.items():
        print(f"  {cls}: {acc:.1f}%")
    
    # Final memory estimate
    final_memory_estimate = estimate_int8_memory(cil_model, input_shape=(32, C, 100))
    print(f"\nFinal INT8 Memory Usage:")
    print(f"  SRAM: {final_memory_estimate['total_sram_mb']:.2f} MB")
    print(f"  Flash: {final_memory_estimate['total_flash_mb']:.2f} MB")
    print(f"  Parameters: {final_memory_estimate['total_params']:,}")
    print(f"  Replay Buffer: {final_memory_estimate['replay_memory_mb']:.2f} MB")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy progression
    stages = ['Backbone'] + [f'Task {i+1}' for i in range(len(accuracy_history))]
    accuracies = [overall_acc] + accuracy_history
    
    ax1.plot(stages, accuracies, 'o-', linewidth=2, markersize=8)
    ax1.set_title('CIL Accuracy Progression (INT8 Quantized)')
    ax1.set_ylabel('Overall Accuracy (%)')
    ax1.set_xlabel('CIL Stage')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Per-class final accuracy
    classes = list(final_class_acc.keys())
    accs = list(final_class_acc.values())
    
    bars = ax2.bar(classes, accs, color='skyblue', alpha=0.7)
    ax2.set_title('Final Per-Class Accuracy (INT8 Quantized)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_xlabel('Behavior Class')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('int8_cnn_results.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    print(f"\nResults saved to int8_cnn_results.png")

if __name__ == "__main__":
    main()
