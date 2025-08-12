#!/usr/bin/env python3
"""
Simplified TDM Pipeline with Intelligent Sampling: Following SparCL paper more closely
Uses exact same logic as dog_tdm.py but with intelligent sampling data loading
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from cwr_head import CWRHead
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tinyml_metrics import TinyMLMetrics
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
            "Processed data not found! Please run dog_processing_tdm_intelligent.py first."
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
        X = data['X']
        y = data['y']
        
        # Convert to tensors
        dog_data[dog_id] = {
            'X': torch.from_numpy(X.astype(np.float32)),
            'y': torch.from_numpy(y.astype(np.int64)),
            'behaviors': list(data['behaviors']),
            'sensor_cols': list(data['sensor_cols'])
        }
        
        total_samples += len(X)
    
    print(f"\nTotal samples loaded: {total_samples:,}")
    
    # Analyze sample distribution
    all_behaviors = []
    for dog_id, data in dog_data.items():
        all_behaviors.extend([behaviors[y] for y in data['y'].numpy()])
    
    behavior_counts = pd.Series(all_behaviors).value_counts()
    print(f"\nSample behavior distribution:")
    for behavior, count in behavior_counts.items():
        print(f"  {behavior}: {count:,} samples")
    
    return dog_data, behavior_to_idx, behaviors

class SimplifiedTDMModel(nn.Module):
    """Simplified TDM model following SparCL paper more closely."""
    
    def __init__(self, input_dim, num_classes, device, sparsity_ratio=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.device = device
        self.sparsity_ratio = sparsity_ratio
        
        # Backbone (feature extractor)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Classifier
        self.classifier = nn.Linear(64, num_classes)
        
        # Initialize binary mask for TDM
        self.mask = torch.ones_like(self.classifier.weight.data, device=device)
        self.apply_initial_sparsity()
        
        # Evaluation classifier (for all classes)
        self.eval_classifier = None
        
        print(f"Simplified TDM initialized: {self.mask.sum().item()}/{self.mask.numel()} parameters active ({100*self.mask.sum().item()/self.mask.numel():.1f}% density)")
    
    def apply_initial_sparsity(self):
        """Apply initial sparsity to the classifier."""
        with torch.no_grad():
            # Randomly mask weights
            mask_flat = self.mask.view(-1)
            num_to_mask = int(mask_flat.numel() * self.sparsity_ratio)
            indices_to_mask = torch.randperm(mask_flat.numel())[:num_to_mask]
            mask_flat[indices_to_mask] = 0
            self.mask = mask_flat.view_as(self.classifier.weight.data)
            
            # Apply mask
            self.classifier.weight.data *= self.mask
    
    def apply_mask(self):
        """Apply the current mask to the classifier weights."""
        with torch.no_grad():
            self.classifier.weight.data *= self.mask
    
    def update_mask_intra_task(self, p_intra=0.05):
        """Intra-task mask update: shrink-and-expand."""
        with torch.no_grad():
            # Calculate weight importance (magnitude)
            weight_importance = torch.abs(self.classifier.weight.data)
            
            # Remove least important weights
            num_to_remove = int(self.mask.numel() * p_intra)
            flat_importance = weight_importance.view(-1)
            flat_mask = self.mask.view(-1)
            
            # Find least important active weights
            active_indices = torch.where(flat_mask > 0)[0]
            if len(active_indices) > num_to_remove:
                least_important = torch.argsort(flat_importance[active_indices])[:num_to_remove]
                flat_mask[active_indices[least_important]] = 0
            
            # Randomly add back weights
            inactive_indices = torch.where(flat_mask == 0)[0]
            if len(inactive_indices) > num_to_remove:
                to_activate = torch.randperm(len(inactive_indices))[:num_to_remove]
                flat_mask[inactive_indices[to_activate]] = 1
            
            self.mask = flat_mask.view_as(self.classifier.weight.data)
            self.apply_mask()
    
    def update_mask_inter_task(self, p_inter=0.1):
        """Inter-task mask update: expand-and-shrink."""
        with torch.no_grad():
            # Expand: randomly add weights
            flat_mask = self.mask.view(-1)
            inactive_indices = torch.where(flat_mask == 0)[0]
            num_to_add = int(self.mask.numel() * p_inter)
            
            if len(inactive_indices) > num_to_add:
                to_activate = torch.randperm(len(inactive_indices))[:num_to_add]
                flat_mask[inactive_indices[to_activate]] = 1
            
            self.mask = flat_mask.view_as(self.classifier.weight.data)
            self.apply_mask()
            print(f"  Simplified TDM Inter-task: expand-and-shrink (p_inter={p_inter})")
    
    def shrink_after_warmup(self, p_inter=0.1):
        """Shrink back to target sparsity after warm-up."""
        with torch.no_grad():
            # Calculate weight importance
            weight_importance = torch.abs(self.classifier.weight.data)
            
            # Remove least important weights
            num_to_remove = int(self.mask.numel() * p_inter)
            flat_importance = weight_importance.view(-1)
            flat_mask = self.mask.view(-1)
            
            # Find least important active weights
            active_indices = torch.where(flat_mask > 0)[0]
            if len(active_indices) > num_to_remove:
                least_important = torch.argsort(flat_importance[active_indices])[:num_to_remove]
                flat_mask[active_indices[least_important]] = 0
            
            self.mask = flat_mask.view_as(self.classifier.weight.data)
            self.apply_mask()
    
    def expand_classifier(self, new_classes):
        """Expand classifier for new classes."""
        old_num_classes = self.num_classes
        self.num_classes += new_classes
        
        # Create new classifier
        old_classifier = self.classifier
        self.classifier = nn.Linear(64, self.num_classes).to(self.device)
        
        # Copy old weights
        self.classifier.weight.data[:old_num_classes] = old_classifier.weight.data
        self.classifier.bias.data[:old_num_classes] = old_classifier.bias.data
        
        # Initialize new weights
        nn.init.xavier_uniform_(self.classifier.weight.data[old_num_classes:])
        nn.init.zeros_(self.classifier.bias.data[old_num_classes:])
        
        # Expand mask
        old_mask = self.mask
        self.mask = torch.ones_like(self.classifier.weight.data, device=self.device)
        self.mask[:old_num_classes] = old_mask
        
        # Apply initial sparsity to new weights
        new_mask_flat = self.mask[old_num_classes:].view(-1)
        num_to_mask = int(new_mask_flat.numel() * self.sparsity_ratio)
        indices_to_mask = torch.randperm(new_mask_flat.numel())[:num_to_mask]
        new_mask_flat[indices_to_mask] = 0
        self.mask[old_num_classes:] = new_mask_flat.view_as(self.mask[old_num_classes:])
        
        self.apply_mask()
        
    def update_eval_classifier(self):
        """Update evaluation classifier for all classes."""
        self.eval_classifier = nn.Linear(64, 7).to(self.device)  # All 7 classes
        self.eval_classifier.weight.data[:self.num_classes] = self.classifier.weight.data
        self.eval_classifier.bias.data[:self.num_classes] = self.classifier.bias.data
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def forward_eval(self, x):
        """Forward pass for evaluation on all classes."""
        features = self.backbone(x)
        return self.eval_classifier(features)

def train_with_simplified_tdm(model, train_loader, optimizer, criterion, device, 
                             replay_buffer, replay_labels, task_idx, epoch, num_classes, buffer_size,
                             p_intra=0.05, p_inter=0.1, delta_k=5):
    """Training function with simplified TDM and latent replay."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Replay ratio - how often to use replay
    replay_ratio = 0.5 if task_idx > 0 else 0.0  # No replay in first task
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Extract features from backbone
        features = model.backbone(data)
        
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

def main():
    """Main function - exact same logic as dog_tdm.py but with intelligent sampling."""
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
    
    # Define behaviors exactly like dog_tdm.py (but use "Lying chest" to match raw data)
    backbone_behaviors = ['Standing', 'Walking', 'Sitting', 'Lying chest']
    cil_behaviors = ['Sniffing', 'Trotting', 'Galloping']
    all_behaviors = backbone_behaviors + cil_behaviors
    
    # CIL tasks exactly like dog_tdm.py (but use "Lying chest" to match raw data)
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
    
    # Initialize TinyML metrics
    # tinyml_tracker = TinyMLMetrics(save_path="logs/tdm_intelligent_metrics.csv")
    
    # STEP 1: BACKBONE TRAINING (exact same as dog_tdm.py)
    print(f"\n{'='*50}")
    print("STEP 1: BACKBONE TRAINING")
    print(f"{'='*50}")
    
    # Create backbone model
    backbone_num_classes = len(backbone_behaviors)
    backbone_model = SimplifiedTDMModel(input_dim, backbone_num_classes, device, sparsity_ratio=0.3)
    backbone_model.to(device)
    
    # Analyze backbone model
    input_shape = (1, 1, input_dim)  # 1D input for TinyML
    # tinyml_tracker.analyze_model(backbone_model, input_shape=input_shape, device=device)
    
    # Prepare backbone training data (exact same logic as dog_tdm.py)
    backbone_dogs = list(dog_data.keys())[:10]  # Use first 10 dogs for backbone
    backbone_X = []
    backbone_y = []
    
    for dog_id in backbone_dogs:
        dog_X = dog_data[dog_id]['X']
        dog_y = dog_data[dog_id]['y']
        
        # Only include backbone behaviors
        backbone_indices = [behavior_to_idx[cls] for cls in backbone_behaviors]
        mask = torch.isin(dog_y, torch.tensor(backbone_indices))
        if mask.sum() > 0:
            backbone_X.append(dog_X[mask])
            # Remap labels to backbone classes
            remapped_y = torch.zeros(mask.sum(), dtype=torch.long)
            for i, cls in enumerate(backbone_behaviors):
                cls_idx = behavior_to_idx[cls]
                remapped_y[dog_y[mask] == cls_idx] = i
            backbone_y.append(remapped_y)
    
    if len(backbone_X) == 0:
        print("No backbone training data found!")
        return
    
    backbone_X = torch.cat(backbone_X, dim=0)
    backbone_y = torch.cat(backbone_y, dim=0)
    
    # Train backbone (exact same as dog_tdm.py)
    backbone_loader = DataLoader(TensorDataset(backbone_X, backbone_y), 
                                batch_size=32, shuffle=True)
    
    backbone_optimizer = torch.optim.Adam(backbone_model.parameters(), lr=0.001)
    backbone_criterion = nn.CrossEntropyLoss()
    
    print("Training backbone model...")
    backbone_model.train()
    for epoch in range(50):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(backbone_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            backbone_optimizer.zero_grad()
            outputs = backbone_model(X_batch)
            loss = backbone_criterion(outputs, y_batch)
            loss.backward()
            backbone_optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        
        if epoch % 10 == 0:
            acc = 100 * correct / total
            print(f"[Backbone] epoch {epoch} acc={acc:.3f}")
    
    print("Backbone training complete!")
    
    # Freeze backbone
    for param in backbone_model.backbone.parameters():
        param.requires_grad = False
    
    # Create CIL model (exact same as dog_tdm.py)
    cil_model = SimplifiedTDMModel(input_dim, len(backbone_behaviors), device, sparsity_ratio=0.3).to(device)
    cil_model.backbone.load_state_dict(backbone_model.backbone.state_dict())
    
    # Freeze CIL backbone
    for param in cil_model.backbone.parameters():
        param.requires_grad = False
    
    # Analyze CIL model
    # tinyml_tracker.analyze_model(cil_model, input_shape=input_shape, device=device)
    
    # Split target dog data with stratified sampling to ensure all behaviors are represented
    target_dog = 47  # Target dog for personalization
    if target_dog not in dog_data:
        print(f"Target dog {target_dog} not found!")
        return
    
    X_target = dog_data[target_dog]['X']
    y_target = dog_data[target_dog]['y']
    
    # Use stratified sampling to ensure all behaviors are in both train and test
    from sklearn.model_selection import train_test_split
    
    # Convert to numpy for sklearn
    X_np = X_target.numpy()
    y_np = y_target.numpy()
    
    # Stratified split to preserve behavior distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=0.3, random_state=42, stratify=y_np
    )
    
    # Convert back to tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float()
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
    
    # Simplified TDM parameters (exact same as dog_tdm.py)
    p_intra = 0.02   # Much smaller
    p_inter = 0.05   # Much smaller
    delta_k = 10     # Much less frequent
    
    # CIL Training loop (exact same as dog_tdm.py)
    for task_idx, task_classes in enumerate(cil_tasks):
        # Determine which new class is being added
        if task_idx == 0:
            new_class = "Sniffing"
        elif task_idx == 1:
            new_class = "Trotting"
        else:
            new_class = "Galloping"
        
        
        print(f"\n--- CIL TASK {task_idx + 1}: Adding '{new_class}' ---")
        
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
        
        # Prepare training data (exact same as dog_tdm.py)
        task_indices = [behavior_to_idx[cls] for cls in task_classes]
        mask = torch.isin(y_train, torch.tensor(task_indices))
        X_task = X_train[mask]
        y_task = y_train[mask]
        
        y_task_remapped = torch.zeros(len(y_task), dtype=torch.long)
        for i, cls in enumerate(task_classes):
            cls_idx = behavior_to_idx[cls]
            y_task_remapped[y_task == cls_idx] = i
        
        # Create data loader
        train_loader = DataLoader(TensorDataset(X_task, y_task_remapped), 
                                batch_size=16, shuffle=True)
        
        # Training with simplified TDM and replay (exact same as dog_tdm.py)
        optimizer = torch.optim.Adam(cil_model.classifier.parameters(), lr=cfg['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        # Train the model
        for epoch in range(15):
            # Learning rate scheduling
            if epoch == 5:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
            elif epoch == 10:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
            
            # Intra-task TDM adjustment
            if epoch > 0 and epoch % delta_k == 0:
                cil_model.update_mask_intra_task(p_intra=p_intra)
            
            # Inter-task warm-up end (shrink back to target sparsity)
            if task_idx > 0 and epoch == delta_k:
                cil_model.shrink_after_warmup(p_inter=p_inter)
            
            loss, acc = train_with_simplified_tdm(
                cil_model, train_loader, optimizer, criterion, device, 
                replay_buffer, replay_labels, task_idx, epoch, num_classes=len(task_classes), buffer_size=buffer_size,
                p_intra=p_intra, p_inter=p_inter, delta_k=delta_k
            )
        
        # Update evaluation classifier
        cil_model.update_eval_classifier()
        
        # EVALUATE ON ALL 7 CLASSES
        # Remap test labels to 0-6 range for evaluation
        y_test_remapped = torch.zeros(len(y_test), dtype=torch.long)
        for i, cls in enumerate(all_behaviors):
            cls_idx = behavior_to_idx[cls]
            y_test_remapped[y_test == cls_idx] = i
        
        test_loader = DataLoader(TensorDataset(X_test, y_test_remapped), 
                               batch_size=32, shuffle=False)
        
        overall_acc, class_acc, class_counts = evaluate_all_classes(
            cil_model, test_loader, device, all_behaviors
        )
        
        print(f"Task {task_idx + 1} complete - Overall accuracy: {overall_acc:.1f}%")
        print(f"Per-class accuracy after task {task_idx + 1}:") 
        for cls, acc in class_acc.items(): 
            print(f"  {cls}: {acc:.1f}%")
        
        # Store accuracy history
        accuracy_history.append(class_acc)
    
    # Print final summary
    print(f"\n{'='*50}")
    print("FINAL SUMMARY")
    print(f"{'='*50}")
    
    print(f"Final overall accuracy: {overall_acc:.1f}%")
    print(f"Final per-class accuracy:")
    for cls, acc in class_acc.items():
        print(f"  {cls}: {acc:.1f}%")
    
    # Save results
    results = {
        'accuracy_history': accuracy_history,
        'final_accuracy': overall_acc,
        'final_class_acc': class_acc
    }
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Task accuracy progression
    plt.subplot(2, 3, 1)
    task_accs = [acc.get('Sniffing', 0) for acc in accuracy_history]
    plt.plot(range(1, len(task_accs) + 1), task_accs, 'o-', label='Sniffing')
    plt.xlabel('Task')
    plt.ylabel('Accuracy (%)')
    plt.title('Sniffing Accuracy Progression')
    plt.legend()
    
    # Overall accuracy
    plt.subplot(2, 3, 2)
    overall_accs = [sum(acc.values()) / len(acc) for acc in accuracy_history]
    plt.plot(range(1, len(overall_accs) + 1), overall_accs, 'o-', label='Overall')
    plt.xlabel('Task')
    plt.ylabel('Accuracy (%)')
    plt.title('Overall Accuracy Progression')
    plt.legend()
    
    # Final class accuracy
    plt.subplot(2, 3, 3)
    classes = list(class_acc.keys())
    accs = list(class_acc.values())
    plt.bar(classes, accs)
    plt.xlabel('Behavior')
    plt.ylabel('Accuracy (%)')
    plt.title('Final Per-Class Accuracy')
    plt.xticks(rotation=45)
    
    # Behavior distribution
    plt.subplot(2, 3, 4)
    all_behaviors = []
    for dog_id, data in dog_data.items():
        all_behaviors.extend([behaviors[y] for y in data['y'].numpy()])
    
    behavior_counts = pd.Series(all_behaviors).value_counts()
    plt.bar(behavior_counts.index, behavior_counts.values)
    plt.xlabel('Behavior')
    plt.ylabel('Number of Samples')
    plt.title('Behavior Distribution in Intelligent Sampling')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('intelligent_tdm_fixed_results.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    print(f"\nResults saved to intelligent_tdm_fixed_results.png")
    # print(f"TinyML metrics saved to logs/tdm_intelligent_metrics.csv")

if __name__ == "__main__":
    main()
