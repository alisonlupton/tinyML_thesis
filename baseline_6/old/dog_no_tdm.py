#!/usr/bin/env python3
"""
EVOLUTIONARY CIL pipeline: Track accuracy on all 7 classes across tasks
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

def load_config():
    """Load configuration."""
    with open("dog_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config['dogmove']

def load_dog_data(cfg):
    """Load raw dog data with only 7 main behaviors."""
    print("Loading raw dog data...")
    
    df = pd.read_csv(cfg['data_path'])
    
    # Select sensor columns
    sensor_cols = []
    if cfg['use_acc']:
        sensor_cols.extend([col for col in df.columns if col.startswith('ABack_') or col.startswith('ANeck_')])
    if cfg['use_gyro']:
        sensor_cols.extend([col for col in df.columns if col.startswith('GBack_') or col.startswith('GNeck_')])
    
    print(f"Using {len(sensor_cols)} sensor columns")
    
    # Define the 7 main behaviors
    main_behaviors = [
        'Standing', 'Walking', 'Sitting', 'Lying on chest', 
        'Sniffing', 'Trotting', 'Galloping'
    ]
    
    # Filter data to only include these 7 behaviors
    df_filtered = df[df['Behavior_1'].isin(main_behaviors)].copy()
    
    # Get behavior counts
    behavior_counts = df_filtered['Behavior_1'].value_counts()
    print(f"Behavior counts after filtering:")
    for behavior, count in behavior_counts.items():
        print(f"  {behavior}: {count} samples")
    
    # Create behavior mapping - use consistent order for CIL
    # Order: backbone behaviors first, then CIL behaviors
    backbone_behaviors = ['Standing', 'Walking', 'Sitting', 'Lying on chest']
    cil_behaviors = ['Sniffing', 'Trotting', 'Galloping']
    behaviors = backbone_behaviors + cil_behaviors  # Consistent order
    behavior_to_idx = {b: i for i, b in enumerate(behaviors)}
    print(f"Using {len(behaviors)} behaviors: {behaviors}")
    
    # Prepare data per dog
    dog_data = {}
    for dog_id in sorted(df_filtered['DogID'].unique()):
        dog_df = df_filtered[df_filtered['DogID'] == dog_id]
        
        # Sample data to manageable size
        if len(dog_df) > 10000:
            dog_df = dog_df.sample(n=10000, random_state=42)
        
        # Extract features and labels
        X = dog_df[sensor_cols].values.astype(np.float32)
        y = dog_df['Behavior_1'].map(behavior_to_idx).values
        
        # Normalize features
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        dog_data[dog_id] = {
            'X': torch.from_numpy(X),
            'y': torch.from_numpy(y),
            'behaviors': sorted(dog_df['Behavior_1'].unique())
        }
    
    return dog_data, behavior_to_idx, behaviors

class EvolutionaryCILModel(nn.Module):
    """Evolutionary CIL model with separate evaluation head."""
    
    def __init__(self, input_dim, num_classes, device):
        super().__init__()
        # Backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # Training classifier (expandable)
        self.classifier = CWRHead(64, num_classes, device)
        # Evaluation classifier (always 7 classes)
        self.eval_classifier = nn.Linear(64, 7)
        self.num_classes = num_classes
        self.device = device
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def forward_eval(self, x):
        """Forward pass for evaluation on all 7 classes."""
        features = self.backbone(x)
        return self.eval_classifier(features)
    
    def get_features(self, x):
        """Get backbone features."""
        return self.backbone(x)
    
    def expand_classifier(self, new_classes):
        """Expand the training classifier."""
        old_classifier = self.classifier
        new_num_classes = self.num_classes + new_classes
        
        # Create new classifier
        self.classifier = CWRHead(64, new_num_classes, self.device)
        
        # Copy old weights
        with torch.no_grad():
            self.classifier.weight[:self.num_classes] = old_classifier.weight
            self.classifier.bias[:self.num_classes] = old_classifier.bias
            self.classifier.cw[:self.num_classes] = old_classifier.cw
            self.classifier.cb[:self.num_classes] = old_classifier.cb
            self.classifier.hist_count[:self.num_classes] = old_classifier.hist_count
            
            # Initialize new weights with proper scaling
            self.classifier.weight[self.num_classes:] = torch.randn(new_classes, 64) * (1.0 / np.sqrt(64))
            self.classifier.bias[self.num_classes:] = torch.zeros(new_classes)
        
        self.num_classes = new_num_classes
    
    def update_eval_classifier(self):
        """Update evaluation classifier with current training classifier weights."""
        with torch.no_grad():
            # Copy weights from training classifier to evaluation classifier
            # For classes that have been learned, copy the weights
            self.eval_classifier.weight[:self.num_classes] = self.classifier.weight
            self.eval_classifier.bias[:self.num_classes] = self.classifier.bias
            
            # For remaining classes (not yet learned), keep random initialization
            # This ensures unseen classes show 0% accuracy as expected

def evaluate_all_classes(model, loader, device, behavior_names):
    """Evaluate model on all 7 classes using evaluation head."""
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 7
    class_total = [0] * 7
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model.forward_eval(data)  # Use evaluation head
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Per-class accuracy for all 7 classes
            for i in range(7):
                mask = (target == i)
                class_total[i] += mask.sum().item()
                class_correct[i] += (pred.squeeze()[mask] == target[mask]).sum().item()
    
    overall_acc = correct / total if total > 0 else 0
    class_acc = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                 for i in range(7)]
    
    # Create per-class accuracy dictionary
    class_acc_dict = {}
    for i, acc in enumerate(class_acc):
        if class_total[i] > 0:
            class_acc_dict[behavior_names[i]] = f"{acc:.1%}"
    
    return overall_acc, class_acc_dict, class_total

def train_with_adaptive_replay(model, train_loader, optimizer, criterion, device, 
                             replay_buffer, task_idx, num_classes, buffer_size=5000):
    """Train with adaptive replay ratio based on task progress."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Adaptive replay ratio: lower for early tasks (focus on learning), higher for later tasks (focus on remembering)
    if task_idx == 0:
        replay_ratio = 0.1  # Low replay for first task (focus on learning new class)
    elif task_idx == 1:
        replay_ratio = 0.2  # Medium replay for second task
    else:
        replay_ratio = 0.3  # Higher replay for later tasks (focus on remembering)
    
    print(f"  Using replay ratio: {replay_ratio:.1f}")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Get features
        features = model.get_features(data)
        
        # Sample replay data
        replay_data = None
        if len(replay_buffer) > 0 and random.random() < replay_ratio:
            replay_features, replay_targets = sample_replay(replay_buffer, batch_size=len(data))
            replay_data = (replay_features.to(device), replay_targets.to(device))
        
        # Forward pass
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
        
        # Update replay buffer with larger size for sensor data
        update_replay_reservoir(replay_buffer, features.detach().cpu(), target.detach().cpu(), buffer_size)
        
        # Statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    return total_loss / len(train_loader), correct / total

def update_replay_reservoir(replay_buffer, features, targets, buffer_size=5000):
    """Update replay buffer using reservoir sampling with larger buffer for sensor data."""
    for feat, target in zip(features, targets):
        if len(replay_buffer) < buffer_size:
            replay_buffer.append((feat, target))
        else:
            # Reservoir sampling
            j = random.randint(0, len(replay_buffer) - 1)
            if j < buffer_size:
                replay_buffer[j] = (feat, target)

def sample_replay(replay_buffer, batch_size=32):
    """Sample from replay buffer."""
    if len(replay_buffer) == 0:
        return None, None
    
    indices = random.sample(range(len(replay_buffer)), min(batch_size, len(replay_buffer)))
    features = torch.stack([replay_buffer[i][0] for i in indices])
    targets = torch.stack([replay_buffer[i][1] for i in indices])
    return features, targets

def plot_accuracy_evolution(accuracy_history, behavior_names, save_path="accuracy_evolution.png"):
    """Plot overall accuracy evolution across tasks as a line plot."""
    plt.figure(figsize=(10, 6))
    
    # Calculate overall accuracy for each task
    overall_accuracies = []
    task_labels = []
    
    for task_idx, task_acc in enumerate(accuracy_history):
        # Calculate weighted average accuracy across all 7 classes
        total_correct = 0
        total_samples = 0
        
        for behavior, acc_str in task_acc.items():
            acc = float(acc_str.strip('%')) / 100
            # Get class index to find sample count
            class_idx = behavior_names.index(behavior)
            # For simplicity, assume equal weight per class (or you can add actual sample counts)
            total_correct += acc
            total_samples += 1
        
        overall_acc = total_correct / total_samples if total_samples > 0 else 0
        overall_accuracies.append(overall_acc)
        task_labels.append(f'Task {task_idx + 1}')
    
    # Create line plot
    plt.plot(task_labels, overall_accuracies, marker='o', linewidth=3, markersize=10, 
             color='#2E86AB', label='Overall Accuracy')
    
    # Add value labels on points
    for i, acc in enumerate(overall_accuracies):
        plt.text(i, acc + 0.02, f'{acc:.1%}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    # Customize the plot
    plt.xlabel('CIL Tasks', fontsize=14, fontweight='bold')
    plt.ylabel('Overall Accuracy (All 7 Classes)', fontsize=14, fontweight='bold')
    plt.title('Overall Accuracy Evolution Across CIL Tasks\n(All 7 Behavior Classes)', 
             fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(0, 1.05)
    
    # Add horizontal lines for reference
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% Reference')
    plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='80% Reference')
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Overall accuracy evolution line plot saved to {save_path}")
    
    # Print the progression
    print(f"\nOverall Accuracy Progression:")
    for i, acc in enumerate(overall_accuracies):
        print(f"  {task_labels[i]}: {acc:.1%}")
    
    # Create second plot: New class accuracy evolution
    plot_new_class_evolution(accuracy_history, behavior_names, "new_class_evolution.png")

def plot_new_class_evolution(accuracy_history, behavior_names, save_path="new_class_evolution.png"):
    """Plot accuracy evolution of newly introduced classes across tasks."""
    plt.figure(figsize=(12, 8))
    
    # Define which classes are introduced in each task
    new_classes_per_task = {
        'Task 1': ['Sniffing'],  # First new class introduced
        'Task 2': ['Trotting'],  # Second new class introduced  
        'Task 3': ['Galloping']  # Third new class introduced
    }
    
    # Colors for each new class
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    
    # Track each new class's performance across tasks
    class_performance = {}
    
    for task_idx, task_acc in enumerate(accuracy_history):
        task_name = f'Task {task_idx + 1}'
        
        for class_name, acc_str in task_acc.items():
            acc = float(acc_str.strip('%')) / 100
            
            # If this is a new class, track its performance
            if class_name in ['Sniffing', 'Trotting', 'Galloping']:
                if class_name not in class_performance:
                    class_performance[class_name] = []
                class_performance[class_name].append(acc)
    
    # Plot each new class's evolution
    task_labels = ['Task 1', 'Task 2', 'Task 3']
    
    for i, (class_name, performances) in enumerate(class_performance.items()):
        # Pad with zeros for tasks before the class was introduced
        padded_performances = []
        if class_name == 'Sniffing':
            padded_performances = performances  # All 3 tasks
        elif class_name == 'Trotting':
            padded_performances = [0.0] + performances  # Task 1 = 0, then actual values
        elif class_name == 'Galloping':
            padded_performances = [0.0, 0.0] + performances  # Tasks 1&2 = 0, then actual value
        
        # Ensure we have exactly 3 values for 3 tasks
        while len(padded_performances) < 3:
            padded_performances.append(0.0)
        padded_performances = padded_performances[:3]  # Truncate to exactly 3
        
        # Plot the line
        plt.plot(task_labels, padded_performances, marker='o', linewidth=3, markersize=10,
                color=colors[i], label=f'{class_name} (Introduced in {new_classes_per_task[f"Task {i+1}"][0]})')
        
        # Add value labels on points
        for j, acc in enumerate(padded_performances):
            if acc > 0:  # Only show labels for non-zero values
                plt.text(j, acc + 0.02, f'{acc:.1%}', ha='center', va='bottom',
                        fontsize=11, fontweight='bold')
    
    # Customize the plot
    plt.xlabel('CIL Tasks', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy of New Classes', fontsize=14, fontweight='bold')
    plt.title('New Class Accuracy Evolution Across CIL Tasks\n(Shows learning and retention of newly introduced classes)', 
             fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(-0.05, 1.05)
    
    # Add horizontal lines for reference
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% Reference')
    plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='80% Reference')
    
    plt.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"New class accuracy evolution plot saved to {save_path}")
    
    # Print the new class progression
    print(f"\nNew Class Accuracy Progression:")
    for class_name, performances in class_performance.items():
        print(f"  {class_name}: {performances}")

def main():
    """Evolutionary CIL pipeline that tracks accuracy on all 7 classes."""
    cfg = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== EVOLUTIONARY CIL PIPELINE ===")
    print("Track accuracy on all 7 classes across tasks")
    
    # Load data
    dog_data, behavior_to_idx, all_behaviors = load_dog_data(cfg)
    
    # DEFINE EVOLUTIONARY CIL SETUP
    # Backbone behaviors (subset of 4 behaviors) - same order as in load_dog_data
    backbone_behaviors = ['Standing', 'Walking', 'Sitting', 'Lying on chest']
    
    # CIL behaviors (truly new behaviors for backbone) - same order as in load_dog_data
    cil_behaviors = ['Sniffing', 'Trotting', 'Galloping']
    
    print(f"\n=== EVOLUTIONARY CIL SETUP ===")
    print(f"Backbone behaviors (4): {backbone_behaviors}")
    print(f"CIL behaviors (3): {cil_behaviors}")
    print(f"Buffer size: 5,000 samples")
    print(f"Adaptive replay ratios: 0.1 → 0.2 → 0.3")
    
    # Select target dog (ensure it has ALL behaviors)
    target_dog = 47
    target_behaviors = dog_data[target_dog]['behaviors']
    print(f"\nTarget dog {target_dog} has behaviors: {target_behaviors}")
    
    # Verify target dog has all behaviors
    missing_behaviors = set(backbone_behaviors + cil_behaviors) - set(target_behaviors)
    if missing_behaviors:
        print(f"WARNING: Target dog missing behaviors: {missing_behaviors}")
        return
    
    # Define CIL tasks (REALISTIC: each task includes new classes + all backbone behaviors)
    cil_tasks = [
        backbone_behaviors + ['Sniffing'],  # 5 classes: 4 backbone + 1 new
        backbone_behaviors + ['Sniffing', 'Trotting'],  # 6 classes: 4 backbone + 2 new
        backbone_behaviors + ['Sniffing', 'Trotting', 'Galloping']  # 7 classes: 4 backbone + 3 new
    ]
    
    print(f"\n=== EVOLUTIONARY CIL TASKS ===")
    for i, task in enumerate(cil_tasks):
        print(f"Task {i+1}: {len(task)} classes - {task}")
    
    # STEP 1: BACKBONE PRETRAINING
    print(f"\n{'='*50}")
    print("STEP 1: BACKBONE PRETRAINING")
    print(f"{'='*50}")
    
    # Use multiple dogs for backbone pretraining (excluding target dog)
    backbone_dogs = [dog_id for dog_id in dog_data.keys() if dog_id != target_dog]
    print(f"Using {len(backbone_dogs)} dogs for backbone pretraining: {backbone_dogs[:5]}...")
    
    # Collect backbone training data
    X_backbone = []
    y_backbone = []
    
    for dog_id in backbone_dogs:
        dog_X = dog_data[dog_id]['X']
        dog_y = dog_data[dog_id]['y']
        
        # Filter for backbone behaviors only
        backbone_indices = [behavior_to_idx[b] for b in backbone_behaviors]
        mask = torch.isin(dog_y, torch.tensor(backbone_indices))
        
        if mask.sum() > 0:
            X_backbone.append(dog_X[mask])
            y_backbone.append(dog_y[mask])
    
    X_backbone = torch.cat(X_backbone, dim=0)
    y_backbone = torch.cat(y_backbone, dim=0)
    
    # Remap backbone labels to 0-based indices
    backbone_behavior_to_idx = {b: i for i, b in enumerate(backbone_behaviors)}
    y_backbone_remapped = torch.zeros(len(y_backbone), dtype=torch.long)
    for i, behavior in enumerate(backbone_behaviors):
        original_idx = behavior_to_idx[behavior]
        y_backbone_remapped[y_backbone == original_idx] = i
    
    print(f"Backbone training data: {len(X_backbone)} samples")
    print(f"Backbone behaviors: {backbone_behaviors}")
    
    # Create backbone model
    input_dim = X_backbone.shape[1]
    backbone_model = EvolutionaryCILModel(input_dim, len(backbone_behaviors), device).to(device)
    
    # Train backbone
    backbone_loader = DataLoader(TensorDataset(X_backbone, y_backbone_remapped), 
                               batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(backbone_model.parameters(), lr=cfg['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining backbone...")
    for epoch in range(10):
        backbone_model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in backbone_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = backbone_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        loss = total_loss / len(backbone_loader)
        acc = correct / total
        print(f"  Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.3f}")
    
    # Evaluate backbone
    backbone_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in backbone_loader:
            data, target = data.to(device), target.to(device)
            output = backbone_model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        backbone_acc = correct / total
    
    print(f"Backbone final accuracy: {backbone_acc:.3f}")
    
    # STEP 2: EVOLUTIONARY CIL WITH FROZEN BACKBONE
    print(f"\n{'='*50}")
    print("STEP 2: EVOLUTIONARY CIL WITH FROZEN BACKBONE")
    print(f"{'='*50}")
    
    # Freeze backbone
    for param in backbone_model.backbone.parameters():
        param.requires_grad = False
    print("Backbone frozen!")
    
    # Create CIL model (copy backbone, new classifier)
    cil_model = EvolutionaryCILModel(input_dim, len(backbone_behaviors), device).to(device)  # Start with backbone classes
    cil_model.backbone.load_state_dict(backbone_model.backbone.state_dict())
    
    # Freeze CIL backbone
    for param in cil_model.backbone.parameters():
        param.requires_grad = False
    
    # Split target dog data
    X_target = dog_data[target_dog]['X']
    y_target = dog_data[target_dog]['y']
    
    # Create fixed test set (last 30% of data)
    n_test = int(0.3 * len(X_target))
    X_test = X_target[-n_test:]
    y_test = y_target[-n_test:]
    X_train = X_target[:-n_test]
    y_train = y_target[:-n_test]
    
    print(f"Target dog data split: {len(X_train)} train, {len(X_test)} test")
    
    # Initialize replay buffer with larger size for sensor data
    replay_buffer = []
    buffer_size = 5000  # Increased from 1000 to 5000 for sensor data
    
    # Store accuracy history for plotting
    accuracy_history = []
    
    # CIL Training loop
    for task_idx, task_classes in enumerate(cil_tasks):
        print(f"\n{'='*50}")
        print(f"EVOLUTIONARY CIL TASK {task_idx + 1}: {len(task_classes)} classes")
        print(f"Classes: {task_classes}")
        print(f"{'='*50}")
        
        # Expand classifier if needed
        if task_idx == 0:
            # First task: expand from 4 backbone classes to 5 classes (4 + 1 new)
            new_classes = len(task_classes) - cil_model.num_classes
            cil_model.expand_classifier(new_classes)
            print(f"Expanded classifier from {cil_model.num_classes - new_classes} to {cil_model.num_classes} classes")
        else:
            # Subsequent tasks: expand by the difference
            new_classes = len(task_classes) - len(cil_tasks[task_idx - 1])
            cil_model.expand_classifier(new_classes)
            print(f"Expanded classifier to {cil_model.num_classes} classes")
        
        # Prepare training data for this task (ALL classes in this task)
        task_indices = [behavior_to_idx[cls] for cls in task_classes]
        mask = torch.isin(y_train, torch.tensor(task_indices))
        X_task = X_train[mask]
        y_task = y_train[mask]
        
        # Remap labels to 0-based indices for this task
        y_task_remapped = torch.zeros(len(y_task), dtype=torch.long)
        for i, cls in enumerate(task_classes):
            cls_idx = behavior_to_idx[cls]
            y_task_remapped[y_task == cls_idx] = i
        
        print(f"Task {task_idx + 1} training data: {len(X_task)} samples")
        print(f"Classes in this task: {task_classes}")
        
        # Show class distribution
        unique, counts = torch.unique(y_task_remapped, return_counts=True)
        print(f"Class distribution:")
        for cls_idx, count in zip(unique, counts):
            print(f"  {task_classes[cls_idx]}: {count} samples")
        
        # Create data loader
        train_loader = DataLoader(TensorDataset(X_task, y_task_remapped), 
                                batch_size=16, shuffle=True)
        
        # Training with adaptive replay
        optimizer = torch.optim.Adam(cil_model.classifier.parameters(), lr=cfg['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        print(f"\nTraining CIL Task {task_idx + 1}...")
        for epoch in range(15):  # 15 epochs per task
            # Learning rate scheduling
            if epoch == 5:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                print(f"  Reducing learning rate to {optimizer.param_groups[0]['lr']:.6f}")
            elif epoch == 10:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                print(f"  Reducing learning rate to {optimizer.param_groups[0]['lr']:.6f}")
            
            loss, acc = train_with_adaptive_replay(
                cil_model, train_loader, optimizer, criterion, device, 
                replay_buffer, task_idx, num_classes=len(task_classes), buffer_size=buffer_size
            )
            print(f"  Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.3f}")
        
        # Update evaluation classifier with current weights
        cil_model.update_eval_classifier()
        
        # EVALUATE ON ALL 7 CLASSES
        print(f"\n--- EVALUATION ON ALL 7 CLASSES FOR TASK {task_idx + 1} ---")
        
        # Create test loader for all 7 classes with CORRECT label mapping
        # Use the same label mapping as training (all_behaviors order)
        test_loader = DataLoader(TensorDataset(X_test, y_test), 
                               batch_size=32, shuffle=False)
        
        # Evaluate on all 7 classes
        overall_acc, class_acc, class_counts = evaluate_all_classes(
            cil_model, test_loader, device, all_behaviors
        )
        
        print(f"Overall accuracy on all 7 classes: {overall_acc:.3f}")
        print(f"Per-class accuracy: {class_acc}")
        print(f"Class counts: {class_counts}")
        
        # Store accuracy history for plotting
        accuracy_history.append(class_acc)
    
    # Plot accuracy evolution
    print(f"\n{'='*50}")
    print("PLOTTING ACCURACY EVOLUTION")
    print(f"{'='*50}")
    
    plot_accuracy_evolution(accuracy_history, all_behaviors)
    
    print(f"\n{'='*50}")
    print("EVOLUTIONARY CIL PIPELINE COMPLETED!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
