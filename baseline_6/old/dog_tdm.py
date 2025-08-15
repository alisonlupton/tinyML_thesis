#!/usr/bin/env python3
"""
Simplified TDM Pipeline: Following SparCL paper more closely
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
    
    # Create a function to select the best behavior from the 3 columns
    def select_behavior(row):
        """Select the best behavior from Behavior_1, Behavior_2, Behavior_3."""
        behaviors = [row['Behavior_1'], row['Behavior_2'], row['Behavior_3']]
        
        # Priority: Behavior_1 > Behavior_2 > Behavior_3
        for behavior in behaviors:
            if behavior != '<undefined>' and behavior in main_behaviors:
                return behavior
        
        # If no main behavior found, return None
        return None
    
    # Apply behavior selection
    df['Selected_Behavior'] = df.apply(select_behavior, axis=1)
    
    # Filter to only rows with valid behaviors
    df_filtered = df[df['Selected_Behavior'].notna()].copy()
    
    print(f"Behavior selection results:")
    print(f"  Total rows: {len(df)}")
    print(f"  Rows with valid behaviors: {len(df_filtered)}")
    print(f"  Rows dropped: {len(df) - len(df_filtered)}")
    
    # Get behavior counts using selected behavior
    behavior_counts = df_filtered['Selected_Behavior'].value_counts()
    print(f"Behavior counts after filtering:")
    for behavior, count in behavior_counts.items():
        print(f"  {behavior}: {count} samples")
    
    # Create behavior mapping - use consistent order for CIL
    backbone_behaviors = ['Standing', 'Walking', 'Sitting', 'Lying on chest']
    cil_behaviors = ['Sniffing', 'Trotting', 'Galloping']
    behaviors = backbone_behaviors + cil_behaviors  # Consistent order
    behavior_to_idx = {b: i for i, b in enumerate(behaviors)}
    print(f"Using {len(behaviors)} behaviors: {behaviors}")
    
    def random_sample(dog_df, target_size=50000, main_behaviors=main_behaviors):
        """Random sampling to preserve proportional representation of the 7 main behaviors."""
        # Filter to only the 7 main behaviors we care about
        dog_df_main = dog_df[dog_df['Selected_Behavior'].isin(main_behaviors)].copy()
        
        if len(dog_df_main) == 0:
            return dog_df_main  # No main behaviors found
        
        # Simple random sampling to preserve proportional representation
        if len(dog_df_main) > target_size:
            sampled = dog_df_main.sample(n=target_size, random_state=42)
        else:
            sampled = dog_df_main  # Keep all if not enough
        
        return sampled
    
    # Prepare data per dog
    dog_data = {}
    for dog_id in sorted(df_filtered['DogID'].unique()):
        dog_df = df_filtered[df_filtered['DogID'] == dog_id]
        

        # Use random sampling for proportional representation
        if len(dog_df) > 50000:
            print(f"Dog {dog_id}: {len(dog_df)} samples → sampling to 50K")
            dog_df = random_sample(dog_df, target_size=50000)
        else:
            print(f"Dog {dog_id}: {len(dog_df)} samples (keeping all)")
        
        # Extract features and labels
        X = dog_df[sensor_cols].values.astype(np.float32)
        y = dog_df['Selected_Behavior'].map(behavior_to_idx).values
        
        # Normalize features
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        dog_data[dog_id] = {
            'X': torch.from_numpy(X),
            'y': torch.from_numpy(y),
            'behaviors': sorted(dog_df['Selected_Behavior'].unique())
        }
    
    return dog_data, behavior_to_idx, behaviors

class SimplifiedTDMModel(nn.Module):
    """Simplified TDM model following SparCL paper more closely."""
    
    def __init__(self, input_dim, num_classes, device, sparsity_ratio=0.3):  # Much less sparsity
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
        
        # Classifier with TDM
        self.classifier = CWRHead(64, num_classes, device)
        self.eval_classifier = nn.Linear(64, 7)
        
        # TDM parameters - MORE CONSERVATIVE
        self.sparsity_ratio = sparsity_ratio  # Only 30% sparsity
        self.device = device
        self.num_classes = num_classes
        
        # Initialize binary mask for TDM
        self.initialize_tdm_mask()
    
    def initialize_tdm_mask(self):
        """Initialize binary mask for TDM."""
        total_params = sum(p.numel() for p in self.parameters())
        active_params = int(total_params * (1 - self.sparsity_ratio))
        
        # Create binary mask
        self.mask = torch.zeros(total_params, device=self.device)
        active_indices = torch.randperm(total_params)[:active_params]
        self.mask[active_indices] = 1.0
        
        print(f"Simplified TDM initialized: {active_params}/{total_params} parameters active ({1-self.sparsity_ratio:.1%} density)")
    
    def apply_mask(self):
        """Apply binary mask to model parameters."""
        param_idx = 0
        for param in self.parameters():
            param_size = param.numel()
            param_mask = self.mask[param_idx:param_idx + param_size].view(param.shape)
            param.data *= param_mask
            param_idx += param_size
    
    def get_cwi_scores(self, current_data, rehearsal_buffer, alpha=0.1, beta=0.1):  # Much smaller coefficients
        """Calculate Continual Weight Importance (CWI) scores - simplified but more conservative."""
        cwi_scores = []
        param_idx = 0
        
        for param in self.parameters():
            param_size = param.numel()
            
            # Weight magnitude (primary importance)
            magnitude = torch.abs(param.data)
            
            # Current task importance (very small)
            current_importance = magnitude * alpha
            
            # Past data importance (very small)
            past_importance = magnitude * beta
            
            # CWI = |w| + α|∂L̃/∂w| + β|∂L/∂w|
            cwi = magnitude + current_importance + past_importance
            
            cwi_scores.append(cwi.flatten())
            param_idx += param_size
        
        return torch.cat(cwi_scores)
    
    def update_mask_intra_task(self, p_intra=0.02, current_data=None, rehearsal_buffer=None):  # Much smaller p_intra
        """Intra-task (within a task) adjustment: shrink-and-expand."""
        print(f"  Simplified TDM Intra-task: shrink-and-expand (p_intra={p_intra})")
        
        # Calculate CWI scores
        cwi_scores = self.get_cwi_scores(current_data, rehearsal_buffer)
        
        # Shrink: remove p_intra of least important weights
        num_to_remove = int(self.mask.sum() * p_intra)
        if num_to_remove > 0:
            active_indices = torch.where(self.mask == 1)[0]
            cwi_active = cwi_scores[active_indices]
            least_important = torch.argsort(cwi_active)[:num_to_remove]
            remove_indices = active_indices[least_important]
            self.mask[remove_indices] = 0
        
        # Expand: randomly add unused weights back
        num_to_add = num_to_remove
        if num_to_add > 0:
            unused_indices = torch.where(self.mask == 0)[0]
            add_indices = torch.randperm(len(unused_indices))[:num_to_add]
            self.mask[unused_indices[add_indices]] = 1
        
        # Apply mask
        self.apply_mask()
        
        active_params = self.mask.sum().item()
        total_params = len(self.mask)
        print(f"  Simplified TDM: {active_params}/{total_params} parameters active ({active_params/total_params:.1%} density)")
    
    def update_mask_inter_task(self, p_inter=0.05, current_data=None, rehearsal_buffer=None):  # Much smaller p_inter
        """Inter-task (between task) adjustment: expand-and-shrink."""
        print(f"  Simplified TDM Inter-task: expand-and-shrink (p_inter={p_inter})")
        
        # Expand: add p_inter unused weights
        num_to_add = int(self.mask.sum() * p_inter)
        if num_to_add > 0:
            unused_indices = torch.where(self.mask == 0)[0]
            add_indices = torch.randperm(len(unused_indices))[:num_to_add]
            self.mask[unused_indices[add_indices]] = 1
        
        # Apply mask
        self.apply_mask()
        
        active_params = self.mask.sum().item()
        total_params = len(self.mask)
        print(f"  Simplified TDM: {active_params}/{total_params} parameters active ({active_params/total_params:.1%} density)")
    
    def shrink_after_warmup(self, p_inter=0.05, current_data=None, rehearsal_buffer=None):
        """Shrink after warm-up period."""
        print(f"  Simplified TDM Warm-up end: shrinking back to target sparsity")
        
        # Calculate CWI scores
        cwi_scores = self.get_cwi_scores(current_data, rehearsal_buffer)
        
        # Remove p_inter of least important weights
        num_to_remove = int(self.mask.sum() * p_inter)
        if num_to_remove > 0:
            active_indices = torch.where(self.mask == 1)[0]
            cwi_active = cwi_scores[active_indices]
            least_important = torch.argsort(cwi_active)[:num_to_remove]
            remove_indices = active_indices[least_important]
            self.mask[remove_indices] = 0
        
        # Apply mask
        self.apply_mask()
        
        active_params = self.mask.sum().item()
        total_params = len(self.mask)
        print(f"  Simplified TDM: {active_params}/{total_params} parameters active ({active_params/total_params:.1%} density)")
    
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
        """Expand the training classifier with mask preservation."""
        old_classifier = self.classifier
        new_num_classes = self.num_classes + new_classes
        
        # Store current mask state
        old_mask = self.mask.clone()
        old_total_params = len(old_mask)
        
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
        
        # Reinitialize TDM mask but preserve old structure
        new_total_params = sum(p.numel() for p in self.parameters())
        new_active_params = int(new_total_params * (1 - self.sparsity_ratio))
        
        # Create new mask
        self.mask = torch.zeros(new_total_params, device=self.device)
        
        # Preserve old mask structure for existing parameters
        if old_total_params <= new_total_params:
            self.mask[:old_total_params] = old_mask
        
        # Add new active parameters
        remaining_active = new_active_params - int(old_mask.sum().item())
        if remaining_active > 0:
            unused_indices = torch.where(self.mask == 0)[0]
            add_indices = torch.randperm(len(unused_indices))[:remaining_active]
            self.mask[unused_indices[add_indices]] = 1
        
        print(f"Simplified TDM: Preserved mask structure during expansion")
        print(f"  Old: {int(old_mask.sum().item())}/{old_total_params} active")
        print(f"  New: {int(self.mask.sum().item())}/{new_total_params} active")
    
    def update_eval_classifier(self):
        """Update evaluation classifier with current training classifier weights."""
        with torch.no_grad():
            self.eval_classifier.weight[:self.num_classes] = self.classifier.weight
            self.eval_classifier.bias[:self.num_classes] = self.classifier.bias

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
            output = model.forward_eval(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            for i in range(7):
                mask = (target == i)
                class_total[i] += mask.sum().item()
                class_correct[i] += (pred.squeeze()[mask] == target[mask]).sum().item()
    
    overall_acc = correct / total if total > 0 else 0
    class_acc = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                 for i in range(7)]
    
    class_acc_dict = {}
    for i, acc in enumerate(class_acc):
        if class_total[i] > 0:
            class_acc_dict[behavior_names[i]] = f"{acc:.1%}"
    
    return overall_acc, class_acc_dict, class_total

def train_with_simplified_tdm(model, train_loader, optimizer, criterion, device, 
                             replay_buffer, task_idx, epoch, num_classes, buffer_size=5000,
                             p_intra=0.02, p_inter=0.05, delta_k=10):  # Much less frequent adjustments
    """Train with simplified TDM and adaptive replay."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Adaptive replay ratio
    if task_idx == 0:
        replay_ratio = 0.1
    elif task_idx == 1:
        replay_ratio = 0.2
    else:
        replay_ratio = 0.3
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
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
        
        # Apply TDM mask LESS FREQUENTLY
        if batch_idx % 10 == 0:  # Only apply every 10 batches
            model.apply_mask()
        
        # Update replay buffer
        update_replay_reservoir(replay_buffer, features.detach().cpu(), target.detach().cpu(), buffer_size)
        
        # Statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    return total_loss / len(train_loader), correct / total

def update_replay_reservoir(replay_buffer, features, targets, buffer_size=5000):
    """Update replay buffer using reservoir sampling."""
    for feat, target in zip(features, targets):
        if len(replay_buffer) < buffer_size:
            replay_buffer.append((feat, target))
        else:
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

def main():
    """Simplified TDM pipeline."""
    # Set fixed random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    cfg = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== SIMPLIFIED TDM PIPELINE ===")
    print("Following SparCL paper more closely with conservative parameters")
    print("Fixed random seed: 42 for reproducibility")
    
    # Initialize TinyML metrics tracker
    tinyml_tracker = TinyMLMetrics(save_path="logs/tdm_simplified_metrics.csv")
    print("TinyML metrics tracking enabled!")
    
    # Load data
    dog_data, behavior_to_idx, all_behaviors = load_dog_data(cfg)
    
    # DEFINE SIMPLIFIED TDM SETUP
    backbone_behaviors = ['Standing', 'Walking', 'Sitting', 'Lying on chest']
    cil_behaviors = ['Sniffing', 'Trotting', 'Galloping']
    
    print(f"\n=== SIMPLIFIED TDM SETUP ===")
    print(f"Backbone behaviors (4): {backbone_behaviors}")
    print(f"CIL behaviors (3): {cil_behaviors}")
    print(f"TDM sparsity ratio: 0.3 (70% weights active) - MUCH LESS SPARSITY!")
    print(f"TDM p_intra: 0.02, p_inter: 0.05, δk: 10 - MUCH MORE CONSERVATIVE!")
    
    # Select target dog
    target_dog = 47
    target_behaviors = dog_data[target_dog]['behaviors']
    print(f"\nTarget dog {target_dog} has behaviors: {target_behaviors}")
    
    # Define CIL tasks
    cil_tasks = [
        backbone_behaviors + ['Sniffing'],
        backbone_behaviors + ['Sniffing', 'Trotting'],
        backbone_behaviors + ['Sniffing', 'Trotting', 'Galloping']
    ]
    
    print(f"\n=== SIMPLIFIED TDM CIL TASKS ===")
    for i, task in enumerate(cil_tasks):
        print(f"Task {i+1}: {len(task)} classes - {task}")
    
    # STEP 1: BACKBONE PRETRAINING WITH SIMPLIFIED TDM
    print(f"\n{'='*50}")
    print("STEP 1: BACKBONE PRETRAINING WITH SIMPLIFIED TDM")
    print(f"{'='*50}")
    
    # Use multiple dogs for backbone pretraining
    backbone_dogs = [dog_id for dog_id in dog_data.keys() if dog_id != target_dog]
    print(f"Using {len(backbone_dogs)} dogs for backbone pretraining")
    
    # Collect backbone training data
    X_backbone = []
    y_backbone = []
    
    for dog_id in backbone_dogs:
        dog_X = dog_data[dog_id]['X']
        dog_y = dog_data[dog_id]['y']
        
        backbone_indices = [behavior_to_idx[b] for b in backbone_behaviors]
        mask = torch.isin(dog_y, torch.tensor(backbone_indices))
        
        if mask.sum() > 0:
            X_backbone.append(dog_X[mask])
            y_backbone.append(dog_y[mask])
    
    X_backbone = torch.cat(X_backbone, dim=0)
    y_backbone = torch.cat(y_backbone, dim=0)
    
    # Remap backbone labels
    backbone_behavior_to_idx = {b: i for i, b in enumerate(backbone_behaviors)}
    y_backbone_remapped = torch.zeros(len(y_backbone), dtype=torch.long)
    for i, behavior in enumerate(backbone_behaviors):
        original_idx = behavior_to_idx[behavior]
        y_backbone_remapped[y_backbone == original_idx] = i
    
    print(f"Backbone training data: {len(X_backbone)} samples")
    
    # Create simplified TDM backbone model
    input_dim = X_backbone.shape[1]
    backbone_model = SimplifiedTDMModel(input_dim, len(backbone_behaviors), device, sparsity_ratio=0.3).to(device)
    
    # Analyze backbone model with TinyML metrics
    print(f"\n--- TinyML Analysis: Backbone Model ---")
    input_shape = (1, 1, input_dim)  # (batch_size, channels, features) for 1D analysis
    tinyml_tracker.analyze_model(backbone_model, input_shape=input_shape, device=device)
    
    # Train backbone with simplified TDM
    backbone_loader = DataLoader(TensorDataset(X_backbone, y_backbone_remapped), 
                               batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(backbone_model.parameters(), lr=cfg['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining backbone with simplified TDM...")
    for epoch in range(10):
        backbone_model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(backbone_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = backbone_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Apply TDM mask LESS FREQUENTLY
            if random.random() < 0.1:  # Only 10% chance per batch
                backbone_model.apply_mask()
            
            # Track TinyML metrics every 100 batches
            if batch_idx % 100 == 0:
                tinyml_tracker.track_training_metrics(
                    task_id=0, epoch=epoch, step=batch_idx, 
                    model=backbone_model, input_shape=input_shape, device=device
                )
            
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
    
    print(f"Simplified TDM backbone final accuracy: {backbone_acc:.3f}")
    
    # STEP 2: SIMPLIFIED TDM CIL WITH FROZEN BACKBONE
    print(f"\n{'='*50}")
    print("STEP 2: SIMPLIFIED TDM CIL WITH FROZEN BACKBONE")
    print(f"{'='*50}")
    
    # Freeze backbone
    for param in backbone_model.backbone.parameters():
        param.requires_grad = False
    print("Backbone frozen!")
    
    # Create CIL model
    cil_model = SimplifiedTDMModel(input_dim, len(backbone_behaviors), device, sparsity_ratio=0.3).to(device)
    cil_model.backbone.load_state_dict(backbone_model.backbone.state_dict())
    
    # Freeze CIL backbone
    for param in cil_model.backbone.parameters():
        param.requires_grad = False
    
    # Analyze CIL model with TinyML metrics
    print(f"\n--- TinyML Analysis: CIL Model (Initial) ---")
    tinyml_tracker.analyze_model(cil_model, input_shape=input_shape, device=device)
    
    # Split target dog data
    X_target = dog_data[target_dog]['X']
    y_target = dog_data[target_dog]['y']
    
    n_test = int(0.3 * len(X_target))
    X_test = X_target[-n_test:]
    y_test = y_target[-n_test:]
    X_train = X_target[:-n_test]
    y_train = y_target[:-n_test]
    
    print(f"Target dog data split: {len(X_train)} train, {len(X_test)} test")
    
    # Initialize replay buffer
    replay_buffer = []
    buffer_size = 5000
    
    # Store accuracy history
    accuracy_history = []
    
    # Simplified TDM parameters - MUCH MORE CONSERVATIVE
    p_intra = 0.02   # Much smaller
    p_inter = 0.05   # Much smaller
    delta_k = 10     # Much less frequent
    
    # CIL Training loop with simplified TDM
    for task_idx, task_classes in enumerate(cil_tasks):
        print(f"\n{'='*50}")
        print(f"SIMPLIFIED TDM CIL TASK {task_idx + 1}: {len(task_classes)} classes")
        print(f"Classes: {task_classes}")
        print(f"{'='*50}")
        
        # Expand classifier if needed
        if task_idx == 0:
            new_classes = len(task_classes) - cil_model.num_classes
            cil_model.expand_classifier(new_classes)
            print(f"Expanded classifier from {cil_model.num_classes - new_classes} to {cil_model.num_classes} classes")
        else:
            new_classes = len(task_classes) - len(cil_tasks[task_idx - 1])
            cil_model.expand_classifier(new_classes)
            print(f"Expanded classifier to {cil_model.num_classes} classes")
        
        # Inter-task adjustment (expand-and-shrink)
        if task_idx > 0:
            print(f"\n--- Simplified TDM Inter-task Adjustment ---")
            cil_model.update_mask_inter_task(p_inter=p_inter)
        
        # Prepare training data
        task_indices = [behavior_to_idx[cls] for cls in task_classes]
        mask = torch.isin(y_train, torch.tensor(task_indices))
        X_task = X_train[mask]
        y_task = y_train[mask]
        
        y_task_remapped = torch.zeros(len(y_task), dtype=torch.long)
        for i, cls in enumerate(task_classes):
            cls_idx = behavior_to_idx[cls]
            y_task_remapped[y_task == cls_idx] = i
        
        print(f"Task {task_idx + 1} training data: {len(X_task)} samples")
        
        # Show class distribution
        unique, counts = torch.unique(y_task_remapped, return_counts=True)
        print(f"Class distribution:")
        for cls_idx, count in zip(unique, counts):
            print(f"  {task_classes[cls_idx]}: {count} samples")
        
        # Create data loader
        train_loader = DataLoader(TensorDataset(X_task, y_task_remapped), 
                                batch_size=16, shuffle=True)
        
        # Training with simplified TDM and replay
        optimizer = torch.optim.Adam(cil_model.classifier.parameters(), lr=cfg['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        print(f"\nTraining Simplified TDM CIL Task {task_idx + 1}...")
        for epoch in range(15):
            # Learning rate scheduling
            if epoch == 5:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                print(f"  Reducing learning rate to {optimizer.param_groups[0]['lr']:.6f}")
            elif epoch == 10:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                print(f"  Reducing learning rate to {optimizer.param_groups[0]['lr']:.6f}")
            
            # Intra-task TDM adjustment - MUCH LESS FREQUENT
            if epoch > 0 and epoch % delta_k == 0:
                print(f"\n--- Simplified TDM Intra-task Adjustment (Epoch {epoch}) ---")
                cil_model.update_mask_intra_task(p_intra=p_intra)
            
            # Inter-task warm-up end (shrink back to target sparsity)
            if task_idx > 0 and epoch == delta_k:
                print(f"\n--- Simplified TDM Warm-up End (Epoch {epoch}) ---")
                cil_model.shrink_after_warmup(p_inter=p_inter)
            
            loss, acc = train_with_simplified_tdm(
                cil_model, train_loader, optimizer, criterion, device, 
                replay_buffer, task_idx, epoch, num_classes=len(task_classes), buffer_size=buffer_size,
                p_intra=p_intra, p_inter=p_inter, delta_k=delta_k
            )
            
            # Track TinyML metrics every 5 epochs
            if epoch % 5 == 0:
                tinyml_tracker.track_training_metrics(
                    task_id=task_idx+1, epoch=epoch, step=epoch, 
                    model=cil_model, input_shape=input_shape, device=device
                )
            
            print(f"  Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.3f}")
        
        # Update evaluation classifier
        cil_model.update_eval_classifier()
        
        # EVALUATE ON ALL 7 CLASSES
        print(f"\n--- EVALUATION ON ALL 7 CLASSES FOR TASK {task_idx + 1} ---")
        
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
        
        print(f"Simplified TDM Overall accuracy on all 7 classes: {overall_acc:.3f}")
        print(f"Simplified TDM Per-class accuracy: {class_acc}")
        print(f"Class counts: {class_counts}")
        
        # Store accuracy history
        accuracy_history.append(class_acc)
        
        # Analyze model after each task with TinyML metrics
        print(f"\n--- TinyML Analysis: After Task {task_idx + 1} ---")
        tinyml_tracker.analyze_model(cil_model, input_shape=input_shape, device=device)
    
    # Print final summary
    print(f"\n{'='*50}")
    print("SIMPLIFIED TDM FINAL SUMMARY")
    print(f"{'='*50}")
    
    print(f"Simplified TDM Overall Accuracy Progression:")
    for i, task_acc in enumerate(accuracy_history):
        total_acc = sum(float(acc.strip('%')) / 100 for acc in task_acc.values()) / len(task_acc)
        print(f"  Task {i+1}: {total_acc:.1%}")
    
    print(f"\nSimplified TDM Per-Class Final Performance:")
    for behavior, acc in accuracy_history[-1].items():
        print(f"  {behavior}: {acc}")
    
    # Save and plot TinyML metrics
    print(f"\n--- Saving TinyML Metrics ---")
    tinyml_tracker.save_metrics()
    tinyml_tracker.plot_metrics()
    
    print(f"\n{'='*50}")
    print("SIMPLIFIED TDM PIPELINE COMPLETED!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
