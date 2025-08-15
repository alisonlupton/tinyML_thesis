#!/usr/bin/env python3
"""
Windowed TDM Pipeline: Using 2-second windows to eliminate temporal bias
Following the original paper's approach of windowing sensor data.
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

def identify_behavior_segments(df, min_segment_length=50):
    """
    Identify continuous segments of the same behavior.
    
    Args:
        df: DataFrame with 'Selected_Behavior' and 't_sec' columns
        min_segment_length: Minimum samples to consider a segment
    
    Returns:
        List of segment dictionaries with start_idx, end_idx, behavior, length
    """
    segments = []
    current_behavior = None
    start_idx = 0
    
    # Reset index to ensure proper indexing
    df = df.reset_index(drop=True)
    
    for i in range(len(df)):
        behavior = df.iloc[i]['Selected_Behavior']
        
        if behavior != current_behavior:
            # End current segment
            if current_behavior is not None and i - start_idx >= min_segment_length:
                segments.append({
                    'start_idx': start_idx,
                    'end_idx': i - 1,
                    'behavior': current_behavior,
                    'length': i - start_idx,
                    'start_time': df.iloc[start_idx]['t_sec'],
                    'end_time': df.iloc[i-1]['t_sec']
                })
            
            # Start new segment
            current_behavior = behavior
            start_idx = i
    
    # Handle last segment
    if current_behavior is not None and len(df) - start_idx >= min_segment_length:
        segments.append({
            'start_idx': start_idx,
            'end_idx': len(df) - 1,
            'behavior': current_behavior,
            'length': len(df) - start_idx,
            'start_time': df.iloc[start_idx]['t_sec'],
            'end_time': df.iloc[-1]['t_sec']
        })
    
    return segments

def intelligent_sample_segment(df, segment, target_samples_per_segment=100):
    """
    Intelligently sample from a behavior segment.
    
    Strategy:
    - Take samples from beginning (25%)
    - Take samples from middle (50%) 
    - Take samples from end (25%)
    """
    start_idx = segment['start_idx']
    end_idx = segment['end_idx']
    segment_length = segment['length']
    
    if segment_length <= target_samples_per_segment:
        # If segment is small, take all samples
        indices = list(range(start_idx, end_idx + 1))
    else:
        # Calculate sampling points
        n_beginning = max(1, int(target_samples_per_segment * 0.25))
        n_middle = max(1, int(target_samples_per_segment * 0.50))
        n_end = max(1, int(target_samples_per_segment * 0.25))
        
        # Beginning samples
        beginning_indices = list(range(start_idx, start_idx + n_beginning))
        
        # Middle samples (evenly spaced)
        middle_start = start_idx + (segment_length - n_middle) // 2
        middle_indices = list(range(middle_start, middle_start + n_middle))
        
        # End samples
        end_indices = list(range(end_idx - n_end + 1, end_idx + 1))
        
        indices = beginning_indices + middle_indices + end_indices
    
    return df.iloc[indices]

def intelligent_sample_dog_data(df, target_total_samples=50000, min_segment_length=50):
    """
    Intelligently sample dog data while preserving temporal structure.
    
    Args:
        df: DataFrame with dog data
        target_total_samples: Target number of samples to keep
        min_segment_length: Minimum samples to consider a behavior segment
    
    Returns:
        Sampled DataFrame
    """
    print(f"Original data: {len(df):,} samples")
    
    # Sort by time
    df = df.sort_values(['TestNum', 't_sec'])
    
    # Identify behavior segments
    segments = identify_behavior_segments(df, min_segment_length)
    print(f"Identified {len(segments)} behavior segments")
    
    # Analyze segments
    behavior_segments = {}
    for segment in segments:
        behavior = segment['behavior']
        if behavior not in behavior_segments:
            behavior_segments[behavior] = []
        behavior_segments[behavior].append(segment)
    
    print(f"Behavior segment analysis:")
    for behavior, segs in behavior_segments.items():
        total_length = sum(seg['length'] for seg in segs)
        avg_length = total_length / len(segs)
        print(f"  {behavior}: {len(segs)} segments, {total_length:,} total samples, {avg_length:.1f} avg length")
    
    # Calculate proportional targets
    total_original_samples = sum(seg['length'] for seg in segments)
    behavior_proportions = {}
    for behavior, segs in behavior_segments.items():
        total_length = sum(seg['length'] for seg in segs)
        behavior_proportions[behavior] = total_length / total_original_samples
    
    print(f"\nBehavior proportions:")
    for behavior, prop in behavior_proportions.items():
        print(f"  {behavior}: {prop:.3f} ({prop * target_total_samples:.0f} target samples)")
    
    # Sample from each behavior proportionally
    sampled_dfs = []
    for behavior, segs in behavior_segments.items():
        target_samples = int(behavior_proportions[behavior] * target_total_samples)
        samples_per_segment = max(1, target_samples // len(segs))
        
        print(f"\nSampling {behavior}: {target_samples} total samples, ~{samples_per_segment} per segment")
        
        behavior_samples = []
        for i, segment in enumerate(segs):
            segment_df = intelligent_sample_segment(df, segment, samples_per_segment)
            behavior_samples.append(segment_df)
            
            if i < 3:  # Show first few segments
                print(f"  Segment {i+1}: {segment['length']:,} → {len(segment_df):,} samples")
        
        # Combine all segments for this behavior
        if behavior_samples:
            behavior_df = pd.concat(behavior_samples, ignore_index=True)
            sampled_dfs.append(behavior_df)
            print(f"  {behavior} total: {len(behavior_df):,} samples")
    
    # Combine all behaviors
    final_df = pd.concat(sampled_dfs, ignore_index=True)
    
    print(f"\nFinal result: {len(final_df):,} samples")
    
    # Verify behavior distribution
    final_behavior_counts = final_df['Selected_Behavior'].value_counts()
    print(f"Final behavior distribution:")
    for behavior, count in final_behavior_counts.items():
        print(f"  {behavior}: {count:,} samples ({count/len(final_df)*100:.1f}%)")
    
    return final_df

def load_dog_data_intelligent(cfg):
    """Load dog data using intelligent sampling to preserve temporal structure."""
    print("Loading dog data with intelligent sampling...")
    
    df = pd.read_csv(cfg['data_path'])
    
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
    print(f"  Total rows: {len(df):,}")
    print(f"  Rows with valid behaviors: {len(df_filtered):,}")
    print(f"  Rows dropped: {len(df) - len(df_filtered):,}")
    
    # Create behavior mapping - use consistent order for CIL
    backbone_behaviors = ['Standing', 'Walking', 'Sitting', 'Lying on chest']
    cil_behaviors = ['Sniffing', 'Trotting', 'Galloping']
    behaviors = backbone_behaviors + cil_behaviors  # Consistent order
    behavior_to_idx = {b: i for i, b in enumerate(behaviors)}
    print(f"Using {len(behaviors)} behaviors: {behaviors}")
    
    # Intelligently sample data for each dog
    dog_data = {}
    total_samples = 0
    
    for dog_id in sorted(df_filtered['DogID'].unique()):
        dog_df = df_filtered[df_filtered['DogID'] == dog_id].copy()
        
        print(f"Processing Dog {dog_id}: {len(dog_df):,} raw samples")
        
        # Intelligently sample the data
        sampled_df = intelligent_sample_dog_data(dog_df, target_total_samples=50000, min_segment_length=50)
        
        if len(sampled_df) > 0:
            # Get sensor columns
            sensor_cols = []
            for col in sampled_df.columns:
                if col.startswith('ABack_') or col.startswith('ANeck_') or col.startswith('GBack_') or col.startswith('GNeck_'):
                    sensor_cols.append(col)
            
            # Extract features and labels
            X = sampled_df[sensor_cols].values.astype(np.float32)
            y = sampled_df['Selected_Behavior'].map(behavior_to_idx).values
            
            # Normalize features
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
            
            dog_data[dog_id] = {
                'X': torch.from_numpy(X),
                'y': torch.from_numpy(y.astype(np.int64)),
                'behaviors': sorted(sampled_df['Selected_Behavior'].unique()),
                'sampled_df': sampled_df
            }
            
            total_samples += len(sampled_df)
            print(f"  Intelligently sampled to {len(sampled_df):,} samples")
        else:
            print(f"  No valid samples created")
    
    print(f"\nTotal samples created: {total_samples:,}")
    
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
        self.sparsity_ratio = sparsity_ratio
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
    
    def get_cwi_scores(self, current_data, rehearsal_buffer, alpha=0.1, beta=0.1):
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
    
    def update_mask_intra_task(self, p_intra=0.02, current_data=None, rehearsal_buffer=None):
        """Intra-task (within a task) adjustment: shrink-and-expand."""
        print(f"  Simplified TDM Intra-task: shrink-and-expand (p_intra={p_intra})")
        
        # Get current CWI scores
        cwi_scores = self.get_cwi_scores(current_data, rehearsal_buffer)
        
        # Remove p_intra of least important weights
        num_to_remove = int(len(cwi_scores) * p_intra)
        if num_to_remove > 0:
            _, remove_indices = torch.topk(cwi_scores, k=num_to_remove, largest=False)
            self.mask[remove_indices] = 0.0
        
        # Randomly add back the same number
        inactive_indices = torch.where(self.mask == 0.0)[0]
        if len(inactive_indices) > 0:
            add_indices = inactive_indices[torch.randperm(len(inactive_indices))[:num_to_remove]]
            self.mask[add_indices] = 1.0
        
        # Apply updated mask
        self.apply_mask()
    
    def update_mask_inter_task(self, p_inter=0.05, current_data=None, rehearsal_buffer=None):
        """Inter-task (between tasks) adjustment: expand-and-shrink."""
        print(f"  Simplified TDM Inter-task: expand-and-shrink (p_inter={p_inter})")
        
        # Expand: randomly add p_inter of weights
        num_to_add = int(len(self.mask) * p_inter)
        inactive_indices = torch.where(self.mask == 0.0)[0]
        if len(inactive_indices) > 0:
            add_indices = inactive_indices[torch.randperm(len(inactive_indices))[:num_to_add]]
            self.mask[add_indices] = 1.0
        
        # Apply expanded mask
        self.apply_mask()
        
        # After warm-up period, shrink back
        # (This would be done after some training epochs)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def forward_eval(self, x):
        features = self.backbone(x)
        return self.eval_classifier(features)

def main():
    """Intelligent sampling TDM pipeline with temporal structure preservation."""
    # Set fixed random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Load configuration
    cfg = load_config()
    
    # Load intelligently sampled data
    dog_data, behavior_to_idx, behaviors = load_dog_data_intelligent(cfg)
    
    if len(dog_data) == 0:
        print("No valid dog data found!")
        return
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize TinyML metrics
    tinyml_tracker = TinyMLMetrics(save_path="logs/tdm_intelligent_metrics.csv")
    
    # Get input dimension from first dog's data
    first_dog_id = list(dog_data.keys())[0]
    input_dim = dog_data[first_dog_id]['X'].shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Create backbone model (trained on subset of dogs)
    backbone_behaviors = ['Standing', 'Walking', 'Sitting', 'Lying on chest']
    backbone_num_classes = len(backbone_behaviors)
    
    backbone_model = SimplifiedTDMModel(input_dim, backbone_num_classes, device)
    backbone_model.to(device)
    
    # Analyze backbone model
    input_shape = (1, 1, input_dim)  # 1D input for TinyML
    tinyml_tracker.analyze_model(backbone_model, input_shape=input_shape, device=device)
    
    # Prepare backbone training data
    backbone_dogs = list(dog_data.keys())[:2]  # Use first 2 dogs for backbone
    print(f"Backbone training dogs: {backbone_dogs}")
    
    backbone_X = []
    backbone_y = []
    
    for dog_id in backbone_dogs:
        dog_X = dog_data[dog_id]['X']
        dog_y = dog_data[dog_id]['y']
        
        # Only include backbone behaviors
        backbone_mask = torch.tensor([behaviors[y] in backbone_behaviors for y in dog_y])
        if backbone_mask.sum() > 0:
            backbone_X.append(dog_X[backbone_mask])
            # Remap labels to backbone classes
            remapped_y = torch.tensor([backbone_behaviors.index(behaviors[y]) for y in dog_y[backbone_mask]])
            backbone_y.append(remapped_y)
    
    if len(backbone_X) == 0:
        print("No backbone training data found!")
        return
    
    backbone_X = torch.cat(backbone_X, dim=0)
    backbone_y = torch.cat(backbone_y, dim=0)
    
    print(f"Backbone training data: {len(backbone_X):,} windows")
    
    # Train backbone
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
            
            # Track TinyML metrics every 100 batches
            if batch_idx % 100 == 0:
                tinyml_tracker.track_training_metrics(
                    task_id=0, epoch=epoch, step=batch_idx, 
                    model=backbone_model, input_shape=input_shape, device=device
                )
        
        if epoch % 10 == 0:
            acc = 100 * correct / total
            print(f"[Backbone] epoch {epoch} acc={acc:.3f}")
    
    print("Backbone training complete!")
    
    # Freeze backbone
    for param in backbone_model.backbone.parameters():
        param.requires_grad = False
    
    # Create CIL model with expandable classifier
    cil_model = SimplifiedTDMModel(input_dim, 7, device)  # All 7 behaviors
    cil_model.to(device)
    
    # Copy backbone weights
    cil_model.backbone.load_state_dict(backbone_model.backbone.state_dict())
    
    # Analyze CIL model
    tinyml_tracker.analyze_model(cil_model, input_shape=input_shape, device=device)
    
    # CIL training setup
    target_dog = 47  # Target dog for personalization
    if target_dog not in dog_data:
        print(f"Target dog {target_dog} not found!")
        return
    
    target_X = dog_data[target_dog]['X']
    target_y = dog_data[target_dog]['y']
    
    print(f"Target dog {target_dog}: {len(target_X):,} samples")
    
    # Split target dog data using stratified sampling (not temporal!)
    from sklearn.model_selection import train_test_split
    
    # Convert to numpy for sklearn
    X_np = target_X.numpy()
    y_np = target_y.numpy()
    
    # Stratified split to ensure all behaviors in both train and test
    X_stream, X_test, y_stream, y_test = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
    )
    
    # Convert back to tensors
    X_stream = torch.from_numpy(X_stream)
    y_stream = torch.from_numpy(y_stream)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)
    
    print(f"Stream data: {len(X_stream):,} samples, Test data: {len(X_test):,} samples")
    
    # CIL tasks: introduce new behaviors incrementally
    cil_behaviors = ['Sniffing', 'Trotting', 'Galloping']
    num_tasks = len(cil_behaviors)
    
    # Replay buffer for latent replay
    replay_buffer_size = 1000
    replay_buffer = []
    replay_labels = []
    
    # Training history
    task_accuracies = []
    overall_accuracies = []
    
    print(f"\n=== Starting CIL with {num_tasks} tasks ===")
    
    for task_idx in range(num_tasks):
        new_behavior = cil_behaviors[task_idx]
        print(f"\n--- Task {task_idx + 1}: Introducing '{new_behavior}' ---")
        
        # Get data for this task (new behavior + all previous behaviors)
        task_behaviors = backbone_behaviors + cil_behaviors[:task_idx + 1]
        task_mask = torch.tensor([behaviors[y] in task_behaviors for y in y_stream])
        
        if task_mask.sum() == 0:
            print(f"No data for task {task_idx + 1}")
            continue
        
        X_task = X_stream[task_mask]
        y_task = y_stream[task_mask]
        
        print(f"Task {task_idx + 1} data: {len(X_task):,} samples")
        
        # Create task loader
        task_loader = DataLoader(TensorDataset(X_task, y_task), 
                                batch_size=32, shuffle=True)
        
        # Train on this task
        cil_optimizer = torch.optim.Adam(cil_model.parameters(), lr=0.001)
        cil_criterion = nn.CrossEntropyLoss()
        
        cil_model.train()
        for epoch in range(20):  # More epochs per task
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (X_batch, y_batch) in enumerate(task_loader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Latent replay
                if len(replay_buffer) > 0 and task_idx > 0:
                    # Sample from replay buffer
                    replay_size = min(16, len(replay_buffer))
                    replay_indices = random.sample(range(len(replay_buffer)), replay_size)
                    replay_features = torch.stack([replay_buffer[i] for i in replay_indices])
                    replay_labels_batch = torch.tensor([replay_labels[i] for i in replay_indices])
                    
                    # Process current batch through backbone first
                    current_features = cil_model.backbone(X_batch)
                    
                    # Combine features (both are now 64-dimensional)
                    X_combined = torch.cat([current_features, replay_features], dim=0)
                    y_combined = torch.cat([y_batch, replay_labels_batch], dim=0)
                    
                    # Use classifier directly on combined features
                    outputs = cil_model.classifier(X_combined)
                else:
                    # Normal forward pass
                    outputs = cil_model(X_batch)
                    y_combined = y_batch
                
                cil_optimizer.zero_grad()
                loss = cil_criterion(outputs, y_combined)
                loss.backward()
                cil_optimizer.step()
                
                # Apply TDM mask
                cil_model.apply_mask()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_combined.size(0)
                correct += (predicted == y_combined).sum().item()
            
            if epoch % 5 == 0:
                acc = 100 * correct / total
                print(f"[Task {task_idx + 1}] epoch {epoch} acc={acc:.3f}")
                
                # Track TinyML metrics
                tinyml_tracker.track_training_metrics(
                    task_id=task_idx+1, epoch=epoch, step=epoch, 
                    model=cil_model, input_shape=input_shape, device=device
                )
        
        # Update replay buffer with features from this task
        with torch.no_grad():
            features = cil_model.backbone(X_task)
            for i in range(len(features)):
                if len(replay_buffer) < replay_buffer_size:
                    replay_buffer.append(features[i].cpu())
                    replay_labels.append(y_task[i].item())
                else:
                    # Reservoir sampling
                    j = random.randint(0, len(replay_buffer) - 1)
                    replay_buffer[j] = features[i].cpu()
                    replay_labels[j] = y_task[i].item()
        
        # TDM inter-task adjustment
        if task_idx < num_tasks - 1:
            cil_model.update_mask_inter_task()
        
        # Evaluate on test set
        cil_model.eval()
        with torch.no_grad():
            test_loader = DataLoader(TensorDataset(X_test, y_test), 
                                    batch_size=32, shuffle=False)
            
            correct = 0
            total = 0
            all_predictions = []
            all_labels = []
            
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = cil_model.forward_eval(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
            
            test_acc = 100 * correct / total
            task_accuracies.append(test_acc)
            
            print(f"Task {task_idx + 1} test accuracy: {test_acc:.3f}%")
            
            # Per-class accuracy
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            
            print("Per-class accuracy:")
            for i, behavior in enumerate(behaviors):
                mask = all_labels == i
                if mask.sum() > 0:
                    class_acc = 100 * (all_predictions[mask] == all_labels[mask]).sum() / mask.sum()
                    print(f"  {behavior}: {class_acc:.1f}% ({mask.sum()} samples)")
    
    # Final evaluation on all seen data
    print(f"\n=== Final Evaluation ===")
    
    # Evaluate on all data (backbone + CIL)
    all_X = torch.cat([backbone_X, X_test], dim=0)
    all_y = torch.cat([backbone_y, y_test], dim=0)
    
    cil_model.eval()
    with torch.no_grad():
        all_loader = DataLoader(TensorDataset(all_X, all_y), 
                               batch_size=32, shuffle=False)
        
        correct = 0
        total = 0
        for X_batch, y_batch in all_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = cil_model.forward_eval(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        
        final_acc = 100 * correct / total
        print(f"Final overall accuracy: {final_acc:.3f}%")
    
    # Save results
    results = {
        'task_accuracies': task_accuracies,
        'final_accuracy': final_acc,
        'total_samples': sum(len(dog_data[dog_id]['X']) for dog_id in dog_data)
    }
    
    # Save TinyML metrics
    tinyml_tracker.save_metrics()
    tinyml_tracker.plot_metrics()
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Task accuracy plot
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(task_accuracies) + 1), task_accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Task')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Task Accuracy Progression')
    plt.grid(True, alpha=0.3)
    
    # Overall accuracy plot
    plt.subplot(2, 2, 2)
    overall_accs = [task_accuracies[0]] + task_accuracies
    plt.plot(range(len(overall_accs)), overall_accs, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Task')
    plt.ylabel('Overall Accuracy (%)')
    plt.title('Overall Accuracy on All Classes')
    plt.grid(True, alpha=0.3)
    
    # Behavior distribution
    plt.subplot(2, 2, 3)
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
    plt.savefig('intelligent_tdm_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nResults saved to intelligent_tdm_results.png")
    print(f"TinyML metrics saved to logs/tdm_intelligent_metrics.csv")

if __name__ == "__main__":
    main()
