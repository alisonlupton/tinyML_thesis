import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import pandas as pd
import seaborn as sns

class BatchDebugger:
    def __init__(self, save_path="logs/batch_analysis.csv"):
        self.save_path = save_path
        self.batch_stats = []
        self.class_distributions = []
        
    def analyze_batch(self, x, y, task_id, step, epoch, batch_type="combined"):
        """Analyze a single batch and record statistics"""
        batch_size = x.size(0)
        unique_classes = torch.unique(y).tolist()
        class_counts = Counter(y.tolist())
        
        # Calculate batch statistics
        batch_mean = x.mean().item()
        batch_std = x.std().item()
        batch_min = x.min().item()
        batch_max = x.max().item()
        
        # Class distribution analysis
        class_entropy = self._calculate_class_entropy(class_counts, batch_size)
        class_balance = self._calculate_class_balance(class_counts, batch_size)
        
        # Record statistics
        stats = {
            'task_id': task_id,
            'epoch': epoch,
            'step': step,
            'batch_type': batch_type,
            'batch_size': batch_size,
            'num_classes': len(unique_classes),
            'classes': str(unique_classes),
            'class_counts': str(dict(class_counts)),
            'batch_mean': batch_mean,
            'batch_std': batch_std,
            'batch_min': batch_min,
            'batch_max': batch_max,
            'class_entropy': class_entropy,
            'class_balance': class_balance,
            'is_single_class': len(unique_classes) == 1,
            'is_balanced': class_balance > 0.8,  # 80% threshold for "balanced"
        }
        
        self.batch_stats.append(stats)
        
        # Print summary for first few batches
        if step < 5:
            print(f"[BATCH DEBUG] {batch_type.upper()} - Task {task_id}, Step {step}")
            print(f"  Size: {batch_size}, Classes: {unique_classes}")
            print(f"  Class counts: {dict(class_counts)}")
            print(f"  Mean: {batch_mean:.4f}, Std: {batch_std:.4f}")
            print(f"  Single class: {stats['is_single_class']}, Balanced: {stats['is_balanced']}")
            print()
    
    def analyze_bn_layers(self, model, x, task_id, step, epoch):
        """Analyze batch normalization layer statistics"""
        bn_stats = []
        
        def hook_fn(module, input, output):
            if isinstance(module, nn.BatchNorm2d):
                # Get running statistics
                running_mean = module.running_mean.clone()
                running_var = module.running_var.clone()
                
                # Calculate current batch statistics
                if input[0].dim() == 4:  # Conv2d input
                    current_mean = input[0].mean(dim=[0, 2, 3])
                    current_var = input[0].var(dim=[0, 2, 3], unbiased=False)
                else:
                    current_mean = input[0].mean(dim=0)
                    current_var = input[0].var(dim=0, unbiased=False)
                
                bn_stats.append({
                    'layer_name': str(module),
                    'running_mean_mean': running_mean.mean().item(),
                    'running_mean_std': running_mean.std().item(),
                    'running_var_mean': running_var.mean().item(),
                    'running_var_std': running_var.std().item(),
                    'current_mean_mean': current_mean.mean().item(),
                    'current_mean_std': current_mean.std().item(),
                    'current_var_mean': current_var.mean().item(),
                    'current_var_std': current_var.std().item(),
                    'mean_diff': (running_mean.mean() - current_mean.mean()).item(),
                    'var_diff': (running_var.mean() - current_var.mean()).item(),
                })
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass
        with torch.no_grad():
            _ = model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Add task/step info
        for stat in bn_stats:
            stat.update({
                'task_id': task_id,
                'epoch': epoch,
                'step': step
            })
        
        return bn_stats
    
    def _calculate_class_entropy(self, class_counts, batch_size):
        """Calculate entropy of class distribution"""
        if len(class_counts) == 1:
            return 0.0
        probs = [count / batch_size for count in class_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs)
        return entropy
    
    def _calculate_class_balance(self, class_counts, batch_size):
        """Calculate balance metric (0=unbalanced, 1=perfectly balanced)"""
        if len(class_counts) == 1:
            return 0.0
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        return min_count / max_count
    
    def save_stats(self):
        """Save all collected statistics to CSV"""
        if self.batch_stats:
            df = pd.DataFrame(self.batch_stats)
            df.to_csv(self.save_path, index=False)
            print(f"Batch statistics saved to {self.save_path}")
            
            # Print summary
            print("\n=== BATCH ANALYSIS SUMMARY ===")
            print(f"Total batches analyzed: {len(df)}")
            print(f"Single-class batches: {df['is_single_class'].sum()} ({df['is_single_class'].mean()*100:.1f}%)")
            print(f"Balanced batches: {df['is_balanced'].sum()} ({df['is_balanced'].mean()*100:.1f}%)")
            print(f"Average batch size: {df['batch_size'].mean():.1f}")
            print(f"Average classes per batch: {df['num_classes'].mean():.1f}")
            print(f"Average class entropy: {df['class_entropy'].mean():.3f}")
            
            return df
        return None
    
    def plot_batch_analysis(self, df=None):
        """Create visualizations of batch statistics"""
        if df is None:
            df = pd.read_csv(self.save_path)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Batch size distribution
        axes[0, 0].hist(df['batch_size'], bins=20, alpha=0.7)
        axes[0, 0].set_title('Batch Size Distribution')
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('Frequency')
        
        # Classes per batch
        axes[0, 1].hist(df['num_classes'], bins=range(1, df['num_classes'].max()+2), alpha=0.7)
        axes[0, 1].set_title('Classes per Batch')
        axes[0, 1].set_xlabel('Number of Classes')
        axes[0, 1].set_ylabel('Frequency')
        
        # Class entropy over time
        axes[0, 2].scatter(df['step'], df['class_entropy'], alpha=0.6)
        axes[0, 2].set_title('Class Entropy vs Step')
        axes[0, 2].set_xlabel('Training Step')
        axes[0, 2].set_ylabel('Class Entropy')
        
        # Batch statistics over time
        axes[1, 0].plot(df['step'], df['batch_mean'], label='Mean', alpha=0.7)
        axes[1, 0].plot(df['step'], df['batch_std'], label='Std', alpha=0.7)
        axes[1, 0].set_title('Batch Statistics Over Time')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        
        # Single class vs multi-class
        single_class = df[df['is_single_class']]
        multi_class = df[~df['is_single_class']]
        axes[1, 1].scatter(single_class['step'], single_class['batch_std'], 
                          label='Single Class', alpha=0.7, color='red')
        axes[1, 1].scatter(multi_class['step'], multi_class['batch_std'], 
                          label='Multi Class', alpha=0.7, color='blue')
        axes[1, 1].set_title('Batch Std: Single vs Multi-Class')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Batch Standard Deviation')
        axes[1, 1].legend()
        
        # Class balance distribution
        axes[1, 2].hist(df['class_balance'], bins=20, alpha=0.7)
        axes[1, 2].set_title('Class Balance Distribution')
        axes[1, 2].set_xlabel('Class Balance (0=unbalanced, 1=balanced)')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('logs/batch_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

# Usage example:
# debugger = BatchDebugger()
# debugger.analyze_batch(x_live, y_live, task_id, step, epoch, "live")
# debugger.analyze_batch(x_feats, y_all, task_id, step, epoch, "combined")
# debugger.save_stats()
# debugger.plot_batch_analysis() 