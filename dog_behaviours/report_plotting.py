import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-deep')

# Data from the intelligent_sampling_summary.txt
behaviors = {
    'Lying chest': {'samples': 11349, 'percentage': 18.5},
    'Sitting': {'samples': 11266, 'percentage': 18.4},
    'Standing': {'samples': 10424, 'percentage': 17.0},
    'Sniffing': {'samples': 10393, 'percentage': 17.0},
    'Walking': {'samples': 7991, 'percentage': 13.1},
    'Trotting': {'samples': 7895, 'percentage': 12.9},
    'Galloping': {'samples': 1885, 'percentage': 3.1}
}

# Create DataFrame
df = pd.DataFrame([
    {'Behavior': behavior, 'Samples': data['samples'], 'Percentage': data['percentage']}
    for behavior, data in behaviors.items()
])

# Sort by percentage for better visualization
df = df.sort_values('Percentage', ascending=False)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Percentage distribution (pie chart)
colors = sns.color_palette("deep", len(df))
wedges, texts, autotexts = ax1.pie(df['Percentage'], labels=df['Behavior'], autopct='%1.1f%%', 
                                   colors=colors, startangle=90)
ax1.set_title('Dog Behavior Distribution (Percentage)', fontsize=14, fontweight='bold', pad=20)

# Plot 2: Sample count distribution (bar chart)
bars = ax2.bar(df['Behavior'], df['Samples'], color=colors, alpha=0.8)
ax2.set_title('Dog Behavior Distribution (Sample Count)', fontsize=14, fontweight='bold', pad=20)
ax2.set_xlabel('Behavior', fontsize=12)
ax2.set_ylabel('Number of Samples', fontsize=12)
ax2.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, sample_count in zip(bars, df['Samples']):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 100,
             f'{sample_count:,}', ha='center', va='bottom', fontweight='bold')

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('plots_and_metrics/behavior_distribution.png', dpi=300, bbox_inches='tight')
print("Plot saved to 'plots_and_metrics/behavior_distribution.png'")

# plt.show()




##### BACKBONE + CIL TRAINING 


class CILTrainingPlotter:
    """Class to handle plotting for CIL training experiments"""
    
    def __init__(self, save_dir='plots_and_metrics'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set seaborn style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("deep")
        
        # Storage for metrics
        self.backbone_metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epochs': []
        }
        
        self.cil_metrics = {
            'task_accuracies': [],  # List of dicts: [{class: acc}, {class: acc}, ...]
            'overall_accuracies': [],  # List of overall accuracies per task
            'task_names': []  # List of task names
        }
    
    def add_backbone_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                          train_acc: float, val_acc: float):
        """Add backbone training metrics for one epoch"""
        self.backbone_metrics['epochs'].append(epoch)
        self.backbone_metrics['train_loss'].append(train_loss)
        self.backbone_metrics['val_loss'].append(val_loss)
        self.backbone_metrics['train_acc'].append(train_acc)
        self.backbone_metrics['val_acc'].append(val_acc)
    
    def add_cil_task(self, task_name: str, class_accuracies: Dict[str, float], 
                     overall_accuracy: float):
        """Add CIL task results"""
        self.cil_metrics['task_names'].append(task_name)
        self.cil_metrics['task_accuracies'].append(class_accuracies.copy())
        self.cil_metrics['overall_accuracies'].append(overall_accuracy)
    
    def plot_backbone_training(self, save_plot=True):
        """Plot backbone training curves"""
        if not self.backbone_metrics['epochs']:
            print("No backbone metrics to plot!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        epochs = self.backbone_metrics['epochs']
        
        # Plot 1: Loss curves
        ax1.plot(epochs, self.backbone_metrics['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.backbone_metrics['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Backbone Training Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        ax2.plot(epochs, self.backbone_metrics['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.backbone_metrics['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Backbone Training Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(f'{self.save_dir}/backbone_training.png', dpi=300, bbox_inches='tight')
            print(f"Backbone training plot saved to {self.save_dir}/backbone_training.png")
        
        plt.show()
    
    def plot_cil_progression(self, save_plot=True):
        """Plot CIL task progression"""
        if not self.cil_metrics['task_names']:
            print("No CIL metrics to plot!")
            return
        
        # Create DataFrame for easier plotting
        all_classes = set()
        for task_acc in self.cil_metrics['task_accuracies']:
            all_classes.update(task_acc.keys())
        
        # Create matrix for heatmap
        class_list = sorted(list(all_classes))
        task_names = self.cil_metrics['task_names']
        
        # Initialize matrix with NaN
        acc_matrix = np.full((len(class_list), len(task_names)), np.nan)
        
        # Fill in the accuracies
        for task_idx, task_acc in enumerate(self.cil_metrics['task_accuracies']):
            for class_idx, class_name in enumerate(class_list):
                if class_name in task_acc:
                    acc_matrix[class_idx, task_idx] = task_acc[class_name]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Overall accuracy progression
        ax1.plot(range(1, len(task_names) + 1), self.cil_metrics['overall_accuracies'], 
                'o-', linewidth=2, markersize=8, color='blue')
        ax1.set_xlabel('Task', fontsize=12)
        ax1.set_ylabel('Overall Accuracy (%)', fontsize=12)
        ax1.set_title('CIL Overall Accuracy Progression', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(1, len(task_names) + 1))
        ax1.set_xticklabels([f'Task {i+1}' for i in range(len(task_names))])
        
        # Add value labels on points
        for i, acc in enumerate(self.cil_metrics['overall_accuracies']):
            ax1.annotate(f'{acc:.1f}%', (i+1, acc), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')
        
        # Plot 2: Per-class accuracy heatmap
        im = ax2.imshow(acc_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax2.set_xlabel('Task', fontsize=12)
        ax2.set_ylabel('Class', fontsize=12)
        ax2.set_title('CIL Per-Class Accuracy Heatmap', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(task_names)))
        ax2.set_xticklabels([f'Task {i+1}' for i in range(len(task_names))])
        ax2.set_yticks(range(len(class_list)))
        ax2.set_yticklabels(class_list)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Accuracy (%)', fontsize=12)
        
        # Add text annotations on heatmap
        for i in range(len(class_list)):
            for j in range(len(task_names)):
                if not np.isnan(acc_matrix[i, j]):
                    ax2.text(j, i, f'{acc_matrix[i, j]:.0f}%', 
                            ha='center', va='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(f'{self.save_dir}/cil_progression.png', dpi=300, bbox_inches='tight')
            print(f"CIL progression plot saved to {self.save_dir}/cil_progression.png")
        
        plt.show()
    
    def plot_final_comparison(self, save_plot=True):
        """Plot final comparison of all classes"""
        if not self.cil_metrics['task_accuracies']:
            print("No CIL metrics to plot!")
            return
        
        # Get final task results
        final_accuracies = self.cil_metrics['task_accuracies'][-1]
        
        # Create bar plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        classes = list(final_accuracies.keys())
        accuracies = list(final_accuracies.values())
        
        # Sort by accuracy for better visualization
        sorted_pairs = sorted(zip(classes, accuracies), key=lambda x: x[1], reverse=True)
        classes, accuracies = zip(*sorted_pairs)
        
        bars = ax.bar(classes, accuracies, color=sns.color_palette("deep", len(classes)), alpha=0.8)
        ax.set_xlabel('Behavior Class', fontsize=12)
        ax.set_ylabel('Final Accuracy (%)', fontsize=12)
        ax.set_title('Final CIL Performance by Class', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(f'{self.save_dir}/final_class_performance.png', dpi=300, bbox_inches='tight')
            print(f"Final class performance plot saved to {self.save_dir}/final_class_performance.png")
        
        plt.show()
    
    def save_metrics(self, filename='cil_training_metrics.json'):
        """Save all metrics to JSON file"""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        metrics_to_save = {
            'backbone_metrics': {k: v for k, v in self.backbone_metrics.items()},
            'cil_metrics': {
                'task_names': self.cil_metrics['task_names'],
                'overall_accuracies': self.cil_metrics['overall_accuracies'],
                'task_accuracies': self.cil_metrics['task_accuracies']
            }
        }
        
        with open(f'{self.save_dir}/{filename}', 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        print(f"Metrics saved to {self.save_dir}/{filename}")
    
    def create_summary_report(self):
        """Create a comprehensive summary report"""
        if not self.cil_metrics['task_names']:
            print("No metrics to report!")
            return
        
        print("\n" + "="*60)
        print("CIL TRAINING SUMMARY REPORT")
        print("="*60)
        
        # Backbone summary
        if self.backbone_metrics['epochs']:
            print(f"\nBACKBONE TRAINING:")
            print(f"  Total epochs: {len(self.backbone_metrics['epochs'])}")
            print(f"  Final training accuracy: {self.backbone_metrics['train_acc'][-1]:.2f}%")
            print(f"  Best validation accuracy: {max(self.backbone_metrics['val_acc']):.2f}%")
        
        # CIL summary
        print(f"\nCIL TRAINING:")
        print(f"  Total tasks: {len(self.cil_metrics['task_names'])}")
        print(f"  Final overall accuracy: {self.cil_metrics['overall_accuracies'][-1]:.2f}%")
        
        print(f"\nTASK PROGRESSION:")
        for i, (task_name, overall_acc) in enumerate(zip(self.cil_metrics['task_names'], 
                                                        self.cil_metrics['overall_accuracies'])):
            print(f"  Task {i+1} ({task_name}): {overall_acc:.2f}%")
        
        # Final per-class performance
        if self.cil_metrics['task_accuracies']:
            final_accs = self.cil_metrics['task_accuracies'][-1]
            print(f"\nFINAL PER-CLASS PERFORMANCE:")
            for class_name, acc in sorted(final_accs.items(), key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {acc:.2f}%")
        
        print("="*60)

