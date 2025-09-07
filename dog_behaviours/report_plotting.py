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
    'Lying chest': {'samples': 22529, 'percentage': 18.8},
    'Sitting': {'samples': 22306, 'percentage': 18.6},
    'Standing': {'samples': 20374, 'percentage': 17.0},
    'Sniffing': {'samples': 20359, 'percentage': 17.0},
    'Walking': {'samples': 15473, 'percentage': 12.9},
    'Trotting': {'samples': 15298, 'percentage': 12.8},
    'Galloping': {'samples': 3324, 'percentage': 2.8}
}



# Create DataFrame
df = pd.DataFrame([
    {'Behavior': behavior, 'Samples': data['samples'], 'Percentage': data['percentage']}
    for behavior, data in behaviors.items()
])

# Sort by percentage for better visualization
df = df.sort_values('Percentage', ascending=False)

# Create figure with single subplot for pie chart
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot: Percentage distribution (pie chart)
colors = sns.color_palette("deep", len(df))
wedges, texts, autotexts = ax.pie(df['Percentage'], labels=df['Behavior'], autopct='%1.1f%%', 
                                   colors=colors, startangle=90)
ax.set_title('Dog Behavior Distribution (Percentage)', fontsize=16, fontweight='bold', pad=20)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('plots_and_metrics/behavior_distribution.png', dpi=300, bbox_inches='tight')
print("Plot saved to 'plots_and_metrics/behavior_distribution.png'")

# plt.show()




##### BACKBONE + CIL TRAINING 


class CILTrainingPlotter:
    """Class to handle plotting for CIL training experiments"""
    
    def __init__(self, save_dir='plots_and_metrics', seed=None):
        self.save_dir = save_dir
        self.seed = seed
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
        
        # New storage for CIL training metrics
        self.cil_training_metrics = {
            'task_training_losses': [],  # List of lists: [[epoch1_loss, epoch2_loss, ...], ...]
            'task_training_accuracies': [],  # List of lists: [[epoch1_acc, epoch2_acc, ...], ...]
            'task_validation_losses': [],  # List of lists: [[epoch1_val_loss, epoch2_val_loss, ...], ...]
            'task_validation_accuracies': [],  # List of lists: [[epoch1_val_acc, epoch2_val_acc, ...], ...]
            'task_epochs': [],  # List of lists: [[1, 2, 3, ...], ...] for each task
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
    
    def add_cil_task_training_epoch(self, task_name: str, epoch: int, train_loss: float, 
                                   train_acc: float, val_loss: float = None, val_acc: float = None):
        """Add CIL training metrics for one epoch of a specific task"""
        # Find or create task index
        if task_name not in self.cil_training_metrics['task_names']:
            self.cil_training_metrics['task_names'].append(task_name)
            self.cil_training_metrics['task_training_losses'].append([])
            self.cil_training_metrics['task_training_accuracies'].append([])
            self.cil_training_metrics['task_validation_losses'].append([])
            self.cil_training_metrics['task_validation_accuracies'].append([])
            self.cil_training_metrics['task_epochs'].append([])
        
        task_idx = self.cil_training_metrics['task_names'].index(task_name)
        
        # Add metrics
        self.cil_training_metrics['task_epochs'][task_idx].append(epoch)
        self.cil_training_metrics['task_training_losses'][task_idx].append(train_loss)
        self.cil_training_metrics['task_training_accuracies'][task_idx].append(train_acc)
        
        if val_loss is not None:
            self.cil_training_metrics['task_validation_losses'][task_idx].append(val_loss)
        if val_acc is not None:
            self.cil_training_metrics['task_validation_accuracies'][task_idx].append(val_acc)

    def plot_backbone_training(self, save_plot=True):
        """Plot backbone training curves"""
        if not self.backbone_metrics['epochs']:
            print("No backbone metrics to plot!")
            return
        
        # Set seaborn style to match behavior distribution plots
        plt.style.use('seaborn-v0_8-deep')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        epochs = self.backbone_metrics['epochs']
        
        # Plot 1: Training Loss
        ax1.plot(epochs, self.backbone_metrics['train_loss'], 'o-', linewidth=2.5, markersize=8, color='blue', label='Training Loss')
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Backbone Training Loss', fontsize=16, fontweight='bold', pad=20)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Validation Accuracy
        ax2.plot(epochs, self.backbone_metrics['val_acc'], 'o-', linewidth=2.5, markersize=8, color='red', label='Validation Accuracy')
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Backbone Validation Accuracy', fontsize=16, fontweight='bold', pad=20)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(f'{self.save_dir}/backbone_training.png', dpi=300, bbox_inches='tight')
            print(f"Backbone training plot saved to {self.save_dir}/backbone_training.png")
            
            # Save backbone training data to CSV
            self.save_backbone_training_to_csv()
        
        # plt.show()
    
    def plot_cil_progression(self, save_plot=False):
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
        
        # plt.show()
    
    def plot_final_comparison(self, save_plot=False):
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
        
        # plt.show()
    
    def plot_cil_training_curves(self, save_plot=True):
        """Plot CIL training curves (loss and accuracy) for all tasks"""
        if not self.cil_training_metrics['task_names']:
            print("No CIL training metrics to plot!")
            return
        
        num_tasks = len(self.cil_training_metrics['task_names'])
        fig, axes = plt.subplots(2, num_tasks, figsize=(5*num_tasks, 10))
        
        # If only one task, make axes 2D
        if num_tasks == 1:
            axes = axes.reshape(2, 1)
        
        colors = sns.color_palette("deep", num_tasks)
        
        for task_idx, task_name in enumerate(self.cil_training_metrics['task_names']):
            epochs = self.cil_training_metrics['task_epochs'][task_idx]
            train_losses = self.cil_training_metrics['task_training_losses'][task_idx]
            train_accs = self.cil_training_metrics['task_training_accuracies'][task_idx]
            val_losses = self.cil_training_metrics['task_validation_losses'][task_idx]
            val_accs = self.cil_training_metrics['task_validation_accuracies'][task_idx]
            
            # Plot training loss
            ax_loss = axes[0, task_idx]
            ax_loss.plot(epochs, train_losses, '-', label='Training Loss', linewidth=2, color=colors[task_idx])
            if val_losses:
                ax_loss.plot(epochs, val_losses, '--', label='Validation Loss', linewidth=2, color=colors[task_idx], alpha=0.7)
            ax_loss.set_xlabel('Epoch', fontsize=10)
            ax_loss.set_ylabel('Loss', fontsize=10)
            ax_loss.set_title(f'{task_name} - Loss', fontsize=12, fontweight='bold')
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)
            
            # Plot training accuracy
            ax_acc = axes[1, task_idx]
            ax_acc.plot(epochs, train_accs, '-', label='Training Accuracy', linewidth=2, color=colors[task_idx])
            if val_accs:
                ax_acc.plot(epochs, val_accs, '--', label='Validation Accuracy', linewidth=2, color=colors[task_idx], alpha=0.7)
            ax_acc.set_xlabel('Epoch', fontsize=10)
            ax_acc.set_ylabel('Accuracy (%)', fontsize=10)
            ax_acc.set_title(f'{task_name} - Accuracy', fontsize=12, fontweight='bold')
            ax_acc.legend()
            ax_acc.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(f'{self.save_dir}/cil_training_curves.png', dpi=300, bbox_inches='tight')
            print(f"CIL training curves plot saved to {self.save_dir}/cil_training_curves.png")
            
            # Save training curves data to CSV
            self.save_training_curves_to_csv()
        
        # plt.show()
    
    def save_training_curves_to_csv(self):
        """Save CIL training curves data to CSV for multi-seed analysis"""
        if not self.cil_training_metrics['task_names']:
            print("No CIL training metrics to save!")
            return
        
        # Get seed from config (you'll need to pass this to the plotter)
        seed = getattr(self, 'seed', 'unknown')
        
        # Prepare data for CSV
        csv_data = []
        
        for task_idx, task_name in enumerate(self.cil_training_metrics['task_names']):
            epochs = self.cil_training_metrics['task_epochs'][task_idx]
            train_losses = self.cil_training_metrics['task_training_losses'][task_idx]
            train_accs = self.cil_training_metrics['task_training_accuracies'][task_idx]
            val_losses = self.cil_training_metrics['task_validation_losses'][task_idx]
            val_accs = self.cil_training_metrics['task_validation_accuracies'][task_idx]
            
            for epoch_idx, epoch in enumerate(epochs):
                # Training data
                csv_data.append({
                    'seed': seed,
                    'task_name': task_name,
                    'task_idx': task_idx,
                    'epoch': epoch,
                    'metric_type': 'training_loss',
                    'value': train_losses[epoch_idx] if epoch_idx < len(train_losses) else None
                })
                
                csv_data.append({
                    'seed': seed,
                    'task_name': task_name,
                    'task_idx': task_idx,
                    'epoch': epoch,
                    'metric_type': 'training_accuracy',
                    'value': train_accs[epoch_idx] if epoch_idx < len(train_accs) else None
                })
                
                # Validation data (if available)
                if val_losses and epoch_idx < len(val_losses):
                    csv_data.append({
                        'seed': seed,
                        'task_name': task_name,
                        'task_idx': task_idx,
                        'epoch': epoch,
                        'metric_type': 'validation_loss',
                        'value': val_losses[epoch_idx]
                    })
                
                if val_accs and epoch_idx < len(val_accs):
                    csv_data.append({
                        'seed': seed,
                        'task_name': task_name,
                        'task_idx': task_idx,
                        'epoch': epoch,
                        'metric_type': 'validation_accuracy',
                        'value': val_accs[epoch_idx]
                    })
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        
        # Append to existing file or create new one
        csv_file = os.path.join(self.save_dir, 'cil_training_curves_data.csv')
        if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
            try:
                existing_df = pd.read_csv(csv_file)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                # File exists but is empty or corrupted, start fresh
                combined_df = df
        else:
            combined_df = df
        
        combined_df.to_csv(csv_file, index=False)
        print(f"CIL training curves data saved to {csv_file}")
    
    def save_backbone_training_to_csv(self):
        """Save backbone training data to CSV for multi-seed analysis"""
        if not self.backbone_metrics['epochs']:
            print("No backbone training metrics to save!")
            return
        
        # Get seed
        seed = getattr(self, 'seed', 'unknown')
        
        # Prepare data for CSV
        csv_data = []
        
        epochs = self.backbone_metrics['epochs']
        train_losses = self.backbone_metrics['train_loss']
        val_losses = self.backbone_metrics['val_loss']
        train_accs = self.backbone_metrics['train_acc']
        val_accs = self.backbone_metrics['val_acc']
        
        for epoch_idx, epoch in enumerate(epochs):
            # Training loss
            csv_data.append({
                'seed': seed,
                'stage': 'Backbone',
                'epoch': epoch,
                'metric_type': 'training_loss',
                'value': train_losses[epoch_idx] if epoch_idx < len(train_losses) else None
            })
            
            # Training accuracy
            csv_data.append({
                'seed': seed,
                'stage': 'Backbone',
                'epoch': epoch,
                'metric_type': 'training_accuracy',
                'value': train_accs[epoch_idx] if epoch_idx < len(train_accs) else None
            })
            
            # Validation loss
            csv_data.append({
                'seed': seed,
                'stage': 'Backbone',
                'epoch': epoch,
                'metric_type': 'validation_loss',
                'value': val_losses[epoch_idx] if epoch_idx < len(val_losses) else None
            })
            
            # Validation accuracy
            csv_data.append({
                'seed': seed,
                'stage': 'Backbone',
                'epoch': epoch,
                'metric_type': 'validation_accuracy',
                'value': val_accs[epoch_idx] if epoch_idx < len(val_accs) else None
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        
        # Append to existing file or create new one
        csv_file = os.path.join(self.save_dir, 'backbone_training_data.csv')
        if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
            try:
                existing_df = pd.read_csv(csv_file)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                # File exists but is empty or corrupted, start fresh
                combined_df = df
        else:
            combined_df = df
        
        combined_df.to_csv(csv_file, index=False)
        print(f"Backbone training data saved to {csv_file}")
    
    def plot_cil_training_summary(self, save_plot=True):
        """Plot a summary of CIL training metrics across all tasks"""
        if not self.cil_training_metrics['task_names']:
            print("No CIL training metrics to plot!")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # # Plot 1: Final training loss per task
        # final_train_losses = []
        # task_labels = []
        # for task_idx, task_name in enumerate(self.cil_training_metrics['task_names']):
        #     if self.cil_training_metrics['task_training_losses'][task_idx]:
        #         final_train_losses.append(self.cil_training_metrics['task_training_losses'][task_idx][-1])
        #         task_labels.append(f'Task {task_idx+1}')
        
        # bars1 = ax1.bar(task_labels, final_train_losses, color=sns.color_palette("deep", len(task_labels)), alpha=0.8)
        # ax1.set_title('Final Training Loss per Task', fontsize=14, fontweight='bold')
        # ax1.set_ylabel('Loss', fontsize=12)
        # ax1.grid(True, alpha=0.3, axis='y')
        
        # # Add value labels on bars
        # for bar, loss in zip(bars1, final_train_losses):
        #     height = bar.get_height()
        #     ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
        #             f'{loss:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # # Plot 2: Final training accuracy per task
        # final_train_accs = []
        # for task_idx in range(len(self.cil_training_metrics['task_names'])):
        #     if self.cil_training_metrics['task_training_accuracies'][task_idx]:
        #         final_train_accs.append(self.cil_training_metrics['task_training_accuracies'][task_idx][-1])
        
        # bars2 = ax2.bar(task_labels, final_train_accs, color=sns.color_palette("deep", len(task_labels)), alpha=0.8)
        # ax2.set_title('Final Training Accuracy per Task', fontsize=14, fontweight='bold')
        # ax2.set_ylabel('Accuracy (%)', fontsize=12)
        # ax2.grid(True, alpha=0.3, axis='y')
        
        # # Add value labels on bars
        # for bar, acc in zip(bars2, final_train_accs):
        #     height = bar.get_height()
        #     ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
        #             f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Training loss progression (all tasks on same plot)
        ax1.set_title('Training Loss Progression', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        colors = sns.color_palette("deep", len(self.cil_training_metrics['task_names']))
        for task_idx, task_name in enumerate(self.cil_training_metrics['task_names']):
            epochs = self.cil_training_metrics['task_epochs'][task_idx]
            losses = self.cil_training_metrics['task_training_losses'][task_idx]
            if epochs and losses:
                ax1.plot(epochs, losses, 'o-', label=f'Task {task_idx+1}', 
                        linewidth=2, markersize=4, color=colors[task_idx])
        ax1.legend()
        
        # Plot 4: Training accuracy progression (all tasks on same plot)
        ax2.set_title('Training Accuracy Progression', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        for task_idx, task_name in enumerate(self.cil_training_metrics['task_names']):
            epochs = self.cil_training_metrics['task_epochs'][task_idx]
            accs = self.cil_training_metrics['task_training_accuracies'][task_idx]
            if epochs and accs:
                ax2.plot(epochs, accs, 'o-', label=f'Task {task_idx+1}', 
                        linewidth=2, markersize=4, color=colors[task_idx])
        ax2.legend()
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(f'{self.save_dir}/cil_training_summary.png', dpi=300, bbox_inches='tight')
            print(f"CIL training summary plot saved to {self.save_dir}/cil_training_summary.png")
        
        # plt.show()
    
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
            },
            'cil_training_metrics': {
                'task_names': self.cil_training_metrics['task_names'],
                'task_training_losses': self.cil_training_metrics['task_training_losses'],
                'task_training_accuracies': self.cil_training_metrics['task_training_accuracies'],
                'task_validation_losses': self.cil_training_metrics['task_validation_losses'],
                'task_validation_accuracies': self.cil_training_metrics['task_validation_accuracies'],
                'task_epochs': self.cil_training_metrics['task_epochs']
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
        
        # CIL Training Metrics Summary
        if self.cil_training_metrics['task_names']:
            print(f"\nCIL TRAINING METRICS:")
            for task_idx, task_name in enumerate(self.cil_training_metrics['task_names']):
                if self.cil_training_metrics['task_training_losses'][task_idx]:
                    final_train_loss = self.cil_training_metrics['task_training_losses'][task_idx][-1]
                    final_train_acc = self.cil_training_metrics['task_training_accuracies'][task_idx][-1]
                    num_epochs = len(self.cil_training_metrics['task_epochs'][task_idx])
                    print(f"  Task {task_idx+1} ({task_name}):")
                    print(f"    Epochs: {num_epochs}")
                    print(f"    Final training loss: {final_train_loss:.4f}")
                    print(f"    Final training accuracy: {final_train_acc:.2f}%")
                    
                    # Add validation metrics if available
                    if self.cil_training_metrics['task_validation_losses'][task_idx]:
                        final_val_loss = self.cil_training_metrics['task_validation_losses'][task_idx][-1]
                        final_val_acc = self.cil_training_metrics['task_validation_accuracies'][task_idx][-1]
                        print(f"    Final validation loss: {final_val_loss:.4f}")
                        print(f"    Final validation accuracy: {final_val_acc:.2f}%")
        
        # Final per-class performance
        if self.cil_metrics['task_accuracies']:
            final_accs = self.cil_metrics['task_accuracies'][-1]
            print(f"\nFINAL PER-CLASS PERFORMANCE:")
            for class_name, acc in sorted(final_accs.items(), key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {acc:.2f}%")
        
        print("="*60)

