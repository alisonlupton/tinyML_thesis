#!/usr/bin/env python3
"""
Dog Breed CIL Training Plotter - Analogous to dog_behaviours report_plotting.py
Creates individual training loss and accuracy plots for each CIL task
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, List

def clean_breed_name(breed_id: str) -> str:
    """
    Convert breed ID like 'n02097209-standard_schnauzer' to 'Standard Schnauzer'
    """
    if '-' in breed_id:
        #Extract the part after the last dash
        breed_part = breed_id.split('-')[-1]
        #Replace underscores with spaces and title case
        return breed_part.replace('_', ' ').title()
    else:
        #If no dash, just clean up underscores and title case
        return breed_id.replace('_', ' ').title()

class DogBreedCILTrainingPlotter:
    """Class to handle plotting for Dog Breed CIL training experiments"""
    
    def __init__(self, save_dir='plots_and_metrics', seed=None):
        self.save_dir = save_dir
        self.seed = seed
        os.makedirs(save_dir, exist_ok=True)
        
        #Set seaborn style to match dog_behaviours plots
        plt.style.use('seaborn-v0_8-deep')
        sns.set_palette("deep")
        
        #Storage for backbone metrics
        self.backbone_metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epochs': []
        }
        
        #Storage for CIL task results (final accuracies)
        self.cil_metrics = {
            'task_accuracies': [],  #List of dicts: [{breed: acc}, {breed: acc}, ...]
            'overall_accuracies': [],  #List of overall accuracies per task
            'task_names': []  #List of task names
        }
        
        #Storage for CIL training metrics (per-epoch data)
        self.cil_training_metrics = {
            'task_training_losses': [],  #List of lists: [[epoch1_loss, epoch2_loss, ...], ...]
            'task_training_accuracies': [],  #List of lists: [[epoch1_acc, epoch2_acc, ...], ...]
            'task_validation_losses': [],  #List of lists: [[epoch1_val_loss, epoch2_val_loss, ...], ...]
            'task_validation_accuracies': [],  #List of lists: [[epoch1_val_acc, epoch2_val_acc, ...], ...]
            'task_epochs': [],  #List of lists: [[1, 2, 3, ...], ...] for each task
            'task_names': []  #List of task names
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
        #Find or create task index
        if task_name not in self.cil_training_metrics['task_names']:
            self.cil_training_metrics['task_names'].append(task_name)
            self.cil_training_metrics['task_training_losses'].append([])
            self.cil_training_metrics['task_training_accuracies'].append([])
            self.cil_training_metrics['task_validation_losses'].append([])
            self.cil_training_metrics['task_validation_accuracies'].append([])
            self.cil_training_metrics['task_epochs'].append([])
        
        task_idx = self.cil_training_metrics['task_names'].index(task_name)
        
        #Add metrics
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
        
        #Set seaborn style to match behavior distribution plots
        plt.style.use('seaborn-v0_8-deep')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        epochs = self.backbone_metrics['epochs']
        
        #Plot 1: Training Loss
        ax1.plot(epochs, self.backbone_metrics['train_loss'], 'o-', linewidth=2.5, markersize=8, color='blue', label='Training Loss')
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Backbone Training Loss', fontsize=16, fontweight='bold', pad=20)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        #Plot 2: Validation Accuracy
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
        
        #plt.show()
    
    def plot_cil_training_curves(self, save_plot=True):
        """Plot CIL training curves (loss and accuracy) for all tasks - creates 6 plots total"""
        if not self.cil_training_metrics['task_names']:
            print("No CIL training metrics to plot!")
            return
        
        num_tasks = len(self.cil_training_metrics['task_names'])
        fig, axes = plt.subplots(2, num_tasks, figsize=(5*num_tasks, 10))
        
        #If only one task, make axes 2D
        if num_tasks == 1:
            axes = axes.reshape(2, 1)
        
        colors = sns.color_palette("deep", num_tasks)
        
        for task_idx, task_name in enumerate(self.cil_training_metrics['task_names']):
            epochs = self.cil_training_metrics['task_epochs'][task_idx]
            train_losses = self.cil_training_metrics['task_training_losses'][task_idx]
            train_accs = self.cil_training_metrics['task_training_accuracies'][task_idx]
            val_losses = self.cil_training_metrics['task_validation_losses'][task_idx]
            val_accs = self.cil_training_metrics['task_validation_accuracies'][task_idx]
            
            
            #Plot training loss
            ax_loss = axes[0, task_idx]
            ax_loss.plot(epochs, train_losses, '-', label='Training Loss', linewidth=2, color=colors[task_idx])
            if val_losses:
                ax_loss.plot(epochs, val_losses, '--', label='Validation Loss', linewidth=2, color=colors[task_idx], alpha=0.7)
            ax_loss.set_xlabel('Epoch', fontsize=10)
            ax_loss.set_ylabel('Loss', fontsize=10)
            ax_loss.set_title(f'{task_name} - Loss', fontsize=12, fontweight='bold')
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)
            
            #Plot training accuracy
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
        
        #plt.show()
    
    def plot_cil_training_summary(self, save_plot=True):
        """Plot a summary of CIL training metrics across all tasks"""
        if not self.cil_training_metrics['task_names']:
            print("No CIL training metrics to plot!")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        #Plot 1: Training loss progression (all tasks on same plot)
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
        
        #Plot 2: Training accuracy progression (all tasks on same plot)
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
        
        #plt.show()
    
    def save_training_curves_to_csv(self):
        """Save CIL training curves data to CSV for multi-seed analysis"""
        if not self.cil_training_metrics['task_names']:
            print("No CIL training metrics to save!")
            return
        
        #Get seed
        seed = getattr(self, 'seed', 'unknown')
        
        #Prepare data for CSV
        csv_data = []
        
        for task_idx, task_name in enumerate(self.cil_training_metrics['task_names']):
            epochs = self.cil_training_metrics['task_epochs'][task_idx]
            train_losses = self.cil_training_metrics['task_training_losses'][task_idx]
            train_accs = self.cil_training_metrics['task_training_accuracies'][task_idx]
            val_losses = self.cil_training_metrics['task_validation_losses'][task_idx]
            val_accs = self.cil_training_metrics['task_validation_accuracies'][task_idx]
            
            for epoch_idx, epoch in enumerate(epochs):
                #Training data
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
                
                #Validation data (if available)
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
        
        #Create DataFrame and save
        df = pd.DataFrame(csv_data)
        
        #Append to existing file or create new one
        csv_file = os.path.join(self.save_dir, 'cil_training_curves_data.csv')
        if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
            try:
                existing_df = pd.read_csv(csv_file)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                #File exists but is empty or corrupted, start fresh
                combined_df = df
        else:
            combined_df = df
        
        combined_df.to_csv(csv_file, index=False)
        print(f"CIL training curves data saved to {csv_file}")
    
    def create_summary_report(self):
        """Create a comprehensive summary report"""
        if not self.cil_metrics['task_names']:
            print("No metrics to report!")
            return
        
        print("\n" + "="*60)
        print("DOG BREED CIL TRAINING SUMMARY REPORT")
        print("="*60)
        
        #Backbone summary
        if self.backbone_metrics['epochs']:
            print(f"\nBACKBONE TRAINING:")
            print(f"  Total epochs: {len(self.backbone_metrics['epochs'])}")
            print(f"  Final training accuracy: {self.backbone_metrics['train_acc'][-1]:.2f}%")
            print(f"  Best validation accuracy: {max(self.backbone_metrics['val_acc']):.2f}%")
        
        #CIL summary
        print(f"\nCIL TRAINING:")
        print(f"  Total tasks: {len(self.cil_metrics['task_names'])}")
        print(f"  Final overall accuracy: {self.cil_metrics['overall_accuracies'][-1]:.2f}%")
        
        print(f"\nTASK PROGRESSION:")
        for i, (task_name, overall_acc) in enumerate(zip(self.cil_metrics['task_names'], 
                                                        self.cil_metrics['overall_accuracies'])):
            print(f"  Task {i+1} ({task_name}): {overall_acc:.2f}%")
        
        #CIL Training Metrics Summary
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
                    
                    #Add validation metrics if available
                    if self.cil_training_metrics['task_validation_losses'][task_idx]:
                        final_val_loss = self.cil_training_metrics['task_validation_losses'][task_idx][-1]
                        final_val_acc = self.cil_training_metrics['task_validation_accuracies'][task_idx][-1]
                        print(f"    Final validation loss: {final_val_loss:.4f}")
                        print(f"    Final validation accuracy: {final_val_acc:.2f}%")
        
        #Final per-class performance
        if self.cil_metrics['task_accuracies']:
            final_accs = self.cil_metrics['task_accuracies'][-1]
            print(f"\nFINAL PER-CLASS PERFORMANCE:")
            for class_name, acc in sorted(final_accs.items(), key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {acc:.2f}%")
        
        print("="*60)


def test_dog_breed_plotter():
    """Test function to demonstrate the plotter with random data"""
    print(" Testing Dog Breed CIL Training Plotter")
    print("=" * 50)
    
    #Create plotter instance
    plotter = DogBreedCILTrainingPlotter(save_dir='test_plots', seed=42)
    
    #Simulate backbone training data
    print("Adding backbone training data...")
    for epoch in range(1, 21):
        train_loss = 2.0 * np.exp(-epoch/10) + 0.1 + np.random.normal(0, 0.05)
        val_loss = 2.2 * np.exp(-epoch/10) + 0.15 + np.random.normal(0, 0.05)
        train_acc = 100 * (1 - train_loss/2.0) + np.random.normal(0, 2)
        val_acc = 100 * (1 - val_loss/2.2) + np.random.normal(0, 2)
        plotter.add_backbone_epoch(epoch, train_loss, val_loss, train_acc, val_acc)
    
    #Simulate CIL task data with realistic breed IDs
    dog_breed_ids = ["n02097209-standard_schnauzer", "n02106662-German_shepherd", "n02099712-Labrador_retriever", "n02113799-standard_poodle", "n02096585-Boston_bull", "n02088364-beagle"]
    
    for task_idx in range(3):  #3 tasks
        breed_id = dog_breed_ids[task_idx*2]
        clean_breed = clean_breed_name(breed_id)
        task_name = f"Task {task_idx+1}: {clean_breed}"
        print(f"Adding CIL task: {task_name}")
        
        #Simulate per-epoch training data
        for epoch in range(1, 11):  #10 epochs per task
            train_loss = 1.5 * np.exp(-epoch/5) + 0.2 + np.random.normal(0, 0.05)
            train_acc = 85 + 10 * (1 - np.exp(-epoch/3)) + np.random.normal(0, 2)
            val_loss = 1.6 * np.exp(-epoch/5) + 0.25 + np.random.normal(0, 0.05)
            val_acc = 83 + 12 * (1 - np.exp(-epoch/3)) + np.random.normal(0, 2)
            
            #Make sure validation data is properly passed
            plotter.add_cil_task_training_epoch(task_name, epoch, train_loss, train_acc, val_loss, val_acc)
        
        #Simulate final task results
        class_accs = {}
        for i, breed_id in enumerate(dog_breed_ids):
            clean_breed = clean_breed_name(breed_id)
            class_accs[clean_breed] = 80 + np.random.uniform(0, 20)
        
        overall_acc = sum(class_accs.values()) / len(class_accs)
        plotter.add_cil_task(task_name, class_accs, overall_acc)
    
    #Generate all plots
    print("\nGenerating plots...")
    plotter.plot_backbone_training()
    plotter.plot_cil_training_curves()  #This creates the 6 individual plots
    plotter.plot_cil_training_summary()
    
    #Create summary report
    plotter.create_summary_report()
    
    print("\n" + "=" * 50)
    print(" Test completed! Check the 'test_plots' directory for generated plots.")
    print("Files created:")
    print("  - backbone_training.png")
    print("  - cil_training_curves.png (6 individual plots)")
    print("  - cil_training_summary.png")
    print("  - cil_training_curves_data.csv")


if __name__ == "__main__":
    test_dog_breed_plotter()
