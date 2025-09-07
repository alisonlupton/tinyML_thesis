# multi_seed_plotting.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def plot_multi_seed_backbone_training(plot_dir='plots_and_metrics_seeded', save_dir='plots_and_metrics_seeded'):
    """
    Plot backbone training curves with mean and std across multiple seeds.
    """
    # Find all backbone training CSV files with seed numbers
    backbone_files = []
    for file in os.listdir(plot_dir):
        if file.startswith('backbone_training_data_') and file.endswith('.csv'):
            backbone_files.append(os.path.join(plot_dir, file))
    
    if not backbone_files:
        print(f"No backbone training CSV files found in {plot_dir}")
        print("Looking for files like: backbone_training_data_42.csv, backbone_training_data_43.csv, etc.")
        return
    
    print(f"Found backbone training files: {[os.path.basename(f) for f in backbone_files]}")
    
    # Load and combine all backbone training data
    dfs = []
    for file in backbone_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Set seaborn style to match your existing plots
    plt.style.use('seaborn-v0_8-deep')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Training Loss
    train_loss_data = df[df['metric_type'] == 'training_loss']
    if not train_loss_data.empty:
        # Calculate mean and std across seeds for each epoch
        loss_stats = train_loss_data.groupby('epoch')['value'].agg(['mean', 'std']).reset_index()
        
        epochs = loss_stats['epoch']
        mean_loss = loss_stats['mean']
        std_loss = loss_stats['std']
        
        # Plot mean line
        ax1.plot(epochs, mean_loss, 'o-', linewidth=2.5, markersize=8, color='blue', label='Mean Training Loss')
        
        # Plot std band
        ax1.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, 
                        alpha=0.3, color='blue', label='±1 Std')
        
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Backbone Training Loss (Multi-Seed)', fontsize=16, fontweight='bold', pad=20)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy
    val_acc_data = df[df['metric_type'] == 'validation_accuracy']
    if not val_acc_data.empty:
        # Calculate mean and std across seeds for each epoch
        acc_stats = val_acc_data.groupby('epoch')['value'].agg(['mean', 'std']).reset_index()
        
        epochs = acc_stats['epoch']
        mean_acc = acc_stats['mean']
        std_acc = acc_stats['std']
        
        # Plot mean line
        ax2.plot(epochs, mean_acc, 'o-', linewidth=2.5, markersize=8, color='red', label='Mean Validation Accuracy')
        
        # Plot std band
        ax2.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, 
                        alpha=0.3, color='red', label='±1 Std')
        
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Backbone Validation Accuracy (Multi-Seed)', fontsize=16, fontweight='bold', pad=20)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(save_dir, 'multi_seed_backbone_training.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Multi-seed backbone training plot saved to {output_file}")
    
    # plt.show()

def plot_multi_seed_cil_progression(plot_dir='plots_and_metrics', save_dir='plots_and_metrics'):
    """
    Plot CIL progression with mean and std across multiple seeds.
    """
    # Find all CIL progression CSV files with seed numbers
    progression_files = []
    for file in os.listdir(plot_dir):
        if file.startswith('cil_progression_data_') and file.endswith('.csv'):
            progression_files.append(os.path.join(plot_dir, file))
    
    if not progression_files:
        print(f"No CIL progression CSV files found in {plot_dir}")
        print("Looking for files like: cil_progression_data_42.csv, cil_progression_data_43.csv, etc.")
        return
    
    print(f"Found CIL progression files: {[os.path.basename(f) for f in progression_files]}")
    
    # Load and combine all CIL progression data
    dfs = []
    for file in progression_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Debug: Show data structure
    print(f"Combined CIL progression data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample data:")
    print(df.head(10))
    print(f"\nUnique seeds: {sorted(df['seed'].unique())}")
    print(f"Unique stages: {sorted(df['stage'].unique())}")
    print(f"Unique behaviors: {sorted(df['behavior'].unique())}")
    print(f"\nOverall accuracy values by stage:")
    for stage in sorted(df['stage'].unique()):
        stage_data = df[df['stage'] == stage]
        print(f"  {stage}: {stage_data['overall_accuracy'].unique()}")
    print(f"\nPer-seed overall accuracy:")
    for seed in sorted(df['seed'].unique()):
        seed_data = df[df['seed'] == seed]
        print(f"  Seed {seed}: {seed_data['overall_accuracy'].unique()}")
    
    # Set seaborn style to match your existing plots
    plt.style.use('seaborn-v0_8-deep')
    
    # Create the main plot
    plt.figure(figsize=(14, 8))
    
    # Get all behaviors and stages from actual data
    behaviors = sorted(df['behavior'].unique())
    stages = sorted(df['stage'].unique())  # Use actual stages from data
    print(f"Found stages in data: {stages}")
    print(f"Found behaviors in data: {behaviors}")
    
    # Plot per-class accuracy progression with std
    for behavior in behaviors:
        behavior_data = df[df['behavior'] == behavior]
        
        # Calculate mean and std across seeds for each stage
        acc_stats = behavior_data.groupby('stage')['accuracy'].agg(['mean', 'std']).reset_index()
        
        # Ensure all stages are present and in correct order
        stage_mapping = {stage: i for i, stage in enumerate(stages)}
        acc_stats['stage_order'] = acc_stats['stage'].map(stage_mapping)
        acc_stats = acc_stats.sort_values('stage_order')
        
        # Only plot if we have data for this behavior
        if acc_stats.empty:
            print(f"No data found for behavior: {behavior}")
            continue
        
        mean_acc = acc_stats['mean']
        std_acc = acc_stats['std']
        
        # Only plot if we have data for all stages
        if len(mean_acc) == len(stages):
            # Plot mean line
            plt.plot(range(len(stages)), mean_acc, 'o-', linewidth=2.5, markersize=8, 
                    label=f'{behavior}', alpha=0.8)
            
            # Plot std band
            plt.fill_between(range(len(stages)), mean_acc - std_acc, mean_acc + std_acc, 
                            alpha=0.2)
        else:
            print(f"Note: {behavior} only has data for {len(mean_acc)} stages (expected {len(stages)}) - this is normal for behaviors introduced in later tasks")
    
    # Customize the plot
    plt.title('Per-Class Accuracy Progression: Backbone → CIL Tasks (Multi-Seed)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Training Stage', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    
    # Set x-axis labels
    plt.xticks(range(len(stages)), stages, rotation=45)
    
    # Add grid and customize legend
    plt.grid(True, alpha=0.3)
    plt.legend(title='Behavior Classes', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # Add value annotations (mean values) - only for behaviors that were plotted
    for behavior in behaviors:
        behavior_data = df[df['behavior'] == behavior]
        acc_stats = behavior_data.groupby('stage')['accuracy'].agg(['mean', 'std']).reset_index()
        stage_mapping = {stage: i for i, stage in enumerate(stages)}
        acc_stats['stage_order'] = acc_stats['stage'].map(stage_mapping)
        acc_stats = acc_stats.sort_values('stage_order')
        
        # Only annotate if we have data for all stages
        if len(acc_stats) == len(stages):
            for i, (_, row) in enumerate(acc_stats.iterrows()):
                plt.annotate(f'{row["mean"]:.1f}%', 
                            xy=(i, row['mean']), 
                            xytext=(0, 10), textcoords='offset points',
                            fontsize=9, fontweight='bold', alpha=0.8,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(save_dir, 'multi_seed_intelligent_tdm_per_class_progression.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Multi-seed per-class progression plot saved to {output_file}")
    
    # plt.show() 
    
    # Create overall accuracy progression plot
    plt.figure(figsize=(10, 6))
    
    # Calculate overall accuracy for each stage - need to get unique values first
    # since each row has the same overall_accuracy repeated
    overall_stats = df.groupby(['seed', 'stage'])['overall_accuracy'].first().reset_index()
    overall_stats = overall_stats.groupby('stage')['overall_accuracy'].agg(['mean', 'std']).reset_index()
    
    # Ensure correct stage order
    stage_mapping = {stage: i for i, stage in enumerate(stages)}
    overall_stats['stage_order'] = overall_stats['stage'].map(stage_mapping)
    overall_stats = overall_stats.sort_values('stage_order')
    
    # Only proceed if we have overall accuracy data
    if overall_stats.empty:
        print("No overall accuracy data found!")
        return
    
    mean_overall = overall_stats['mean']
    std_overall = overall_stats['std']
    
    # Only plot if we have data for all stages
    if len(mean_overall) == len(stages):
        # Plot mean line
        plt.plot(range(len(stages)), mean_overall, 'o-', linewidth=3, markersize=10, 
                color='darkblue', label='Mean Overall Accuracy')
        
        # Plot std band
        plt.fill_between(range(len(stages)), mean_overall - std_overall, mean_overall + std_overall, 
                        alpha=0.3, color='darkblue', label='±1 Std')
    else:
        print(f"Cannot plot overall accuracy: has {len(mean_overall)} data points but {len(stages)} stages")
        return
    
    # Customize
    plt.title('Overall Accuracy Progression: Backbone → CIL Tasks (Multi-Seed)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Training Stage', fontsize=12, fontweight='bold')
    plt.ylabel('Overall Accuracy (%)', fontsize=12, fontweight='bold')
    plt.xticks(range(len(stages)), stages, rotation=45)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.legend(fontsize=12)
    
    # Add value annotations - only if we have data for all stages
    if len(overall_stats) == len(stages):
        for i, (_, row) in enumerate(overall_stats.iterrows()):
            plt.annotate(f'{row["mean"]:.1f}%', 
                        xy=(i, row['mean']), 
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(save_dir, 'multi_seed_intelligent_tdm_overall_progression.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Multi-seed overall progression plot saved to {output_file}")
    
    # plt.show()

def plot_multi_seed_cil_training_curves(plot_dir='plots_and_metrics', save_dir='plots_and_metrics'):
    """
    Plot CIL training curves with mean and std across multiple seeds.
    """
    # Find all CIL training curves CSV files with seed numbers
    training_files = []
    for file in os.listdir(plot_dir):
        if file.startswith('cil_training_curves_data_') and file.endswith('.csv'):
            training_files.append(os.path.join(plot_dir, file))
    
    if not training_files:
        print(f"No CIL training curves CSV files found in {plot_dir}")
        print("Looking for files like: cil_training_curves_data_42.csv, cil_training_curves_data_43.csv, etc.")
        return
    
    print(f"Found CIL training curves files: {[os.path.basename(f) for f in training_files]}")
    
    # Load and combine all CIL training curves data
    dfs = []
    for file in training_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Set seaborn style
    plt.style.use('seaborn-v0_8-deep')
    
    # Get unique tasks
    tasks = sorted(df['task_name'].unique())
    num_tasks = len(tasks)
    
    if num_tasks == 0:
        print("No CIL training data found!")
        return
    
    fig, axes = plt.subplots(2, num_tasks, figsize=(5*num_tasks, 10))
    
    # If only one task, make axes 2D
    if num_tasks == 1:
        axes = axes.reshape(2, 1)
    
    colors = sns.color_palette("deep", num_tasks)
    
    for task_idx, task_name in enumerate(tasks):
        task_data = df[df['task_name'] == task_name]
        
        # Plot training loss
        ax_loss = axes[0, task_idx]
        train_loss_data = task_data[task_data['metric_type'] == 'training_loss']
        
        if not train_loss_data.empty:
            # Calculate mean and std across seeds for each epoch
            loss_stats = train_loss_data.groupby('epoch')['value'].agg(['mean', 'std']).reset_index()
            
            epochs = loss_stats['epoch']
            mean_loss = loss_stats['mean']
            std_loss = loss_stats['std']
            
            # Plot mean line
            ax_loss.plot(epochs, mean_loss, '-', linewidth=2, color=colors[task_idx], 
                        label='Mean Training Loss')
            
            # Plot std band
            ax_loss.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, 
                               alpha=0.3, color=colors[task_idx])
        
        ax_loss.set_xlabel('Epoch', fontsize=10)
        ax_loss.set_ylabel('Loss', fontsize=10)
        ax_loss.set_title(f'{task_name} - Loss (Multi-Seed)', fontsize=12, fontweight='bold')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
        
        # Plot training accuracy
        ax_acc = axes[1, task_idx]
        train_acc_data = task_data[task_data['metric_type'] == 'training_accuracy']
        
        if not train_acc_data.empty:
            # Calculate mean and std across seeds for each epoch
            acc_stats = train_acc_data.groupby('epoch')['value'].agg(['mean', 'std']).reset_index()
            
            epochs = acc_stats['epoch']
            mean_acc = acc_stats['mean']
            std_acc = acc_stats['std']
            
            # Plot mean line
            ax_acc.plot(epochs, mean_acc, '-', linewidth=2, color=colors[task_idx], 
                       label='Mean Training Accuracy')
            
            # Plot std band
            ax_acc.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, 
                              alpha=0.3, color=colors[task_idx])
        
        ax_acc.set_xlabel('Epoch', fontsize=10)
        ax_acc.set_ylabel('Accuracy (%)', fontsize=10)
        ax_acc.set_title(f'{task_name} - Accuracy (Multi-Seed)', fontsize=12, fontweight='bold')
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(save_dir, 'multi_seed_cil_training_curves.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Multi-seed CIL training curves plot saved to {output_file}")
    
    # plt.show()

def generate_all_multi_seed_plots(plot_dir='plots_and_metrics_seeded', save_dir='plots_and_metrics_seeded'):
    """
    Generate all multi-seed plots from CSV data.
    """
    print("Generating multi-seed plots from CSV data...")
    print(f"Loading data from: {plot_dir}")
    print(f"Saving plots to: {save_dir}")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate all plots
    plot_multi_seed_backbone_training(plot_dir, save_dir)
    plot_multi_seed_cil_progression(plot_dir, save_dir)
    plot_multi_seed_cil_training_curves(plot_dir, save_dir)
    
    print("\nAll multi-seed plots generated successfully!")
    print("Generated files:")
    print(f"  - {save_dir}/multi_seed_backbone_training.png")
    print(f"  - {save_dir}/multi_seed_intelligent_tdm_per_class_progression.png")
    print(f"  - {save_dir}/multi_seed_intelligent_tdm_overall_progression.png")
    print(f"  - {save_dir}/multi_seed_cil_training_curves.png")

if __name__ == "__main__":
    # Generate all multi-seed plots
    generate_all_multi_seed_plots()
