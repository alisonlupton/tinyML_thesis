#!/usr/bin/env python3
"""
Ablation plotting script for dog behaviours continual learning experiments.
This script creates comparison plots between different ablation conditions.

Usage:
    python ablation_plotting.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_replay_comparison():
    """
    Plot comparison between replay and no-replay results.
    Loads CSV files from both experiments and creates an overlay plot.
    """
    plot_dir = "plots_and_metrics/ablations"
    #Set seaborn style to match other plots
    plt.style.use('seaborn-v0_8-deep')
    sns.set_palette("Set1")  #This will affect all subsequent plots

    
    #Set LaTeX-style fonts to match document
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 15,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 25,
        'ytick.labelsize': 20,
        'legend.fontsize': 17,
        'figure.titlesize': 17,
        'text.usetex': False,  #Set to True if have LaTeX installed
        'mathtext.fontset': 'cm'  #Use Computer Modern for math text
    })
    

    #Load both CSV files
    replay_csv = os.path.join(plot_dir, 'intelligent_plotting_overall_withreplay.csv')
    replay_no_head_csv = os.path.join(plot_dir, 'intelligent_plotting_overall_withreplay_nohead.csv')
    no_replay_csv = os.path.join(plot_dir, 'intelligent_plotting_overall_noreplay.csv')
    no_replay_no_head_csv = os.path.join(plot_dir, 'intelligent_plotting_overall_noreplay_nohead.csv')
    
    if not os.path.exists(replay_csv):
        print(f"Error: Replay CSV not found at {replay_csv}")
        print("Please run the replay experiment first to generate this file.")
        return
    
    if not os.path.exists(no_replay_csv):
        print(f"Error: No-replay CSV not found at {no_replay_csv}")
        print("Please run the no-replay experiment and save the CSV as 'intelligent_plotting_overall_no_replay.csv'")
        return
        
    if not os.path.exists(no_replay_csv):
        print(f"Error: No-replay no-head CSV not found at {no_replay_csv}")
        print("Please run the no-replay + no-head experiment and save the CSV as 'intelligent_plotting_overall_no_replay.csv'")
        return
    if not os.path.exists(replay_no_head_csv):
        print(f"Error: No-replay no-head CSV not found at {no_replay_csv}")
        print("Please run the no-replay + no-head experiment and save the CSV as 'intelligent_plotting_overall_no_replay.csv'")
        return
    
    #Load data
    replay_df = pd.read_csv(replay_csv)
    replay_no_head_df = pd.read_csv(replay_no_head_csv)
    no_replay_df = pd.read_csv(no_replay_csv)
    no_replay_no_head_df = pd.read_csv(no_replay_no_head_csv)
    
    print("Loaded data:")
    print(f"Replay results: {len(replay_df)} stages")
    print(f"No-replay results: {len(no_replay_df)} stages")
    
    #Create comparison plot
    plt.figure(figsize=(10, 6))
    
    #Plot replay results (solid line)
    plt.plot(range(len(replay_df)), replay_df['Overall_Accuracy'], 
             marker='o', linewidth=3, markersize=10, 
             label='With Replay', linestyle='-')
    
    #Plot no-replay results (dotted line)
    plt.plot(range(len(no_replay_df)), no_replay_df['Overall_Accuracy'], 
             marker='s', linewidth=3, markersize=10, 
             label='Without Replay', linestyle='--')
    
    #Plot no-replay results (dotted line)
    plt.plot(range(len(no_replay_no_head_df)), no_replay_no_head_df['Overall_Accuracy'], 
             marker='^', linewidth=3, markersize=10, 
             label='Without Replay or Adaptive Head', linestyle='-.')
    
    #Plot replay no-head results (dotted line)
    plt.plot(range(len(replay_no_head_df)), replay_no_head_df['Overall_Accuracy'], 
             marker='*', linewidth=3, markersize=10, 
             label='With Replay Without Adaptive Head', linestyle=':')
    
    #Customize plot
    plt.xlabel('Training Stage', fontweight='bold')
    plt.ylabel('Overall Accuracy (%)', fontweight='bold')
    plt.xticks(range(len(replay_df)), replay_df['Stage'], rotation=45)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/replay_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Replay comparison plot saved to {plot_dir}/replay_comparison.pdf")

def plot_p_comparison():
    """
    Plot comparison between replay and no-replay results.
    Loads CSV files from both experiments and creates an overlay plot.
    """
    plot_dir = "plots_and_metrics/ablations"
    #Set seaborn style to match other plots
    plt.style.use('seaborn-v0_8-deep')
    sns.set_palette("Set1")  #This will affect all subsequent plots

    #Get seaborn colors
    colors = sns.color_palette("Set1", 3)
    
    #Set LaTeX-style fonts to match document
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 15,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 25,
        'ytick.labelsize': 20,
        'legend.fontsize': 17,
        'figure.titlesize': 17,
        'text.usetex': False,  #Set to True if have LaTeX installed
        'mathtext.fontset': 'cm'  #Use Computer Modern for math text
    })
    

    #Load both CSV files
    intra10_inter25 = os.path.join(plot_dir, 'intelligent_plotting_overall_10_25.csv')
    intra20_inter45 = os.path.join(plot_dir, 'intelligent_plotting_overall_20_45.csv')
    intra30_inter65 = os.path.join(plot_dir, 'intelligent_plotting_overall_30_65.csv')
    
    
    
    #Load data
    intra10_inter25_df = pd.read_csv(intra10_inter25)
    intra20_inter45_df = pd.read_csv(intra20_inter45)
    intra30_inter65_df = pd.read_csv(intra30_inter65)
    
    #Create comparison plot
    plt.figure(figsize=(10, 6))
    
    #Plot replay results (solid line)
    plt.plot(range(len(intra10_inter25_df)), intra10_inter25_df['Overall_Accuracy'], 
             color = colors[0], marker='o', linewidth=2, markersize=10, 
             label='p_intra = 0.1, p_inter = 0.25', linestyle='-')
    
    #Plot no-replay results (dotted line)
    plt.plot(range(len(intra20_inter45_df)), intra20_inter45_df['Overall_Accuracy'], 
             color = colors[1], marker='s', linewidth=2, markersize=10, 
             label='p_intra = 0.2, p_inter = 0.45', linestyle='--')
    
    #Plot no-replay results (dotted line)
    plt.plot(range(len(intra30_inter65_df)), intra30_inter65_df['Overall_Accuracy'], 
             color = colors[2], marker='^', linewidth=2, markersize=10, 
             label='p_intra = 0.3, p_inter = 0.65', linestyle='-.')
    
    #Customize plot
    plt.xlabel('Training Stage', fontweight='bold')
    plt.ylabel('Overall Accuracy (%)', fontweight='bold')
    plt.xticks(range(len(intra20_inter45_df)), intra20_inter45_df['Stage'], rotation=45)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/p_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"P levels comparison plot saved to {plot_dir}/p_comparison.pdf")

def plot_per_class_comparison():
    """
    Plot per-class accuracy comparison between replay and no-replay.
    """
    plot_dir = "plots_and_metrics"
    
    #Set LaTeX-style fonts
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 15,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 25,
        'ytick.labelsize': 20,
        'legend.fontsize': 17,
        'figure.titlesize': 17,
        'text.usetex': False,
        'mathtext.fontset': 'cm'
    })
    
    plt.style.use('seaborn-v0_8-deep')
    
    #Load per-class data
    replay_csv = os.path.join(plot_dir, 'intelligent_plotting_per_class.csv')
    no_replay_csv = os.path.join(plot_dir, 'intelligent_plotting_per_class_no_replay.csv')
    
    if not os.path.exists(replay_csv):
        print(f"Error: Replay per-class CSV not found at {replay_csv}")
        return
    
    if not os.path.exists(no_replay_csv):
        print(f"Error: No-replay per-class CSV not found at {no_replay_csv}")
        return
    
    #Load data
    replay_df = pd.read_csv(replay_csv)
    no_replay_df = pd.read_csv(no_replay_csv)
    
    #Get unique behaviors
    behaviors = replay_df['Behavior'].unique()
    stages = replay_df['Stage'].unique()
    
    #Create per-class comparison plot
    plt.figure(figsize=(14, 8))
    
    #Plot each behavior
    for behavior in behaviors:
        #Replay data
        replay_behavior = replay_df[replay_df['Behavior'] == behavior]
        replay_behavior = replay_behavior.sort_values('Stage', key=lambda x: x.map({stage: i for i, stage in enumerate(stages)}))
        
        #No-replay data
        no_replay_behavior = no_replay_df[no_replay_df['Behavior'] == behavior]
        no_replay_behavior = no_replay_behavior.sort_values('Stage', key=lambda x: x.map({stage: i for i, stage in enumerate(stages)}))
        
        #Plot replay (solid)
        plt.plot(range(len(stages)), replay_behavior['Accuracy'], 
                marker='o', linewidth=2.5, markersize=8, 
                label=f'{behavior} (Replay)', linestyle='-')
        
        #Plot no-replay (dotted)
        plt.plot(range(len(stages)), no_replay_behavior['Accuracy'], 
                marker='s', linewidth=2.5, markersize=8, 
                label=f'{behavior} (No Replay)', linestyle='--')
    
    #Customize plot
    plt.xlabel('Training Stage', fontweight='bold')
    plt.ylabel('Accuracy (%)', fontweight='bold')
    plt.xticks(range(len(stages)), stages, rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/per_class_replay_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Per-class replay comparison plot saved to {plot_dir}/per_class_replay_comparison.pdf")

def main():
    """
    Main function to generate all ablation comparison plots.
    """
    print("Generating ablation comparison plots...")
    
    #Create plots directory if it doesn't exist
    os.makedirs("plots_and_metrics", exist_ok=True)
    
    #Generate replay comparison plot
    print("\n1. Generating replay vs no-replay overall accuracy comparison...")
    plot_replay_comparison()
    
    plot_p_comparison()
    
    #Generate per-class comparison plot
    print("\n2. Generating per-class replay comparison...")
    plot_per_class_comparison()
    
    print("\nDone! Check the plots_and_metrics/ directory for comparison plots.")

if __name__ == "__main__":
    main()
