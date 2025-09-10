#!/usr/bin/env python3
"""
Script to create a line plot with error bars showing accuracy across seeds
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

#Set seaborn style to match dog behaviors plots
plt.style.use('seaborn-v0_8-deep')

#Results from the three seeded runs
results = {
    'Seed 42': {
        'Backbone': 54.06,
        'Task 1': 34.20,
        'Task 2': 46.73,
        'Task 3': 54.03
    },
    'Seed 123': {
        'Backbone': 54.06,
        'Task 1': 34.20,
        'Task 2': 46.73,
        'Task 3': 54.03
    },
    'Seed 456': {
        'Backbone': 54.06,
        'Task 1': 37.69,
        'Task 2': 47.71,
        'Task 3': 53.16
    }
}

#Convert to DataFrame
df = pd.DataFrame(results).T

#Calculate mean and std across seeds
stages = ['Backbone', 'Task 1', 'Task 2', 'Task 3']
means = []
stds = []

for stage in stages:
    values = df[stage].values
    means.append(np.mean(values))
    stds.append(np.std(values))

#Create the plot
plt.figure(figsize=(10, 6))

#Plot mean line with std band (matching dog behaviors style)
plt.plot(range(len(stages)), means, 'o-', linewidth=2.5, markersize=8, 
         color='darkblue', label='Mean Overall Accuracy')

#Plot std band
plt.fill_between(range(len(stages)), 
                 np.array(means) - np.array(stds), 
                 np.array(means) + np.array(stds), 
                 alpha=0.3, color='darkblue', label='1 Std')

#Customize the plot
plt.title('Overall Accuracy Progression: Backbone  CIL Tasks (Multi-Seed)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Training Stage', fontsize=12, fontweight='bold')
plt.ylabel('Overall Accuracy (%)', fontsize=12, fontweight='bold')

#Set x-axis labels
plt.xticks(range(len(stages)), stages, rotation=45)

#Add grid and customize legend
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

#Set y-axis limits
plt.ylim(30, 60)

#Add value annotations (mean values)
for i, (mean, std) in enumerate(zip(means, stds)):
    plt.annotate(f'{mean:.1f}%', 
                xy=(i, mean), 
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

#Tight layout
plt.tight_layout()

#Save the plot
output_path = 'plots_and_metrics/seeded_results_plot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

#Also save as PDF
output_path_pdf = 'plots_and_metrics/seeded_results_plot.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"Plot also saved as PDF: {output_path_pdf}")

#Show the plot
plt.show()

#Print summary statistics
print("\n" + "="*60)
print("SEEDED RESULTS SUMMARY")
print("="*60)
print(f"{'Stage':<12} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
print("-"*60)
for i, stage in enumerate(stages):
    values = df[stage].values
    print(f"{stage:<12} {means[i]:<8.2f} {stds[i]:<8.2f} {np.min(values):<8.2f} {np.max(values):<8.2f}")

print(f"\nReproducibility: {len(set([tuple(df.loc[seed].values) for seed in df.index]))} unique result patterns")
print("(Lower is better for reproducibility)")
