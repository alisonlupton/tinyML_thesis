#!/usr/bin/env python3
"""
Test script to demonstrate the updated plotting functions with random data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def test_fixed13_overall_plot():
    """Test the updated fixed13_overall plot with random data"""
    print("Testing Fixed-13 Overall Accuracy Plot...")
    
    #Create random data similar to what would come from training
    stages = ["Task 1", "Task 2", "Task 3"]
    overall_accuracies = [85.2, 91.7, 88.9]  #Random but realistic accuracies
    
    #Create DataFrame
    df13_overall = pd.DataFrame({
        "Stage": stages,
        "Overall": overall_accuracies
    })
    
    #Create the plot (same code as in the updated function)
    plt.figure(figsize=(10,6))
    ax = sns.lineplot(data=df13_overall, x="Stage", y="Overall", marker="o", linewidth=3, markersize=10)
    plt.title("Fixed-13 Overall Accuracy", fontsize=15, fontweight='bold', pad=20)
    plt.ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    plt.xlabel("Stage", fontsize=12, fontweight='bold')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    #Add value annotations
    for i, (stage, acc) in enumerate(zip(df13_overall["Stage"], df13_overall["Overall"])):
        plt.annotate(f'{acc:.1f}%', 
                    xy=(i, acc), 
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("test_fixed13_overall.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(" Fixed-13 Overall plot saved as 'test_fixed13_overall.png'")

def test_fixed13_per_class_plot():
    """Test the updated fixed13_per_class plot with random data"""
    print("Testing Fixed-13 Per-Class Accuracy Plot...")
    
    #Create random data for multiple dog breeds
    dog_breeds = ["Golden Retriever", "German Shepherd", "Labrador", "Poodle", "Bulldog", "Beagle"]
    stages = ["Task 1", "Task 2", "Task 3"]
    
    #Generate random but realistic per-class accuracies
    np.random.seed(42)  #For reproducible results
    data = []
    
    for stage in stages:
        for breed in dog_breeds:
            #Generate realistic accuracy values (70-95%)
            accuracy = np.random.uniform(70, 95)
            data.append({
                "Stage": stage,
                "ClassName": breed,
                "Accuracy": accuracy
            })
    
    df13_pc = pd.DataFrame(data)
    
    #Create the plot (same code as in the updated function)
    plt.figure(figsize=(14, 8))
    
    #Plot per-class accuracy progression with legend
    ax = sns.lineplot(data=df13_pc, x="Stage", y="Accuracy", hue="ClassName", 
                     marker="o", linewidth=2.5, markersize=8)
    
    #Customize the plot
    plt.title("Fixed-13 Per-Class Accuracy Progression", 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Stage", fontsize=12, fontweight='bold')
    plt.ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    #Add legend with dog breed names
    plt.legend(title='Dog Breeds', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("test_fixed13_per_class.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(" Fixed-13 Per-Class plot saved as 'test_fixed13_per_class.png'")

def main():
    """Run both test plots"""
    print(" Testing Updated Dog Breeds Plotting Functions")
    print("=" * 50)
    
    #Set seaborn style to match existing plots
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    #Test both plots
    test_fixed13_overall_plot()
    print()
    test_fixed13_per_class_plot()
    
    print("\n" + "=" * 50)
    print(" Test completed! Check the generated PNG files to see your updated plots.")
    print("Files created:")
    print("  - test_fixed13_overall.png")
    print("  - test_fixed13_per_class.png")

if __name__ == "__main__":
    main()
