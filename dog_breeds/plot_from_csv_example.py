#!/usr/bin/env python3
"""
Example script to generate plots from saved CSV data for dog breeds.
This allows for fast plot iteration without re-running training.

Usage:
    python plot_from_csv_example.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_config
from cnn_CIL_training_utils import plot_from_csv

def main():
    #Load config
    cfg = load_config()
    
    #Generate plots from CSV data
    print("Generating dog breed plots from CSV data...")
    plot_from_csv(cfg)
    print("Done! Check the plots_and_metrics/ directory for updated plots.")

if __name__ == "__main__":
    main()
