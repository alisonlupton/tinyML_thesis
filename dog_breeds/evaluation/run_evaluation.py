#!/usr/bin/env python3
"""
Run the complete evaluation pipeline:
1. Compute R matrix from saved artifacts
2. Calculate CL metrics from R matrix
"""

import subprocess
import sys
import os

def run_script(script_name):
    """Run a Python script and return success status."""
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(f"{script_name} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"{script_name} failed with error:")
        print(e.stderr)
        return False

def main():
    print(" Starting evaluation pipeline...")
    print("=" * 50)
    
    #Step 1: Compute R matrix
    print("\nStep 1: Computing R matrix from saved artifacts...")
    if not run_script("computing_R.py"):
        print("Failed to compute R matrix. Exiting.")
        return
    
    #Step 2: Calculate CL metrics
    print("\n Step 2: Calculating CL metrics from R matrix...")
    if not run_script("compute_cl_metrics.py"):
        print("Failed to calculate CL metrics. Exiting.")
        return
    
    print("\n Evaluation pipeline completed successfully!")
    print("=" * 50)
    print("Generated files:")
    print("- R_offline.npy (R matrix)")
    print("- cl_metrics_from_R.json (CL metrics)")
    
    #Show final results
    try:
        import json
        with open('cl_metrics_from_R.json', 'r') as f:
            metrics = json.load(f)
        
        print("\n Final CL Metrics Summary:")
        print("-" * 30)
        for name, value in metrics.items():
            print(f"{name:25}: {value:8.2f}%")
            
    except Exception as e:
        print(f"Could not display final metrics: {e}")

if __name__ == "__main__":
    main()
