#!/usr/bin/env python3
"""
Debug CWR* consolidation for Task 3 scenario
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add the current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from cwr_head_fixed import CWRHeadFixed

def simulate_task3_cwr():
    """Simulate the exact Task 3 scenario to debug CWR*"""
    
    print("🔍 Debugging CWR* for Task 3 Scenario")
    print("=" * 50)
    
    # Setup
    device = torch.device("cpu")
    in_features = 256
    num_classes = 10
    
    # Create CWR head
    classifier = CWRHeadFixed(in_features, num_classes, device)
    
    print(f"✅ Created CWR head with {num_classes} classes, {in_features} features")
    
    # Simulate the progression through tasks 0, 1, 2 to get to task 3
    print("\n🎯 Simulating Task Progression:")
    print("-" * 30)
    
    # Task 0: Classes [0, 1]
    print("Task 0: Classes [0, 1]")
    present_classes = [0, 1]
    counts_present = torch.tensor([40, 40], dtype=torch.float)
    
    # Simulate some training and consolidation
    classifier.preload_cw(present_classes)
    with torch.no_grad():
        classifier.weight[0] = torch.randn(in_features) * 0.1
        classifier.weight[1] = torch.randn(in_features) * 0.1
    
    classifier.consolidate(present_classes, counts_present)
    print(f"  hist_count after Task 0: {classifier.hist_count.tolist()}")
    print(f"  cw norms after Task 0: {[classifier.cw[i].norm().item() for i in range(10)]}")
    
    # Task 1: Classes [2, 3]
    print("\nTask 1: Classes [2, 3]")
    present_classes = [2, 3]
    counts_present = torch.tensor([40, 40], dtype=torch.float)
    
    classifier.preload_cw(present_classes)
    with torch.no_grad():
        classifier.weight[2] = torch.randn(in_features) * 0.1
        classifier.weight[3] = torch.randn(in_features) * 0.1
    
    classifier.consolidate(present_classes, counts_present)
    print(f"  hist_count after Task 1: {classifier.hist_count.tolist()}")
    print(f"  cw norms after Task 1: {[classifier.cw[i].norm().item() for i in range(10)]}")
    
    # Task 2: Classes [4, 5]
    print("\nTask 2: Classes [4, 5]")
    present_classes = [4, 5]
    counts_present = torch.tensor([40, 40], dtype=torch.float)
    
    classifier.preload_cw(present_classes)
    with torch.no_grad():
        classifier.weight[4] = torch.randn(in_features) * 0.1
        classifier.weight[5] = torch.randn(in_features) * 0.1
    
    classifier.consolidate(present_classes, counts_present)
    print(f"  hist_count after Task 2: {classifier.hist_count.tolist()}")
    print(f"  cw norms after Task 2: {[classifier.cw[i].norm().item() for i in range(10)]}")
    
    # Now simulate Task 3: Classes [6, 7] - THE PROBLEMATIC TASK
    print("\n🎯 Task 3: Classes [6, 7] - THE PROBLEMATIC TASK")
    print("-" * 50)
    
    present_classes = [6, 7]
    counts_present = torch.tensor([40, 40], dtype=torch.float)
    
    print(f"Present classes: {present_classes}")
    print(f"Counts present: {counts_present.tolist()}")
    
    # Check state before Task 3
    idx = torch.tensor(present_classes, device=device, dtype=torch.long)
    past_j = classifier.hist_count.index_select(0, idx)
    print(f"past_j for classes [6,7]: {past_j.tolist()}")
    
    # Preload consolidated weights
    classifier.preload_cw(present_classes)
    print(f"tw after preload: {classifier.weight[idx].mean().item():.6f}")
    print(f"cw before consolidation: {classifier.cw[idx].mean().item():.6f}")
    
    # Set some training weights (simulating training)
    with torch.no_grad():
        classifier.weight[6] = torch.randn(in_features) * 0.1
        classifier.weight[7] = torch.randn(in_features) * 0.1
    
    print(f"tw after training simulation: {classifier.weight[idx].mean().item():.6f}")
    
    # NOW TEST CONSOLIDATION - THIS IS WHERE THE ISSUE MIGHT BE
    print("\n🔧 Testing Consolidation:")
    print("-" * 20)
    
    # Reset the debug flag to see consolidation prints
    if hasattr(classifier, '_dbg_printed2'):
        delattr(classifier, '_dbg_printed2')
    
    classifier.consolidate(present_classes, counts_present)
    
    print(f"tw after consolidation: {classifier.weight[idx].mean().item():.6f}")
    print(f"cw after consolidation: {classifier.cw[idx].mean().item():.6f}")
    
    # Check difference
    diff = (classifier.cw[idx] - classifier.weight[idx]).abs().mean().item()
    print(f"|cw - tw|: {diff:.6f}")
    
    if diff < 1e-6:
        print("❌ cw and tw are identical - consolidation failed!")
    else:
        print("✅ cw and tw are different - consolidation worked!")
    
    # Check hist_count update
    print(f"hist_count after Task 3: {classifier.hist_count.tolist()}")
    
    # Test multiple consolidation steps (like in real training)
    print("\n🔄 Testing Multiple Consolidation Steps:")
    print("-" * 35)
    
    for step in range(3):
        print(f"Step {step}:")
        
        # Simulate new training weights
        with torch.no_grad():
            classifier.weight[6] = torch.randn(in_features) * 0.1
            classifier.weight[7] = torch.randn(in_features) * 0.1
        
        print(f"  tw before consolidation: {classifier.weight[idx].mean().item():.6f}")
        print(f"  cw before consolidation: {classifier.cw[idx].mean().item():.6f}")
        
        # Consolidate
        classifier.consolidate(present_classes, counts_present)
        
        print(f"  tw after consolidation: {classifier.weight[idx].mean().item():.6f}")
        print(f"  cw after consolidation: {classifier.cw[idx].mean().item():.6f}")
        
        diff = (classifier.cw[idx] - classifier.weight[idx]).abs().mean().item()
        print(f"  |cw - tw|: {diff:.6f}")
        
        if diff < 1e-6:
            print("  ❌ Consolidation failed!")
        else:
            print("  ✅ Consolidation worked!")
        print()

def test_cwr_config():
    """Test if CWR configuration is correct"""
    
    print("🔧 Testing CWR Configuration")
    print("=" * 30)
    
    # Simulate your config
    config = {
        'CWR': True,
        'num_classes': 10
    }
    
    print(f"CWR enabled: {config['CWR']}")
    print(f"num_classes: {config['num_classes']}")
    
    # Test the consolidation logic
    present_classes = [6, 7]
    counts_present = torch.tensor([40, 40], dtype=torch.float)
    
    print(f"present_classes: {present_classes}")
    print(f"counts_present: {counts_present.tolist()}")
    
    # Test the consolidation call
    if config['CWR']:
        print("✅ CWR is enabled - consolidation should be called")
    else:
        print("❌ CWR is disabled - consolidation will be skipped")

if __name__ == "__main__":
    test_cwr_config()
    print("\n" + "="*60 + "\n")
    simulate_task3_cwr()
