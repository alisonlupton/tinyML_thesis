#!/usr/bin/env python3
"""
Debug script to test CWR* consolidation mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add the current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from cwr_head_fixed import CWRHeadFixed

def test_cwr_consolidation():
    """Test CWR* consolidation step by step"""
    
    print("🔍 Testing CWR* Consolidation Mechanism")
    print("=" * 50)
    
    # Setup
    device = torch.device("cpu")
    in_features = 256
    num_classes = 10
    
    # Create CWR head
    classifier = CWRHeadFixed(in_features, num_classes, device)
    
    print(f"✅ Created CWR head with {num_classes} classes, {in_features} features")
    print(f"Initial cw mean: {classifier.cw.mean().item():.6f}")
    print(f"Initial tw mean: {classifier.weight.mean().item():.6f}")
    print()
    
    # Simulate training for different tasks
    for task in range(3):
        print(f"🎯 Task {task}")
        print("-" * 30)
        
        # Simulate classes present in this task
        if task == 0:
            present_classes = [0, 1]  # First task: classes 0,1
        elif task == 1:
            present_classes = [2, 3]  # Second task: classes 2,3
        else:
            present_classes = [4, 5]  # Third task: classes 4,5
            
        print(f"Present classes: {present_classes}")
        
        # Simulate some training steps
        for step in range(3):
            print(f"  Step {step}:")
            
            # Preload consolidated weights for present classes
            classifier.preload_cw(present_classes)
            
            # Simulate training (update tw)
            with torch.enable_grad():
                # Create fake features and targets
                batch_size = 32
                features = torch.randn(batch_size, in_features, device=device)
                targets = torch.randint(0, len(present_classes), (batch_size,), device=device)
                
                # Forward pass
                logits = classifier(features)
                loss = F.cross_entropy(logits, targets)
                
                # Backward pass (this updates tw)
                loss.backward()
                
                # Simulate optimizer step
                with torch.no_grad():
                    classifier.weight -= 0.01 * classifier.weight.grad
                    classifier.weight.grad.zero_()
            
            print(f"    tw mean after training: {classifier.weight.mean().item():.6f}")
            print(f"    cw mean before consolidation: {classifier.cw.mean().item():.6f}")
            
            # Simulate batch counts
            counts_present = torch.tensor([16, 16], dtype=torch.float)  # Equal counts
            
            # Consolidate
            classifier.consolidate(present_classes, counts_present)
            
            print(f"    cw mean after consolidation: {classifier.cw.mean().item():.6f}")
            
            # Check if cw and tw are different for present classes
            idx = torch.tensor(present_classes, device=device, dtype=torch.long)
            cw_present = classifier.cw.index_select(0, idx)
            tw_present = classifier.weight.index_select(0, idx)
            
            diff = (cw_present - tw_present).abs().mean().item()
            print(f"    |cw - tw| for present classes: {diff:.6f}")
            
            if diff < 1e-6:
                print("    ⚠️  WARNING: cw and tw are identical!")
            else:
                print(f"    ✅ cw and tw are different (good!)")
            
            print()
        
        print(f"📊 After task {task}:")
        print(f"  hist_count: {classifier.hist_count.tolist()}")
        print(f"  cw norms: {classifier.cw.norm(dim=1).tolist()}")
        print(f"  tw norms: {classifier.weight.norm(dim=1).tolist()}")
        print()

def test_specific_consolidation_scenario():
    """Test the specific scenario that might be causing the issue"""
    
    print("🔬 Testing Specific Consolidation Scenario")
    print("=" * 50)
    
    device = torch.device("cpu")
    in_features = 256
    num_classes = 10
    
    classifier = CWRHeadFixed(in_features, num_classes, device)
    
    # Simulate the exact scenario from your training
    present_classes = [8, 9]  # Classes 8,9 (like in Task 4)
    counts_present = torch.tensor([40, 40], dtype=torch.float)  # Equal batch counts
    
    print(f"Present classes: {present_classes}")
    print(f"Counts present: {counts_present.tolist()}")
    
    # Check initial state
    idx = torch.tensor(present_classes, device=device, dtype=torch.long)
    past_j = classifier.hist_count.index_select(0, idx)
    print(f"Initial past_j: {past_j.tolist()}")
    
    # Simulate some training on these classes
    classifier.preload_cw(present_classes)
    
    # Set some arbitrary weights (simulating training)
    with torch.no_grad():
        classifier.weight[idx[0]] = torch.randn(in_features) * 0.1
        classifier.weight[idx[1]] = torch.randn(in_features) * 0.1
    
    print(f"tw before consolidation: {classifier.weight[idx].mean().item():.6f}")
    print(f"cw before consolidation: {classifier.cw[idx].mean().item():.6f}")
    
    # Consolidate
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

if __name__ == "__main__":
    test_cwr_consolidation()
    print("\n" + "="*60 + "\n")
    test_specific_consolidation_scenario()
