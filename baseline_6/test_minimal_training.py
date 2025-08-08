#!/usr/bin/env python3
"""
Minimal training test to check if CWR* consolidation is being called
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add the current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from cwr_head_fixed import CWRHeadFixed

def test_minimal_training():
    """Test minimal training loop to see if CWR* is called"""
    
    print("🔍 Testing Minimal Training Loop")
    print("=" * 40)
    
    # Setup
    device = torch.device("cpu")
    in_features = 256
    num_classes = 10
    
    # Create CWR head
    classifier = CWRHeadFixed(in_features, num_classes, device)
    
    # Simulate config
    config = {'CWR': True, 'num_classes': 10}
    
    print(f"✅ CWR enabled: {config['CWR']}")
    
    # Simulate Task 3 scenario
    present_classes = [6, 7]
    present_idx = torch.tensor(present_classes, device=device, dtype=torch.long)
    
    print(f"Present classes: {present_classes}")
    print(f"Present idx: {present_idx.tolist()}")
    
    # Simulate a few training steps
    for step in range(5):
        print(f"\n--- Step {step} ---")
        
        # Simulate batch data
        batch_size = 32
        y_all = torch.randint(6, 8, (batch_size,))  # Only classes 6,7
        
        # Count classes in batch
        binc = torch.bincount(y_all, minlength=config["num_classes"])
        counts_present = binc[present_idx].float()
        
        print(f"Batch counts: {binc.tolist()}")
        print(f"Counts present: {counts_present.tolist()}")
        
        # Check if CWR is enabled
        if config['CWR']:
            print(f"[DEBUG] CWR enabled, calling consolidate for step {step}")
            print(f"[DEBUG] present_classes: {present_classes}")
            print(f"[DEBUG] counts_present: {counts_present.tolist()}")
            
            # Simulate some training (update tw)
            with torch.no_grad():
                classifier.weight[6] += torch.randn_like(classifier.weight[6]) * 0.01
                classifier.weight[7] += torch.randn_like(classifier.weight[7]) * 0.01
            
            print(f"tw before consolidation: {classifier.weight[present_idx].mean().item():.6f}")
            print(f"cw before consolidation: {classifier.cw[present_idx].mean().item():.6f}")
            
            # Call consolidation
            classifier.consolidate(present_classes, counts_present)
            
            print(f"[DEBUG] Consolidation completed for step {step}")
            print(f"tw after consolidation: {classifier.weight[present_idx].mean().item():.6f}")
            print(f"cw after consolidation: {classifier.cw[present_idx].mean().item():.6f}")
            
            # Check difference
            diff = (classifier.cw[present_idx] - classifier.weight[present_idx]).abs().mean().item()
            print(f"|cw - tw|: {diff:.6f}")
            
            if diff < 1e-6:
                print("❌ Consolidation failed!")
            else:
                print("✅ Consolidation worked!")
        else:
            print(f"[DEBUG] CWR disabled for step {step}")

if __name__ == "__main__":
    test_minimal_training()
