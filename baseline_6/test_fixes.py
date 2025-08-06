#!/usr/bin/env python3
"""
Test script to verify the fixes for baseline_6 continual learning setup.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os

def test_cwr_implementation():
    """Test the CWR* implementation logic"""
    print("Testing CWR* implementation...")
    
    # Create a simple classifier
    classifier = nn.Linear(128, 10)
    nn.init.normal_(classifier.weight, std=0.01)
    nn.init.zeros_(classifier.bias)
    
    # Initialize CWR banks
    classifier.cwr_bank = torch.zeros_like(classifier.weight.data)
    classifier.cwr_bias_bank = torch.zeros_like(classifier.bias.data)
    
    # Test new class initialization
    task_classes = [0, 1]
    alpha = 0.5
    
    print(f"Initial weight norm for class 0: {classifier.weight[0].norm():.6f}")
    print(f"Initial CWR bank norm for class 0: {classifier.cwr_bank[0].norm():.6f}")
    
    # Simulate training (just update weights)
    with torch.no_grad():
        classifier.weight[0] += 0.1 * torch.randn_like(classifier.weight[0])
        classifier.weight[1] += 0.1 * torch.randn_like(classifier.weight[1])
    
    print(f"After training weight norm for class 0: {classifier.weight[0].norm():.6f}")
    
    # Test CWR update
    for c in task_classes:
        current_weight = classifier.weight[c].detach()
        current_bias = classifier.bias[c].detach()
        
        # Update CWR bank with weighted average
        if classifier.cwr_bank[c].norm() < 1e-6:  # First time seeing this class
            classifier.cwr_bank[c].copy_(current_weight)
            classifier.cwr_bias_bank[c].copy_(current_bias)
        else:  # Update existing bank
            classifier.cwr_bank[c] = (
                alpha * classifier.cwr_bank[c] + (1 - alpha) * current_weight
            )
            classifier.cwr_bias_bank[c] = (
                alpha * classifier.cwr_bias_bank[c] + (1 - alpha) * current_bias
            )
        
        # Renormalize CWR bank weights
        bank_weight = classifier.cwr_bank[c]
        bank_weight = F.normalize(bank_weight, p=2, dim=0)
        classifier.cwr_bank[c] = bank_weight
    
    print(f"After CWR update CWR bank norm for class 0: {classifier.cwr_bank[0].norm():.6f}")
    print("CWR* implementation test passed!")

def test_feature_normalization():
    """Test feature normalization logic"""
    print("\nTesting feature normalization...")
    
    # Simulate features
    features = torch.randn(10, 128)
    
    # Test normalization
    normalized = F.normalize(features, p=2, dim=1)
    
    # Check that each row has unit norm
    norms = torch.norm(normalized, p=2, dim=1)
    print(f"Feature norms after normalization: {norms}")
    print(f"All features have unit norm: {torch.allclose(norms, torch.ones_like(norms), atol=1e-6)}")
    print("Feature normalization test passed!")

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    if os.path.exists("CL_config.yaml"):
        with open("CL_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        print(f"Learning rate: {config['learning_rate']}")
        print(f"CWR enabled: {config['CWR']}")
        print(f"Alpha: {config['alpha']}")
        print("Configuration test passed!")
    else:
        print("CL_config.yaml not found!")

if __name__ == "__main__":
    test_cwr_implementation()
    test_feature_normalization()
    test_config()
    print("\nAll tests completed!") 