#!/usr/bin/env python3
"""
Script to compute and analyze model parameters for dog_behaviours models.
"""

import torch
import torch.nn as nn
import sys
import os

#Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from models.cnn import SimplifiedTDMModelCNN
from models.cnn_head import SimplifiedTDMHead, ClassRegistry

def count_parameters(model, name="Model"):
    """Count trainable and total parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{name}:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    return total_params, trainable_params

def analyze_backbone_params(backbone, name="Backbone"):
    """Analyze backbone parameters by layer."""
    print(f"\n{name} Layer Analysis:")
    total_params = 0
    
    for i, layer in enumerate(backbone):
        layer_params = sum(p.numel() for p in layer.parameters())
        total_params += layer_params
        print(f"  Layer {i}: {layer.__class__.__name__} - {layer_params:,} parameters")
        
        #Show layer details for Conv1d layers
        if isinstance(layer, nn.Conv1d):
            print(f"    Input channels: {layer.in_channels}")
            print(f"    Output channels: {layer.out_channels}")
            print(f"    Kernel size: {layer.kernel_size}")
            print(f"    Stride: {layer.stride}")
            print(f"    Padding: {layer.padding}")
    
    print(f"  Total backbone parameters: {total_params:,}")
    return total_params

def analyze_head_params(head, name="Head"):
    """Analyze head parameters."""
    print(f"\n{name} Analysis:")
    
    #Linear layer parameters
    linear_params = sum(p.numel() for p in head.linear.parameters())
    print(f"  Linear layer parameters: {linear_params:,}")
    print(f"    Weight shape: {head.linear.weight.shape}")
    print(f"    Bias shape: {head.linear.bias.shape}")
    print(f"    Input dimension: {head.in_dim}")
    print(f"    Output dimension: {head.out_dim}")
    
    #TDM/CWI parameters (these are not trainable but stored)
    mask_params = head.mask.numel()
    cwi_params = head.cwi.numel()
    cwi_curr_params = head.cwi_curr.numel()
    cwi_mem_params = head.cwi_mem.numel()
    
    print(f"  TDM/CWI storage (non-trainable):")
    print(f"    Mask parameters: {mask_params:,}")
    print(f"    CWI parameters: {cwi_params:,}")
    print(f"    CWI_curr parameters: {cwi_curr_params:,}")
    print(f"    CWI_mem parameters: {cwi_mem_params:,}")
    print(f"    Total TDM/CWI storage: {mask_params + cwi_params + cwi_curr_params + cwi_mem_params:,}")
    
    return linear_params

def create_sample_models():
    """Create sample models with different configurations."""
    device = torch.device('cpu')
    
    #Model configurations
    configs = [
        {"in_channels": 12, "init_num_classes": 4, "sparsity_ratio": 0.3, "name": "Backbone (4 classes)"},
        {"in_channels": 12, "init_num_classes": 7, "sparsity_ratio": 0.3, "name": "Full Model (7 classes)"},
    ]
    
    models = {}
    
    for config in configs:
        name = config["name"]
        model = SimplifiedTDMModelCNN(
            in_channels=config["in_channels"],
            init_num_classes=config["init_num_classes"],
            device=device,
            sparsity_ratio=config["sparsity_ratio"]
        )
        models[name] = model
    
    return models

def analyze_expansion_scenario():
    """Analyze parameter growth during head expansion."""
    print("\n" + "="*60)
    print("HEAD EXPANSION ANALYSIS")
    print("="*60)
    
    device = torch.device('cpu')
    
    #Start with backbone model (4 classes)
    model = SimplifiedTDMModelCNN(
        in_channels=12,
        init_num_classes=4,
        device=device,
        sparsity_ratio=0.3
    )
    
    #Create registry
    all_behaviors = ['Standing', 'Walking', 'Sitting', 'Lying Chest', 'Sniffing', 'Trotting', 'Galloping']
    registry = ClassRegistry(all_behaviors)
    backbone_behaviors = ['Standing', 'Walking', 'Sitting', 'Lying Chest']
    registry.add_classes(backbone_behaviors)
    
    print("Initial model (4 backbone classes):")
    total_params, trainable_params = count_parameters(model, "Initial Model")
    analyze_backbone_params(model.backbone, "Backbone")
    analyze_head_params(model.head, "Head")
    
    #Expand to add new classes
    new_behaviors = ['Sniffing', 'Trotting', 'Galloping']
    print(f"\nExpanding head to add: {new_behaviors}")
    
    model.expand_head(new_behaviors, registry)
    
    print("\nAfter expansion (7 total classes):")
    total_params_expanded, trainable_params_expanded = count_parameters(model, "Expanded Model")
    analyze_head_params(model.head, "Expanded Head")
    
    print(f"\nParameter Growth:")
    print(f"  Total parameter increase: {total_params_expanded - total_params:,}")
    print(f"  Trainable parameter increase: {trainable_params_expanded - trainable_params:,}")
    print(f"  Growth factor: {total_params_expanded / total_params:.2f}x")

def main():
    """Main function to analyze model parameters."""
    print("="*60)
    print("DOG BEHAVIOURS MODEL PARAMETER ANALYSIS")
    print("="*60)
    
    #Create and analyze sample models
    models = create_sample_models()
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 40)
        total_params, trainable_params = count_parameters(model, name)
        analyze_backbone_params(model.backbone, "Backbone")
        analyze_head_params(model.head, "Head")
    
    #Analyze expansion scenario
    analyze_expansion_scenario()
    
    #Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    
    backbone_model = models["Backbone (4 classes)"]
    full_model = models["Full Model (7 classes)"]
    
    backbone_total = sum(p.numel() for p in backbone_model.parameters())
    full_total = sum(p.numel() for p in full_model.parameters())
    
    print(f"Backbone model (4 classes): {backbone_total:,} parameters")
    print(f"Full model (7 classes): {full_total:,} parameters")
    print(f"Difference: {full_total - backbone_total:,} parameters")
    print(f"Head expansion adds: {full_total - backbone_total:,} parameters")
    
    #Calculate parameter breakdown
    backbone_only = sum(p.numel() for p in backbone_model.backbone.parameters())
    backbone_gap = sum(p.numel() for p in backbone_model.gap.parameters())
    head_4_classes = sum(p.numel() for p in backbone_model.head.linear.parameters())
    head_7_classes = sum(p.numel() for p in full_model.head.linear.parameters())
    
    print(f"\nDetailed Breakdown:")
    print(f"  Backbone (Conv layers): {backbone_only:,} parameters")
    print(f"  GAP layer: {backbone_gap:,} parameters")
    print(f"  Head (4 classes): {head_4_classes:,} parameters")
    print(f"  Head (7 classes): {head_7_classes:,} parameters")
    print(f"  Head expansion: {head_7_classes - head_4_classes:,} parameters")

if __name__ == "__main__":
    main()
