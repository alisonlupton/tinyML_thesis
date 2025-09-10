#!/usr/bin/env python3
"""
Model Parameter Counter for Dog Breed CIL Models

This script calculates and displays the parameter counts for:
1. TunedMCUStudentCNN (backbone + projection + classifier)
2. TunedMCUStudentCNN_CIL (backbone + projection + TDM head)
3. Individual components breakdown
"""

import torch
import torch.nn as nn
import sys
import os

#Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from models.cnn import TunedMCUStudentCNN, TunedMCUStudentCNN_CIL
from models.cnn_head import SimplifiedTDMHead

def count_parameters(model, name="Model"):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{name}:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Size (FP32): {total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"  Size (INT8): {total_params / 1024 / 1024:.2f} MB")
    
    return total_params, trainable_params

def count_module_parameters(module, name="Module"):
    """Count parameters in a specific module."""
    total_params = sum(p.numel() for p in module.parameters())
    print(f"  {name}: {total_params:,} params")
    return total_params

def analyze_backbone(backbone, name="Backbone"):
    """Analyze backbone parameters by layer."""
    print(f"\n{name} Layer Analysis:")
    total_params = 0
    
    for i, layer in enumerate(backbone):
        layer_params = sum(p.numel() for p in layer.parameters())
        total_params += layer_params
        
        #Get layer info
        if isinstance(layer, nn.Conv2d):
            in_ch, out_ch, k_size = layer.in_channels, layer.out_channels, layer.kernel_size
            stride = layer.stride
            print(f"  Layer {i}: Conv2d({in_ch}{out_ch}, {k_size}, stride={stride}) = {layer_params:,} params")
        elif isinstance(layer, nn.BatchNorm2d):
            num_features = layer.num_features
            print(f"  Layer {i}: BatchNorm2d({num_features}) = {layer_params:,} params")
        elif isinstance(layer, nn.MaxPool2d):
            print(f"  Layer {i}: MaxPool2d = {layer_params:,} params")
        elif isinstance(layer, nn.ReLU):
            print(f"  Layer {i}: ReLU = {layer_params:,} params")
        else:
            print(f"  Layer {i}: {type(layer).__name__} = {layer_params:,} params")
    
    print(f"  {name} Total: {total_params:,} params")
    return total_params

def analyze_head(head, name="Head"):
    """Analyze head parameters."""
    print(f"\n{name} Analysis:")
    
    #Linear layer
    linear_params = sum(p.numel() for p in head.linear.parameters())
    print(f"  Linear layer: {linear_params:,} params")
    print(f"    Weight: {head.linear.weight.numel():,} params ({head.linear.weight.shape})")
    print(f"    Bias: {head.linear.bias.numel():,} params ({head.linear.bias.shape})")
    
    #TDM components (these are not trainable parameters, but memory)
    mask_params = head.mask.numel()
    cwi_params = head.cwi.numel()
    cwi_curr_params = head.cwi_curr.numel()
    cwi_mem_params = head.cwi_mem.numel()
    
    print(f"  TDM Components (memory only):")
    print(f"    Mask: {mask_params:,} elements")
    print(f"    CWI: {cwi_params:,} elements")
    print(f"    CWI_curr: {cwi_curr_params:,} elements")
    print(f"    CWI_mem: {cwi_mem_params:,} elements")
    print(f"    TDM Total Memory: {(mask_params + cwi_params + cwi_curr_params + cwi_mem_params) * 4 / 1024 / 1024:.2f} MB")
    
    return linear_params

def main():
    print("=" * 60)
    print("DOG BREED MODEL PARAMETER ANALYSIS")
    print("=" * 60)
    
    #Model configurations
    in_channels = 3
    feat_dim = 128
    img_size = 160
    backbone_classes = 10
    cil_init_classes = 10
    sparsity_ratio = 0.0  #No sparsity for parameter counting
    device = torch.device('cpu')
    
    print(f"Configuration:")
    print(f"  Input channels: {in_channels}")
    print(f"  Feature dimension: {feat_dim}")
    print(f"  Image size: {img_size}")
    print(f"  Backbone classes: {backbone_classes}")
    print(f"  CIL init classes: {cil_init_classes}")
    print(f"  Sparsity ratio: {sparsity_ratio}")
    
    #1. TunedMCUStudentCNN (Original Model)
    print("\n" + "=" * 60)
    print("1. TUNED MCU STUDENT CNN (Original Model)")
    print("=" * 60)
    
    original_model = TunedMCUStudentCNN(
        in_channels=in_channels,
        feat_dim=feat_dim,
        num_classes=backbone_classes,
        img_size=img_size
    )
    
    total_orig, trainable_orig = count_parameters(original_model, "TunedMCUStudentCNN")
    
    #Breakdown
    backbone_params = analyze_backbone(original_model.backbone, "Backbone")
    gap_params = sum(p.numel() for p in original_model.gap.parameters())
    proj_params = sum(p.numel() for p in original_model.proj.parameters())
    classifier_params = sum(p.numel() for p in original_model.classifier.parameters())
    
    print(f"\nComponent Breakdown:")
    count_module_parameters(original_model.gap, "GAP")
    count_module_parameters(original_model.proj, "Projection")
    count_module_parameters(original_model.classifier, "Classifier")
    
    #2. TunedMCUStudentCNN_CIL (CIL Model)
    print("\n" + "=" * 60)
    print("2. TUNED MCU STUDENT CNN CIL (CIL Model)")
    print("=" * 60)
    
    cil_model = TunedMCUStudentCNN_CIL(
        in_channels=in_channels,
        init_num_classes=cil_init_classes,
        device=device,
        sparsity_ratio=sparsity_ratio,
        feat_dim=feat_dim
    )
    
    total_cil, trainable_cil = count_parameters(cil_model, "TunedMCUStudentCNN_CIL")
    
    #Breakdown
    cil_backbone_params = analyze_backbone(cil_model.backbone, "Backbone")
    cil_gap_params = sum(p.numel() for p in cil_model.gap.parameters())
    cil_proj_params = sum(p.numel() for p in cil_model.proj.parameters())
    
    print(f"\nComponent Breakdown:")
    count_module_parameters(cil_model.gap, "GAP")
    count_module_parameters(cil_model.proj, "Projection")
    
    #Head analysis
    head_params = analyze_head(cil_model.head, "TDM Head")
    
    #3. Standalone Head Analysis
    print("\n" + "=" * 60)
    print("3. STANDALONE TDM HEAD")
    print("=" * 60)
    
    standalone_head = SimplifiedTDMHead(
        in_dim=feat_dim,
        init_num_classes=cil_init_classes,
        device=device,
        sparsity_ratio=sparsity_ratio
    )
    
    standalone_head_params = analyze_head(standalone_head, "Standalone TDM Head")
    
    #4. Summary Comparison
    print("\n" + "=" * 60)
    print("4. SUMMARY COMPARISON")
    print("=" * 60)
    
    print(f"Original Model (TunedMCUStudentCNN):")
    print(f"  Total: {total_orig:,} params ({total_orig * 4 / 1024 / 1024:.2f} MB)")
    
    print(f"\nCIL Model (TunedMCUStudentCNN_CIL):")
    print(f"  Total: {total_cil:,} params ({total_cil * 4 / 1024 / 1024:.2f} MB)")
    
    print(f"\nDifference (CIL - Original):")
    diff = total_cil - total_orig
    print(f"  Parameters: {diff:,} ({diff * 4 / 1024 / 1024:.2f} MB)")
    
    #5. CIL Head Expansion Analysis
    print("\n" + "=" * 60)
    print("5. CIL HEAD EXPANSION ANALYSIS")
    print("=" * 60)
    
    print("Head parameter growth with new classes:")
    for num_classes in [10, 11, 12, 13]:
        head = SimplifiedTDMHead(feat_dim, num_classes, device, sparsity_ratio)
        params = sum(p.numel() for p in head.parameters())
        print(f"  {num_classes} classes: {params:,} params ({params * 4 / 1024:.2f} KB)")
    
    #6. Memory Usage Analysis
    print("\n" + "=" * 60)
    print("6. MEMORY USAGE ANALYSIS")
    print("=" * 60)
    
    #Calculate memory for different scenarios
    scenarios = [
        ("Backbone only", backbone_params),
        ("Backbone + Proj", backbone_params + proj_params),
        ("Full Original Model", total_orig),
        ("Full CIL Model", total_cil),
    ]
    
    print("Memory usage (FP32):")
    for name, params in scenarios:
        mb = params * 4 / 1024 / 1024
        print(f"  {name}: {params:,} params ({mb:.2f} MB)")
    
    print("\nMemory usage (INT8 quantized):")
    for name, params in scenarios:
        mb = params / 1024 / 1024
        print(f"  {name}: {params:,} params ({mb:.2f} MB)")
    
    #7. Sparsity Analysis
    if sparsity_ratio > 0:
        print("\n" + "=" * 60)
        print("7. SPARSITY ANALYSIS")
        print("=" * 60)
        
        sparse_head = SimplifiedTDMHead(feat_dim, cil_init_classes, device, sparsity_ratio)
        total_head_params = sum(p.numel() for p in sparse_head.parameters())
        active_params = int(total_head_params * (1 - sparsity_ratio))
        pruned_params = total_head_params - active_params
        
        print(f"Sparsity ratio: {sparsity_ratio:.1%}")
        print(f"Total head parameters: {total_head_params:,}")
        print(f"Active parameters: {active_params:,}")
        print(f"Pruned parameters: {pruned_params:,}")
        print(f"Memory reduction: {pruned_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
