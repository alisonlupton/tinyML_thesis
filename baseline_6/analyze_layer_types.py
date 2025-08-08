import torch
import torch.nn as nn
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.quicknet_BNN_BRN import BinarizedQuickNetBRN

def analyze_layer_types(model):
    """Analyze the model to count different layer types"""
    print("🔍 Analyzing Layer Types in Model...")
    
    # Initialize counters
    layer_counts = {
        'fp32_conv': 0,
        'binary_conv': 0,
        'binary_depthwise_separable': 0,
        'fp32_linear': 0,
        'batch_renorm': 0,
        'binary_activation': 0,
        'fp32_activation': 0,
        'pooling': 0,
        'dropout': 0
    }
    
    # Track layer details
    layer_details = []
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        
        if isinstance(module, nn.Conv2d):
            layer_counts['fp32_conv'] += 1
            layer_details.append(f"{name}: {module_type} (FP32)")
        elif module_type == 'BinaryConv2d':
            layer_counts['binary_conv'] += 1
            layer_details.append(f"{name}: {module_type} (Binary)")
        elif module_type == 'BinaryDepthwiseSeparableConv':
            layer_counts['binary_depthwise_separable'] += 1
            layer_details.append(f"{name}: {module_type} (Binary)")
        elif isinstance(module, nn.Linear):
            layer_counts['fp32_linear'] += 1
            layer_details.append(f"{name}: {module_type} (FP32)")
        elif module_type == 'BatchRenorm2d':
            layer_counts['batch_renorm'] += 1
            layer_details.append(f"{name}: {module_type} (FP32)")
        elif module_type == 'BinaryActivation':
            layer_counts['binary_activation'] += 1
            layer_details.append(f"{name}: {module_type} (Binary)")
        elif isinstance(module, nn.ReLU):
            layer_counts['fp32_activation'] += 1
            layer_details.append(f"{name}: {module_type} (FP32)")
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            layer_counts['pooling'] += 1
            layer_details.append(f"{name}: {module_type} (FP32)")
        elif isinstance(module, nn.Dropout):
            layer_counts['dropout'] += 1
            layer_details.append(f"{name}: {module_type} (FP32)")
    
    return layer_counts, layer_details

def main():
    """Analyze the model layer types"""
    print("=== Layer Type Analysis ===")
    
    # Initialize model
    model = BinarizedQuickNetBRN(num_classes=2, first_layer_fp32=True)
    
    # Analyze layer types
    layer_counts, layer_details = analyze_layer_types(model)
    
    # Print results
    print(f"\n📊 LAYER TYPE BREAKDOWN:")
    print(f"   FP32 Convolutional Layers: {layer_counts['fp32_conv']}")
    print(f"   Binary Convolutional Layers: {layer_counts['binary_conv']}")
    print(f"   Binary Depthwise Separable Conv: {layer_counts['binary_depthwise_separable']}")
    print(f"   FP32 Linear Layers: {layer_counts['fp32_linear']}")
    print(f"   Batch Renormalization Layers: {layer_counts['batch_renorm']}")
    print(f"   Binary Activation Layers: {layer_counts['binary_activation']}")
    print(f"   FP32 Activation Layers: {layer_counts['fp32_activation']}")
    print(f"   Pooling Layers: {layer_counts['pooling']}")
    print(f"   Dropout Layers: {layer_counts['dropout']}")
    
    # Calculate totals
    total_fp32_layers = (layer_counts['fp32_conv'] + layer_counts['fp32_linear'] + 
                        layer_counts['batch_renorm'] + layer_counts['fp32_activation'] + 
                        layer_counts['pooling'] + layer_counts['dropout'])
    
    total_binary_layers = (layer_counts['binary_conv'] + layer_counts['binary_depthwise_separable'] + 
                          layer_counts['binary_activation'])
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Total FP32 Layers: {total_fp32_layers}")
    print(f"   Total Binary Layers: {total_binary_layers}")
    print(f"   Total Layers: {total_fp32_layers + total_binary_layers}")
    print(f"   Binary Ratio: {total_binary_layers / (total_fp32_layers + total_binary_layers):.1%}")
    
    print(f"\n📋 DETAILED LAYER LIST:")
    for detail in layer_details:
        print(f"   {detail}")

if __name__ == "__main__":
    main()
