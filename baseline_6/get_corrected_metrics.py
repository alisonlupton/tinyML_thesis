import torch
import torch.nn as nn
import sys
import os
import pandas as pd

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tinyml_metrics import TinyMLMetrics
from models.quicknet_BNN_BRN import BinarizedQuickNetBRN

def analyze_layer_types(model):
    """Analyze the model to count different layer types"""
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
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        
        if isinstance(module, nn.Conv2d):
            layer_counts['fp32_conv'] += 1
        elif module_type == 'BinaryConv2d':
            layer_counts['binary_conv'] += 1
        elif module_type == 'BinaryDepthwiseSeparableConv':
            layer_counts['binary_depthwise_separable'] += 1
        elif isinstance(module, nn.Linear):
            layer_counts['fp32_linear'] += 1
        elif module_type == 'BatchRenorm2d':
            layer_counts['batch_renorm'] += 1
        elif module_type == 'BinaryActivation':
            layer_counts['binary_activation'] += 1
        elif isinstance(module, nn.ReLU):
            layer_counts['fp32_activation'] += 1
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            layer_counts['pooling'] += 1
        elif isinstance(module, nn.Dropout):
            layer_counts['dropout'] += 1
    
    return layer_counts

def analyze_and_save_corrected_metrics(model_path, model_class, num_classes=10):
    """Analyze a saved model and save corrected metrics"""
    print(f"\n=== Analyzing Model: {model_path} ===")
    
    # Initialize the model
    model = model_class(num_classes=num_classes)
    
    # Load the saved weights
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Get layer-based analysis
    layer_counts = analyze_layer_types(model)
    
    # Calculate correct totals
    total_fp32_layers = (layer_counts['fp32_conv'] + layer_counts['fp32_linear'] + 
                        layer_counts['batch_renorm'] + layer_counts['fp32_activation'] + 
                        layer_counts['pooling'] + layer_counts['dropout'])
    
    total_binary_layers = (layer_counts['binary_conv'] + layer_counts['binary_depthwise_separable'] + 
                          layer_counts['binary_activation'])
    
    total_layers = total_fp32_layers + total_binary_layers
    layer_binary_ratio = total_binary_layers / total_layers if total_layers > 0 else 0
    
    # Initialize TinyML analyzer for parameter-based metrics
    tinyml_analyzer = TinyMLMetrics(save_path=f"logs/{os.path.basename(model_path)}_tinyml_metrics_corrected.csv")
    
    # Analyze the model
    metrics = tinyml_analyzer.analyze_model(model, input_shape=(1, 3, 32, 32), device='cpu')
    
    # Add the metrics to history so it can be saved
    tinyml_analyzer.metrics_history.append(metrics)
    
    # Save the metrics
    tinyml_analyzer.save_metrics()
    
    # Create corrected summary with both parameter and layer-based metrics
    summary_data = {
        'model_name': [os.path.basename(model_path)],
        'total_parameters': [metrics['total_parameters']],
        'model_size_mb': [metrics['model_size_mb']],
        'flops': [metrics['flops']],
        'latency_ms': [metrics['latency_ms']],
        'sram_estimate_kb': [metrics['sram_estimate_kb']],
        'flash_estimate_kb': [metrics['flash_estimate_kb']],
        'binary_layers': [total_binary_layers],
        'fp32_layers': [total_fp32_layers],
        'total_layers': [total_layers],
        'layer_binary_ratio': [layer_binary_ratio],
        'parameter_binary_ratio': [metrics['binary_ratio']],
        'brn_layers': [layer_counts['batch_renorm']],
        'conv_layers': [layer_counts['fp32_conv'] + layer_counts['binary_conv']],
        'linear_layers': [layer_counts['fp32_linear']],
        'binary_conv_layers': [layer_counts['binary_conv']],
        'binary_depthwise_layers': [layer_counts['binary_depthwise_separable']],
        'binary_activation_layers': [layer_counts['binary_activation']]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = f"logs/{os.path.basename(model_path)}_summary_corrected.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"💾 Corrected summary saved to: {summary_path}")
    
    # Print the corrected analysis
    print(f"\n📊 CORRECTED LAYER ANALYSIS:")
    print(f"   Total Layers: {total_layers}")
    print(f"   FP32 Layers: {total_fp32_layers}")
    print(f"   Binary Layers: {total_binary_layers}")
    print(f"   Layer Binary Ratio: {layer_binary_ratio:.1%}")
    print(f"   Parameter Binary Ratio: {metrics['binary_ratio']:.1%}")
    print(f"   BRN Layers: {layer_counts['batch_renorm']}")
    print(f"   Binary Conv Layers: {layer_counts['binary_conv']}")
    print(f"   Binary Depthwise Layers: {layer_counts['binary_depthwise_separable']}")
    print(f"   Binary Activation Layers: {layer_counts['binary_activation']}")
    
    return metrics, layer_counts

def main():
    """Analyze all saved models and save corrected metrics"""
    print("🔍 Analyzing Model Parameters with Corrected Layer Analysis...")
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Define model paths and their corresponding classes
    models_to_analyze = [
        ("models/quicknet_bnn_brn_cifar10_2class_best.pth", BinarizedQuickNetBRN, 2),
    ]
    
    all_results = []
    
    for model_path, model_class, num_classes in models_to_analyze:
        if os.path.exists(model_path):
            result = analyze_and_save_corrected_metrics(model_path, model_class, num_classes)
            if result:
                all_results.append((model_path, result[0], result[1]))
        else:
            print(f"⚠️  Model file not found: {model_path}")
    
    # Print summary comparison
    if all_results:
        print("\n" + "="*60)
        print("📊 CORRECTED MODEL PARAMETERS SUMMARY")
        print("="*60)
        
        for model_path, metrics, layer_counts in all_results:
            total_fp32_layers = (layer_counts['fp32_conv'] + layer_counts['fp32_linear'] + 
                               layer_counts['batch_renorm'] + layer_counts['fp32_activation'] + 
                               layer_counts['pooling'] + layer_counts['dropout'])
            total_binary_layers = (layer_counts['binary_conv'] + layer_counts['binary_depthwise_separable'] + 
                                 layer_counts['binary_activation'])
            total_layers = total_fp32_layers + total_binary_layers
            layer_binary_ratio = total_binary_layers / total_layers if total_layers > 0 else 0
            
            print(f"\n📁 {os.path.basename(model_path)}:")
            print(f"   Total Parameters: {metrics['total_parameters']:,}")
            print(f"   Model Size: {metrics['model_size_mb']:.2f} MB")
            print(f"   FLOPS: {metrics['flops']}")
            print(f"   Latency: {metrics['latency_ms']:.2f} ms")
            print(f"   SRAM Estimate: {metrics['sram_estimate_kb']:.1f} KB")
            print(f"   FLASH Estimate: {metrics['flash_estimate_kb']:.1f} KB")
            print(f"   Total Layers: {total_layers}")
            print(f"   FP32 Layers: {total_fp32_layers}")
            print(f"   Binary Layers: {total_binary_layers}")
            print(f"   Layer Binary Ratio: {layer_binary_ratio:.1%}")
            print(f"   Parameter Binary Ratio: {metrics['binary_ratio']:.1%}")
    
    print(f"\n💾 All corrected results saved to logs/ directory")
    print(f"📁 Check these files:")
    for model_path, _, _ in all_results:
        base_name = os.path.basename(model_path)
        print(f"   - logs/{base_name}_tinyml_metrics_corrected.csv")
        print(f"   - logs/{base_name}_summary_corrected.csv")

if __name__ == "__main__":
    main()
