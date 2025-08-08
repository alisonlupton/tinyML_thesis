import torch
import torch.nn as nn
import sys
import os
import pandas as pd

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tinyml_metrics import TinyMLMetrics
from models.quicknet_BNN_BRN import BinarizedQuickNetBRN

def analyze_and_save_model_metrics(model_path, model_class, num_classes=10):
    """Analyze a saved model and save its parameters to CSV"""
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
    
    # Initialize TinyML analyzer
    tinyml_analyzer = TinyMLMetrics(save_path=f"logs/{os.path.basename(model_path)}_tinyml_metrics.csv")
    
    # Analyze the model
    metrics = tinyml_analyzer.analyze_model(model, input_shape=(1, 3, 32, 32), device='cpu')
    
    # Add the metrics to history so it can be saved
    tinyml_analyzer.metrics_history.append(metrics)
    
    # Save the metrics
    tinyml_analyzer.save_metrics()
    
    # Also save a simple summary CSV
    summary_data = {
        'model_name': [os.path.basename(model_path)],
        'total_parameters': [metrics['total_parameters']],
        'model_size_mb': [metrics['model_size_mb']],
        'flops': [metrics['flops']],
        'latency_ms': [metrics['latency_ms']],
        'sram_estimate_kb': [metrics['sram_estimate_kb']],
        'flash_estimate_kb': [metrics['flash_estimate_kb']],
        'binary_layers': [metrics['binary_layers']],
        'brn_layers': [metrics['brn_layers']],
        'binary_ratio': [metrics['binary_ratio']],
        'conv_layers': [metrics['conv_layers']],
        'linear_layers': [metrics['linear_layers']]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = f"logs/{os.path.basename(model_path)}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"💾 Summary saved to: {summary_path}")
    
    return metrics

def main():
    """Analyze all saved models and save metrics"""
    print("🔍 Analyzing Model Parameters and Saving Metrics...")
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Define model paths and their corresponding classes
    models_to_analyze = [
        ("models/quicknet_bnn_brn_cifar10_2class_best.pth", BinarizedQuickNetBRN, 2),
    ]
    
    all_metrics = []
    
    for model_path, model_class, num_classes in models_to_analyze:
        if os.path.exists(model_path):
            metrics = analyze_and_save_model_metrics(model_path, model_class, num_classes)
            if metrics:
                all_metrics.append((model_path, metrics))
        else:
            print(f"⚠️  Model file not found: {model_path}")
    
    # Print summary comparison
    if all_metrics:
        print("\n" + "="*60)
        print("📊 MODEL PARAMETERS SUMMARY")
        print("="*60)
        
        for model_path, metrics in all_metrics:
            print(f"\n📁 {os.path.basename(model_path)}:")
            print(f"   Total Parameters: {metrics['total_parameters']:,}")
            print(f"   Model Size: {metrics['model_size_mb']:.2f} MB")
            print(f"   FLOPS: {metrics['flops']}")
            print(f"   Latency: {metrics['latency_ms']:.2f} ms")
            print(f"   SRAM Estimate: {metrics['sram_estimate_kb']:.1f} KB")
            print(f"   FLASH Estimate: {metrics['flash_estimate_kb']:.1f} KB")
            print(f"   Binary Layers: {metrics['binary_layers']}")
            print(f"   BRN Layers: {metrics['brn_layers']}")
    
    print(f"\n💾 All results saved to logs/ directory")
    print(f"📁 Check these files:")
    for model_path, _ in all_metrics:
        base_name = os.path.basename(model_path)
        print(f"   - logs/{base_name}_tinyml_metrics.csv")
        print(f"   - logs/{base_name}_summary.csv")

if __name__ == "__main__":
    main()
