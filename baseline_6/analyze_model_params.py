import torch
import torch.nn as nn
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tinyml_metrics import TinyMLMetrics
from models.quicknet_BNN_BRN import BinarizedQuickNetBRN

def analyze_saved_model(model_path, model_class, num_classes=10):
    """Analyze a saved model and print its parameters"""
    print(f"\n=== Analyzing Model: {model_path} ===")
    
    # Initialize the model
    model = model_class(num_classes=num_classes)
    
    # Load the saved weights
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
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
    
    # Save the metrics
    tinyml_analyzer.save_metrics()
    
    return metrics

def main():
    """Analyze all saved models"""
    print("🔍 Analyzing Model Parameters...")
    
    # Define model paths and their corresponding classes
    models_to_analyze = [
        ("models/quicknet_bnn_brn_cifar10_2class_best.pth", BinarizedQuickNetBRN, 2),
        ("models/quicknet_bnn_cifar10_2class_best.pth", BinarizedQuickNetBRN, 2),
    ]
    
    all_metrics = []
    
    for model_path, model_class, num_classes in models_to_analyze:
        if os.path.exists(model_path):
            metrics = analyze_saved_model(model_path, model_class, num_classes)
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
    
    print(f"\n💾 Results saved to logs/ directory")

if __name__ == "__main__":
    main()
