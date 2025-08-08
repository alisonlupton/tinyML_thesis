import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import os
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from thop import profile, clever_format

class TinyMLMetrics:
    """
    Comprehensive TinyML metrics analyzer for BNN with BRN models.
    Tracks memory, parameters, FLOPS, latency, and other relevant metrics.
    """
    
    def __init__(self, save_path="logs/tinyml_metrics.csv"):
        self.save_path = save_path
        self.metrics_history = []
        self.model_info = {}
        
    def analyze_model(self, model, input_shape=(1, 3, 32, 32), device='cpu'):
        """Analyze model architecture and compute comprehensive metrics"""
        print("=== TinyML Model Analysis ===")
        
        # Basic model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Count different layer types
        layer_counts = defaultdict(int)
        brn_layers = 0
        binary_layers = 0
        conv_layers = 0
        linear_layers = 0
        
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                conv_layers += 1
                layer_counts['Conv2d'] += 1
            elif isinstance(module, nn.Linear):
                linear_layers += 1
                layer_counts['Linear'] += 1
            elif hasattr(module, '__class__') and 'BatchRenorm' in module.__class__.__name__:
                brn_layers += 1
                layer_counts['BatchRenorm'] += 1
            elif hasattr(module, '__class__') and 'Binary' in module.__class__.__name__:
                binary_layers += 1
                layer_counts['Binary'] += 1
        
        # Memory analysis
        model_size_mb = self._get_model_size_mb(model)
        process_memory_mb = self._get_memory_usage_mb()
        model_memory_mb = self._get_model_memory_mb(model)
        
        # FLOPS and MACs analysis
        input_tensor = torch.randn(input_shape).to(device)
        try:
            flops, params_count = profile(model, inputs=(input_tensor,), verbose=False)
            flops, params_count = clever_format([flops, params_count], "%.3f")
        except Exception as e:
            print(f"Warning: Could not compute FLOPS with thop: {e}")
            # Manual FLOPS estimation as fallback
            flops = self._estimate_flops_manual(model, input_shape)
            params_count = f"{total_params:,}"
        
        # Latency measurement
        latency_ms = self._measure_latency(model, input_tensor, device)
        
        # BNN-specific metrics
        bnn_metrics = self._analyze_bnn_specific(model)
        
        # SRAM and FLASH estimates
        sram_estimate = self._estimate_sram_usage(model, input_shape)
        flash_estimate = self._estimate_flash_usage(model)
        
        # Compile metrics
        metrics = {
            'timestamp': time.time(),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'process_memory_mb': process_memory_mb,
            'model_memory_mb': model_memory_mb,
            'flops': flops,
            'latency_ms': latency_ms,
            'conv_layers': conv_layers,
            'linear_layers': linear_layers,
            'brn_layers': brn_layers,
            'binary_layers': binary_layers,
            'sram_estimate_kb': sram_estimate,
            'flash_estimate_kb': flash_estimate,
            **bnn_metrics
        }
        
        self.model_info = metrics
        self._print_analysis(metrics)
        
        return metrics
    
    def _get_model_size_mb(self, model):
        """Calculate model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def _get_memory_usage_mb(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb
    
    def _get_model_memory_mb(self, model):
        """Get model-specific memory usage in MB"""
        # Calculate model parameters memory
        param_memory = 0
        for param in model.parameters():
            param_memory += param.nelement() * param.element_size()
        
        # Calculate model buffers memory (like BatchNorm running stats)
        buffer_memory = 0
        for buffer in model.buffers():
            buffer_memory += buffer.nelement() * buffer.element_size()
        
        # Calculate gradients memory (if requires_grad)
        grad_memory = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_memory += param.grad.nelement() * param.grad.element_size()
        
        total_model_memory = (param_memory + buffer_memory + grad_memory) / 1024 / 1024
        return total_model_memory
    
    def _measure_latency(self, model, input_tensor, device, num_runs=100):
        """Measure inference latency"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Measure latency
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(input_tensor)
                if device == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return np.mean(times)
    
    def _analyze_bnn_specific(self, model):
        """Analyze BNN-specific metrics"""
        binary_ops = 0
        binary_params = 0
        brn_params = 0
        
        for name, module in model.named_modules():
            if hasattr(module, '__class__') and 'Binary' in module.__class__.__name__:
                if hasattr(module, 'real_weights'):
                    binary_params += module.real_weights.numel()
                    # Estimate binary operations (rough approximation)
                    if hasattr(module, 'stride') and hasattr(module, 'padding'):
                        binary_ops += module.real_weights.numel() * 2  # Binary conv operations
            
            elif hasattr(module, '__class__') and 'BatchRenorm' in module.__class__.__name__:
                brn_params += sum(p.numel() for p in module.parameters())
        
        return {
            'binary_operations': binary_ops,
            'binary_parameters': binary_params,
            'brn_parameters': brn_params,
            'binary_ratio': binary_params / sum(p.numel() for p in model.parameters()) if sum(p.numel() for p in model.parameters()) > 0 else 0
        }
    
    def _estimate_sram_usage(self, model, input_shape):
        """Estimate SRAM usage for embedded deployment"""
        # Rough estimation based on model parameters and activations
        total_params = sum(p.numel() for p in model.parameters())
        
        # Estimate activation memory (rough approximation)
        batch_size, channels, height, width = input_shape
        activation_memory = batch_size * channels * height * width * 4  # 4 bytes per float
        
        # Model parameters in SRAM (assuming 4 bytes per parameter)
        param_memory = total_params * 4
        
        # Buffer for intermediate computations
        buffer_memory = activation_memory * 2  # Rough estimate
        
        total_sram_kb = (param_memory + activation_memory + buffer_memory) / 1024
        return total_sram_kb
    
    def _estimate_flops_manual(self, model, input_shape):
        """Manual FLOPS estimation for models that don't work with thop"""
        batch_size, channels, height, width = input_shape
        total_flops = 0
        
        # Estimate FLOPS for each layer type
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                # Conv2d FLOPS = H_out * W_out * (C_in * K * K * C_out)
                h_out = height // module.stride[0] if isinstance(module.stride, tuple) else height // module.stride
                w_out = width // module.stride[1] if isinstance(module.stride, tuple) else width // module.stride
                k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                total_flops += h_out * w_out * (channels * k * k * module.out_channels)
                height, width = h_out, w_out
                channels = module.out_channels
            elif isinstance(module, nn.Linear):
                # Linear FLOPS = input_features * output_features
                total_flops += module.in_features * module.out_features
        
        # Format FLOPS
        if total_flops > 1e9:
            return f"{total_flops/1e9:.2f}G"
        elif total_flops > 1e6:
            return f"{total_flops/1e6:.2f}M"
        elif total_flops > 1e3:
            return f"{total_flops/1e3:.2f}K"
        else:
            return f"{total_flops:.0f}"
    
    def _estimate_flash_usage(self, model):
        """Estimate FLASH memory usage for model storage"""
        total_params = sum(p.numel() for p in model.parameters())
        
        # Assume 4 bytes per parameter for storage
        flash_kb = (total_params * 4) / 1024
        return flash_kb
    
    def _print_analysis(self, metrics):
        """Print comprehensive analysis"""
        print(f"\n📊 MODEL ARCHITECTURE:")
        print(f"   Total Parameters: {metrics['total_parameters']:,}")
        print(f"   Trainable Parameters: {metrics['trainable_parameters']:,}")
        print(f"   Model Size: {metrics['model_size_mb']:.2f} MB")
        print(f"   Model Memory (params+buffers+grads): {metrics['model_memory_mb']:.2f} MB")
        print(f"   Process Memory (total Python): {metrics['process_memory_mb']:.2f} MB")
        
        print(f"\n🔧 LAYER BREAKDOWN:")
        print(f"   Conv2d Layers: {metrics['conv_layers']}")
        print(f"   Linear Layers: {metrics['linear_layers']}")
        print(f"   BatchRenorm Layers: {metrics['brn_layers']}")
        print(f"   Binary Layers: {metrics['binary_layers']}")
        
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   FLOPS: {metrics['flops']}")
        print(f"   Latency: {metrics['latency_ms']:.2f} ms")
        print(f"   Throughput: {1000/metrics['latency_ms']:.1f} inferences/sec")
        
        print(f"\n🧠 BNN-SPECIFIC METRICS:")
        print(f"   Binary Operations: {metrics['binary_operations']:,}")
        print(f"   Binary Parameters: {metrics['binary_parameters']:,}")
        print(f"   BRN Parameters: {metrics['brn_parameters']:,}")
        print(f"   Binary Ratio: {metrics['binary_ratio']:.2%}")
        
        print(f"\n💾 EMBEDDED MEMORY ESTIMATES:")
        print(f"   SRAM Usage: {metrics['sram_estimate_kb']:.1f} KB")
        print(f"   FLASH Usage: {metrics['flash_estimate_kb']:.1f} KB")
        
        # TinyML suitability assessment
        print(f"\n🎯 TINYML SUITABILITY:")
        sram_suitable = metrics['sram_estimate_kb'] < 1024  # 1MB threshold
        flash_suitable = metrics['flash_estimate_kb'] < 2048  # 2MB threshold
        latency_suitable = metrics['latency_ms'] < 100  # 100ms threshold
        
        print(f"   SRAM: {'✅ Suitable' if sram_suitable else '❌ Too Large'}")
        print(f"   FLASH: {'✅ Suitable' if flash_suitable else '❌ Too Large'}")
        print(f"   Latency: {'✅ Suitable' if latency_suitable else '❌ Too Slow'}")
        
        overall_suitable = sram_suitable and flash_suitable and latency_suitable
        print(f"   Overall: {'✅ TinyML Ready' if overall_suitable else '❌ Needs Optimization'}")
    
    def track_training_metrics(self, task_id, epoch, step, model, input_shape=(1, 3, 32, 32), device='cpu'):
        """Track metrics during training"""
        metrics = self.analyze_model(model, input_shape, device)
        metrics.update({
            'task_id': task_id,
            'epoch': epoch,
            'step': step,
            'phase': 'training'
        })
        
        self.metrics_history.append(metrics)
        
        # Print summary for first few steps
        if step < 5:
            print(f"\n📈 TRAINING METRICS - Task {task_id}, Epoch {epoch}, Step {step}:")
            print(f"   Model Memory: {metrics['model_memory_mb']:.1f} MB")
            print(f"   Process Memory: {metrics['process_memory_mb']:.1f} MB")
            print(f"   Latency: {metrics['latency_ms']:.2f} ms")
            print(f"   SRAM: {metrics['sram_estimate_kb']:.1f} KB")
    
    def save_metrics(self):
        """Save all collected metrics to CSV"""
        if self.metrics_history:
            df = pd.DataFrame(self.metrics_history)
            df.to_csv(self.save_path, index=False)
            print(f"\n💾 TinyML metrics saved to {self.save_path}")
            
            # Print summary statistics
            print(f"\n📊 TINYML METRICS SUMMARY:")
            print(f"   Total measurements: {len(df)}")
            print(f"   Average latency: {df['latency_ms'].mean():.2f} ms")
            print(f"   Average model memory: {df['model_memory_mb'].mean():.1f} MB")
            print(f"   Average process memory: {df['process_memory_mb'].mean():.1f} MB")
            print(f"   Average SRAM: {df['sram_estimate_kb'].mean():.1f} KB")
            print(f"   Average FLASH: {df['flash_estimate_kb'].mean():.1f} KB")
            
            return df
        return None
    
    def plot_metrics(self, df=None):
        """Create visualizations of TinyML metrics"""
        if df is None:
            df = pd.read_csv(self.save_path)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Memory usage over time
        axes[0, 0].plot(df['step'], df['model_memory_mb'], label='Model Memory')
        axes[0, 0].plot(df['step'], df['process_memory_mb'], label='Process Memory')
        axes[0, 0].set_title('Memory Usage Over Time')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Memory (MB)')
        axes[0, 0].legend()
        
        # Latency over time
        axes[0, 1].plot(df['step'], df['latency_ms'])
        axes[0, 1].set_title('Latency Over Time')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Latency (ms)')
        
        # SRAM usage over time
        axes[0, 2].plot(df['step'], df['sram_estimate_kb'])
        axes[0, 2].set_title('SRAM Usage Over Time')
        axes[0, 2].set_xlabel('Training Step')
        axes[0, 2].set_ylabel('SRAM (KB)')
        
        # FLASH usage over time
        axes[1, 0].plot(df['step'], df['flash_estimate_kb'])
        axes[1, 0].set_title('FLASH Usage Over Time')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('FLASH (KB)')
        
        # Binary ratio over time
        axes[1, 1].plot(df['step'], df['binary_ratio'])
        axes[1, 1].set_title('Binary Parameter Ratio')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Binary Ratio')
        
        # Parameter count over time
        axes[1, 2].plot(df['step'], df['total_parameters'])
        axes[1, 2].set_title('Total Parameters')
        axes[1, 2].set_xlabel('Training Step')
        axes[1, 2].set_ylabel('Parameter Count')
        
        plt.tight_layout()
        plt.savefig('logs/tinyml_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()

# Usage example:
# tinyml_analyzer = TinyMLMetrics()
# tinyml_analyzer.analyze_model(model, input_shape=(1, 3, 32, 32), device='cpu')
# tinyml_analyzer.track_training_metrics(task_id, epoch, step, model)
# tinyml_analyzer.save_metrics()
# tinyml_analyzer.plot_metrics() 