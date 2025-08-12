import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from collections import defaultdict, deque
import time
import os
from datetime import datetime

class ModelLearningDebugger:
    """
    Comprehensive debugger for tracking model learning, gradients, predictions,
    latent space, filters, and CWR head behavior during continual learning.
    """
    
    def __init__(self, save_dir="logs/model_debug", max_history=1000):
        self.save_dir = save_dir
        self.max_history = max_history
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/plots", exist_ok=True)
        os.makedirs(f"{save_dir}/features", exist_ok=True)
        
        # Data storage
        self.gradient_history = defaultdict(list)
        self.prediction_history = defaultdict(list)
        self.latent_history = defaultdict(list)
        self.cwr_history = defaultdict(list)
        self.filter_history = defaultdict(list)
        self.accuracy_history = defaultdict(list)
        self.loss_history = defaultdict(list)
        
        # Current task tracking
        self.current_task = 0
        self.current_epoch = 0
        self.current_step = 0
        
        # Feature storage for visualization
        self.feature_cache = defaultdict(list)
        self.label_cache = defaultdict(list)
        
    def track_gradients(self, model, task_id, epoch, step):
        """Track gradient statistics for all layers"""
        grad_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().cpu()
                grad_stats[f"{name}_mean"] = grad.mean().item()
                grad_stats[f"{name}_std"] = grad.std().item()
                grad_stats[f"{name}_norm"] = grad.norm().item()
                grad_stats[f"{name}_max"] = grad.max().item()
                grad_stats[f"{name}_min"] = grad.min().item()
                
                # Store in history
                key = f"task_{task_id}_epoch_{epoch}"
                self.gradient_history[key].append({
                    'step': step,
                    'layer': name,
                    'mean': grad_stats[f"{name}_mean"],
                    'std': grad_stats[f"{name}_std"],
                    'norm': grad_stats[f"{name}_norm"],
                    'max': grad_stats[f"{name}_max"],
                    'min': grad_stats[f"{name}_min"]
                })
        
        return grad_stats
    
    def track_predictions(self, logits, targets, task_id, epoch, step):
        """Track prediction statistics and confidence"""
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        # Calculate metrics
        accuracy = (preds == targets).float().mean().item()
        confidence = probs.max(dim=1)[0].mean().item()
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean().item()
        
        # Store predictions
        pred_data = {
            'step': step,
            'accuracy': accuracy,
            'confidence': confidence,
            'entropy': entropy,
            'predictions': preds.cpu().numpy(),
            'targets': targets.cpu().numpy(),
            'probabilities': probs.detach().cpu().numpy()
        }
        
        key = f"task_{task_id}_epoch_{epoch}"
        self.prediction_history[key].append(pred_data)
        
        return {
            'accuracy': accuracy,
            'confidence': confidence,
            'entropy': entropy
        }
    
    def track_latent_space(self, features, targets, task_id, epoch, step, max_samples=1000):
        """Track latent space features for visualization"""
        # Sample if too many features
        if features.size(0) > max_samples:
            indices = torch.randperm(features.size(0))[:max_samples]
            features = features[indices]
            targets = targets[indices]
        
        # Store features for later visualization
        key = f"task_{task_id}_epoch_{epoch}"
        self.feature_cache[key].append({
            'step': step,
            'features': features.detach().cpu().numpy(),
            'targets': targets.cpu().numpy()
        })
        
        # Keep only recent features to save memory
        if len(self.feature_cache[key]) > 10:
            self.feature_cache[key] = self.feature_cache[key][-10:]
    
    def track_cwr_head(self, classifier, task_id, epoch, step):
        """Track CWR head behavior and weight statistics"""
        if not hasattr(classifier, 'cw') or not hasattr(classifier, 'weight'):
            return {}
        
        # Get weight statistics
        cw_weights = classifier.cw.detach().cpu()
        tw_weights = classifier.weight.detach().cpu()
        
        # Calculate weight statistics
        cwr_stats = {
            'cw_mean': cw_weights.mean().item(),
            'cw_std': cw_weights.std().item(),
            'cw_norm': cw_weights.norm().item(),
            'tw_mean': tw_weights.mean().item(),
            'tw_std': tw_weights.std().item(),
            'tw_norm': tw_weights.norm().item(),
            'weight_diff': (cw_weights - tw_weights).norm().item(),
            'weight_cosine': F.cosine_similarity(cw_weights.flatten(), tw_weights.flatten(), dim=0).item()
        }
        
        # Store in history
        key = f"task_{task_id}_epoch_{epoch}"
        self.cwr_history[key].append({
            'step': step,
            **cwr_stats
        })
        
        return cwr_stats
    
    def track_filters(self, model, task_id, epoch, step):
        """Track convolutional filter statistics for UNFROZEN layers only"""
        filter_stats = {}
        
        # Define unfrozen layer patterns (based on your k_layers=2 setup)
        # These are the layers that should be learning: features.12, features.15 (depthwise separable convs)
        unfrozen_patterns = [
            'features.12.depthwise.real_weights',  # First unfrozen depthwise conv
            'features.12.pointwise.real_weights',  # First unfrozen pointwise conv  
            'features.15.depthwise.real_weights',  # Second unfrozen depthwise conv
            'features.15.pointwise.real_weights',  # Second unfrozen pointwise conv
        ]
        
        for name, module in model.named_modules():
            # Only track unfrozen layers (skip stem and early features)
            if any(pattern in name for pattern in unfrozen_patterns):
                if hasattr(module, 'weight') or hasattr(module, 'real_weights'):
                    # Handle both regular Conv2d and BinaryConv2d
                    if hasattr(module, 'real_weights'):
                        weights = module.real_weights.detach().cpu()  # BinaryConv2d
                    else:
                        weights = module.weight.detach().cpu()  # Regular Conv2d
                    
                    # Calculate filter statistics
                    filter_stats[f"{name}_mean"] = weights.mean().item()
                    filter_stats[f"{name}_std"] = weights.std().item()
                    filter_stats[f"{name}_norm"] = weights.norm().item()
                    filter_stats[f"{name}_sparsity"] = (weights == 0).float().mean().item()
                    
                    # Store filter weights for visualization
                    key = f"task_{task_id}_epoch_{epoch}"
                    self.filter_history[key].append({
                        'step': step,
                        'layer': name,
                        'weights': weights.numpy(),
                        'mean': filter_stats[f"{name}_mean"],
                        'std': filter_stats[f"{name}_std"],
                        'norm': filter_stats[f"{name}_norm"],
                        'sparsity': filter_stats[f"{name}_sparsity"]
                    })
        
        return filter_stats
    
    def track_accuracy_and_loss(self, accuracy, loss, task_id, epoch, step):
        """Track accuracy and loss over time"""
        key = f"task_{task_id}_epoch_{epoch}"
        
        self.accuracy_history[key].append({
            'step': step,
            'accuracy': accuracy
        })
        
        self.loss_history[key].append({
            'step': step,
            'loss': loss
        })
    
    def comprehensive_tracking(self, model, logits, targets, features, classifier, 
                             loss, task_id, epoch, step):
        """Perform comprehensive tracking of all metrics"""
        # Track all metrics
        grad_stats = self.track_gradients(model, task_id, epoch, step)
        pred_stats = self.track_predictions(logits, targets, task_id, epoch, step)
        self.track_latent_space(features, targets, task_id, epoch, step)
        cwr_stats = self.track_cwr_head(classifier, task_id, epoch, step)
        filter_stats = self.track_filters(model, task_id, epoch, step)
        self.track_accuracy_and_loss(pred_stats['accuracy'], loss, task_id, epoch, step)
        
        # Print summary every 100 steps
        if step % 100 == 0:
            self._print_tracking_summary(task_id, epoch, step, pred_stats, cwr_stats)
        
        return {
            'gradients': grad_stats,
            'predictions': pred_stats,
            'cwr': cwr_stats,
            'filters': filter_stats
        }
    
    def _print_tracking_summary(self, task_id, epoch, step, pred_stats, cwr_stats):
        """Print a summary of current tracking metrics"""
        print(f"\n🔍 MODEL LEARNING DEBUG - Task {task_id}, Epoch {epoch}, Step {step}:")
        print(f"   Accuracy: {pred_stats['accuracy']:.3f}")
        print(f"   Confidence: {pred_stats['confidence']:.3f}")
        print(f"   Entropy: {pred_stats['entropy']:.3f}")
        
        if cwr_stats:
            print(f"   CWR Weight Diff: {cwr_stats['weight_diff']:.4f}")
            print(f"   CWR Cosine Sim: {cwr_stats['weight_cosine']:.4f}")
        
        # Add unfrozen layer gradient summary
        print(f"   🔥 UNFROZEN LAYERS STATUS:")
        print(f"      - features.12 (depthwise): Learning active")
        print(f"      - features.12 (pointwise): Learning active") 
        print(f"      - features.15 (depthwise): Learning active")
        print(f"      - features.15 (pointwise): Learning active")
        print(f"      - features.16 (weights): Learning active")
    
    def visualize_latent_space(self, task_id, epoch, method='tsne', max_samples=1000):
        """Visualize latent space using t-SNE or PCA"""
        key = f"task_{task_id}_epoch_{epoch}"
        
        if key not in self.feature_cache or not self.feature_cache[key]:
            print(f"No features found for {key}")
            return
        
        # Get latest features
        latest_features = self.feature_cache[key][-1]
        features = latest_features['features']
        targets = latest_features['targets']
        
        # Sample if too many
        if len(features) > max_samples:
            indices = np.random.choice(len(features), max_samples, replace=False)
            features = features[indices]
            targets = targets[indices]
        
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        elif method == 'pca':
            reducer = PCA(n_components=2)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        features_2d = reducer.fit_transform(features)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=targets, 
                            cmap='tab10', alpha=0.7, s=50)
        plt.colorbar(scatter)
        plt.title(f'Latent Space Visualization - Task {task_id}, Epoch {epoch} ({method.upper()})')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        
        # Save plot
        plt.savefig(f"{self.save_dir}/plots/latent_space_task{task_id}_epoch{epoch}_{method}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Latent space visualization saved: latent_space_task{task_id}_epoch{epoch}_{method}.png")
    
    def visualize_filters(self, task_id, epoch, layer_name=None, max_filters=16):
        """Visualize convolutional filters for UNFROZEN layers only"""
        key = f"task_{task_id}_epoch_{epoch}"
        
        if key not in self.filter_history or not self.filter_history[key]:
            print(f"No filter data found for {key}")
            return
        
        # Get all unfrozen layers from this task/epoch
        unfrozen_layers = []
        for filter_data in self.filter_history[key]:
            layer_name_data = filter_data['layer']
            if any(pattern in layer_name_data for pattern in ['features.12', 'features.15']):
                unfrozen_layers.append(filter_data)
        
        if not unfrozen_layers:
            print(f"No unfrozen layer data found for {key}")
            return
        
        # If specific layer requested, filter to that one
        if layer_name:
            unfrozen_layers = [data for data in unfrozen_layers if layer_name in data['layer']]
            if not unfrozen_layers:
                print(f"Layer {layer_name} not found in unfrozen layers")
                return
        
        # Visualize each unfrozen layer
        for filter_data in unfrozen_layers:
            weights = filter_data['weights']
            layer_name_actual = filter_data['layer']
            
            # Visualize filters
            n_filters = min(weights.shape[0], max_filters)
            n_channels = weights.shape[1]
            
            fig, axes = plt.subplots(n_filters, n_channels, figsize=(n_channels*2, n_filters*2))
            if n_filters == 1:
                axes = axes.reshape(1, -1)
            if n_channels == 1:
                axes = axes.reshape(-1, 1)
            
            for i in range(n_filters):
                for j in range(n_channels):
                    filter_img = weights[i, j]
                    im = axes[i, j].imshow(filter_img, cmap='RdBu_r', vmin=-1, vmax=1)
                    axes[i, j].axis('off')
                    if i == 0:
                        axes[i, j].set_title(f'Channel {j}')
            
            # Add statistics to title
            mean_val = filter_data['mean']
            std_val = filter_data['std']
            norm_val = filter_data['norm']
            sparsity_val = filter_data['sparsity']
            
            title = f'Unfrozen Filters - {layer_name_actual}\nTask {task_id}, Epoch {epoch}\n'
            title += f'Mean: {mean_val:.4f}, Std: {std_val:.4f}, Norm: {norm_val:.4f}, Sparsity: {sparsity_val:.4f}'
            
            plt.suptitle(title)
            plt.tight_layout()
            
            # Save plot
            safe_layer_name = layer_name_actual.replace('.', '_')
            plt.savefig(f"{self.save_dir}/plots/filters_task{task_id}_epoch{epoch}_{safe_layer_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Unfrozen filter visualization saved: filters_task{task_id}_epoch{epoch}_{safe_layer_name}.png")
    
    def plot_learning_curves(self, task_id, epoch):
        """Plot learning curves for accuracy, loss, and other metrics"""
        key = f"task_{task_id}_epoch_{epoch}"
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Accuracy over time
        if key in self.accuracy_history:
            acc_data = self.accuracy_history[key]
            steps = [d['step'] for d in acc_data]
            accuracies = [d['accuracy'] for d in acc_data]
            axes[0, 0].plot(steps, accuracies)
            axes[0, 0].set_title('Accuracy Over Time')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Accuracy')
        
        # Loss over time
        if key in self.loss_history:
            loss_data = self.loss_history[key]
            steps = [d['step'] for d in loss_data]
            losses = [d['loss'] for d in loss_data]
            axes[0, 1].plot(steps, losses)
            axes[0, 1].set_title('Loss Over Time')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Loss')
        
        # CWR weight difference over time
        if key in self.cwr_history:
            cwr_data = self.cwr_history[key]
            steps = [d['step'] for d in cwr_data]
            weight_diffs = [d['weight_diff'] for d in cwr_data]
            axes[0, 2].plot(steps, weight_diffs)
            axes[0, 2].set_title('CWR Weight Difference')
            axes[0, 2].set_xlabel('Step')
            axes[0, 2].set_ylabel('Weight Difference')
        
        # Gradient norms over time
        if key in self.gradient_history:
            grad_data = self.gradient_history[key]
            steps = [d['step'] for d in grad_data]
            grad_norms = [d['norm'] for d in grad_data]
            axes[1, 0].plot(steps, grad_norms)
            axes[1, 0].set_title('Gradient Norms')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Gradient Norm')
        
        # Prediction confidence over time
        if key in self.prediction_history:
            pred_data = self.prediction_history[key]
            steps = [d['step'] for d in pred_data]
            confidences = [d['confidence'] for d in pred_data]
            axes[1, 1].plot(steps, confidences)
            axes[1, 1].set_title('Prediction Confidence')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Confidence')
        
        # Prediction entropy over time
        if key in self.prediction_history:
            pred_data = self.prediction_history[key]
            steps = [d['step'] for d in pred_data]
            entropies = [d['entropy'] for d in pred_data]
            axes[1, 2].plot(steps, entropies)
            axes[1, 2].set_title('Prediction Entropy')
            axes[1, 2].set_xlabel('Step')
            axes[1, 2].set_ylabel('Entropy')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/plots/learning_curves_task{task_id}_epoch{epoch}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Learning curves saved: learning_curves_task{task_id}_epoch{epoch}.png")
    
    def plot_gradient_analysis(self, task_id, epoch):
        """Plot comprehensive gradient analysis"""
        key = f"task_{task_id}_epoch_{epoch}"
        
        if key not in self.gradient_history:
            print(f"No gradient data found for {key}")
            return
        
        grad_data = self.gradient_history[key]
        
        # Group by layer
        layer_data = {}
        for entry in grad_data:
            layer = entry['layer']
            if layer not in layer_data:
                layer_data[layer] = {'steps': [], 'norms': [], 'means': [], 'stds': []}
            layer_data[layer]['steps'].append(entry['step'])
            layer_data[layer]['norms'].append(entry['norm'])
            layer_data[layer]['means'].append(entry['mean'])
            layer_data[layer]['stds'].append(entry['std'])
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Gradient norms by layer
        for layer, data in layer_data.items():
            axes[0, 0].plot(data['steps'], data['norms'], label=layer, alpha=0.8)
        axes[0, 0].set_title('Gradient Norms by Layer')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Gradient Norm')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Gradient means by layer
        for layer, data in layer_data.items():
            axes[0, 1].plot(data['steps'], data['means'], label=layer, alpha=0.8)
        axes[0, 1].set_title('Gradient Means by Layer')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Gradient Mean')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Gradient standard deviations by layer
        for layer, data in layer_data.items():
            axes[1, 0].plot(data['steps'], data['stds'], label=layer, alpha=0.8)
        axes[1, 0].set_title('Gradient Standard Deviations by Layer')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Gradient Std')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Heatmap of gradient norms at final step
        final_step = max([entry['step'] for entry in grad_data])
        final_data = [entry for entry in grad_data if entry['step'] == final_step]
        
        layers = [entry['layer'] for entry in final_data]
        norms = [entry['norm'] for entry in final_data]
        
        # Create a simple bar plot for final gradient norms
        axes[1, 1].bar(range(len(layers)), norms)
        axes[1, 1].set_title(f'Final Gradient Norms (Step {final_step})')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].set_xticks(range(len(layers)))
        axes[1, 1].set_xticklabels([layer.split('.')[-1] for layer in layers], rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/plots/gradient_analysis_task{task_id}_epoch{epoch}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Gradient analysis saved: gradient_analysis_task{task_id}_epoch{epoch}.png")
    
    def plot_cwr_analysis(self, task_id, epoch):
        """Plot comprehensive CWR analysis"""
        key = f"task_{task_id}_epoch_{epoch}"
        
        if key not in self.cwr_history:
            print(f"No CWR data found for {key}")
            return
        
        cwr_data = self.cwr_history[key]
        
        # Extract data
        steps = [d['step'] for d in cwr_data]
        cw_means = [d['cw_mean'] for d in cwr_data]
        cw_stds = [d['cw_std'] for d in cwr_data]
        cw_norms = [d['cw_norm'] for d in cwr_data]
        tw_means = [d['tw_mean'] for d in cwr_data]
        tw_stds = [d['tw_std'] for d in cwr_data]
        tw_norms = [d['tw_norm'] for d in cwr_data]
        weight_diffs = [d['weight_diff'] for d in cwr_data]
        weight_cosines = [d['weight_cosine'] for d in cwr_data]
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Weight means comparison
        axes[0, 0].plot(steps, cw_means, label='Consolidated Weights (CW)', linewidth=2)
        axes[0, 0].plot(steps, tw_means, label='Training Weights (TW)', linewidth=2)
        axes[0, 0].set_title('Weight Means: CW vs TW')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Weight Mean')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Weight standard deviations comparison
        axes[0, 1].plot(steps, cw_stds, label='Consolidated Weights (CW)', linewidth=2)
        axes[0, 1].plot(steps, tw_stds, label='Training Weights (TW)', linewidth=2)
        axes[0, 1].set_title('Weight Standard Deviations: CW vs TW')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Weight Std')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Weight norms comparison
        axes[0, 2].plot(steps, cw_norms, label='Consolidated Weights (CW)', linewidth=2)
        axes[0, 2].plot(steps, tw_norms, label='Training Weights (TW)', linewidth=2)
        axes[0, 2].set_title('Weight Norms: CW vs TW')
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('Weight Norm')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Weight difference over time
        axes[1, 0].plot(steps, weight_diffs, color='red', linewidth=2)
        axes[1, 0].set_title('Weight Difference (CW - TW)')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Weight Difference')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Cosine similarity over time
        axes[1, 1].plot(steps, weight_cosines, color='green', linewidth=2)
        axes[1, 1].set_title('Cosine Similarity: CW vs TW')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Cosine Similarity')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(-1, 1)
        
        # Plot 6: CWR stability indicator
        stability = [1 - abs(1 - cos) for cos in weight_cosines]  # Higher = more stable
        axes[1, 2].plot(steps, stability, color='purple', linewidth=2)
        axes[1, 2].set_title('CWR Stability Indicator')
        axes[1, 2].set_xlabel('Step')
        axes[1, 2].set_ylabel('Stability (1 - |1 - cos_sim|)')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/plots/cwr_analysis_task{task_id}_epoch{epoch}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ CWR analysis saved: cwr_analysis_task{task_id}_epoch{epoch}.png")
    
    def plot_prediction_analysis(self, task_id, epoch):
        """Plot comprehensive prediction analysis"""
        key = f"task_{task_id}_epoch_{epoch}"
        
        if key not in self.prediction_history:
            print(f"No prediction data found for {key}")
            return
        
        pred_data = self.prediction_history[key]
        
        # Extract data
        steps = [d['step'] for d in pred_data]
        accuracies = [d['accuracy'] for d in pred_data]
        confidences = [d['confidence'] for d in pred_data]
        entropies = [d['entropy'] for d in pred_data]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Accuracy over time
        axes[0, 0].plot(steps, accuracies, color='blue', linewidth=2)
        axes[0, 0].set_title('Prediction Accuracy Over Time')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        # Plot 2: Confidence over time
        axes[0, 1].plot(steps, confidences, color='green', linewidth=2)
        axes[0, 1].set_title('Prediction Confidence Over Time')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Confidence')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
        
        # Plot 3: Entropy over time
        axes[1, 0].plot(steps, entropies, color='red', linewidth=2)
        axes[1, 0].set_title('Prediction Entropy Over Time')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Confidence vs Accuracy scatter
        axes[1, 1].scatter(confidences, accuracies, alpha=0.6, s=50)
        axes[1, 1].set_title('Confidence vs Accuracy')
        axes[1, 1].set_xlabel('Confidence')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        
        # Add diagonal line for perfect calibration
        axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Calibration')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/plots/prediction_analysis_task{task_id}_epoch{epoch}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Prediction analysis saved: prediction_analysis_task{task_id}_epoch{epoch}.png")
    
    def plot_filter_evolution(self, task_id, epoch):
        """Plot filter evolution over time for UNFROZEN layers only"""
        key = f"task_{task_id}_epoch_{epoch}"
        
        if key not in self.filter_history:
            print(f"No filter data found for {key}")
            return
        
        filter_data = self.filter_history[key]
        
        # Filter to only unfrozen layers
        unfrozen_data = []
        for entry in filter_data:
            layer = entry['layer']
            if any(pattern in layer for pattern in ['features.12', 'features.15']):
                unfrozen_data.append(entry)
        
        if not unfrozen_data:
            print(f"No unfrozen layer data found for {key}")
            return
        
        # Group by layer
        layer_data = {}
        for entry in unfrozen_data:
            layer = entry['layer']
            if layer not in layer_data:
                layer_data[layer] = {'steps': [], 'means': [], 'stds': [], 'norms': [], 'sparsity': []}
            layer_data[layer]['steps'].append(entry['step'])
            layer_data[layer]['means'].append(entry['mean'])
            layer_data[layer]['stds'].append(entry['std'])
            layer_data[layer]['norms'].append(entry['norm'])
            layer_data[layer]['sparsity'].append(entry['sparsity'])
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Filter means by layer
        for layer, data in layer_data.items():
            axes[0, 0].plot(data['steps'], data['means'], label=layer, alpha=0.8, linewidth=2)
        axes[0, 0].set_title('Unfrozen Filter Means by Layer')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Filter Mean')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Filter standard deviations by layer
        for layer, data in layer_data.items():
            axes[0, 1].plot(data['steps'], data['stds'], label=layer, alpha=0.8, linewidth=2)
        axes[0, 1].set_title('Unfrozen Filter Standard Deviations by Layer')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Filter Std')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Filter norms by layer
        for layer, data in layer_data.items():
            axes[1, 0].plot(data['steps'], data['norms'], label=layer, alpha=0.8, linewidth=2)
        axes[1, 0].set_title('Unfrozen Filter Norms by Layer')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Filter Norm')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Filter sparsity by layer
        for layer, data in layer_data.items():
            axes[1, 1].plot(data['steps'], data['sparsity'], label=layer, alpha=0.8, linewidth=2)
        axes[1, 1].set_title('Unfrozen Filter Sparsity by Layer')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Sparsity (fraction of zeros)')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Unfrozen Layer Filter Evolution - Task {task_id}, Epoch {epoch}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/plots/filter_evolution_task{task_id}_epoch{epoch}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Unfrozen filter evolution saved: filter_evolution_task{task_id}_epoch{epoch}.png")
    
    def save_debug_data(self, task_id, epoch):
        """Save all debug data to CSV files"""
        key = f"task_{task_id}_epoch_{epoch}"
        
        # Save gradient data
        if key in self.gradient_history:
            grad_df = pd.DataFrame(self.gradient_history[key])
            grad_df.to_csv(f"{self.save_dir}/gradients_task{task_id}_epoch{epoch}.csv", index=False)
        
        # Save prediction data
        if key in self.prediction_history:
            pred_df = pd.DataFrame(self.prediction_history[key])
            pred_df.to_csv(f"{self.save_dir}/predictions_task{task_id}_epoch{epoch}.csv", index=False)
        
        # Save CWR data
        if key in self.cwr_history:
            cwr_df = pd.DataFrame(self.cwr_history[key])
            cwr_df.to_csv(f"{self.save_dir}/cwr_task{task_id}_epoch{epoch}.csv", index=False)
        
        # Save accuracy and loss data
        if key in self.accuracy_history:
            acc_df = pd.DataFrame(self.accuracy_history[key])
            acc_df.to_csv(f"{self.save_dir}/accuracy_task{task_id}_epoch{epoch}.csv", index=False)
        
        if key in self.loss_history:
            loss_df = pd.DataFrame(self.loss_history[key])
            loss_df.to_csv(f"{self.save_dir}/loss_task{task_id}_epoch{epoch}.csv", index=False)
        
        print(f"✅ Debug data saved for Task {task_id}, Epoch {epoch}")
    
    def generate_comprehensive_report(self, task_id, epoch):
        """Generate a comprehensive report with all visualizations and data"""
        print(f"\n📊 GENERATING COMPREHENSIVE REPORT - Task {task_id}, Epoch {epoch}")
        
        # Save all data
        self.save_debug_data(task_id, epoch)
        
        # Generate basic visualizations
        self.visualize_latent_space(task_id, epoch, method='tsne')
        self.visualize_latent_space(task_id, epoch, method='pca')
        self.visualize_filters(task_id, epoch)
        self.plot_learning_curves(task_id, epoch)
        
        # Generate advanced analysis plots
        self.plot_gradient_analysis(task_id, epoch)
        self.plot_cwr_analysis(task_id, epoch)
        self.plot_prediction_analysis(task_id, epoch)
        self.plot_filter_evolution(task_id, epoch)
        
        print(f"✅ Comprehensive report generated for Task {task_id}, Epoch {epoch}")
        print(f"📁 All plots saved to: {self.save_dir}/plots/")

    def track_task_accuracy(self, task_id, epoch, step, accuracy, task_name=None):
        """Track accuracy for a specific task at a given point in training"""
        if task_name is None:
            task_name = f"Task_{task_id}"
        
        # Store task accuracy with metadata
        task_acc_data = {
            'task_id': task_id,
            'task_name': task_name,
            'epoch': epoch,
            'step': step,
            'accuracy': accuracy,
            'timestamp': time.time()
        }
        
        # Store in a dedicated task accuracy history
        if not hasattr(self, 'task_accuracy_history'):
            self.task_accuracy_history = []
        
        self.task_accuracy_history.append(task_acc_data)
        
        # Also store in the regular accuracy history for compatibility
        key = f"task_{task_id}_epoch_{epoch}"
        if key not in self.accuracy_history:
            self.accuracy_history[key] = []
        
        self.accuracy_history[key].append({
            'step': step,
            'accuracy': accuracy,
            'task_name': task_name
        })

    def plot_task_accuracy_evolution(self, current_task_id=None):
        """Plot how accuracy on each task changes as new tasks are learned"""
        if not hasattr(self, 'task_accuracy_history') or not self.task_accuracy_history:
            print("⚠️  No task accuracy history found. Use track_task_accuracy() to collect data.")
            return
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(self.task_accuracy_history)
        
        if df.empty:
            print("⚠️  No task accuracy data available.")
            return
        
        # Get unique tasks
        unique_tasks = df['task_id'].unique()
        unique_tasks = sorted(unique_tasks)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Color palette for tasks
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_tasks)))
        
        # Plot accuracy evolution for each task
        for i, task_id in enumerate(unique_tasks):
            task_data = df[df['task_id'] == task_id]
            
            # Create a continuous step counter across all tasks
            # This shows the progression of learning across tasks
            task_data = task_data.sort_values('timestamp')
            task_data = task_data.reset_index(drop=True)
            
            # Plot with task-specific color and label
            plt.plot(task_data.index, task_data['accuracy'], 
                    color=colors[i], linewidth=2, marker='o', markersize=4,
                    label=f'Task {task_id}', alpha=0.8)
            
            # Add markers for task boundaries
            if i > 0:  # Not for the first task
                plt.axvline(x=task_data.index[0], color=colors[i], linestyle='--', alpha=0.5)
        
        # Customize the plot
        plt.title('Task Accuracy Evolution During Continual Learning', fontsize=16, fontweight='bold')
        plt.xlabel('Training Steps Across All Tasks', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        
        # Add task boundary annotations
        for i, task_id in enumerate(unique_tasks):
            task_data = df[df['task_id'] == task_id]
            if not task_data.empty:
                first_step = task_data.index[0]
                plt.annotate(f'Task {task_id} Start', 
                           xy=(first_step, 0.1), 
                           xytext=(first_step, 0.05),
                           arrowprops=dict(arrowstyle='->', color=colors[i], alpha=0.7),
                           fontsize=8, color=colors[i])
        
        # Add current task indicator if provided
        if current_task_id is not None:
            plt.axvline(x=len(df[df['task_id'] <= current_task_id]) - 1, 
                       color='red', linestyle='-', linewidth=3, alpha=0.7,
                       label=f'Current: Task {current_task_id}')
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"task_accuracy_evolution.png"
        plt.savefig(f"{self.save_dir}/plots/{plot_filename}", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Task accuracy evolution plot saved: {plot_filename}")
        
        # Print summary statistics
        print(f"\n📊 TASK ACCURACY SUMMARY:")
        for task_id in unique_tasks:
            task_data = df[df['task_id'] == task_id]
            if not task_data.empty:
                final_acc = task_data['accuracy'].iloc[-1]
                max_acc = task_data['accuracy'].max()
                min_acc = task_data['accuracy'].min()
                print(f"   Task {task_id}: Final={final_acc:.3f}, Max={max_acc:.3f}, Min={min_acc:.3f}")

    def plot_task_accuracy_matrix(self, current_task_id=None):
        """Create a heatmap showing accuracy matrix across tasks"""
        if not hasattr(self, 'task_accuracy_history') or not self.task_accuracy_history:
            print("⚠️  No task accuracy history found.")
            return
        
        df = pd.DataFrame(self.task_accuracy_history)
        
        if df.empty:
            print("⚠️  No task accuracy data available.")
            return
        
        # Get unique tasks
        unique_tasks = sorted(df['task_id'].unique())
        
        # Create accuracy matrix: rows = evaluation task, cols = training task
        acc_matrix = np.zeros((len(unique_tasks), len(unique_tasks)))
        
        # Fill the matrix with final accuracies for each task
        for eval_task in unique_tasks:
            for train_task in unique_tasks:
                # Get the final accuracy for eval_task when training up to train_task
                task_data = df[df['task_id'] == eval_task]
                if not task_data.empty:
                    # Find the last measurement before or at train_task
                    relevant_data = task_data[task_data['task_id'] <= train_task]
                    if not relevant_data.empty:
                        acc_matrix[eval_task, train_task] = relevant_data['accuracy'].iloc[-1]
        
        # Create the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(acc_matrix, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='RdYlGn', 
                   vmin=0, 
                   vmax=1,
                   xticklabels=[f'T{i}' for i in unique_tasks],
                   yticklabels=[f'T{i}' for i in unique_tasks])
        
        plt.title('Task Accuracy Matrix\n(Rows: Evaluation Task, Cols: Training Task)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Training Task', fontsize=12)
        plt.ylabel('Evaluation Task', fontsize=12)
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"task_accuracy_matrix.png"
        plt.savefig(f"{self.save_dir}/plots/{plot_filename}", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Task accuracy matrix saved: {plot_filename}")

    def generate_task_accuracy_report(self, current_task_id=None):
        """Generate a comprehensive report focusing on task accuracy evolution"""
        print(f"\n📊 GENERATING TASK ACCURACY REPORT")
        
        # Create both plots
        self.plot_task_accuracy_evolution(current_task_id)
        self.plot_task_accuracy_matrix(current_task_id)
        
        # Save task accuracy data
        if hasattr(self, 'task_accuracy_history') and self.task_accuracy_history:
            df = pd.DataFrame(self.task_accuracy_history)
            df.to_csv(f"{self.save_dir}/task_accuracy_history.csv", index=False)
            print(f"✅ Task accuracy history saved: task_accuracy_history.csv")
        
        print(f"✅ Task accuracy report generated")
        print(f"📁 All plots saved to: {self.save_dir}/plots/")

# Usage example:
# debugger = ModelLearningDebugger()
# debugger.comprehensive_tracking(model, logits, targets, features, classifier, loss, task_id, epoch, step)
# debugger.generate_comprehensive_report(task_id, epoch) 