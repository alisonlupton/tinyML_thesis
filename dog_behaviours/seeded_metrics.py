# seeded_metrics.py

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from typing import Dict, List, Any, Tuple
import warnings
import csv
warnings.filterwarnings('ignore')

# Add the current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_config, set_seed
from metrics import profile_cil_resources
import torch

def continual_learning_metrics_extended(eval_exp, train_exp, acc, forgetting, model_name='model'):
    """
    Calculate continual learning metrics inspired by Towards Lifelong Deep Learning.
    
    Args:
        eval_exp: List of evaluation experience indices
        train_exp: List of training experience indices  
        acc: List of accuracies
        forgetting: List of forgetting values
        model_name: Name of the model for CSV output
        
    Returns:
        Dictionary of CL metrics
    """
    eval_exp = np.array(eval_exp, dtype=int)
    train_exp = np.array(train_exp, dtype=int)
    acc = np.array(acc, dtype=float)
    forgetting = np.array(forgetting, dtype=float)

    # number of experiences
    K = max(eval_exp) + 1
    
    # create KxK matrix as the paper suggests 
    R = np.full((K, K), np.nan, dtype=float)
    
    # populate R (only for valid entries where data has been seen before)
    # i  <-- number of experiences seen  =>  training_exp (rows)
    # j  <-- which task we eval         =>  eval_exp (cols)
    valid_acc = (train_exp >= eval_exp)
    R[train_exp[valid_acc], eval_exp[valid_acc]] = acc[valid_acc]

    final_mask = (train_exp == (K-1)) # get indices for only the accuracies after all data has been seen 
    final_accs = acc[final_mask] 
    avg_acc = np.nanmean(final_accs)

    valid_forg = (train_exp > eval_exp)
    avg_forg = np.nanmean(forgetting[valid_forg])
    
    # --- Average Incremental Accuracy A_K = 2/(K*(K+1)) * sum_{i>=j} R[i,j]
    # mask lower triangle including diagonal
    mask = np.tril(np.ones((K, K), dtype=bool))
    sum_lower = np.nansum(R[mask])
    avg_inc_acc = (2.0 / (K*(K+1))) * sum_lower
    
    # --- Backward Transfer BWT = avg_{j < K} [ R[K-1,j] - R[j,j] ]
    last_row = R[K-1, :K-1]          # R[K-1, j] for j=0..K-2
    diagonal = np.diag(R)[:K-1]      # R[j,j] for j=0..K-2
    bwt = np.nanmean(last_row - diagonal)
    
    # --- Forward Transfer FWT = avg_{i < j} [ R[i,j] - R[0,j] ]
    # Here R[0,j] is the "initial" accuracy before training on any task
    # We compare performance of model after task i on unseen task j>i vs that initial baseline
    baseline = R[0, :]               # shape (K,)
    # get indices i<j
    tri_i, tri_j = np.triu_indices(K, k=1)
    fwt_values = R[tri_i, tri_j] - baseline[tri_j]
    fwt = np.nanmean(fwt_values)

    CL_dic = {
        "avg_acc": avg_acc,
        "avg_forg": avg_forg,
        "avg_inc_acc": avg_inc_acc,
        "fwt": fwt, 
        "bwt": bwt,
    }
    
    return CL_dic

class SeededMetricsCollector:
    """
    Collects metrics from multiple CIL training runs with different seeds.
    Saves results to CSV and generates plots for analysis.
    """
    
    def __init__(self, config_path: str = "dog_behaviors_config.yaml", 
                 output_dir: str = "plots_and_metrics_seeded", 
                 seeds: List[int] = None):
        """
        Initialize the metrics collector.
        
        Args:
            config_path: Path to the configuration file
            output_dir: Directory to save results and plots
            seeds: List of seeds to run. If None, uses [42]
        """
        self.config_path = config_path
        self.output_dir = output_dir
        self.seeds = seeds if seeds is not None else [42]
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load base configuration
        self.base_config = load_config()
        
        # Storage for results
        self.results = []
        self.resource_metrics = []
        
    def run_single_seed(self, seed: int) -> Dict[str, Any]:
        """
        Run CIL training for a single seed and collect metrics.
        Uses the exact same training pipeline as cnn_CIL_training.py
        
        Args:
            seed: Random seed for this run
            
        Returns:
            Dictionary containing all collected metrics
        """
        print(f"\n{'='*60}")
        print(f"RUNNING SEED {seed}")
        print(f"{'='*60}")
        
        # Set seed for reproducibility
        set_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create a copy of config with this seed
        config = self.base_config.copy()
        config['random_seed'] = seed
        
        # Update metrics path to be seed-specific
        config['metrics_path'] = f"{self.output_dir}/dog_cil_metrics_seed_{seed}.txt"
        
        try:
            # Import the exact same modules as cnn_CIL_training.py
            from cnn_CIL_training_utils import load_CIL_data, CIL_post_task_eval, train_with_simplified_tdm, make_CIL_plots, check_CIL_validity
            from cnn_backbone_training_utils import train_cnn_backbone, load_data_cnn_backbone
            from models.cnn import SimplifiedTDMModelCNN
            from utils import load_processed_dog_data, make_global_to_local_map, ClassRegistry
            from replay import BalancedQuantReplayDynamic
            from report_plotting import CILTrainingPlotter
            import random
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            from torch.optim import Adam
            from torch.nn import CrossEntropyLoss
            
            # Load data (exact same as cnn_CIL_training.py)
            dog_data, behavior_to_idx, behaviors = load_processed_dog_data()
            
            if len(dog_data) == 0:
                raise ValueError("No dog data loaded!")
            
            # Define behaviors (exact same as cnn_CIL_training.py)
            backbone_behaviors = config['backbone_behaviors']
            cil_behaviors = config['cil_behaviors']
            all_behaviors = backbone_behaviors + cil_behaviors
            cil_tasks = config['cil_tasks']
            
            # Device setup (exact same as cnn_CIL_training.py)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
            
            # Select dogs for this seed (exact same logic as cnn_CIL_training.py)
            all_possible_dogs = sorted(dog_data.keys())
            valid_CIL_dogs_for_this_seed = [
                d for d in all_possible_dogs
                if check_CIL_validity(dog_data, d, all_behaviors, behavior_to_idx, verbose=True)
            ] 
            
            print(f"Valid CIL dogs for seed {seed}: {valid_CIL_dogs_for_this_seed}")
            if not valid_CIL_dogs_for_this_seed:
                raise ValueError("No dogs satisfy nonzero train+test counts for all 7 classes under this seed/split.")
            
            rng = random.Random(seed)  
            target_dog = rng.choice(valid_CIL_dogs_for_this_seed)
            print(f"Target Dog Chosen: {target_dog}")
            
            remaining = sorted([d for d in all_possible_dogs if d != target_dog])
            validation_dog = rng.choice(remaining)
            print(f"Validation Dog Chosen: {validation_dog}")
            
            remaining = sorted([dog for dog in all_possible_dogs if dog not in [target_dog, validation_dog]])
            backbone_dogs = rng.sample(remaining, config['num_pretrain_dogs'])
            print(f"Backbone Dogs Chosen: {backbone_dogs}")
            
            # Backbone training (exact same as cnn_CIL_training.py)
            backbone_num_classes = len(backbone_behaviors)
            C = len(dog_data[list(dog_data.keys())[0]]['sensor_cols'])
            L = int(dog_data[list(dog_data.keys())[0]]['window_len'])
            backbone_model = SimplifiedTDMModelCNN(C, backbone_num_classes, device, 
                                                 sparsity_ratio=config['backbone_sparsity_ratio'])
            backbone_model.to(device)
            
            backbone_data = load_data_cnn_backbone(backbone_dogs, dog_data, behavior_to_idx, 
                                                 backbone_behaviors, validation_dog)
            
            # Create data loaders (exact same as cnn_CIL_training.py)
            gen_seed = torch.Generator()
            gen_seed.manual_seed(seed)
            backbone_train_loader = DataLoader(TensorDataset(backbone_data.X_train, backbone_data.y_train), 
                                            batch_size=config['backbone_batch_size_train'], shuffle=True, generator=gen_seed)
            backbone_val_loader = DataLoader(TensorDataset(backbone_data.X_val, backbone_data.y_val), 
                                          batch_size=config['backbone_batch_size_val'], shuffle=False)
            
            backbone_optimizer = Adam(backbone_model.parameters(), lr=config['backbone_learning_rate'], 
                                    weight_decay=config['backbone_weight_decay'])
            backbone_criterion = CrossEntropyLoss()
            
            # Train backbone (exact same as cnn_CIL_training.py)
            backbone_model, backbone_class_acc = train_cnn_backbone(
                backbone_model, backbone_behaviors, backbone_optimizer, backbone_criterion, 
                backbone_train_loader, backbone_val_loader, device, config
            )
            
            # CIL Training setup (exact same as cnn_CIL_training.py)
            cil_model = SimplifiedTDMModelCNN(C, len(all_behaviors), device, 
                                            sparsity_ratio=config['sparsity_ratio'])
            cil_model.to(device)
            
            # Copy backbone weights (exact same as cnn_CIL_training.py)
            cil_model.backbone.load_state_dict(backbone_model.backbone.state_dict())
            
            # Initialize head for all classes (exact same as cnn_CIL_training.py)
            cil_model.head.linear.weight.data.zero_()
            if cil_model.head.linear.bias is not None:
                cil_model.head.linear.bias.data.zero_()
            
            # CIL training (exact same as cnn_CIL_training.py)
            cil_metrics = self._run_cil_training_exact(cil_model, cil_tasks, target_dog, dog_data, 
                                                     behavior_to_idx, config, device, all_behaviors, 
                                                     backbone_behaviors, gen_seed)
            
            # Collect resource metrics (exact same as cnn_CIL_training.py)
            resource_report = profile_cil_resources(
                config, cil_model, C, L,
                batch_size_train=config['CIL_batch_size_train'],
                batch_size_infer=config['CIL_batch_size_test'],
                replay_size=config['buffer_size'],
                optimizer_kind="adam",
                deployed_bits_backbone=32,
                deployed_bits_classifier=8,
                training_bits_classifier=32,
                replay_bits=8,
                activation_bits_train=32,
                activation_bits_infer=32,
                replay_batch_size=config['replay_batch_size'],
                kd_prev_rows=len(backbone_behaviors),
                kd_enabled=True,
                autograd_grad_for_cwi=True
            )
            
            # Calculate continual learning metrics
            cl_metrics = self._calculate_cl_metrics(cil_metrics)
            
            # Compile results for this seed
            seed_results = {
                'seed': seed,
                'target_dog': target_dog,
                'validation_dog': validation_dog,
                'backbone_dogs': backbone_dogs,
                'backbone_metrics': {'best_val_acc': max(backbone_class_acc.values()) if backbone_class_acc else 0, 
                                   'final_val_class_acc': backbone_class_acc},
                'cil_metrics': cil_metrics,
                'cl_metrics': cl_metrics,
                'resource_metrics': resource_report.to_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"\nSeed {seed} completed successfully!")
            return seed_results
            
        except Exception as e:
            print(f"Error running seed {seed}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'seed': seed,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    
    def _run_cil_training_exact(self, cil_model, cil_tasks, target_dog, dog_data, behavior_to_idx, 
                               config, device, all_behaviors, backbone_behaviors, gen_seed):
        """
        Run CIL training using the exact same approach as cnn_CIL_training.py
        """
        from cnn_CIL_training_utils import CIL_post_task_eval, train_with_simplified_tdm, split_by_testnum_for_CIL
        from utils import make_global_to_local_map, ClassRegistry
        from replay import BalancedQuantReplayDynamic
        from torch.utils.data import DataLoader, TensorDataset
        import torch.nn as nn
        import torch
        
        # Load target dog data (exact same as cnn_CIL_training.py)
        target_X = dog_data[target_dog]['X']
        target_y = dog_data[target_dog]['y']
        
        # Split by test number (exact same as cnn_CIL_training.py)
        train_mask, test_mask, train_session, test_session = split_by_testnum_for_CIL(dog_data, target_dog, verbose=False)
        
        X_train = target_X[train_mask]
        y_train = target_y[train_mask]
        X_test = target_X[test_mask]
        y_test = target_y[test_mask]
        
        # Initialize replay buffer (exact same as cnn_CIL_training.py)
        replay_buffer_q = BalancedQuantReplayDynamic(buffer_size=config['buffer_size'])
        
        # Initialize registry (exact same as cnn_CIL_training.py)
        registry = ClassRegistry(all_behaviors)
        registry.add_classes(backbone_behaviors)
        
        seen_names = registry.seen_classes()
        _, seen_gids = make_global_to_local_map(seen_names, behavior_to_idx, device="cpu")
        replay_buffer_q.on_new_task(seen_gids.tolist())
        
        # TDM parameters (exact same as cnn_CIL_training.py)
        p_intra = config['p_intra']
        p_inter = config['p_inter']
        delta_k = config['delta_k']
        
        cil_metrics = {
            'task_metrics': [],
            'accuracy_history': []
        }
        
        # CIL training loop (exact same as cnn_CIL_training.py)
        for task_idx, task_classes in enumerate(cil_tasks):
            # Determine which new class is being added (exact same as cnn_CIL_training.py)
            if task_idx == 0:
                new_class = "Sniffing"
            elif task_idx == 1:
                new_class = "Trotting"
            else:
                new_class = "Galloping"
            
            print(f"\n--- CIL TASK {task_idx + 1}: Added '{new_class}' ---")
            
            # 1) Expand head for new classes (exact same as cnn_CIL_training.py)
            prev_num = registry.num_classes()
            if len(task_classes) > prev_num:
                new_cls = task_classes[prev_num:]
                cil_model.expand_head(new_cls, registry)
            
            # 2) Inter-task expansion (exact same as cnn_CIL_training.py)
            if task_idx > 0:
                cil_model.head.update_mask_inter(p=p_inter)
            
            # Refresh seen set (exact same as cnn_CIL_training.py)
            seen_names = registry.seen_classes()
            _, seen_gids = make_global_to_local_map(seen_names, behavior_to_idx, device="cpu")
            replay_buffer_q.on_new_task(seen_gids.tolist())
            
            # 3) Prepare training data (exact same as cnn_CIL_training.py)
            map_task, task_gids = make_global_to_local_map(task_classes, behavior_to_idx, device=y_train.device)
            mask = torch.isin(y_train, task_gids)
            X_task = X_train[mask]
            y_task = y_train[mask]
            y_task_remapped = map_task[y_task]
            
            # 4) Create data loader (exact same as cnn_CIL_training.py)
            train_loader = DataLoader(TensorDataset(X_task, y_task_remapped), 
                                    batch_size=config['CIL_batch_size_train'], shuffle=True, generator=gen_seed)
            
            # 5) Create teacher snapshot (exact same as cnn_CIL_training.py)
            teacher = None
            if prev_num > 0:
                teacher = nn.Linear(cil_model.head.in_dim, prev_num, bias=True).to(device)
                with torch.no_grad():
                    teacher.weight.copy_(cil_model.head.linear.weight[:prev_num])
                    teacher.bias.copy_(cil_model.head.linear.bias[:prev_num])
                teacher.eval()
            
            optimizer = torch.optim.Adam(cil_model.head.parameters(), lr=config['CIL_learning_rate'])
            criterion = nn.CrossEntropyLoss()
            
            # Training metrics for this task
            task_train_metrics = []
            
            # CIL training epochs (exact same as cnn_CIL_training.py)
            for epoch in range(config['CIL_epochs']):
                # Ensure frozen (exact same as cnn_CIL_training.py)
                cil_model.head.train()
                cil_model.backbone.eval()
                
                # Learning rate scheduling (exact same as cnn_CIL_training.py)
                if epoch == 5:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                elif epoch == 10:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                
                # Intra-task TDM adjustment (exact same as cnn_CIL_training.py)
                if epoch > 0 and epoch % delta_k == 0:
                    cil_model.head.update_mask_intra(p=p_intra)
                
                # Inter-task warm-up end (exact same as cnn_CIL_training.py)
                if task_idx > 0 and epoch == delta_k:
                    cil_model.head.shrink_after_warmup(p=p_inter)
                
                # Train and get metrics (exact same as cnn_CIL_training.py)
                train_loss, train_acc = train_with_simplified_tdm(
                    cil_model, config, registry, task_classes, teacher, prev_num,
                    train_loader, behavior_to_idx, optimizer, criterion, device,
                    replay_buffer_q, seen_gids.tolist(), seen_names, task_idx, epoch
                )
                
                task_train_metrics.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc
                })
            
            # Evaluate on test set (exact same as cnn_CIL_training.py)
            cil_model.eval()
            test_loader = DataLoader(TensorDataset(X_test, y_test), 
                                   batch_size=config['CIL_batch_size_test'], shuffle=False)
            
            class_acc, overall_acc = CIL_post_task_eval(task_classes, test_loader, device, 
                                                      all_behaviors, cil_model, registry)
            
            # Store task metrics
            task_metrics = {
                'task_idx': task_idx,
                'task_classes': task_classes,
                'new_class': new_class,
                'train_metrics': task_train_metrics,
                'test_class_acc': class_acc.copy(),
                'test_overall_acc': overall_acc
            }
            
            cil_metrics['task_metrics'].append(task_metrics)
            cil_metrics['accuracy_history'].append({
                'class_acc': class_acc.copy(),
                'overall_acc': overall_acc
            })
            
            print(f"Task {task_idx + 1} - Overall Accuracy: {overall_acc:.2f}%")
            for behavior, acc in class_acc.items():
                print(f"  {behavior}: {acc:.2f}%")
        
        return cil_metrics
    
    def _calculate_cl_metrics(self, cil_metrics):
        """
        Calculate continual learning metrics from CIL training results.
        """
        # Extract accuracies for CL metrics calculation
        eval_exp = []
        train_exp = []
        acc = []
        forgetting = []
        
        for task_idx, task_metrics in enumerate(cil_metrics['task_metrics']):
            # For each task, we evaluate on all previous tasks + current task
            for eval_task_idx in range(task_idx + 1):
                eval_exp.append(eval_task_idx)
                train_exp.append(task_idx)  # After training on task_idx
                acc.append(task_metrics['test_overall_acc'])  # Overall accuracy
                forgetting.append(0.0)  # Placeholder - would need to track per-class forgetting
        
        # Calculate CL metrics
        cl_metrics = continual_learning_metrics_extended(eval_exp, train_exp, acc, forgetting, 'cil_model')
        
        return cl_metrics
    
    def run_all_seeds(self) -> None:
        """
        Run CIL training for all seeds and collect metrics.
        """
        print(f"Starting seeded metrics collection with {len(self.seeds)} seeds: {self.seeds}")
        
        for seed in self.seeds:
            result = self.run_single_seed(seed)
            self.results.append(result)
            
            # Save intermediate results
            self.save_results()
        
        print(f"\n{'='*60}")
        print("ALL SEEDS COMPLETED")
        print(f"{'='*60}")
    
    def save_results(self) -> None:
        """
        Save all collected results to CSV files.
        """
        if not self.results:
            print("No results to save.")
            return
        
        # Save detailed results as JSON
        results_file = os.path.join(self.output_dir, f"seeded_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Detailed results saved to: {results_file}")
        
        # Create summary CSV for accuracy metrics
        self._create_accuracy_summary_csv()
        
        # Create summary CSV for resource metrics
        self._create_resource_summary_csv()
        
        # Create overall performance CSV
        self._create_performance_summary_csv()
        
        # Create continual learning metrics CSV
        self._create_cl_metrics_csv()
    
    def _create_accuracy_summary_csv(self) -> None:
        """Create CSV with comprehensive accuracy metrics across all seeds."""
        accuracy_data = []
        
        for result in self.results:
            if 'error' in result:
                continue
                
            seed = result['seed']
            
            # Backbone metrics
            if 'backbone_metrics' in result:
                backbone_metrics = result['backbone_metrics']
                
                # Backbone final validation accuracy
                backbone_acc = backbone_metrics['final_val_class_acc']
                for behavior, acc in backbone_acc.items():
                    accuracy_data.append({
                        'seed': seed,
                        'stage': 'Backbone_Val',
                        'behavior': behavior,
                        'accuracy': acc,
                        'overall_accuracy': backbone_metrics['best_val_acc'],
                        'metric_type': 'validation'
                    })
            
            # CIL task metrics
            if 'cil_metrics' in result:
                cil_metrics = result['cil_metrics']
                
                for task_metrics in cil_metrics['task_metrics']:
                    task_idx = task_metrics['task_idx']
                    stage = f'Task_{task_idx + 1}'
                    
                    # CIL test accuracy
                    test_class_acc = task_metrics['test_class_acc']
                    for behavior, acc in test_class_acc.items():
                        accuracy_data.append({
                            'seed': seed,
                            'stage': f'{stage}_Test',
                            'behavior': behavior,
                            'accuracy': acc,
                            'overall_accuracy': task_metrics['test_overall_acc'],
                            'metric_type': 'test'
                        })
                    
                    # CIL training accuracy (final epoch)
                    if task_metrics['train_metrics']:
                        final_train_epoch = task_metrics['train_metrics'][-1]
                        accuracy_data.append({
                            'seed': seed,
                            'stage': f'{stage}_Train',
                            'behavior': 'Overall',
                            'accuracy': final_train_epoch['train_acc'],
                            'overall_accuracy': final_train_epoch['train_acc'],
                            'metric_type': 'training'
                        })
        
        df = pd.DataFrame(accuracy_data)
        csv_file = os.path.join(self.output_dir, "seeded_accuracy_metrics.csv")
        df.to_csv(csv_file, index=False)
        print(f"Accuracy metrics saved to: {csv_file}")
    
    def _create_resource_summary_csv(self) -> None:
        """Create CSV with resource metrics across all seeds."""
        resource_data = []
        
        for result in self.results:
            if 'error' in result or 'resource_metrics' not in result:
                continue
                
            seed = result['seed']
            metrics = result['resource_metrics']
            
            # Add seed to metrics
            metrics['seed'] = seed
            resource_data.append(metrics)
        
        if resource_data:
            df = pd.DataFrame(resource_data)
            csv_file = os.path.join(self.output_dir, "seeded_resource_metrics.csv")
            df.to_csv(csv_file, index=False)
            print(f"Resource metrics saved to: {csv_file}")
    
    def _create_performance_summary_csv(self) -> None:
        """Create CSV with overall performance summary across seeds."""
        performance_data = []
        
        for result in self.results:
            if 'error' in result:
                performance_data.append({
                    'seed': result['seed'],
                    'status': 'error',
                    'error_message': result['error'],
                    'final_overall_accuracy': None,
                    'backbone_avg_accuracy': None,
                    'task1_accuracy': None,
                    'task2_accuracy': None,
                    'task3_accuracy': None
                })
                continue
            
            seed = result['seed']
            
            # Calculate backbone average accuracy
            backbone_avg = None
            if 'backbone_metrics' in result:
                backbone_metrics = result['backbone_metrics']
                backbone_avg = backbone_metrics['best_val_acc']
            
            # Get final overall accuracy
            final_overall_acc = None
            if 'cil_metrics' in result and result['cil_metrics']['task_metrics']:
                final_overall_acc = result['cil_metrics']['task_metrics'][-1]['test_overall_acc']
            
            # Get task-specific accuracies
            task_accuracies = {}
            if 'cil_metrics' in result:
                for i, task_metrics in enumerate(result['cil_metrics']['task_metrics']):
                    task_accuracies[f'task{i+1}_accuracy'] = task_metrics['test_overall_acc']
            
            performance_data.append({
                'seed': seed,
                'status': 'success',
                'error_message': None,
                'final_overall_accuracy': final_overall_acc,
                'backbone_avg_accuracy': backbone_avg,
                'task1_accuracy': task_accuracies.get('task1_accuracy'),
                'task2_accuracy': task_accuracies.get('task2_accuracy'),
                'task3_accuracy': task_accuracies.get('task3_accuracy'),
                'target_dog': result['target_dog'],
                'validation_dog': result['validation_dog']
            })
        
        df = pd.DataFrame(performance_data)
        csv_file = os.path.join(self.output_dir, "seeded_performance_summary.csv")
        df.to_csv(csv_file, index=False)
        print(f"Performance summary saved to: {csv_file}")
    
    def _create_cl_metrics_csv(self) -> None:
        """Create CSV with continual learning metrics across all seeds."""
        cl_data = []
        
        for result in self.results:
            if 'error' in result or 'cl_metrics' not in result:
                continue
                
            seed = result['seed']
            cl_metrics = result['cl_metrics']
            
            cl_data.append({
                'seed': seed,
                'avg_acc': cl_metrics.get('avg_acc'),
                'avg_forg': cl_metrics.get('avg_forg'),
                'avg_inc_acc': cl_metrics.get('avg_inc_acc'),
                'fwt': cl_metrics.get('fwt'),
                'bwt': cl_metrics.get('bwt')
            })
        
        if cl_data:
            df = pd.DataFrame(cl_data)
            csv_file = os.path.join(self.output_dir, "seeded_cl_metrics.csv")
            df.to_csv(csv_file, index=False)
            print(f"Continual learning metrics saved to: {csv_file}")
    
    def create_plots(self) -> None:
        """
        Create plots showing results across different seeds.
        """
        if not self.results:
            print("No results to plot.")
            return
        
        # Set seaborn style for consistency [[memory:7843758]]
        plt.style.use('seaborn-v0_8-deep')
        
        # 1. Overall accuracy progression across seeds
        self._plot_overall_accuracy_progression()
        
        # 2. Per-class accuracy heatmap
        self._plot_per_class_accuracy_heatmap()
        
        # 3. Resource metrics comparison
        self._plot_resource_metrics_comparison()
        
        # 4. Performance variability across seeds
        self._plot_performance_variability()
    
    def _plot_overall_accuracy_progression(self) -> None:
        """Plot overall accuracy progression for each seed."""
        plt.figure(figsize=(12, 8))
        
        stages = ['Backbone', 'Task 1', 'Task 2', 'Task 3']
        
        for result in self.results:
            if 'error' in result:
                continue
                
            seed = result['seed']
            
            # Calculate overall accuracies
            overall_accs = []
            
            # Backbone
            if 'backbone_metrics' in result:
                backbone_overall = result['backbone_metrics']['best_val_acc']
                overall_accs.append(backbone_overall)
            
            # CIL tasks
            if 'cil_metrics' in result:
                for task_metrics in result['cil_metrics']['task_metrics']:
                    overall_accs.append(task_metrics['test_overall_acc'])
            
            plt.plot(stages[:len(overall_accs)], overall_accs, 
                    marker='o', linewidth=2, markersize=6, 
                    label=f'Seed {seed}', alpha=0.8)
        
        plt.title('Overall Accuracy Progression Across Seeds', fontsize=16, fontweight='bold')
        plt.xlabel('Training Stage', fontsize=12, fontweight='bold')
        plt.ylabel('Overall Accuracy (%)', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, "seeded_overall_accuracy_progression.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Overall accuracy progression plot saved to: {plot_file}")
    
    def _plot_per_class_accuracy_heatmap(self) -> None:
        """Create heatmap showing per-class accuracy across seeds and stages."""
        # Collect data
        heatmap_data = []
        
        for result in self.results:
            if 'error' in result:
                continue
                
            seed = result['seed']
            
            # Get all behaviors
            all_behaviors = set()
            if 'backbone_metrics' in result:
                backbone_acc = result['backbone_metrics']['final_val_class_acc']
                all_behaviors.update(backbone_acc.keys())
            
            if 'cil_metrics' in result:
                for task_metrics in result['cil_metrics']['task_metrics']:
                    all_behaviors.update(task_metrics['test_class_acc'].keys())
            
            all_behaviors = sorted(list(all_behaviors))
            
            # Create row for this seed
            for behavior in all_behaviors:
                row = {'seed': seed, 'behavior': behavior}
                
                # Backbone accuracy
                if 'backbone_metrics' in result:
                    backbone_acc = result['backbone_metrics']['final_val_class_acc']
                    row['backbone'] = backbone_acc.get(behavior, 0)
                else:
                    row['backbone'] = 0
                
                # Task accuracies
                if 'cil_metrics' in result:
                    for i, task_metrics in enumerate(result['cil_metrics']['task_metrics']):
                        row[f'task{i+1}'] = task_metrics['test_class_acc'].get(behavior, 0)
                else:
                    for i in range(3):  # Assume 3 tasks
                        row[f'task{i+1}'] = 0
                
                heatmap_data.append(row)
        
        if not heatmap_data:
            return
        
        df = pd.DataFrame(heatmap_data)
        
        # Create heatmap for each behavior
        behaviors = df['behavior'].unique()
        n_behaviors = len(behaviors)
        
        fig, axes = plt.subplots(2, (n_behaviors + 1) // 2, figsize=(15, 10))
        if n_behaviors == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, behavior in enumerate(behaviors):
            behavior_data = df[df['behavior'] == behavior]
            
            # Prepare data for heatmap
            heatmap_matrix = []
            stages = ['backbone', 'task1', 'task2', 'task3']
            
            for seed in sorted(behavior_data['seed'].unique()):
                seed_data = behavior_data[behavior_data['seed'] == seed]
                if len(seed_data) > 0:
                    row = [seed_data.iloc[0][stage] for stage in stages]
                    heatmap_matrix.append(row)
            
            if heatmap_matrix:
                heatmap_matrix = np.array(heatmap_matrix)
                
                im = axes[i].imshow(heatmap_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
                axes[i].set_title(f'{behavior}', fontweight='bold')
                axes[i].set_xlabel('Training Stage')
                axes[i].set_ylabel('Seed')
                axes[i].set_xticks(range(len(stages)))
                axes[i].set_xticklabels(stages, rotation=45)
                axes[i].set_yticks(range(len(sorted(behavior_data['seed'].unique()))))
                axes[i].set_yticklabels(sorted(behavior_data['seed'].unique()))
                
                # Add text annotations
                for j in range(len(heatmap_matrix)):
                    for k in range(len(heatmap_matrix[0])):
                        text = axes[i].text(k, j, f'{heatmap_matrix[j, k]:.1f}',
                                          ha="center", va="center", color="black", fontweight='bold')
        
        # Hide unused subplots
        for i in range(len(behaviors), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Per-Class Accuracy Heatmap Across Seeds', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, "seeded_per_class_accuracy_heatmap.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Per-class accuracy heatmap saved to: {plot_file}")
    
    def _plot_resource_metrics_comparison(self) -> None:
        """Plot resource metrics comparison across seeds."""
        resource_data = []
        
        for result in self.results:
            if 'error' in result or 'resource_metrics' not in result:
                continue
                
            seed = result['seed']
            metrics = result['resource_metrics']
            
            resource_data.append({
                'seed': seed,
                'flash_total_int8_sparse': metrics.get('flash_total_int8_sparse', 0),
                'train_sram_total_bytes': metrics.get('train_sram_total_bytes', 0),
                'forward_macs_total_sparse_per_sample': metrics.get('forward_macs_total_sparse_per_sample', 0),
                'head_nonzero_frac': metrics.get('head_nonzero_frac', 0)
            })
        
        if not resource_data:
            return
        
        df = pd.DataFrame(resource_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Flash memory
        axes[0, 0].bar(df['seed'], df['flash_total_int8_sparse'] / 1024, alpha=0.7)
        axes[0, 0].set_title('Flash Memory Usage (KB)', fontweight='bold')
        axes[0, 0].set_xlabel('Seed')
        axes[0, 0].set_ylabel('Flash Memory (KB)')
        
        # Training SRAM
        axes[0, 1].bar(df['seed'], df['train_sram_total_bytes'] / 1024, alpha=0.7)
        axes[0, 1].set_title('Training SRAM Usage (KB)', fontweight='bold')
        axes[0, 1].set_xlabel('Seed')
        axes[0, 1].set_ylabel('Training SRAM (KB)')
        
        # MACs
        axes[1, 0].bar(df['seed'], df['forward_macs_total_sparse_per_sample'], alpha=0.7)
        axes[1, 0].set_title('Forward MACs per Sample', fontweight='bold')
        axes[1, 0].set_xlabel('Seed')
        axes[1, 0].set_ylabel('MACs')
        
        # Sparsity
        axes[1, 1].bar(df['seed'], df['head_nonzero_frac'], alpha=0.7)
        axes[1, 1].set_title('Head Sparsity (Non-zero Fraction)', fontweight='bold')
        axes[1, 1].set_xlabel('Seed')
        axes[1, 1].set_ylabel('Non-zero Fraction')
        
        plt.suptitle('Resource Metrics Comparison Across Seeds', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, "seeded_resource_metrics_comparison.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Resource metrics comparison plot saved to: {plot_file}")
    
    def _plot_performance_variability(self) -> None:
        """Plot performance variability across seeds."""
        performance_data = []
        
        for result in self.results:
            if 'error' in result:
                continue
                
            seed = result['seed']
            
            # Backbone accuracy
            backbone_avg = None
            if 'backbone_metrics' in result:
                backbone_avg = result['backbone_metrics']['best_val_acc']
            
            # Final task accuracy
            final_acc = None
            if 'cil_metrics' in result and result['cil_metrics']['task_metrics']:
                final_acc = result['cil_metrics']['task_metrics'][-1]['test_overall_acc']
            
            performance_data.append({
                'seed': seed,
                'backbone_accuracy': backbone_avg,
                'final_task_accuracy': final_acc
            })
        
        if not performance_data:
            return
        
        df = pd.DataFrame(performance_data)
        
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(df))
        width = 0.35
        
        plt.bar(x - width/2, df['backbone_accuracy'], width, label='Backbone', alpha=0.8)
        plt.bar(x + width/2, df['final_task_accuracy'], width, label='Final Task', alpha=0.8)
        
        plt.xlabel('Seed', fontweight='bold')
        plt.ylabel('Accuracy (%)', fontweight='bold')
        plt.title('Performance Variability Across Seeds', fontsize=16, fontweight='bold')
        plt.xticks(x, df['seed'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for i, (backbone, final) in enumerate(zip(df['backbone_accuracy'], df['final_task_accuracy'])):
            plt.text(i - width/2, backbone + 1, f'{backbone:.1f}%', ha='center', va='bottom', fontweight='bold')
            plt.text(i + width/2, final + 1, f'{final:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, "seeded_performance_variability.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance variability plot saved to: {plot_file}")


def main():
    """
    Main function to run seeded metrics collection.
    """
    # Configuration
    config_path = "dog_behaviors_config.yaml"
    output_dir = "plots_and_metrics_seeded"
    seeds = [42]  # You can modify these seeds
    
    print("Starting Seeded Metrics Collection for Dog Behavior CIL Pipeline")
    print(f"Seeds to run: {seeds}")
    print(f"Output directory: {output_dir}")
    
    # Create collector
    collector = SeededMetricsCollector(
        config_path=config_path,
        output_dir=output_dir,
        seeds=seeds
    )
    
    # Run all seeds
    collector.run_all_seeds()
    
    # Create plots
    print("\nCreating plots...")
    # collector.create_plots()
    
    print(f"\n{'='*60}")
    print("SEEDED METRICS COLLECTION COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}/")
    print("Generated files:")
    print("  - seeded_results_*.json (detailed results)")
    print("  - seeded_accuracy_metrics.csv (comprehensive accuracy data)")
    print("  - seeded_resource_metrics.csv (resource data)")
    print("  - seeded_performance_summary.csv (summary)")
    print("  - seeded_cl_metrics.csv (continual learning metrics)")
    # print("  - seeded_*.png (plots)")


if __name__ == "__main__":
    main()
