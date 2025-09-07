# cnn_CIL_training_with_metrics.py


import torch
torch.use_deterministic_algorithms(True)  # PyTorch ≥1.8
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils import load_config, load_processed_dog_data, evaluate_all_classes, set_seed, make_global_to_local_map, ClassRegistry
# from metrics import profile_cil_memory, print_cil_memory_report
from metrics import profile_cil_resources
from models.cnn import SimplifiedTDMModelCNN
from cnn_backbone_training_utils import train_cnn_backbone, load_data_cnn_backbone
import logging
from cnn_CIL_training_utils import load_CIL_data, CIL_post_task_eval, train_with_simplified_tdm, make_CIL_plots, check_CIL_validity
from replay import BalancedQuantReplayDynamic
from report_plotting import CILTrainingPlotter
import random
import pandas as pd
import numpy as np
from datetime import datetime
import os

def calculate_cl_metrics_from_matrix(R_full, random_baseline_full, K_plus_1):
    # R_full: (K+1)x(K+1) including backbone at index 0.
    # K_plus_1: number of rows/cols to consider (backbone + K tasks).
    R_tasks = R_full[1:K_plus_1, 1:K_plus_1]       # K x K tasks-only
    K = R_tasks.shape[0]
    baselines = random_baseline_full[1:K_plus_1]   # length K, in %

    # ACC_K: last row over tasks only
    avg_acc = float(np.mean(R_tasks[-1,:])) if K > 0 else 0.0

    # A_K
    if K > 0:
        s = 0.0
        for i in range(K):
            for j in range(i+1):
                s += R_tasks[i,j]
        avg_inc_acc = (2.0/(K*(K+1))) * s
    else:
        avg_inc_acc = 0.0

    # AF_K
    if K > 1:
        s = 0.0
        for j in range(K-1):
            s += float(np.max(R_tasks[:K-1, j])) - float(R_tasks[K-1, j])
        avg_forg = s / (K-1)
    else:
        avg_forg = 0.0

    # BWT_K
    if K > 1:
        s = 0.0
        for j in range(K-1):
            s += float(R_tasks[K-1, j]) - float(R_tasks[j, j])
        bwt = s / (K-1)
    else:
        bwt = 0.0

    # FWT_K uses R_full[j-1, j]
    if K > 1:
        s = 0.0
        for j in range(1, K+1):  # j=1..K in task indexing
            s += float(R_full[j-1, j]) - float(baselines[j-1])
        fwt = s / (K-1)
    else:
        fwt = 0.0

    # κ (modified BWT)
    if K > 1:
        s = 0.0
        for i in range(1, K):
            for j in range(i):
                s += float(R_tasks[i, j]) - float(R_tasks[j, j])
        kappa = (2.0/(K*(K-1))) * s
    else:
        kappa = 0.0

    # ζ (modified FWT)
    if K > 1:
        s = 0.0
        for i in range(K):
            for j in range(i+1, K):
                s += float(R_tasks[i, j])
        zeta = (2.0/(K*(K-1))) * s
    else:
        zeta = 0.0

    # Intransigence
    if K > 0:
        optimal = np.max(R_tasks, axis=0)
        actual  = R_tasks[-1, :]
        intransigence = float(np.mean(optimal - actual))
    else:
        intransigence = 0.0
    print(f"R: {R_full}")
    return {"avg_acc": avg_acc, "avg_inc_acc": avg_inc_acc, "avg_forg": avg_forg,
            "bwt": bwt, "fwt": fwt, "kappa": kappa, "zeta": zeta, "intransigence": intransigence}
    

def save_metrics_to_csv(seed, backbone_class_acc, accuracy_history, training_loss_history, cl_metrics, resource_report, output_dir="plots_and_metrics_seeded"):
    """
    Save all metrics to CSV files for analysis across multiple seeds.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save backbone validation metrics
    backbone_data = []
    for behavior, acc in backbone_class_acc.items():
        backbone_data.append({
            'seed': seed,
            'stage': 'Backbone_Val',
            'behavior': behavior,
            'accuracy': acc,
            'overall_accuracy': backbone_class_acc.get('overall_acc', max(backbone_class_acc.values())),
            'metric_type': 'validation'
        })
    
    # 2. Save CIL task metrics
    cil_data = []
    for task_idx, task_metrics in enumerate(accuracy_history):
        task_name = f"Task_{task_idx + 1}"
        
        # Test metrics (per-class and overall)
        for behavior, acc in task_metrics['class_acc'].items():
            cil_data.append({
                'seed': seed,
                'stage': f'{task_name}_Test',
                'behavior': behavior,
                'accuracy': acc,
                'overall_accuracy': task_metrics['overall_acc'],
                'metric_type': 'test'
            })
    
    # 2.5. Save CIL training loss metrics
    training_loss_data = []
    for task_loss_metrics in training_loss_history:
        task_name = task_loss_metrics['task_name']
        
        # Add summary training loss metrics
        training_loss_data.append({
            'seed': seed,
            'stage': f'{task_name}_Training',
            'task_idx': task_loss_metrics['task_idx'],
            'final_training_loss': task_loss_metrics['final_training_loss'],
            'avg_training_loss': task_loss_metrics['avg_training_loss'],
            'metric_type': 'training_loss'
        })
        
        # Add per-epoch training losses
        for epoch, loss in enumerate(task_loss_metrics['training_losses']):
            training_loss_data.append({
                'seed': seed,
                'stage': f'{task_name}_Training_Epoch_{epoch}',
                'task_idx': task_loss_metrics['task_idx'],
                'epoch': epoch,
                'training_loss': loss,
                'metric_type': 'training_loss_epoch'
            })
    
    # 3. Save continual learning metrics (per-task)
    cl_data = []
    for task_metrics in cl_metrics:
        task_id = task_metrics.get('task', 'unknown')
        for metric_name, value in task_metrics.items():
            if metric_name != 'task':
                cl_data.append({
                    'seed': seed,
                    'task': task_id,
                    'metric': metric_name,
                    'value': value
                })
    
    # 4. Save resource metrics
    resource_data = []
    if hasattr(resource_report, 'to_dict'):
        resource_dict = resource_report.to_dict()
        for metric_name, value in resource_dict.items():
            resource_data.append({
                'seed': seed,
                'metric': metric_name,
                'value': value
            })
    
    # Save to CSV files
    backbone_df = pd.DataFrame(backbone_data)
    cil_df = pd.DataFrame(cil_data)
    training_loss_df = pd.DataFrame(training_loss_data)
    cl_df = pd.DataFrame(cl_data)
    resource_df = pd.DataFrame(resource_data)
    
    # Append to existing files or create new ones
    backbone_file = os.path.join(output_dir, "seeded_accuracy_metrics.csv")
    training_loss_file = os.path.join(output_dir, "seeded_training_loss_metrics.csv")
    cl_file = os.path.join(output_dir, "seeded_cl_metrics.csv")
    resource_file = os.path.join(output_dir, "seeded_resource_metrics.csv")
    
    # Append backbone and CIL data to accuracy file
    accuracy_df = pd.concat([backbone_df, cil_df], ignore_index=True)
    if os.path.exists(backbone_file) and os.path.getsize(backbone_file) > 0:
        try:
            existing_df = pd.read_csv(backbone_file)
            combined_df = pd.concat([existing_df, accuracy_df], ignore_index=True)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            # File exists but is empty or corrupted, start fresh
            combined_df = accuracy_df
    else:
        combined_df = accuracy_df
    combined_df.to_csv(backbone_file, index=False)
    
    # Append training loss metrics
    if os.path.exists(training_loss_file) and os.path.getsize(training_loss_file) > 0:
        try:
            existing_df = pd.read_csv(training_loss_file)
            combined_df = pd.concat([existing_df, training_loss_df], ignore_index=True)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            # File exists but is empty or corrupted, start fresh
            combined_df = training_loss_df
    else:
        combined_df = training_loss_df
    combined_df.to_csv(training_loss_file, index=False)
    
    # Append CL metrics
    if os.path.exists(cl_file) and os.path.getsize(cl_file) > 0:
        try:
            existing_df = pd.read_csv(cl_file)
            combined_df = pd.concat([existing_df, cl_df], ignore_index=True)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            # File exists but is empty or corrupted, start fresh
            combined_df = cl_df
    else:
        combined_df = cl_df
    combined_df.to_csv(cl_file, index=False)
    
    # Append resource metrics
    if os.path.exists(resource_file) and os.path.getsize(resource_file) > 0:
        try:
            existing_df = pd.read_csv(resource_file)
            combined_df = pd.concat([existing_df, resource_df], ignore_index=True)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            # File exists but is empty or corrupted, start fresh
            combined_df = resource_df
    else:
        combined_df = resource_df
    combined_df.to_csv(resource_file, index=False)
    
    print(f"Metrics saved to CSV files in {output_dir}/")

def main():
    """Main function"""
    # Set fixed random seeds for reproducibility
    
    # Load configuration
    cfg = load_config()
    seed = cfg['random_seed']
    set_seed(seed)
    gen_seed = torch.Generator()
    gen_seed.manual_seed(seed)
    
    # Load intelligent sampling data
    dog_data, behavior_to_idx, behaviors = load_processed_dog_data()

    if len(dog_data) == 0:
        print("No dog data loaded!")
        return
    
    # Define behaviors 
    backbone_behaviors = cfg['backbone_behaviors']
    cil_behaviors = cfg['cil_behaviors']
    all_behaviors = backbone_behaviors + cil_behaviors

    # CIL task progression
    cil_tasks = cfg['cil_tasks']
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    #####################################################################################################################
    #--------------------------------------- STEP 1: BACKBONE SETTUP 
    #####################################################################################################################
    
    print(f"\n{'='*50}")
    print("STEP 1: BACKBONE TRAINING")
    print(f"{'='*50}")
 
    
    # Create backbone model
    backbone_num_classes = len(backbone_behaviors)
    C = len(dog_data[list(dog_data.keys())[0]]['sensor_cols'])
    L = int(dog_data[list(dog_data.keys())[0]]['window_len'])
    backbone_model = SimplifiedTDMModelCNN(C, backbone_num_classes, device, sparsity_ratio=cfg['backbone_sparsity_ratio'])
    backbone_model.to(device)
    
    # Prepare backbone training data 
    # Exclude target dog and validation dog from backbone training
    all_possible_dogs = sorted(dog_data.keys())
    valid_CIL_dogs_for_this_seed = [
        d for d in all_possible_dogs
        if check_CIL_validity(dog_data, d, all_behaviors, behavior_to_idx, verbose=True)
    ] 
    
    print(f"Valid CIL dogs for seed {seed}: {valid_CIL_dogs_for_this_seed}")
    if not valid_CIL_dogs_for_this_seed:
        raise ValueError("No dogs satisfy nonzero train+test counts for all 7 classes under this seed/split.")
    
    rng = random.Random(seed)  
    
    # pick CIL target dog
    target_dog = rng.choice(valid_CIL_dogs_for_this_seed)
    print(f"Target Dog Chosen: {target_dog}")
    
    # pick validation dog (not same as target)
    remaining = sorted([d for d in all_possible_dogs if d != target_dog])
    validation_dog = rng.choice(remaining)
    print(f"Validation Dog Chosen: {validation_dog}")
    
    # Exclude target dog and validation dog from backbone training
    remaining = sorted([dog for dog in all_possible_dogs if dog not in [target_dog, validation_dog]])
    backbone_dogs = rng.sample(remaining, cfg['num_pretrain_dogs']) # randomly selection num of pretraining dogs
    print(f"Backbone Dogs Chosen: {backbone_dogs}")

    logging.info(f"Backbone training dogs: {backbone_dogs}")
    logging.info(f"Backbone validation dog: {validation_dog}")
    logging.info(f"Target dog for CIL: {target_dog}")
    
    backbone_data = load_data_cnn_backbone(backbone_dogs, dog_data, behavior_to_idx, backbone_behaviors, validation_dog)
    
    # Create data loaders
    backbone_train_loader = DataLoader(TensorDataset(backbone_data.X_train, backbone_data.y_train), 
                                      batch_size=cfg['backbone_batch_size_train'], shuffle=True, generator = gen_seed)
    backbone_val_loader = DataLoader(TensorDataset(backbone_data.X_val, backbone_data.y_val), 
                                    batch_size=cfg['backbone_batch_size_val'], shuffle=False)
    
    backbone_optimizer = torch.optim.Adam(backbone_model.parameters(), lr=cfg['backbone_learning_rate'], weight_decay = cfg['backbone_weight_decay'])
    backbone_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # Initialize plotter for tracking metrics
    plotter = CILTrainingPlotter()
    
    #####################################################################################################################
    #--------------------------------------- STEP 2: BACKBONE TRAINING AND EVALUATION
    #####################################################################################################################
    
    backbone_model, _ = train_cnn_backbone(backbone_model, backbone_behaviors, backbone_optimizer, backbone_criterion, backbone_train_loader, backbone_val_loader, device, cfg, plotter)
    
    #####################################################################################################################
    #--------------------------------------- STEP 3: SET UP CIL MODEL
    #####################################################################################################################
    
    print(f"\n{'='*50}")
    print("STEP 2: CIL TRAINING")
    print(f"{'='*50}")
 
    # Freeze backbone
    for param in backbone_model.backbone.parameters():
        param.requires_grad = False
    
    # Create CIL model
    cil_model = SimplifiedTDMModelCNN(C, backbone_num_classes, device, sparsity_ratio=cfg['sparsity_ratio']).to(device)
    cil_model.backbone.load_state_dict(backbone_model.backbone.state_dict())
    
    # Freeze CIL backbone
    for param in cil_model.backbone.parameters():
        param.requires_grad = False
        
    cil_model.backbone.eval()
    
    #####################################################################################################################        
    #--------------------------------------- STEP 4: SET UP CIL SCENARIO
    ##################################################################################################################### 
    if target_dog not in dog_data:
        print(f"Target dog {target_dog} not found!")
        return
    
    X_train, y_train, X_test, y_test = load_CIL_data(dog_data, target_dog, all_behaviors, behavior_to_idx, backbone_data.train_mean, backbone_data.train_std, seed, verbose = True)
    
    # Create evaluation classifier for all 7 classes
    backbone_registry = ClassRegistry(all_behaviors)
    backbone_registry.add_classes(backbone_behaviors)
    
    # Remap test labels to 0-6 range for evaluation
    m_test, gids_all = make_global_to_local_map(all_behaviors, behavior_to_idx, device=y_test.device)
    y_test_remapped = m_test[y_test]   # vectorized 0..6 mapping
        
    test_loader = DataLoader(TensorDataset(X_test, y_test_remapped), batch_size=cfg['CIL_batch_size_test'], shuffle=False)
    
    # Evaluate backbone on all 7 classes for target dog
    overall_acc, class_acc = evaluate_all_classes(backbone_model, test_loader, backbone_registry, all_behaviors, device)
    
    print(f"Backbone overall accuracy on all 7 classes: {overall_acc:.2f}%")
    print(f"Backbone per-class accuracy:")
    for cls, acc in class_acc.items():
        print(f"  {cls}: {acc:.2f}%")
    
    
    # Initialize accuracy matrix R for CL metrics calculation
    # R[i,j] = accuracy on task j after training on task i
    # R[0,:] = backbone performance on all tasks
    # R[1,:] = performance after task 1, etc.
    num_tasks = len(cil_tasks)
    accuracy_matrix = np.zeros((num_tasks + 1, num_tasks + 1))  # 4x4 matrix (backbone + 3 tasks)
    random_baseline = np.zeros(num_tasks + 1)  # Random accuracy for each task (including backbone)
    
    
    # calculate backbone first! (only on 4 classes instead of 7)
    # R[0,0] = backbone accuracy on backbone task (4 classes)
    bb_map, bb_gids = make_global_to_local_map(backbone_behaviors, behavior_to_idx, device=y_test.device)
    bb_mask = torch.isin(y_test, bb_gids)
    bb_X = X_test[bb_mask]
    bb_y_global = m_test[y_test[bb_mask]]  # GLOBAL labels 0..6

    if len(bb_X) > 0:
        bb_loader = DataLoader(TensorDataset(bb_X, bb_y_global),
                            batch_size=cfg['CIL_batch_size_test'], shuffle=False)
        bb_class_acc, bb_overall_acc = CIL_post_task_eval(backbone_behaviors, bb_loader,
                                                        device, all_behaviors, backbone_model, backbone_registry)
        accuracy_matrix[0, 0] = bb_overall_acc
    
  
    
    
    
    #####################################################################################################################
    #--------------------------------------- STEP 5: PERFORM CIL SCENARIO
    #####################################################################################################################
    
    # Initialize replay buffer
    replay_buffer_q = BalancedQuantReplayDynamic(buffer_size=cfg['buffer_size'])
    
    registry = ClassRegistry(all_behaviors)
    registry.add_classes(backbone_behaviors)  # seed with 4 base classes
    
    seen_names = registry.seen_classes()              # ['Standing', 'Walking', 'Sitting', 'Lying chest']
    _, seen_gids = make_global_to_local_map(seen_names, behavior_to_idx, device="cpu")
    replay_buffer_q.on_new_task(seen_gids.tolist())   # buffer wants global class IDs
    
    accuracy_history = []
    training_loss_history = []  # Store training losses for each task
    backbone_class_acc = class_acc
    
    
        
    # Measure Sizing! 
    report = profile_cil_resources(
        cfg,
        cil_model, C=C, L=L,
        batch_size_train=cfg['CIL_batch_size_train'],
        batch_size_infer=cfg['CIL_batch_size_test'],
        replay_size=cfg['buffer_size'],
        optimizer_kind="adam",
        deployed_bits_backbone=32,
        deployed_bits_classifier=8,
        training_bits_classifier=32,
        replay_bits=8,
        activation_bits_train=32,
        activation_bits_infer=32,
        replay_batch_size=cfg['replay_batch_size'],  
        kd_prev_rows=len(backbone_behaviors),         
        kd_enabled=True,                              
        autograd_grad_for_cwi=True                    
    )
    
    # TDM parameters 
    p_intra = cfg['p_intra']   
    p_inter = cfg['p_inter']   
    delta_k = cfg['delta_k']    
    
    
    #---------------------------- CIL PRE TASK PIPELINE (TASK NUMBER LEVEL)
    for task_idx, task_classes in enumerate(cil_tasks):
        # Determine which new class is being added
        if task_idx == 0:
            new_class = "Sniffing"
        elif task_idx == 1:
            new_class = "Trotting"
        else:
            new_class = "Galloping"
        
        print(f"\n--- CIL TASK {task_idx + 1}: Added '{new_class}' ---")
        
        # 1) Expand head for new classes
        prev_num = registry.num_classes()
        if len(task_classes) > prev_num:
            new_cls = task_classes[prev_num:]  # true new names
            cil_model.expand_head(new_cls, registry)

        cil_model.eval()
        
        #----------------- FWT pre-eval for new task j
        # Pre-eval BEFORE adding the new class
        eval_map, eval_gids = make_global_to_local_map([new_class], behavior_to_idx, device=y_test.device)
        mask = torch.isin(y_test, eval_gids)
        X_pre = X_test[mask]
        y_pre = m_test[y_test[mask]]

        loader = DataLoader(TensorDataset(X_pre, y_pre), batch_size=cfg['CIL_batch_size_test'], shuffle=False)

        # Evaluate with current head (no row for Tj yet → accuracy ≈ random baseline)
        _, pre_overall = CIL_post_task_eval(registry.seen_classes(), loader, device, all_behaviors, cil_model, registry)

        accuracy_matrix[task_idx, task_idx+1] = pre_overall
        print(f"FWT pre-eval on {new_class}: {pre_overall:.2f}%")
        
        
        # 2) Inter-task expansion (warm up grow before epochs)
        if task_idx > 0:
            cil_model.head.update_mask_inter(p = p_inter)
         
         # reresh seen set  
        seen_names = registry.seen_classes()
        _, seen_gids = make_global_to_local_map(seen_names, behavior_to_idx, device="cpu")
        replay_buffer_q.on_new_task(seen_gids.tolist())
        
        # 3) Prepare training data 
        map_task, task_gids = make_global_to_local_map(task_classes, behavior_to_idx, device=y_train.device)

        mask = torch.isin(y_train, task_gids)
        X_task = X_train[mask]
        y_task = y_train[mask]
        y_task_remapped = map_task[y_task]   # vectorized mapping to 0..len(task_classes)-1

        
        # 4) Create teacher snapshot over the previous classes for DDR/KD
        train_loader = DataLoader(TensorDataset(X_task, y_task_remapped), 
                                batch_size=cfg['CIL_batch_size_train'], shuffle=True, generator = gen_seed)
        
        teacher = None  
        if prev_num > 0:  
            teacher = nn.Linear(cil_model.head.in_dim, prev_num, bias=True).to(device)
            with torch.no_grad():
                teacher.weight.copy_(cil_model.head.linear.weight[:prev_num])
                teacher.bias.copy_(cil_model.head.linear.bias[:prev_num])
            teacher.eval()
        
        optimizer = torch.optim.Adam(cil_model.head.parameters(), lr=cfg['CIL_learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        #---------------------------- CIL DURING TASK PIPELINE (EPOCH NUMBER LEVEL)
        task_training_losses = []  # Store losses for this task
        for epoch in range(cfg['CIL_epochs']):
            # Ensure frozen
            cil_model.head.train()
            cil_model.backbone.eval()
            
            # Learning rate scheduling
            if epoch == 5:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
            elif epoch == 10:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                    
            # Intra-task TDM adjustment
            if epoch > 0 and epoch % delta_k == 0:
                cil_model.head.update_mask_intra(p = p_intra)
   
            # Inter-task warm-up end (shrink back to target sparsity)
            if task_idx > 0 and epoch == delta_k:
                cil_model.head.shrink_after_warmup(p=p_inter)
                
            # Train and get metrics
            train_loss, train_acc = train_with_simplified_tdm(cil_model, cfg, registry, task_classes, teacher, prev_num, train_loader, behavior_to_idx, optimizer, criterion, device, 
                                    replay_buffer_q, seen_gids.tolist(), seen_names, task_idx, epoch)
            
            # Store training loss for this epoch
            task_training_losses.append(train_loss)
            
            # Add training metrics to plotter
            task_name = f"Task {task_idx+1}: {new_class}"
            plotter.add_cil_task_training_epoch(task_name, epoch, train_loss, train_acc)

        #---------------------------- CIL POST TASK PIPELINE (TASK NUMBER LEVEL)
        # Evaluate on all *seen* classes (no future classes!)
        
        cil_model.eval()
        test_loader = DataLoader(TensorDataset(X_test, y_test_remapped), batch_size=cfg['CIL_batch_size_test'], shuffle=False)
        
        # note task_classes = seen_classes --> classes up through the finished task
        class_acc, overall_acc = CIL_post_task_eval(task_classes, test_loader, device, all_behaviors, cil_model, registry)
        
        # Store both class accuracies and overall accuracy for proper weighted calculation
        accuracy_history.append({
            'class_acc': class_acc.copy(),
            'overall_acc': overall_acc
        })
        
        # Store training losses for this task
        training_loss_history.append({
            'task_idx': task_idx,
            'task_name': f"Task_{task_idx + 1}",
            'training_losses': task_training_losses.copy(),
            'final_training_loss': task_training_losses[-1] if task_training_losses else 0.0,
            'avg_training_loss': sum(task_training_losses) / len(task_training_losses) if task_training_losses else 0.0
        })
        
        # Fill accuracy matrix R[i,j] = accuracy on task j after training on task i
        # For each task j that we've seen so far, evaluate on task j's classes
        for eval_task_idx in range(task_idx + 1):
            eval_task_classes = cil_tasks[eval_task_idx]
            
            # Create test loader for this specific task
            eval_map, eval_gids = make_global_to_local_map(eval_task_classes, behavior_to_idx, device=y_test.device)
            eval_mask = torch.isin(y_test, eval_gids)
            eval_X = X_test[eval_mask]
            eval_y_global = m_test[y_test[eval_mask]]  # GLOBAL labels
                        
            if len(eval_X) > 0:  # Make sure we have data for this task
                eval_test_loader = DataLoader(TensorDataset(eval_X, eval_y_global), 
                                            batch_size=cfg['CIL_batch_size_test'], shuffle=False)
                
                # Evaluate current model on this task
                eval_class_acc, eval_overall_acc = CIL_post_task_eval(eval_task_classes, eval_test_loader, 
                                                                     device, all_behaviors, cil_model, registry)
                
                # Store in accuracy matrix (shift indices by 1 to account for backbone as task 0)
                accuracy_matrix[task_idx + 1, eval_task_idx + 1] = eval_overall_acc
                
                # Calculate random baseline for this task (if not already done)
                if random_baseline[eval_task_idx + 1] == 0:
                    random_baseline[eval_task_idx + 1] = 100.0 / len(eval_task_classes)  # Random accuracy in percentage
        
        # Also evaluate on backbone task (task 0) after each CIL task
        # R[task_idx+1, 0] = accuracy on backbone classes after training on task i
        backbone_map, backbone_gids = make_global_to_local_map(backbone_behaviors, behavior_to_idx, device=y_test.device)
        backbone_mask = torch.isin(y_test, backbone_gids)
        backbone_X = X_test[backbone_mask]
        backbone_y = y_test[backbone_mask]
        backbone_y_global = m_test[backbone_y]  # not backbone_map[..]
        
        if len(backbone_X) > 0:
            backbone_test_loader = DataLoader(TensorDataset(backbone_X, backbone_y_global), 
                                            batch_size=cfg['CIL_batch_size_test'], shuffle=False)
            
            # Evaluate current model on backbone classes
            cil_model.eval()
            backbone_class_acc, backbone_overall_acc = CIL_post_task_eval(backbone_behaviors, backbone_test_loader, 
                                                                         device, all_behaviors, cil_model, registry)
            
            # Store in accuracy matrix: R[task_idx+1, 0] = accuracy on backbone after task i
            accuracy_matrix[task_idx + 1, 0] = backbone_overall_acc
            
            print(f"  Backbone classes: {backbone_overall_acc:.2f}%")
        
        # --- fill R[i,j] for j>i
        for future_idx in range(task_idx+1, num_tasks):
            future_classes = cil_tasks[future_idx]
            f_map, f_gids = make_global_to_local_map(future_classes, behavior_to_idx, device=y_test.device)
            f_mask = torch.isin(y_test, f_gids)
            Xf = X_test[f_mask]
            yf_global = m_test[y_test[f_mask]]

            if len(Xf) > 0:
                f_loader = DataLoader(
                    TensorDataset(Xf, yf_global),
                    batch_size=cfg['CIL_batch_size_test'], shuffle=False
                )

                cil_model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for xb, yb in f_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = cil_model(xb)        # use current full head
                        preds = logits.argmax(dim=1)  # pick max logit
                        correct += (preds == yb).sum().item()
                        total += yb.size(0)

                f_overall = (correct / total) * 100 if total > 0 else 0.0
                accuracy_matrix[task_idx+1, future_idx+1] = f_overall
        
        # Add to plotter
        task_name = f"Task {task_idx+1}: {new_class}"
        plotter.add_cil_task(task_name, class_acc, overall_acc)
        
        if task_idx + 1 == len(cil_tasks):
            for cls, acc in class_acc.items():
                print(f"  {cls}: {acc:.2f}%")
            print("Final overall accuracy:", overall_acc)
            
        else:
            print("Eval so far:", overall_acc)
            for cls, acc in class_acc.items():
                print(f"  {cls}: {acc:.2f}%")
    
    print("CIL PIPELINE FINISHED!")

    # Calculate continual learning metrics from accuracy matrix
    print("\n" + "="*50)
    print("CALCULATING CONTINUAL LEARNING METRICS FROM ACCURACY MATRIX")
    print("="*50)
    
    # Print accuracy matrix for debugging
    print("Accuracy Matrix R (rows=after training on task, cols=accuracy on task):")
    print("Task\\After:", end="")
    print("  B", end="")  # Backbone
    for j in range(num_tasks):
        print(f"  T{j+1}", end="")
    print()
    print("After B:", end="")  # Backbone row
    for j in range(num_tasks + 1):
        print(f"  {accuracy_matrix[0,j]:.2f}", end="")
    print()
    for i in range(num_tasks):
        print(f"After T{i+1}:", end="")
        for j in range(num_tasks + 1):
            print(f"  {accuracy_matrix[i+1,j]:.2f}", end="")
        print()
    
    # Calculate CL metrics for each task (progressive)
    all_cl_metrics = []
    
    for K in range(1, num_tasks + 1):  # K = 1, 2, 3
        print(f"\n--- CL Metrics after Task {K} ---")
        
        # Use submatrix R[0:K+1, 0:K+1] for calculation (including backbone)
        R_sub = accuracy_matrix[:K+1, :K+1]
        baseline_sub = random_baseline[:K+1]
        
        # Calculate CL metrics for this K (K+1 because we include backbone)
        task_cl_metrics = calculate_cl_metrics_from_matrix(R_sub, baseline_sub, K+1)
        task_cl_metrics['task'] = K
        all_cl_metrics.append(task_cl_metrics)
        
        print(f"Task {K} CL Metrics:")
        for metric_name, value in task_cl_metrics.items():
            if metric_name != 'task':
                print(f"  {metric_name}: {value:.2f}%")
    
    # Save all metrics to CSV (including per-task CL metrics)
    save_metrics_to_csv(seed, backbone_class_acc, accuracy_history, training_loss_history, all_cl_metrics, report)
    
    # Save results
    results = {
        'accuracy_history': accuracy_history,
        'final_accuracy': overall_acc,
        'final_class_acc': class_acc,
        'cl_metrics': all_cl_metrics,
        'backbone_class_acc': backbone_class_acc
    }
    
    # Generate all plots and reports
    print("\n" + "="*50)
    print("GENERATING PLOTS AND REPORTS")
    print("="*50)
    
    # Plot backbone training
    plotter.plot_backbone_training()
    
    # Plot CIL progression
    # plotter.plot_cil_progression()
    
    # Plot CIL training curves (new)
    plotter.plot_cil_training_curves()
    
    # Plot CIL training summary (new)
    # plotter.plot_cil_training_summary()
    
    # Plot final class comparison
    # plotter.plot_final_comparison()
    
    # Save metrics to JSON
    # plotter.save_metrics()
    
    # Create summary report
    # plotter.create_summary_report()
    
    # Call original plotting function as well
    make_CIL_plots(cfg, backbone_class_acc, accuracy_history)
    

    
if __name__ == "__main__":
    main()
