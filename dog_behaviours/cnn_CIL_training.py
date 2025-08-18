#!/usr/bin/env python3
"""
Simplified TDM Pipeline with Intelligent Sampling: Following SparCL paper more closely
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# from metrics/tinyml_metrics import TinyMLMetrics
from utils import load_config, load_processed_dog_data, evaluate_all_classes, set_seed, make_global_to_local_map, ClassRegistry
# from metrics import profile_cil_memory, print_cil_memory_report
from metrics import profile_cil_resources
from models.cnn import SimplifiedTDMModelCNN
from cnn_backbone_training_utils import train_cnn_backbone, load_data_cnn_backbone
import logging
from cnn_CIL_training_utils import load_CIL_data, CIL_post_task_eval, train_with_simplified_tdm, make_CIL_plots
from replay import BalancedQuantReplayDynamic

def main():
    """Main function - exact same logic as dog_tdm.py but with intelligent sampling."""
    # Set fixed random seeds for reproducibility
    
    # Load configuration
    cfg = load_config()
    seed = cfg['random_seed']
    set_seed(seed)
    
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
    backbone_model = SimplifiedTDMModelCNN(C, backbone_num_classes, device, sparsity_ratio=0.3)
    backbone_model.to(device)
    
    
    # Prepare backbone training data 
    target_dog = cfg['target_dog']  # Target dog for personalization
    validation_dog = cfg['backbone_val_dog']  # Separate validation dog (different from target and training)
    
    # Exclude target dog and validation dog from backbone training
    all_dogs = list(dog_data.keys())
    available_dogs = [dog for dog in all_dogs if dog not in [target_dog, validation_dog]]
    num_backbone_dogs = cfg['num_pretrain_dogs']
    backbone_dogs = available_dogs[:num_backbone_dogs]  # Use first num available dogs
    
    logging.info(f"Backbone training dogs: {backbone_dogs}")
    logging.info(f"Backbone validation dog: {validation_dog}")
    logging.info(f"Target dog for CIL: {target_dog}")
    
    backbone_data = load_data_cnn_backbone(backbone_dogs, dog_data, behavior_to_idx, backbone_behaviors, validation_dog)
    
    # Create data loaders
    backbone_train_loader = DataLoader(TensorDataset(backbone_data.X_train, backbone_data.y_train), 
                                      batch_size=cfg['backbone_batch_size_train'], shuffle=True)
    backbone_val_loader = DataLoader(TensorDataset(backbone_data.X_val, backbone_data.y_val), 
                                    batch_size=cfg['backbone_batch_size_val'], shuffle=False)
    
    backbone_optimizer = torch.optim.Adam(backbone_model.parameters(), lr=cfg['backbone_learning_rate'], weight_decay = cfg['backbone_weight_decay'])
    backbone_criterion = nn.CrossEntropyLoss()
    
    #####################################################################################################################
    #--------------------------------------- STEP 2: BACKBONE TRAINING AND EVALUATION
    #####################################################################################################################
    
    backbone_model, _ = train_cnn_backbone(backbone_model, backbone_behaviors, backbone_optimizer, backbone_criterion, backbone_train_loader, backbone_val_loader, device, cfg)
    
    #####################################################################################################################
    #--------------------------------------- STEP 3: SET UP CIL MODEL
    #####################################################################################################################

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
    
    X_train, y_train, X_test, y_test = load_CIL_data(dog_data, target_dog, all_behaviors, behavior_to_idx, backbone_data.train_mean, backbone_data.train_std)
    
    # Create evaluation classifier for all 7 classes
    backbone_registry = ClassRegistry(all_behaviors)
    backbone_registry.add_classes(backbone_behaviors)
    
    # Remap test labels to 0-6 range for evaluation
    m_test, gids_all = make_global_to_local_map(all_behaviors, behavior_to_idx, device=y_test.device)
    y_test_remapped = m_test[y_test]   # vectorized 0..6 mapping
        
    test_loader = DataLoader(TensorDataset(X_test, y_test_remapped), batch_size=cfg['CIL_batch_size_test'], shuffle=False)
    
    # Evaluate backbone on all 7 classes for target dog
    overall_acc, class_acc = evaluate_all_classes(backbone_model, test_loader, backbone_registry, all_behaviors, device)
    
    print(f"Backbone overall accuracy on all 7 classes: {overall_acc:.1f}%")
    print(f"Backbone per-class accuracy:")
    for cls, acc in class_acc.items():
        print(f"  {cls}: {acc:.1f}%")
        
    
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
    backbone_class_acc = class_acc
    
    
        
    # Measure Sizing! 
    report = profile_cil_resources(
        cfg,
        cil_model, C=C, L=L,
        batch_size_train=cfg['CIL_batch_size_train'],
        batch_size_infer=cfg['CIL_batch_size_test'],
        replay_size=cfg['buffer_size'],
        optimizer_kind="adam",
        deployed_bits_backbone=8,
        deployed_bits_classifier=8,
        training_bits_classifier=32,
        replay_bits=8,
        activation_bits_train=32,
        activation_bits_infer=8,
        replay_batch_size=cfg['replay_batch_size'],   # NEW
        kd_prev_rows=len(backbone_behaviors),         # for Task 1 this is the teacher size
        kd_enabled=True,                              # you do KD after Task 1
        autograd_grad_for_cwi=True                    # you use autograd.grad for CWI
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
                                batch_size=cfg['CIL_batch_size_train'], shuffle=True)
        
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
                
            train_with_simplified_tdm(cil_model, cfg, registry, task_classes, teacher, prev_num, train_loader, behavior_to_idx, optimizer, criterion, device, 
                                    replay_buffer_q, seen_gids.tolist(), seen_names, task_idx, epoch)

        #---------------------------- CIL POST TASK PIPELINE (TASK NUMBER LEVEL)
        # Evaluate on all *seen* classes (no future classes!)
        
        cil_model.eval()
        test_loader = DataLoader(TensorDataset(X_test, y_test_remapped), batch_size=cfg['CIL_batch_size_test'], shuffle=False)
        
        # note task_classes = seen_classes --> classes up through the finished task
        class_acc, overall_acc = CIL_post_task_eval(task_classes, test_loader, device, all_behaviors, cil_model, registry)
        accuracy_history.append(class_acc.copy())
        
        
        if task_idx + 1 == len(cil_tasks):
            for cls, acc in class_acc.items():
                print(f"  {cls}: {acc:.1f}%")
            print("Final overall accuracy:", overall_acc)
            
        else:
            print("Eval so far:", overall_acc)
            for cls, acc in class_acc.items():
                print(f"  {cls}: {acc:.1f}%")
    
    print("CIL PIPELINE FINISHED!")

    
    # Save results
    results = {
        'accuracy_history': accuracy_history,
        'final_accuracy': overall_acc,
        'final_class_acc': class_acc
    }
    
    
    # Call plotting function 
    make_CIL_plots(cfg, backbone_class_acc, accuracy_history)
    

    
if __name__ == "__main__":
    main()
