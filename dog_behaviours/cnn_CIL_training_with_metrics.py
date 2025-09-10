#cnn_CIL_training_with_metrics.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
torch.use_deterministic_algorithms(True)  #PyTorch 1.8
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils import load_config, load_processed_dog_data, evaluate_all_classes, set_seed, make_global_to_local_map, ClassRegistry
#from metrics import profile_cil_memory, print_cil_memory_report
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
import os
import json
def serialize_registry(registry):
    return {
        "all_behaviors": registry.all_behaviors,          #the fixed universe
        "row_for_global": registry.row_for_global,        #mapping {class_name -> row_idx}
        "global_for_row": registry.global_for_row,        #list of class_names in row order
        "seen_names": registry.seen_classes(),            #redundant but handy
        "num_classes": registry.num_classes(),
    }

def main():
    """Main function"""
    #Set fixed random seeds for reproducibility
    
    #Load configuration
    cfg = load_config()
    seed = cfg['random_seed']
    set_seed(seed)
    gen_seed = torch.Generator()
    gen_seed.manual_seed(seed)
    
    #Load intelligent sampling data
    dog_data, behavior_to_idx, behaviors = load_processed_dog_data()

    if len(dog_data) == 0:
        print("No dog data loaded!")
        return
    
    #Define behaviors
    backbone_behaviors = cfg['backbone_behaviors']
    cil_behaviors = cfg['cil_behaviors']
    all_behaviors = backbone_behaviors + cil_behaviors

    #CIL task progression
    cil_tasks = cfg['cil_tasks']
    
    #metric on time
    rand_baseline = {f"T{k+1}": 100.0/len(task) for k, task in enumerate(cil_tasks)}
    os.makedirs("evaluation/metrics_meta", exist_ok=True)
    with open("evaluation/metrics_meta/random_baseline.json", "w") as f:
        json.dump(rand_baseline, f, indent=2)
        
    #Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    #####################################################################################################################
    #--------------------------------------- STEP 1: BACKBONE SETTUP
    #####################################################################################################################
    
    print(f"\n{'='*50}")
    print("STEP 1: BACKBONE TRAINING")
    print(f"{'='*50}")
 
    
    #Create backbone model
    backbone_num_classes = len(backbone_behaviors)
    C = len(dog_data[list(dog_data.keys())[0]]['sensor_cols'])
    L = int(dog_data[list(dog_data.keys())[0]]['window_len'])
    backbone_model = SimplifiedTDMModelCNN(C, backbone_num_classes, device, sparsity_ratio=cfg['backbone_sparsity_ratio'])
    backbone_model.to(device)
    
    #Prepare backbone training data
    #Exclude target dog and validation dog from backbone training
    all_possible_dogs = sorted(dog_data.keys())
    valid_CIL_dogs_for_this_seed = [
        d for d in all_possible_dogs
        if check_CIL_validity(dog_data, d, all_behaviors, behavior_to_idx, verbose=True)
    ] 
    
    print(f"Valid CIL dogs for seed {seed}: {valid_CIL_dogs_for_this_seed}")
    if not valid_CIL_dogs_for_this_seed:
        raise ValueError("No dogs satisfy nonzero train+test counts for all 7 classes under this seed/split.")
    
    rng = random.Random(seed)  
    
    #pick CIL target dog
    target_dog = rng.choice(valid_CIL_dogs_for_this_seed)
    print(f"Target Dog Chosen: {target_dog}")
    
    #pick validation dog (not same as target)
    remaining = sorted([d for d in all_possible_dogs if d != target_dog])
    validation_dog = rng.choice(remaining)
    print(f"Validation Dog Chosen: {validation_dog}")
    
    #Exclude target dog and validation dog from backbone training
    remaining = sorted([dog for dog in all_possible_dogs if dog not in [target_dog, validation_dog]])
    backbone_dogs = rng.sample(remaining, cfg['num_pretrain_dogs']) #randomly selection num of pretraining dogs
    print(f"Backbone Dogs Chosen: {backbone_dogs}")

    logging.info(f"Backbone training dogs: {backbone_dogs}")
    logging.info(f"Backbone validation dog: {validation_dog}")
    logging.info(f"Target dog for CIL: {target_dog}")
    
    backbone_data = load_data_cnn_backbone(backbone_dogs, dog_data, behavior_to_idx, backbone_behaviors, validation_dog)
    
    #Create data loaders
    backbone_train_loader = DataLoader(TensorDataset(backbone_data.X_train, backbone_data.y_train), 
                                      batch_size=cfg['backbone_batch_size_train'], shuffle=True, generator = gen_seed)
    backbone_val_loader = DataLoader(TensorDataset(backbone_data.X_val, backbone_data.y_val), 
                                    batch_size=cfg['backbone_batch_size_val'], shuffle=False)
    
    backbone_optimizer = torch.optim.Adam(backbone_model.parameters(), lr=cfg['backbone_learning_rate'], weight_decay = cfg['backbone_weight_decay'])
    backbone_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    #Initialize plotter for tracking metrics
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
 
    #Freeze backbone
    for param in backbone_model.backbone.parameters():
        param.requires_grad = False
    
    #Create CIL model
    cil_model = SimplifiedTDMModelCNN(C, backbone_num_classes, device, sparsity_ratio=cfg['sparsity_ratio']).to(device)
    cil_model.backbone.load_state_dict(backbone_model.backbone.state_dict())
    
    #Freeze CIL backbone
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
    
    #Create evaluation classifier for all 7 classes
    backbone_registry = ClassRegistry(all_behaviors)
    backbone_registry.add_classes(backbone_behaviors)
    
    #Remap test labels to 0-6 range for evaluation
    m_test, gids_all = make_global_to_local_map(all_behaviors, behavior_to_idx, device=y_test.device)
    y_test_remapped = m_test[y_test]   #vectorized 0..6 mapping
        
    test_loader = DataLoader(TensorDataset(X_test, y_test_remapped), batch_size=cfg['CIL_batch_size_test'], shuffle=False)
    
    #Evaluate backbone on all 7 classes for target dog
    backbone_all7_acc, class_acc, _, _ = evaluate_all_classes(backbone_model, test_loader, backbone_registry, all_behaviors, device)
    
    print(f"Backbone overall accuracy on all 7 classes: {backbone_all7_acc:.2f}%")
    print(f"Backbone per-class accuracy:")
    for cls, acc in class_acc.items():
        print(f"  {cls}: {acc:.2f}%")
    
       
    #####################################################################################################################
    #--------------------------------------- STEP 5: PERFORM CIL SCENARIO
    #####################################################################################################################
    
    #Initialize replay buffer
    replay_buffer_q = BalancedQuantReplayDynamic(buffer_size=cfg['buffer_size'])
    
    registry = ClassRegistry(all_behaviors)
    registry.add_classes(backbone_behaviors)  #seed with 4 base classes
    
    seen_names = registry.seen_classes()              #['Standing', 'Walking', 'Sitting', 'Lying chest']
    _, seen_gids = make_global_to_local_map(seen_names, behavior_to_idx, device="cpu")
    replay_buffer_q.on_new_task(seen_gids.tolist())   #buffer wants global class IDs
    
    accuracy_history = []
    training_loss_history = []  #Store training losses for each task
    backbone_class_acc = class_acc
    
    
    #Measure Sizing!
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
    
    #TDM parameters
    p_intra = cfg['p_intra']   
    p_inter = cfg['p_inter']   
    delta_k = cfg['delta_k']    
    
    #Pre-CIL metrics
    test_feats = cil_model.backbone(X_test).detach().cpu()  #shape [N_test, D]
    torch.save({"feats": test_feats, "y_global": y_test.cpu(), "m_test": m_test.cpu()}, "evaluation/test_feats.pt")
    
    #--- also cache TRAIN features for oracle heads ---
    train_feats = cil_model.backbone(X_train).detach().cpu()  #shape [N_train, ...]
    torch.save({"feats": train_feats,
                "y_global": y_train.cpu(),
                "m_all": m_test.cpu()},  #reuse mapping to 0..6 if convenient
            "evaluation/train_feats.pt")
    #right after build the CIL head and before Task 1 starts
    os.makedirs("evaluation/checkpoints_cil", exist_ok=True)
    torch.save({
        "task_idx": -1,
        "head_state": cil_model.head.state_dict(),
        "registry": serialize_registry(registry),  #same helper wrote below
        "seed": seed,
    }, "evaluation/checkpoints_cil/head_after_T0.pt")
    
    #---------------------------- CIL PRE TASK PIPELINE (TASK NUMBER LEVEL)
    for task_idx, task_classes in enumerate(cil_tasks):
        #Determine which new class is being added
        if task_idx == 0:
            new_class = "Sniffing"
        elif task_idx == 1:
            new_class = "Trotting"
        else:
            new_class = "Galloping"
        
        print(f"\n--- CIL TASK {task_idx + 1}: Added '{new_class}' ---")
        
        #1) Expand head for new classes
        prev_num = registry.num_classes()
        if len(task_classes) > prev_num:
            new_cls = task_classes[prev_num:]  #true new names
            cil_model.expand_head(new_cls, registry)

        cil_model.eval()

        #2) Inter-task expansion (warm up grow before epochs)
        if task_idx > 0:
            cil_model.head.update_mask_inter(p = p_inter)
         
         #reresh seen set
        seen_names = registry.seen_classes()
        _, seen_gids = make_global_to_local_map(seen_names, behavior_to_idx, device="cpu")
        replay_buffer_q.on_new_task(seen_gids.tolist())
        
        #3) Prepare training data
        map_task, task_gids = make_global_to_local_map(task_classes, behavior_to_idx, device=y_train.device)

        mask = torch.isin(y_train, task_gids)
        X_task = X_train[mask]
        y_task = y_train[mask]
        y_task_remapped = map_task[y_task]   #vectorized mapping to 0..len(task_classes)-1

        
        #4) Create teacher snapshot over the previous classes for DDR/KD
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
        task_training_losses = []  #Store losses for this task
        for epoch in range(cfg['CIL_epochs']):
            #Ensure frozen
            cil_model.head.train()
            cil_model.backbone.eval()
            
            #Learning rate scheduling
            if epoch == 5:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
            elif epoch == 10:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                    
            #Intra-task TDM adjustment
            if epoch > 0 and epoch % delta_k == 0:
                cil_model.head.update_mask_intra(p = p_intra)
   
            #Inter-task warm-up end (shrink back to target sparsity)
            if task_idx > 0 and epoch == delta_k:
                cil_model.head.shrink_after_warmup(p=p_inter)
                
            #Train and get metrics
            train_loss, train_acc = train_with_simplified_tdm(cil_model, cfg, registry, task_classes, teacher, prev_num, train_loader, behavior_to_idx, optimizer, criterion, device, 
                                    replay_buffer_q, seen_gids.tolist(), seen_names, task_idx, epoch)
            
            #Store training loss for this epoch
            task_training_losses.append(train_loss)
            
            #Add training metrics to plotter
            task_name = f"Task {task_idx+1}: {new_class}"
            plotter.add_cil_task_training_epoch(task_name, epoch, train_loss, train_acc)

        #---------------------------- CIL POST TASK PIPELINE (TASK NUMBER LEVEL)
        #Evaluate on all *seen* classes (no future classes!)
        
        cil_model.eval()
        test_loader = DataLoader(TensorDataset(X_test, y_test_remapped), batch_size=cfg['CIL_batch_size_test'], shuffle=False)
        
        #Evaluate using forward_eval7 for consistency, then filter results
        all_classes_overall, all_classes_acc, sample_counts, correct_counts = evaluate_all_classes(cil_model, test_loader, registry, all_behaviors, device)
        
        #Filter to get "seen classes only" results
        seen_classes_acc = {cls: all_classes_acc[cls] for cls in task_classes}
        
        #Calculate proper weighted overall accuracy for seen classes only
        seen_total_samples = sum([sample_counts[cls] for cls in task_classes if cls in sample_counts])
        seen_correct_samples = sum([correct_counts[cls] for cls in task_classes if cls in correct_counts])
        seen_overall_acc = 100.0 * seen_correct_samples / max(seen_total_samples, 1) if seen_total_samples > 0 else 0.0
        
        #Use the filtered results for consistency with existing code
        class_acc = seen_classes_acc
        overall_acc = seen_overall_acc
        
        #Store both class accuracies and overall accuracy for proper weighted calculation
        accuracy_history.append({
            'class_acc': all_classes_acc.copy(),
            'overall_acc': all_classes_overall
        })
        
        #Store training losses for this task
        training_loss_history.append({
            'task_idx': task_idx,
            'task_name': f"Task_{task_idx + 1}",
            'training_losses': task_training_losses.copy(),
            'final_training_loss': task_training_losses[-1] if task_training_losses else 0.0,
            'avg_training_loss': sum(task_training_losses) / len(task_training_losses) if task_training_losses else 0.0
        })


        #Add to plotter
        task_name = f"Task {task_idx+1}: {new_class}"
        plotter.add_cil_task(task_name, all_classes_acc, all_classes_overall)
        
        if task_idx + 1 == len(cil_tasks):
            print(f"[Seen Classes Only] Overall: {overall_acc:.2f}%")
            print(f"[All 7 Classes] Overall: {all_classes_overall:.2f}%")
            for cls, acc in all_classes_acc.items():
                print(f"  {cls}: {acc:.2f}%")
            print("Final overall accuracy:", overall_acc)
            
        else:
            print(f"[Seen Classes Only] Overall: {overall_acc:.2f}%")
            print(f"[All 7 Classes] Overall: {all_classes_overall:.2f}%")
            for cls, acc in all_classes_acc.items():
                print(f"  {cls}: {acc:.2f}%")
                
        #---------------------------------------- End of task metric updating
        #=== after finishing training Task i ===
        #1) save head checkpoint
        ckpt_dir = "evaluation/checkpoints_cil"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt = {
            "task_idx": task_idx,
            "head_state": cil_model.head.state_dict(),
            "registry": serialize_registry(registry),
            "seed": seed,
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f"head_after_T{task_idx+1}.pt"))

        ## MASKS
        task_mask_dir = "evaluation/task_masks"
        os.makedirs(task_mask_dir, exist_ok=True)
        
        #--- cumulative TEST mask
        eval_map, eval_gids = make_global_to_local_map(task_classes, behavior_to_idx, device="cpu")
        eval_mask = torch.isin(y_test.cpu(), eval_gids.cpu()).numpy()
        np.save(os.path.join(task_mask_dir, f"mask_T{task_idx+1}.npy"), eval_mask)

        #--- NEW-CLASS-ONLY TEST mask (for pure FWT)
        new_cls = task_classes[prev_num:]  #classes introduced at this task
        if len(new_cls) > 0:
            _, new_gids = make_global_to_local_map(new_cls, behavior_to_idx, device="cpu")
            new_mask = torch.isin(y_test.cpu(), new_gids.cpu()).numpy()
            np.save(os.path.join(task_mask_dir, f"mask_T{task_idx+1}_new.npy"), new_mask)

        #--- cumulative TRAIN mask (already in code)
        train_eval_map, train_eval_gids = make_global_to_local_map(task_classes, behavior_to_idx, device="cpu")
        train_mask = torch.isin(y_train.cpu(), train_eval_gids.cpu()).numpy()
        np.save(os.path.join(task_mask_dir, f"mask_train_T{task_idx+1}.npy"), train_mask)

        #--- NEW-CLASS-ONLY TRAIN mask (optional, symmetry / diagnostics)
        if len(new_cls) > 0:
            _, new_train_gids = make_global_to_local_map(new_cls, behavior_to_idx, device="cpu")
            new_train_mask = torch.isin(y_train.cpu(), new_train_gids.cpu()).numpy()
            np.save(os.path.join(task_mask_dir, f"mask_train_T{task_idx+1}_new.npy"), new_train_mask)
        #record random baseline for this task (optional: collect to a JSON/CSV)
        rand_baseline = 100.0 / len(task_classes)
    
    print("CIL PIPELINE FINISHED!")

    #Save results
    results = {
        'accuracy_history': accuracy_history,
        'final_accuracy': overall_acc,
        'final_class_acc': class_acc,
        'backbone_class_acc': backbone_class_acc
    }
    meta = {
        "seed": seed,
        "backbone_behaviors": backbone_behaviors,
        "cil_tasks": cil_tasks,
        "all_behaviors": all_behaviors,
        "train_mean": getattr(backbone_data, "train_mean", None),
        "train_std": getattr(backbone_data, "train_std", None),
        "cfg": cfg,
    }
    with open("evaluation/metrics_meta/run_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=lambda o: o if isinstance(o,(int,float,str,bool,list,dict)) else None)
        
    
    #Generate all plots and reports
    print("\n" + "="*50)
    print("GENERATING PLOTS AND REPORTS")
    print("="*50)
    
    #Plot backbone training
    plotter.plot_backbone_training()
    
    #Plot CIL progression
    #plotter.plot_cil_progression()
    
    #Plot CIL training curves (new)
    plotter.plot_cil_training_curves()
    
    #Plot CIL training summary (new)
    #plotter.plot_cil_training_summary()
    
    #Plot final class comparison
    #plotter.plot_final_comparison()
    
    #Save metrics to JSON
    #plotter.save_metrics()
    
    #Create summary report
    #plotter.create_summary_report()
    
    #Call original plotting function as well
    make_CIL_plots(cfg, backbone_class_acc, accuracy_history, backbone_overall_acc=backbone_all7_acc)
    

    
if __name__ == "__main__":
    main()
