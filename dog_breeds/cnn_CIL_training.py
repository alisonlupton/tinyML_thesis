# cnn_CIL_training.py
"""
Simplified TDM Pipeline with Intelligent Sampling: Following SparCL paper more closely
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from process_dog_data import process_dog_data
from utils import load_config, set_seed , ClassRegistry
from metrics import profile_cil_resources
from models.cnn import SimplifiedTDMModelCNN
from cnn_backbone_training_utils import train_cnn_backbone, load_data_cnn_backbone, eval_new_classes_on_backbone
import logging
from cnn_CIL_training_utils import seen_and_new, load_CIL_data, CIL_post_task_eval, train_with_simplified_tdm, make_CIL_plots
from replay import BalancedQuantReplayDynamic
import json
from pathlib import Path
import pandas as pd

def build_seen_df(full_df: pd.DataFrame, seen_gids: list[int]) -> pd.DataFrame:
    """
    Return a dataframe filtered to SEEN gids and with a 'local' label
    consistent with sorted(seen_gids).
    """
    df = full_df[full_df["gid"].isin(seen_gids)].copy()
    local_to_gid = sorted(int(g) for g in seen_gids)
    gid_to_local = {g: i for i, g in enumerate(local_to_gid)}
    df["local"] = [gid_to_local[int(g)] for g in df["gid"].tolist()]
    return df, torch.tensor(local_to_gid, dtype=torch.long)

def main():
    """Main function - exact same logic as dog_tdm.py but with intelligent sampling."""
    # Set fixed random seeds for reproducibility
    # gids = global dog ids (see index files for definitions)
    
    # Load configuration
    cfg = load_config()
    
    seed = cfg['random_seed']
    set_seed(seed)
    
    backbone_gids, schedule = process_dog_data(cfg) # these are lists of gids
    index_json = cfg['cleaned_data_path_json']
    index_csv = cfg['cleaned_data_path_csv']
    gid2breed  = json.loads(Path(index_json).read_text())
    full_df = pd.read_csv(index_csv)
    all_training_gids = sorted(set(list(backbone_gids) + [g for task in schedule for g in task]))
        
    if len(backbone_gids) == 0:
        print("No classes selected for backbone!")
        return
    
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
    backbone_num_classes = len(backbone_gids)

    backbone_model = SimplifiedTDMModelCNN(3, backbone_num_classes, device, cfg['sparsity_ratio'], cfg['feat_dim'])
    backbone_model.to(device)

    backbone_data = load_data_cnn_backbone(cfg, backbone_gids, full_df)
    
    # Create data loaders
    backbone_train_loader = DataLoader(TensorDataset(backbone_data.X_train, backbone_data.y_train), batch_size=cfg['backbone_batch_size_train'], shuffle=True)
    backbone_val_loader = DataLoader(TensorDataset(backbone_data.X_val, backbone_data.y_val), batch_size=cfg['backbone_batch_size_val'], shuffle=False)
    
    backbone_optimizer = torch.optim.Adam(backbone_model.parameters(), lr=cfg['backbone_learning_rate'], weight_decay = cfg['backbone_weight_decay'])
    backbone_criterion = nn.CrossEntropyLoss()
    
    #####################################################################################################################
    #--------------------------------------- STEP 2: BACKBONE TRAINING AND EVALUATION
    #####################################################################################################################
    
    backbone_model, _ = train_cnn_backbone(backbone_model, backbone_data, backbone_optimizer, backbone_criterion, backbone_train_loader, backbone_val_loader, device, cfg, gid2breed)
    
    #####################################################################################################################
    #--------------------------------------- STEP 3: SET UP CIL MODEL
    #####################################################################################################################

    # Freeze backbone
    for param in backbone_model.backbone.parameters():
        param.requires_grad = False
    
    # Create CIL model
    cil_model = SimplifiedTDMModelCNN(3, backbone_num_classes, device, sparsity_ratio=cfg['sparsity_ratio']).to(device)
    cil_model.backbone.load_state_dict(backbone_model.backbone.state_dict())
    
    # Freeze CIL backbone
    for param in cil_model.backbone.parameters():
        param.requires_grad = False
        
    cil_model.backbone.eval()
    for m in cil_model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False  # fully freeze
    
    #####################################################################################################################        
    #--------------------------------------- STEP 4: SET UP CIL SCENARIO
    ##################################################################################################################### 
        
    # Create evaluation classifier for all 7 classes
    backbone_registry = ClassRegistry(gid2name=gid2breed)
    backbone_registry.add_gids(backbone_gids) 
    
    # Initialize replay buffer
    replay_buffer_q = BalancedQuantReplayDynamic(buffer_size=cfg['buffer_size'])
    
    registry = ClassRegistry(gid2name=gid2breed)
    registry.add_gids(backbone_gids)  # seed with base backbone dogs
    
    replay_buffer_q.on_new_task(backbone_gids)
    
    accuracy_history = []
    
    # # Measure Sizing! 
    # report = profile_cil_resources(
    #     cfg,
    #     cil_model, C=C, L=L,
    #     batch_size_train=cfg['CIL_batch_size_train'],
    #     batch_size_infer=cfg['CIL_batch_size_test'],
    #     replay_size=cfg['buffer_size'],
    #     optimizer_kind="adam",
    #     deployed_bits_backbone=8,
    #     deployed_bits_classifier=8,
    #     training_bits_classifier=32,
    #     replay_bits=8,
    #     activation_bits_train=32,
    #     activation_bits_infer=8,
    #     replay_batch_size=cfg['replay_batch_size'],   # NEW
    #     kd_prev_rows=len(backbone_behaviors),         # for Task 1 this is the teacher size
    #     kd_enabled=True,                              # you do KD after Task 1
    #     autograd_grad_for_cwi=True                    # you use autograd.grad for CWI
    # )
    
    # TDM parameters 
    p_intra = cfg['p_intra']   
    p_inter = cfg['p_inter']   
    delta_k = cfg['delta_k']    
    
    #####################################################################################################################
    #--------------------------------------- STEP 5: PERFORM CIL SCENARIO
    #####################################################################################################################
    
    #---------------------------- CIL PRE TASK PIPELINE (TASK NUMBER LEVEL)
    for task_idx, _task_gids in enumerate(schedule):        
        
        # seen classes up to this task
        seen_gids, new_gids = seen_and_new(schedule, task_idx)
        seen_plus_backbone = sorted(set(backbone_gids + seen_gids))
        print(f"\n--- CIL TASK {task_idx + 1}: Added '{new_gids}' ---")
 
        # 1) Prepare task data and do cold evaluation
        seen_df, task_local_to_gid = build_seen_df(full_df, seen_plus_backbone)
        X_train, y_train, X_test, y_test = load_CIL_data(seen_df, backbone_data.train_tf, backbone_data.val_tf)
        
        task_train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=cfg['CIL_batch_size_train'], shuffle=True
        )
        task_test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=cfg['CIL_batch_size_test'], shuffle=False
        )
        #------ Cold evaluation on backbone
        cold_overall, cold_per_class = eval_new_classes_on_backbone(backbone_model, task_test_loader, task_local_to_gid, backbone_registry, device)
        print(f"[Pre-Task Eval] Backbone on Task {task_idx+1} new gids {new_gids}: {cold_overall:.1f}%")
        print("  Per-class accuracy:")
        for local_idx, acc in cold_per_class.items():
            gid = int(task_local_to_gid[local_idx].item())
            name = gid2breed.get(str(gid), f"gid_{gid}")
            print(f"    {name:30s}: {acc:.1f}%")
        
        if task_idx == 0:
            backbone_class_acc = cold_per_class
        
        for i, gid in enumerate(task_local_to_gid.tolist()):
            name = gid2breed.get(str(int(gid)), f'gid_{int(gid)}')
            # Per-class (optional): count only samples of this local i
            # (quick calc without another pass)
            pass
        
        # 2) Expand head for new classes
        prev_num = registry.num_classes()
        if len(new_gids) > 0:
            registry.add_gids(new_gids)
            cil_model.head.expand(len(new_gids))
    
               
        # 3) Inter-task expansion (warm up grow before epochs)
        if task_idx > 0:
            cil_model.head.update_mask_inter(p = p_inter)
            
        # update replay buffer
        replay_buffer_q.on_new_task(seen_plus_backbone)
                
        
        # 4) Create teacher snapshot over the previous classes for DDR/KD
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
            for m in cil_model.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False  # can repeat
            
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
                
            train_with_simplified_tdm(
                model=cil_model, cfg=cfg, registry=registry,
                teacher=teacher, prev_num=prev_num,
                train_loader=task_train_loader, optimizer=optimizer, criterion=criterion, device=device,
                replay_buffer_q=replay_buffer_q,
                seen_global_ids=seen_plus_backbone, local_to_gid=task_local_to_gid,
                task_idx=task_idx, epoch=epoch
            )
        #---------------------------- CIL POST TASK PIPELINE (TASK NUMBER LEVEL)
        # Evaluate on all *seen* classes (no future classes!)
        
        cil_model.eval()

        # note task_classes = seen_classes --> classes up through the finished task
        class_acc_local, overall_acc = CIL_post_task_eval(
            task_test_loader, device, cil_model, registry, task_local_to_gid
        )
        accuracy_history.append(class_acc_local.copy())
        
    
        if task_idx + 1 == len(schedule):
            for cls, acc in class_acc_local.items():
                print(f"  {cls}: {acc:.1f}%")
            print("Final overall accuracy:", overall_acc)
            
        else:
            print("Eval so far:", overall_acc)
            for cls, acc in class_acc_local.items():
                print(f"  {cls}: {acc:.1f}%")
    
    print("CIL PIPELINE FINISHED!")

    
    # Save results
    results = {
        'accuracy_history': accuracy_history,
        'final_accuracy': overall_acc,
        'final_class_acc': class_acc_local
    }
    
    
    # Call plotting function 
    make_CIL_plots(cfg, backbone_class_acc, accuracy_history)
    

    
if __name__ == "__main__":
    main()
