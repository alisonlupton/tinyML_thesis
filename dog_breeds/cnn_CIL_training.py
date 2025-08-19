# cnn_CIL_training.py
"""
Simplified TDM Pipeline with Intelligent Sampling: Following SparCL paper more closely
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from process_dog_data import process_dog_data
from utils import load_config, set_seed , ClassRegistry
from metrics import profile_cil_resources_2d
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
    return gid_to_local, df, torch.tensor(local_to_gid, dtype=torch.long)

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
    
    # Create backbone data loaders
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
    print(f"\n{'='*50}")
    print("STEP 2: CIL TRAINING")
    print(f"{'='*50}")
    
    # Freeze backbone
    for param in backbone_model.backbone.parameters():
        param.requires_grad = False
    for p in backbone_model.proj.parameters():   # proj has params; freeze it too
        p.requires_grad = False
    
    # Create CIL model
    cil_model = SimplifiedTDMModelCNN(3, backbone_num_classes, device, cfg['sparsity_ratio'], cfg['feat_dim']).to(device)
    cil_model.backbone.load_state_dict(backbone_model.backbone.state_dict())
    cil_model.proj.load_state_dict(backbone_model.proj.state_dict())   # have to load proj also

    # Freeze CIL backbone
    for param in cil_model.backbone.parameters():
        param.requires_grad = False
    for p in cil_model.proj.parameters():
        p.requires_grad = False
        
    # batchnorm still runs, but doesn't update 
    cil_model.backbone.eval()
    for m in cil_model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    
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
        new_task_breeds = gid2breed[str(new_gids[0])]
        seen_plus_backbone = sorted(set(backbone_gids + seen_gids))
        print(f"\n--- CIL TASK {task_idx + 1}: Added '{new_task_breeds}' ---")
 
        # 1) Prepare task data and do cold evaluation
        gid_to_local, seen_df, task_local_to_gid = build_seen_df(full_df, seen_plus_backbone)
        X_train, y_train, X_test, y_test = load_CIL_data(seen_df, backbone_data.CIL_train_tf, backbone_data.val_tf, gid2breed, task_local_to_gid)
        
        task_train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=cfg['CIL_batch_size_train'], shuffle=True
        )
        task_test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=cfg['CIL_batch_size_test'], shuffle=False
        )
        #------ Cold evaluation on backbone
        cold_overall, cold_per_class = eval_new_classes_on_backbone(backbone_model,
                                                                    task_train_loader, task_test_loader,
                                                                    device, len(seen_plus_backbone))
                                                                
        print(f"[Pre-Task Eval] Overall Untrained Backbone Accuracy: {cold_overall:.1f}%")
        #TODO: potentially change this to update model each task!
        local = gid_to_local[new_gids[0]]
        for local_idx, acc in cold_per_class.items():
            if local_idx == local:
                print(f"Pre-training backbone accuracy on newly added {new_task_breeds}: {acc:.1f}%")
        
        if task_idx == 0:
            backbone_class_acc = cold_per_class
        
        # 2) Expand head for new classes
        prev_num = registry.num_classes()
        if len(new_gids) > 0:
            registry.add_gids(new_gids)
            cil_model.head.expand(len(new_gids))
        
        if task_idx == 0:
            # # Measure Sizing! 
            prof = profile_cil_resources_2d(cfg, cil_model)
               
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
                    m.eval() # no longer freezing running stats
            
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
                task_idx=task_idx, epoch=epoch, gid2breed=gid2breed
            )
        #---------------------------- CIL POST TASK PIPELINE (TASK NUMBER LEVEL)
        # Evaluate on all *seen* classes (no future classes!)
        
        cil_model.eval()

        # note task_classes = seen_classes --> classes up through the finished task
        class_acc_local, overall_acc = CIL_post_task_eval(
            task_test_loader, device, cil_model, registry, task_local_to_gid, gid2breed
        )
        accuracy_history.append(class_acc_local.copy())
        
        if task_idx + 1 == len(schedule):
            for cls, acc in class_acc_local.items():
                gid = int(task_local_to_gid[cls].item())
                name = gid2breed[str(gid)]
                print(f"  {cls} ({name}): {acc:.1f}%")
            print("Final overall accuracy:", overall_acc)
            
        else:
            print("Overall Accuracy so far:", overall_acc)
            for cls, acc in class_acc_local.items():
                gid = int(task_local_to_gid[cls].item())
                name = gid2breed[str(gid)]
                print(f" local id: {cls}, gid: {gid}, breed: ({name}): {acc:.1f}%")
    
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
