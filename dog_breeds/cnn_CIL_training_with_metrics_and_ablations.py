#cnn_CIL_training_with_metrics.py

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
torch.backends.quantized.engine = 'qnnpack'  
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from process_dog_data import process_dog_data
from utils import load_config, set_seed, ClassRegistry, build_tensors
from metrics import profile_tinyml_cil, save_tinyml_report_csv
from models.cnn import TunedMCUStudentCNN_CIL
from cnn_backbone_training_utils import load_data_cnn_backbone, eval_on_fixed_gids, quantise_frozen_backbone, features_quantized, compute_perchannel_scales
from cnn_CIL_training_utils import seen_and_new, load_CIL_data, CIL_post_task_eval, train_with_simplified_tdm, make_CIL_plots
from dog_breed_training_plotter import DogBreedCILTrainingPlotter, clean_breed_name
from replay import BalancedQuantReplayDynamic
import json
from pathlib import Path
import pandas as pd
import types
import numpy as np


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
    """Main function"""
    #Set fixed random seeds for reproducibility
    #gids = global dog ids (see index files for definitions)
    
    #Load configuration
    cfg = load_config()
    
    seed = cfg['random_seed']
    set_seed(seed)
    
    #Create evaluation directories
    os.makedirs("evaluation/checkpoints_cil", exist_ok=True)
    os.makedirs("evaluation/task_masks", exist_ok=True)
    os.makedirs("evaluation/metrics_meta", exist_ok=True)
    
    #Save random baseline for CL metrics calculation
    random_baseline = {}
    for task_idx in range(1, 4):  #T1, T2, T3
        random_baseline[f"T{task_idx}"] = 1.0 / 13.0  #1/13 for 13 dog breeds
    with open("evaluation/metrics_meta/random_baseline.json", "w") as f:
        json.dump(random_baseline, f, indent=2)
    
    #Initialize training plotter
    plotter = DogBreedCILTrainingPlotter(save_dir=cfg['plot_dir'], seed=seed)
    
    backbone_gids, original_schedule = process_dog_data(cfg) #these are lists of gids
    
    #CIL task progression - modify schedule based on replay setting
    if cfg['replay']:
        schedule = original_schedule  #cumulative learning (all seen + new)
    else:
        #new-class-only learning (just the new breed)
        schedule = original_schedule  #Keep the same schedule structure but will modify training logic
    
    index_json = cfg['cleaned_data_path_json']
    index_csv = cfg['cleaned_data_path_csv']
    
    student_path = cfg['student_path']
    student_meta_path = cfg['student_meta_path']
    gid2breed  = json.loads(Path(index_json).read_text())
    full_df = pd.read_csv(index_csv)
        
    if len(backbone_gids) == 0:
        print("No classes selected for backbone!")
        return
    
    #Device setup
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    #####################################################################################################################
    #--------------------------------------- STEP 1: LOAD TRAINED STUDENT MODEL
    #####################################################################################################################
    
    print(f"\n{'='*50}")
    print("STEP 1: LOADING TRAINED STUDENT MODEL")
    print(f"{'='*50}")
    
    #Load trained student model (tuned MCU version)
    print(f"Loading tuned MCU student model from: {student_path}")
    print(f"Loading student metadata from: {student_meta_path}")
    
    #Load student metadata
    with open(student_meta_path, 'r') as f:
        student_meta = json.load(f)
    
    #Create backbone model with tuned MCU architecture
    backbone_num_classes = len(backbone_gids)
    backbone_model = TunedMCUStudentCNN_CIL(3, backbone_num_classes, device, cfg['sparsity_ratio'], student_meta['feature_dim'])
    backbone_model.to(device)
    
    #Load trained student weights
    student_state_dict = torch.load(student_path, map_location=device)
    
    #The tuned student has backbone, proj, classifier structure
    backbone_model.backbone.load_state_dict(student_state_dict['backbone'])
    backbone_model.proj.load_state_dict(student_state_dict['proj'])
    
    print(f"Successfully loaded student model!")
    print(f"Student was trained on {len(student_meta['backbone_gids'])} classes")
    print(f"Student backbone_gids: {student_meta['backbone_gids']}")
    print(f"Current backbone_gids: {sorted(backbone_gids)}")
    
    #Load data for evaluation (we still need this for transforms and evaluation)
    backbone_data = load_data_cnn_backbone(cfg, backbone_gids, full_df)
    
    #####################################################################################################################
    #--------------------------------------- STEP 2: SET UP CIL MODEL AND QUANTISE BACKBONE
    #####################################################################################################################
    print(f"\n{'='*50}")
    print("STEP 2: CIL TRAINING")
    print(f"{'='*50}")
    
    #Freeze backbone
    for param in backbone_model.backbone.parameters():
        param.requires_grad = False
    for p in backbone_model.proj.parameters():   #proj has params; freeze it too
        p.requires_grad = False

    #Create CIL model
    cil_model = TunedMCUStudentCNN_CIL(3, backbone_num_classes, device, cfg['backbone_sparsity_ratio'], student_meta['feature_dim']).to(device)
    cil_model.backbone.load_state_dict(backbone_model.backbone.state_dict())
    cil_model.proj.load_state_dict(backbone_model.proj.state_dict())   #have to load proj also
    
    #Create registry for CL metrics calculation
    registry = ClassRegistry(gid2name=gid2breed)
    registry.add_gids(backbone_gids)  #seed with base backbone dogs
    
    #Test features will be generated after each task training to ensure they use the current model state
    
    ## Extract train features (for prototype building)
    #X_train13, y_train13 = build_tensors(eval13_df[eval13_df["split"]=="train"], backbone_data.val_tf)
    #train13_loader = DataLoader(TensorDataset(X_train13, y_train13), batch_size=cfg['CIL_batch_size_test'], shuffle=False)
    
    #train_feats = []
    #y_train = []
    #m_train = []
    #with torch.no_grad():
    #for xb, yb in train13_loader:
    #xb = xb.to(device)
    #feats = cil_model._features(xb)
    #proj_feats = cil_model.proj(feats)
    #train_feats.append(proj_feats.cpu())
    #y_train.append(yb)
    #m_train.append(torch.ones_like(yb))
    
    #train_feats = torch.cat(train_feats, dim=0)
    #y_train = torch.cat(y_train, dim=0)
    #m_train = torch.cat(m_train, dim=0)
    
    ## Save train features
    #torch.save({"feats": train_feats, "y_global": y_train.cpu(), "m_train": m_train.cpu()}, "evaluation/train_feats.pt")
    
    ## Save initial checkpoint (after T0/backbone)
    #torch.save({
    #"head_state_dict": cil_model.head.state_dict(),
    #"registry": registry,
    #"checkpoint_in_dim": cil_model.head.in_dim,
    #}, "evaluation/checkpoints_cil/head_after_T0.pt")

    
    #----------------------------------------------------- QUANTISATION
    #calibration (using training transforms)
    max_calib = 1024
    calib_imgs = backbone_data.X_train[:min(max_calib, len(backbone_data.X_train))]
    #ensure eval + no grad
    cil_model.eval()
    for m in cil_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()  #lock BN stats during calib
    
    calib_loader = DataLoader(
        TensorDataset(calib_imgs, torch.zeros(len(calib_imgs))),
        batch_size=32,
        shuffle=False
    )
    
    #for profiler quantization / replacement
    float_backbone_params = sum(p.numel() for p in cil_model.backbone.parameters())
    cil_model._float_backbone_params = int(float_backbone_params)
    
        #build INT8 CPU backbone
    with torch.no_grad():
        qb_int8 = quantise_frozen_backbone(cil_model.backbone, calib_loader)
    with torch.no_grad():
        #switch models feature path to the quantized one (CPU float out)
        cil_model.quant_backbone = qb_int8           #lives on CPU
        cil_model.backbone = nn.Identity()  #ensure I'm only using int8

        cil_model._features = types.MethodType(features_quantized, cil_model)
        #compute per-channel scales once
        scales = compute_perchannel_scales(calib_loader, cil_model, pct=99.9)   

    for p in cil_model.proj.parameters():
        p.requires_grad = False
    
    #####################################################################################################################
    #--------------------------------------- STEP 3: SET UP CIL SCENARIO
    #####################################################################################################################
        
    #Create evaluation classifier for all seen classes
    backbone_registry = ClassRegistry(gid2name=gid2breed)
    backbone_registry.add_gids(backbone_gids) 

        
    #Initialize replay buffer with known feat_dim and set scales
    with torch.no_grad():
        dummy = torch.zeros(1, 3, cfg['img_size'], cfg['img_size'])
        feat_dim_backbone = cil_model._features(dummy).shape[1]  #should be 160

    replay_buffer_q = BalancedQuantReplayDynamic(cfg, buffer_size=cfg['buffer_size'], feat_dim=feat_dim_backbone)
    replay_buffer_q.set_scales(scales)
    
    #Registry already created above for CL metrics calculation
    
    replay_buffer_q.on_new_task(backbone_gids)
    
    #Debug: Check replay buffer population
    print(f"DEBUG: Replay buffer initialized with backbone classes: {backbone_gids}")
    print(f"DEBUG: Replay buffer size: {len(replay_buffer_q)}")
    #if hasattr(replay_buffer_q, 'debug_summary'):
    #replay_buffer_q.debug_summary(title="After backbone initialization")

    
    accuracy_history = []
    
    #TDM parameters
    p_intra = cfg['p_intra']   
    p_inter = cfg['p_inter']   
    delta_k = cfg['delta_k']    
    
    #####################################################################################################################
    #--------------------------------------- STEP 4: PERFORM CIL SCENARIO
    #####################################################################################################################
    
        
    #------------------- Create evaluation for all 13 classes -------------------
    target_13_gids = sorted(set(backbone_gids + [g for task in schedule for g in task]))[:13]
    eval13_gid_to_local = {g:i for i,g in enumerate(target_13_gids)}
    eval13_df = full_df[full_df["gid"].isin(target_13_gids)].copy()
    eval13_df["local"] = [eval13_gid_to_local[int(g)] for g in eval13_df["gid"].tolist()]
    X_eval13, y_eval13 = build_tensors(eval13_df[eval13_df["split"]=="test"], backbone_data.val_tf)
    
    #Debug: Print the complete evaluation mapping
    print("DEBUG: Complete evaluation mapping (GID  local ID):")
    for i, gid in enumerate(target_13_gids):
        breed_name = gid2breed[str(gid)]
        print(f"  GID {gid} ({breed_name})  local ID {i}")
    print()
    eval13_loader = DataLoader(TensorDataset(X_eval13, y_eval13),
                            batch_size=cfg['CIL_batch_size_test'], shuffle=False)
    #----------------------------------------------------------------------------
    
    #Populate replay buffer with backbone training data
    print("DEBUG: Populating replay buffer with backbone training data...")
    backbone_train_df = full_df[full_df["gid"].isin(backbone_gids) & (full_df["split"]=="train")]
    if len(backbone_train_df) > 0:
        #Use the same label mapping as evaluation data for consistency
        backbone_train_df = backbone_train_df.copy()
        backbone_train_df["local"] = [eval13_gid_to_local[int(g)] for g in backbone_train_df["gid"].tolist()]
        
        X_backbone_train, y_backbone_train = build_tensors(backbone_train_df, backbone_data.CIL_train_tf)
        backbone_train_loader = DataLoader(TensorDataset(X_backbone_train, y_backbone_train), 
                                         batch_size=cfg['CIL_batch_size_train'], shuffle=False)
        
        #Extract features and add to replay buffer
        cil_model.eval()
        with torch.no_grad():
            for xb, yb in backbone_train_loader:
                xb = xb.to(device)
                backbone_feats = cil_model._features(xb)
                proj_feats = cil_model.proj(backbone_feats)
                
                #Convert local labels to global for replay buffer
                global_targets = torch.tensor([target_13_gids[i] for i in yb.tolist()], device=device)
                
                #Calculate proper hardness scores using backbone model predictions
                #For backbone samples, we'll use a simple hardness based on confidence
                #Use forward_rows with the backbone classes only
                backbone_rows = torch.tensor([registry.row_for_gid[int(g)] for g in backbone_gids], device=device)
                logits = cil_model.head.forward_rows(proj_feats, backbone_rows)
                probs = torch.softmax(logits, dim=1)
                top2 = torch.topk(probs, k=2, dim=1).values
                margin = (top2[:,0] - top2[:,1])
                entropy = -(probs * (probs.clamp_min(1e-8)).log()).sum(dim=1)
                
                #Map evaluation labels to backbone labels for cross entropy
                backbone_gid_to_local = {g: i for i, g in enumerate(backbone_gids)}
                backbone_labels = torch.tensor([backbone_gid_to_local[target_13_gids[i]] for i in yb.tolist()], device=device)
                ce_i = F.cross_entropy(logits, backbone_labels, reduction='none')
                hardness = ce_i + (1.0 - margin) + 0.5 * entropy
                
                #Add to replay buffer
                replay_buffer_q.add_batch(backbone_feats, global_targets, raw_scores=hardness)
        
        #print(f"DEBUG: Added {len(backbone_train_df)} backbone samples to replay buffer")
        #print(f"DEBUG: Replay buffer size after backbone population: {len(replay_buffer_q)}")
        #if hasattr(replay_buffer_q, 'debug_summary'):
        #replay_buffer_q.debug_summary(title="After backbone population")
    
    seen_pre_hist=[]
    seen_post_hist = []
    fixed13_hist = []
    
    
        
    #put this below in a func later
    #Build a "seen-only" dataset consisting of just backbone classes (Task 0)
    gid_to_local_0, seen_df_0, local_to_gid_0 = build_seen_df(full_df, backbone_gids)
    Xtr0, ytr0, Xte0, yte0 = load_CIL_data(seen_df_0, backbone_data.CIL_train_tf, backbone_data.val_tf, gid2breed, local_to_gid_0)

    test_loader0 = DataLoader(
        TensorDataset(Xte0, yte0),
        batch_size=cfg['CIL_batch_size_test'],
        shuffle=False
    )

    #Evaluate with the frozen backbone+proj + current head (this is "cold" baseline)
    cil_model.eval()
    class_acc_local_0, overall_0 = CIL_post_task_eval(
        test_loader0, device, cil_model, registry, local_to_gid_0, gid2breed
    )

    #Log as a PRE point for Task 0 ( can also log POST the same since nothing was trained yet)
    seen_pre_hist.append({
        "stage": "Task 0 (Backbone)",
        "per_class": class_acc_local_0,     #dict: local_idx -> acc
        "local_to_gid": local_to_gid_0.cpu()
    })
    seen_post_hist.append({
        "stage": "Task 0 (Backbone)",
        "per_class": class_acc_local_0,
        "local_to_gid": local_to_gid_0.cpu()
    })
    #If 're doing fixed-13 reporting, log that too for Task 0:
    #Use proper backbone evaluation instead of TDM head evaluation
    from cnn_backbone_training_utils import eval_new_classes_on_backbone
    
    #Create backbone-specific data loaders for the 13 target classes
    backbone_13_gid_to_local = {g:i for i,g in enumerate(target_13_gids)}
    backbone_13_df = full_df[full_df["gid"].isin(target_13_gids)].copy()
    backbone_13_df["local"] = [backbone_13_gid_to_local[int(g)] for g in backbone_13_df["gid"].tolist()]
    
    #Create train loader for prototype building (use backbone dogs only)
    backbone_train_13_df = backbone_13_df[backbone_13_df["gid"].isin(backbone_gids) & (backbone_13_df["split"]=="train")]
    X_train_13, y_train_13 = build_tensors(backbone_train_13_df, backbone_data.val_tf)
    train_13_loader = DataLoader(TensorDataset(X_train_13, y_train_13), batch_size=cfg['CIL_batch_size_test'], shuffle=False)
    
    #Use the test loader we already created
    overall13_0, per_class_local = eval_new_classes_on_backbone(
        model=cil_model,
        train_loader=train_13_loader,  #For prototype building
        test_loader=eval13_loader,     #For evaluation
        device=device,
        num_classes=len(target_13_gids),
        use_proj=True
    )
    
    #Convert local indices back to GIDs for consistency
    per13_0 = {target_13_gids[i]: per_class_local[i] for i in range(len(target_13_gids))}
    
    #Calculate the "deflated" overall accuracy for the backbone
    #This should be the average of backbone dogs + 0% for unseen dogs
    backbone_accuracies = [per_class_local[i] for i, gid in enumerate(target_13_gids) if gid in backbone_gids]
    unseen_accuracies = [0.0 for i, gid in enumerate(target_13_gids) if gid not in backbone_gids]
    all_accuracies = backbone_accuracies + unseen_accuracies
    deflated_overall = sum(all_accuracies) / len(all_accuracies)
    
    print(f"Backbone overall accuracy on all 13 classes: {deflated_overall:.2f}%")
    print(f"Backbone per-class accuracy:")
    for i, gid in enumerate(target_13_gids):
        breed_name = gid2breed.get(str(gid), f"gid_{gid}")
        print(f"  {breed_name}: {per_class_local[i]:.2f}%")
    
    fixed13_hist.append({
        "stage": "Backbone",
        "overall": deflated_overall,  #Use deflated accuracy for overall plot
        "per_class": per13_0
    })
    
    #Store backbone data in accuracy_history format for consistency
    backbone_all13_acc = {i: per_class_local[i] for i in range(len(target_13_gids))}
    accuracy_history.append({
        'class_acc': backbone_all13_acc.copy(),
        'overall_acc': deflated_overall
    })
    
    #Track all seen classes across all tasks (starts with backbone classes)
    all_seen_gids = set(backbone_gids)

     
    #---------------------------- CIL PRE TASK PIPELINE (TASK NUMBER LEVEL)
    for task_idx, _task_gids in enumerate(schedule):        
        
        #seen classes up to this task
        seen_gids, new_gids = seen_and_new(schedule, task_idx)
        new_task_breeds = gid2breed[str(new_gids[0])]
        seen_plus_backbone = sorted(set(backbone_gids + seen_gids))
        
        #For ablation: modify training classes based on replay setting
        if cfg['replay']:
            #With replay: train on cumulative new classes, backbone classes come from replay
            training_classes = sorted(set(seen_gids + new_gids))  #All new classes learned so far
            print(f"\n--- CIL TASK {task_idx + 1}: Added '{new_task_breeds}' (REPLAY LEARNING) ---")
            print(f"Training on cumulative new classes: {[gid2breed[str(gid)] for gid in training_classes]}")
        else:
            #Without replay: train only on new class (new-class-only learning)
            training_classes = new_gids
            print(f"\n--- CIL TASK {task_idx + 1}: Added '{new_task_breeds}' (NEW-CLASS-ONLY LEARNING) ---")
            print(f"Training on new class only: {[gid2breed[str(gid)] for gid in training_classes]}")

        #1) Prepare task data and do cold evaluation
        gid_to_local, seen_df, task_local_to_gid = build_seen_df(full_df, seen_plus_backbone)
        X_train, y_train, X_test, y_test = load_CIL_data(seen_df, backbone_data.CIL_train_tf, backbone_data.val_tf, gid2breed, task_local_to_gid)
        
        #For replay: use cumulative new classes in training data loader, backbone classes come from replay buffer
        #For no-replay: use new class only in training data loader
        
        #Create training data with the specified classes (cumulative new classes for replay, or new class only for no-replay)
        new_class_df = full_df[full_df["gid"].isin(training_classes)]
        
        #CRITICAL FIX: Use the same label mapping as evaluation data
        #Instead of creating new mapping, use the evaluation mapping for consistency
        new_class_df = new_class_df.copy()
        new_class_df["local"] = [eval13_gid_to_local[int(g)] for g in new_class_df["gid"].tolist()]
        
        #Debug: Verify the mapping is correct
        #print(f"DEBUG: Evaluation mapping for training classes:")
        #for gid in training_classes:
        #local_id = eval13_gid_to_local[int(gid)]
        #breed_name = gid2breed[str(gid)]
        #print(f" GID {gid} ({breed_name}) local ID {local_id}")
        
        #Build tensors for new class only
        X_train_new, y_train_new = build_tensors(
            new_class_df[new_class_df["split"]=="train"], 
            backbone_data.CIL_train_tf
        )
        
        #Use new class training data instead
        X_train = X_train_new
        y_train = y_train_new
        print(f"Training data: {len(X_train)} samples for training classes")
        
        #Debug: print unique classes in training data
        unique_classes = torch.unique(y_train).tolist()
        print(f"DEBUG: Unique classes in training data: {unique_classes}")
        print(f"DEBUG: Training on breeds: {[gid2breed[str(target_13_gids[i])] for i in unique_classes]}")
        print(f"DEBUG: Training data size: {len(X_train)} samples")
        print(f"DEBUG: Training label mapping: {[(i, target_13_gids[i], gid2breed[str(target_13_gids[i])]) for i in unique_classes]}")
        
        if cfg['replay']:
            print(f"DEBUG: Replay enabled - old classes will come from replay buffer")
        else:
            print(f"DEBUG: No replay - only new class will be trained on")
        
        #Update all_seen_gids to include only the new classes from this task
        all_seen_gids.update(new_gids)
        
        task_train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=cfg['CIL_batch_size_train'], shuffle=True
        )
        task_test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=cfg['CIL_batch_size_test'], shuffle=False
        )
   
        #2) Expand head for new classes
        prev_num = registry.num_classes()
        if len(new_gids) > 0:
            registry.add_gids(new_gids)
            cil_model.head.expand(len(new_gids))
        
        #3) PRE-TASK cold eval (seen classes only)
        cil_model.eval()
        
        #For the first task, use backbone evaluation since TDM head is random
        if task_idx == 0:
            #Use backbone evaluation for initial cold evaluation
            from cnn_backbone_training_utils import eval_new_classes_on_backbone
            
            #Create backbone-specific data loaders for seen classes
            seen_gid_to_local = {g:i for i,g in enumerate(seen_plus_backbone)}
            seen_df_eval = full_df[full_df["gid"].isin(seen_plus_backbone)].copy()
            seen_df_eval["local"] = [seen_gid_to_local[int(g)] for g in seen_df_eval["gid"].tolist()]
            
            #Create train loader for prototype building (use backbone dogs only)
            backbone_train_seen_df = seen_df_eval[seen_df_eval["gid"].isin(backbone_gids) & (seen_df_eval["split"]=="train")]
            X_train_seen, y_train_seen = build_tensors(backbone_train_seen_df, backbone_data.val_tf)
            train_seen_loader = DataLoader(TensorDataset(X_train_seen, y_train_seen), batch_size=cfg['CIL_batch_size_test'], shuffle=False)
            
            #Use the task test loader for evaluation
            overall_acc_cold, per_class_local_cold = eval_new_classes_on_backbone(
                model=cil_model,
                train_loader=train_seen_loader,  #For prototype building
                test_loader=task_test_loader,    #For evaluation
                device=device,
                num_classes=len(seen_plus_backbone),
                use_proj=True
            )
            
            #Convert to the expected format
            class_acc_local_cold = {i: per_class_local_cold[i] for i in range(len(seen_plus_backbone))}
        else:
            #For subsequent tasks, use TDM head evaluation
            class_acc_local_cold, overall_acc_cold = CIL_post_task_eval(
                task_test_loader, device, cil_model, registry, task_local_to_gid, gid2breed
            )
        print(f"[Seen-only PRE] overall: {overall_acc_cold:.1f}%")
        for cls, acc in class_acc_local_cold.items():
                gid = int(task_local_to_gid[cls].item())
                name = gid2breed[str(gid)]
                print(f"  {cls} ({name}): {acc:.1f}%")
        seen_pre_hist.append({
            "stage": f"Task {task_idx+1}",
            "per_class": class_acc_local_cold,   #dict local->acc
            "local_to_gid": task_local_to_gid.cpu(),  #LongTensor
        })
        
    
        if task_idx == 0:
            backbone_class_acc = overall_acc_cold
        if task_idx == 2:
            report = profile_tinyml_cil(
                cil_model,
                activation_bits_infer=8,
                activation_bits_train=32,
                backbone_weight_bits_deployed=8,
                proj_weight_bits_deployed=32,
                head_weight_bits_deployed=32,
                replay_bits=8,
                include_replay_scores=True,
                include_perchannel_scales=True,
                num_seen_classes=None,
                optimizer="adam",
                include_elementwise=False,
                config=cfg,   #< this picks up img_size, CIL_batch_size_train, replay_batch_size, replay, buffer_size
            )
            #pretty_print_tinyml_report(report)
            save_tinyml_report_csv(report, "plots_and_metrics/dog_breed_cil_tinyml_report.csv")
                        
        #4) Inter-task expansion (warm up grow before epochs)
        if cfg['adaptive_head']:
            if task_idx > 0:
                cil_model.head.update_mask_inter(p = p_inter)
                
        #update replay buffer
        replay_buffer_q.on_new_task(seen_plus_backbone)
        
        #Debug: Check replay buffer after task update
        #print(f"DEBUG: Replay buffer updated for task {task_idx+1}")
        #print(f"DEBUG: Replay buffer now contains classes: {seen_plus_backbone}")
        #print(f"DEBUG: Replay buffer size: {len(replay_buffer_q)}")
        #if hasattr(replay_buffer_q, 'debug_summary'):
        #replay_buffer_q.debug_summary(title=f"After on_new_task (Task {task_idx+1})")
        
        #5) Create teacher snapshot over the previous classes for DDR/KD
        teacher = None  
        if prev_num > 0:  
            teacher = nn.Linear(cil_model.head.in_dim, prev_num, bias=True).to(device)
            with torch.no_grad():
                teacher.weight.copy_(cil_model.head.linear.weight[:prev_num])
                teacher.bias.copy_(cil_model.head.linear.bias[:prev_num])
            teacher.eval()
        
        optimizer = torch.optim.Adam(cil_model.head.parameters(), lr=cfg['CIL_learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        #6) Train current task
        #---------------------------- CIL DURING TASK PIPELINE (EPOCH NUMBER LEVEL)
        for epoch in range(cfg['CIL_epochs']):
            #Ensure frozen
            cil_model.head.train()
            #cil_model.backbone.eval()
            #for m in cil_model.backbone.modules():
            #if isinstance(m, nn.BatchNorm2d):
            #m.eval() # no longer freezing running stats
            
            #Learning rate scheduling
            if epoch == 5:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
            elif epoch == 10:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                    
            #Intra-task TDM adjustment
            if cfg['adaptive_head']:
                if epoch > 0 and epoch % delta_k == 0:
                    cil_model.head.update_mask_intra(p = p_intra)
    
            
            #Inter-task warm-up end (shrink back to target sparsity)
            if cfg['adaptive_head']:
                if task_idx > 0 and epoch == delta_k:
                    cil_model.head.shrink_after_warmup(p=p_inter)
                    
            #Train and get metrics
            train_loss, train_acc = train_with_simplified_tdm(
                model=cil_model, cfg=cfg, registry=registry,
                teacher=teacher, prev_num=prev_num,
                train_loader=task_train_loader, optimizer=optimizer, criterion=criterion, device=device,
                replay_buffer_q=replay_buffer_q if cfg['replay'] else None,
                seen_global_ids=seen_plus_backbone, local_to_gid=task_local_to_gid,
                task_idx=task_idx, epoch=epoch, gid2breed=gid2breed
            )
            
            #Add training metrics to plotter
            clean_breed = clean_breed_name(new_task_breeds)
            task_name = f"Task {task_idx+1}: {clean_breed}"
            plotter.add_cil_task_training_epoch(task_name, epoch, train_loss, train_acc)
        #---------------------------- CIL POST TASK PIPELINE (TASK NUMBER LEVEL)
        
        #note task_classes = seen_classes --> classes up through the finished task
        cil_model.eval()

        #7) POST-TASK eval using consistent approach
        from utils import evaluate_all_classes
        
        #Debug: Print which classes are being evaluated on
        #print(f"DEBUG: Evaluating on {len(target_13_gids)} total classes:")
        #for i, gid in enumerate(target_13_gids):
        #breed_name = gid2breed.get(str(gid), f"gid_{gid}")
        #is_seen = gid in all_seen_gids
        #in_registry = int(gid) in registry.row_for_gid
        #print(f" Class {i}: {breed_name} (GID {gid}) - {'SEEN' if is_seen else 'UNSEEN'} - {'IN_MODEL' if in_registry else 'NOT_IN_MODEL'}")
        
        #print(f"DEBUG: Model registry contains {len(registry.row_for_gid)} classes: {list(registry.row_for_gid.keys())}")
        #print("DEBUG: Model registry mapping (GID registry row):")
        #for gid, row in registry.row_for_gid.items():
        #breed_name = gid2breed.get(str(gid), f"gid_{gid}")
        #print(f" GID {gid} ({breed_name}) registry row {row}")
        
        #Debug: Print unique classes in the evaluation loader
        #print("DEBUG: Checking unique classes in evaluation loader...")
        #eval_classes = set()
        #eval_samples_per_class = {}
        #for xb, yb in eval13_loader:
        #unique_in_batch = torch.unique(yb).tolist()
        #eval_classes.update(unique_in_batch)
        #for class_id in unique_in_batch:
        #eval_samples_per_class[class_id] = eval_samples_per_class.get(class_id, 0) + (yb == class_id).sum().item()
        
        #print(f"DEBUG: Evaluation loader contains {len(eval_classes)} unique classes: {sorted(eval_classes)}")
        #for class_id in sorted(eval_classes):
        #breed_name = gid2breed.get(str(target_13_gids[class_id]), f"gid_{target_13_gids[class_id]}")
        #print(f" Class {class_id}: {breed_name} - {eval_samples_per_class[class_id]} samples")
        
        ## Debug: Check model's ability to predict the new class specifically
        #if not cfg['replay']: # Only for no-replay ablation
        #print(f"DEBUG: Checking model's ability to predict new class: {new_task_breeds}")
        #new_class_gid = new_gids[0]
        #new_class_idx = target_13_gids.index(new_class_gid)
        #print(f"DEBUG: New class GID: {new_class_gid}, Index in target_13_gids: {new_class_idx}")
        #print(f"DEBUG: Is new class in registry? {int(new_class_gid) in registry.row_for_gid}")
        #if int(new_class_gid) in registry.row_for_gid:
        #print(f"DEBUG: New class registry row: {registry.row_for_gid[int(new_class_gid)]}")
        
        #Evaluate using eval_on_fixed_gids for consistency, then filter results
        all_classes_overall, all_classes_acc, sample_counts, correct_counts = evaluate_all_classes(
            cil_model, eval13_loader, registry, target_13_gids, device)

        #Filter to get "seen classes only" results using all_seen_gids
        seen_classes_acc = {}
        for i, gid in enumerate(target_13_gids):
            if gid in all_seen_gids:
                #This is a seen class, find its local index in the current task
                for cls in range(len(task_local_to_gid)):
                    if int(task_local_to_gid[cls].item()) == gid:
                        seen_classes_acc[cls] = all_classes_acc[i]
                        break

        #Calculate proper weighted overall accuracy for seen classes only
        seen_total_samples = sum([sample_counts[i] for i in range(len(target_13_gids)) 
                                 if target_13_gids[i] in all_seen_gids])
        seen_correct_samples = sum([correct_counts[i] for i in range(len(target_13_gids)) 
                                   if target_13_gids[i] in all_seen_gids])
        seen_overall_acc = 100.0 * seen_correct_samples / max(seen_total_samples, 1) if seen_total_samples > 0 else 0.0

        #Use the filtered results for consistency with existing code
        class_acc_local_post = seen_classes_acc
        overall_acc_post = seen_overall_acc
        
        #Print both metrics like dog behaviours
        if task_idx + 1 == len(schedule):
            print(f"DEBUG: all_seen_gids = {sorted(all_seen_gids)}")
            print(f"DEBUG: target_13_gids = {target_13_gids}")
            print(f"DEBUG: seen_total_samples = {seen_total_samples}, all_total_samples = {sum(sample_counts.values())}")
            print(f"[Seen Classes Only] Overall: {overall_acc_post:.2f}%")
            print(f"[All 13 Classes] Overall: {all_classes_overall:.2f}%")
            for i, gid in enumerate(target_13_gids):
                breed_name = gid2breed.get(str(gid), f"gid_{gid}")
                print(f"  {breed_name}: {all_classes_acc[i]:.2f}%")
            print("Final overall accuracy:", overall_acc_post)
        else:
            print(f"[Seen Classes Only] Overall: {overall_acc_post:.2f}%")
            print(f"[All 13 Classes] Overall: {all_classes_overall:.2f}%")
            for i, gid in enumerate(target_13_gids):
                breed_name = gid2breed.get(str(gid), f"gid_{gid}")
                print(f"  {breed_name}: {all_classes_acc[i]:.2f}%")
        
        #Store both class accuracies and overall accuracy for proper weighted calculation
        accuracy_history.append({
            'class_acc': all_classes_acc.copy(),
            'overall_acc': all_classes_overall
        })
        seen_post_hist.append({
            "stage": f"Task {task_idx+1}",
            "per_class": class_acc_local_post,   #dict local->acc
            "local_to_gid": task_local_to_gid.cpu(),
        })
        
        #Add task results to plotter (use deflated data like dog behaviours)
        clean_breed = clean_breed_name(new_task_breeds)
        task_name = f"Task {task_idx+1}: {clean_breed}"
        plotter.add_cil_task(task_name, all_classes_acc, all_classes_overall)
        
        #Save checkpoint and masks for CL metrics calculation
        ckpt_dir = "evaluation/checkpoints_cil"
        task_mask_dir = "evaluation/task_masks"
        
        #Save checkpoint
        ckpt = {
            "head_state_dict": cil_model.head.state_dict(),
            "registry": registry,
            "checkpoint_in_dim": cil_model.head.in_dim,
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f"head_after_T{task_idx+1}.pt"))
        
        #Regenerate test features with current model state for R matrix computation
        print("Regenerating test features with current model state...")
        cil_model.eval()
        
        #Get test features for all 13 target dogs
        eval13_gid_to_local = {g:i for i,g in enumerate(target_13_gids)}
        eval13_df = full_df[full_df["gid"].isin(target_13_gids)].copy()
        eval13_df["local"] = [eval13_gid_to_local[int(g)] for g in eval13_df["gid"].tolist()]
        X_eval13, y_eval13 = build_tensors(eval13_df[eval13_df["split"]=="test"], backbone_data.val_tf)
        eval13_loader = DataLoader(TensorDataset(X_eval13, y_eval13), batch_size=cfg['CIL_batch_size_test'], shuffle=False)
        
        #Extract test features with current model state
        test_feats = []
        y_test = []
        m_test = []
        with torch.no_grad():
            for xb, yb in eval13_loader:
                xb = xb.to(device)
                feats = cil_model._features(xb)
                proj_feats = cil_model.proj(feats)
                test_feats.append(proj_feats.cpu())
                y_test.append(yb)
                m_test.append(torch.ones_like(yb))  #All samples are valid
        
        test_feats = torch.cat(test_feats, dim=0)
        y_test = torch.cat(y_test, dim=0)
        m_test = torch.cat(m_test, dim=0)
        
        #Save test features with current model state
        torch.save({"feats": test_feats, "y_global": y_test.cpu(), "m_test": m_test.cpu()}, "evaluation/test_feats.pt")
        
        #Save evaluation mask (all seen classes)
        eval_mask = np.zeros(13, dtype=bool)
        for i, gid in enumerate(target_13_gids):
            if gid in seen_plus_backbone:
                eval_mask[i] = True
        np.save(os.path.join(task_mask_dir, f"mask_T{task_idx+1}.npy"), eval_mask)
        
        #Save new-class-only mask
        new_mask = np.zeros(13, dtype=bool)
        for i, gid in enumerate(target_13_gids):
            if gid in new_gids:
                new_mask[i] = True
        np.save(os.path.join(task_mask_dir, f"mask_T{task_idx+1}_new.npy"), new_mask)
        
        #Save training mask (all seen classes)
        train_mask = np.zeros(13, dtype=bool)
        for i, gid in enumerate(target_13_gids):
            if gid in seen_plus_backbone:
                train_mask[i] = True
        np.save(os.path.join(task_mask_dir, f"mask_train_T{task_idx+1}.npy"), train_mask)
        
        #Save new-class-only training mask
        new_train_mask = np.zeros(13, dtype=bool)
        for i, gid in enumerate(target_13_gids):
            if gid in new_gids:
                new_train_mask[i] = True
        np.save(os.path.join(task_mask_dir, f"mask_train_T{task_idx+1}_new.npy"), new_train_mask)
        
        #Store the all-classes results for plotting (already calculated above)
        fixed13_hist.append({
            "stage": f"Task {task_idx+1}",
            "overall": all_classes_overall,
            "per_class": all_classes_acc,   #dict 0..12 -> acc
        })
                
        if task_idx + 1 == len(schedule):
            for cls, acc in class_acc_local_post.items():
                gid = int(task_local_to_gid[cls].item())
                name = gid2breed[str(gid)]
                print(f"  {cls} ({name}): {acc:.1f}%")
            print("Final overall accuracy:", overall_acc_post)
            
        else:
            print("Overall Accuracy so far:", overall_acc_post)
            for cls, acc in class_acc_local_post.items():
                gid = int(task_local_to_gid[cls].item())
                name = gid2breed[str(gid)]
                print(f" local id: {cls}, gid: {gid}, breed: ({name}): {acc:.1f}%")
    
    print("CIL PIPELINE FINISHED!")
    
    #Save run metadata for CL metrics calculation
    run_meta = {
        "target_13_gids": target_13_gids,
        "backbone_gids": backbone_gids,
        "schedule": schedule,
        "seed": seed,
        "gid2breed": gid2breed
    }
    with open("evaluation/metrics_meta/run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    
    #Save results
    results = {
        'accuracy_history': accuracy_history,
        'final_accuracy': overall_acc_post,
        'final_class_acc': class_acc_local_post
    }
    
    #Call plotting function
    make_CIL_plots(cfg, gid2breed, seen_pre_hist, seen_post_hist, fixed13_hist, target_13_gids)
    
    #Save plotting data for fast plot iteration
    from cnn_CIL_training_utils import save_plotting_data_to_csv
    save_plotting_data_to_csv(cfg, backbone_all13_acc, accuracy_history, deflated_overall)
    
    #Generate individual training curves plots
    print("\n" + "="*50)
    print("GENERATING INDIVIDUAL TRAINING CURVES PLOTS")
    print("="*50)
    plotter.plot_cil_training_curves()  #Creates the 6 individual plots
    plotter.plot_cil_training_summary()
    plotter.create_summary_report()
    
if __name__ == "__main__":
    main()
