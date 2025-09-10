#cnn_CIL_training.py


import torch
torch.backends.quantized.engine = 'qnnpack'  
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from process_dog_data import process_dog_data
from utils import load_config, set_seed, ClassRegistry, build_tensors
from metrics import profile_tinyml_cil, pretty_print_tinyml_report
from models.cnn import TunedMCUStudentCNN_CIL
from cnn_backbone_training_utils import load_data_cnn_backbone, eval_on_fixed_gids, quantise_frozen_backbone, features_quantized, compute_perchannel_scales
from cnn_CIL_training_utils import seen_and_new, load_CIL_data, CIL_post_task_eval, train_with_simplified_tdm, make_CIL_plots
from dog_breed_training_plotter import DogBreedCILTrainingPlotter, clean_breed_name
from replay import BalancedQuantReplayDynamic
import json
from pathlib import Path
import pandas as pd
import types


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
    
    #Initialize training plotter
    plotter = DogBreedCILTrainingPlotter(save_dir=cfg['plot_dir'], seed=seed)
    
    backbone_gids, schedule = process_dog_data(cfg) #these are lists of gids
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
    
    registry = ClassRegistry(gid2name=gid2breed)
    registry.add_gids(backbone_gids)  #seed with base backbone dogs
    
    replay_buffer_q.on_new_task(backbone_gids)

    
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
    eval13_loader = DataLoader(TensorDataset(X_eval13, y_eval13),
                            batch_size=cfg['CIL_batch_size_test'], shuffle=False)
    #----------------------------------------------------------------------------
    
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
    
    print(f"Backbone evaluation on 13 dogs:")
    print(f"  Backbone dogs (10): {[f'{acc:.1f}%' for acc in backbone_accuracies]}")
    print(f"  Unseen dogs (3): {[f'{acc:.1f}%' for acc in unseen_accuracies]}")
    print(f"  Deflated overall: {deflated_overall:.1f}%")
    
    fixed13_hist.append({
        "stage": "Backbone",
        "overall": deflated_overall,  #Use deflated accuracy for overall plot
        "per_class": per13_0
})
    #put this above in a func later

     
    #---------------------------- CIL PRE TASK PIPELINE (TASK NUMBER LEVEL)
    for task_idx, _task_gids in enumerate(schedule):        
        
        #seen classes up to this task
        seen_gids, new_gids = seen_and_new(schedule, task_idx)
        new_task_breeds = gid2breed[str(new_gids[0])]
        seen_plus_backbone = sorted(set(backbone_gids + seen_gids))
        print(f"\n--- CIL TASK {task_idx + 1}: Added '{new_task_breeds}' ---")
 
        #1) Prepare task data and do cold evaluation
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
                
        if task_idx == 0:
            ## Measure Sizing!
                
            report = profile_tinyml_cil(
                model=cil_model,
                input_size=(1,3,160,160),
                activation_bits_infer=8,   #inference path is still FP32
                activation_bits_train=32,   #head training in FP32
                weight_bits_deployed=32,    #model params are stored in FP32 right now
                batch_size_train=cfg['CIL_batch_size_train'],
                replay_size=cfg['buffer_size'],
                replay_bits=8,             #caching latents size
                optimizer="adam"
            )
            
            pretty_print_tinyml_report(report)
               
        #4) Inter-task expansion (warm up grow before epochs)
        if task_idx > 0:
            cil_model.head.update_mask_inter(p = p_inter)
            
        #update replay buffer
        replay_buffer_q.on_new_task(seen_plus_backbone)
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
            if epoch > 0 and epoch % delta_k == 0:
                cil_model.head.update_mask_intra(p = p_intra)
   
            #Inter-task warm-up end (shrink back to target sparsity)
            if task_idx > 0 and epoch == delta_k:
                cil_model.head.shrink_after_warmup(p=p_inter)
                
            #Train and get metrics
            train_loss, train_acc = train_with_simplified_tdm(
                model=cil_model, cfg=cfg, registry=registry,
                teacher=teacher, prev_num=prev_num,
                train_loader=task_train_loader, optimizer=optimizer, criterion=criterion, device=device,
                replay_buffer_q=replay_buffer_q,
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

        #7) POST-TASK eval (seen-only) comparable to (#3)
        class_acc_local_post, overall_acc_post = CIL_post_task_eval(
            task_test_loader, device, cil_model, registry, task_local_to_gid, gid2breed)
        print(f"[Seen-only POST] overall: {overall_acc_post:.1f}%")
        accuracy_history.append(class_acc_local_post.copy())
        seen_post_hist.append({
            "stage": f"Task {task_idx+1}",
            "per_class": class_acc_local_post,   #dict local->acc
            "local_to_gid": task_local_to_gid.cpu(),
        })
        
        #Add task results to plotter
        clean_breed = clean_breed_name(new_task_breeds)
        task_name = f"Task {task_idx+1}: {clean_breed}"
        plotter.add_cil_task(task_name, class_acc_local_post, overall_acc_post)
        
        #8) evaluation on all classes (inlcuding future ones)
        fixed13_overall, fixed13_per = eval_on_fixed_gids(
            model=cil_model,
            loader=eval13_loader,               #labels 0..12 map to target_13_gids order
            registry=registry,            #IMPORTANT: same registry the head uses
            fixed_gids=target_13_gids,         #fixed label set (length 13)
            device=device
        )
        print(f"[Fixed-13 Eval] Overall: {fixed13_overall:.1f}%")
        fixed13_hist.append({
            "stage": f"Task {task_idx+1}",
            "overall": fixed13_overall,
            "per_class": fixed13_per,   #dict 0..12 -> acc
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

    
    #Save results
    results = {
        'accuracy_history': accuracy_history,
        'final_accuracy': overall_acc_post,
        'final_class_acc': class_acc_local_post
    }
    
    #Call plotting function
    make_CIL_plots(cfg, gid2breed, seen_pre_hist, seen_post_hist, fixed13_hist, target_13_gids)
    
    #Generate individual training curves plots
    print("\n" + "="*50)
    print("GENERATING INDIVIDUAL TRAINING CURVES PLOTS")
    print("="*50)
    plotter.plot_cil_training_curves()  #Creates the 6 individual plots
    plotter.plot_cil_training_summary()
    plotter.create_summary_report()
    
if __name__ == "__main__":
    main()
