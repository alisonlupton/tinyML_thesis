#cnn_CIL_training_utils.py

#from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import torch
from utils import normalize_features, kd_loss_ce, make_global_to_local_map
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random 
from collections import defaultdict
from math import ceil
import os

def split_by_testnum_for_CIL(dog_data, target_dog, verbose=True):
    
    """
    Put ALL windows from one TestNum into train, ALL from another TestNum into test.
    By default, chooses the LARGER session (more windows) as train, the smaller as test.
    """

    sess_ids = dog_data[target_dog]['session_ids']   #(N, 2) [DogID, TestNum]
    if sess_ids is None:
        raise ValueError(f"Dog {target_dog} has no session_ids saved in the NPZ.")
    if sess_ids.shape[1] < 2:
        raise ValueError(f"Dog {target_dog} session_ids should be (N,2) = [DogID, TestNum]. Got shape {sess_ids.shape}.")

    testnums = np.unique(sess_ids[:, 1])
    if len(testnums) < 2:
        raise ValueError(f"Dog {target_dog} has only {len(testnums)} session(s): {testnums}. Need at least 2 to split.")

    train_session = test_session = None
    counts = {tn: (sess_ids[:, 1] == tn).sum() for tn in testnums}
    sorted_sessions = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    train_session = sorted_sessions[0][0]  #session with most windows
    test_session  = sorted_sessions[1][0]  #session with second-most windows

    if train_session == test_session:
        raise ValueError("train_session and test_session must be different.")

    train_mask = (sess_ids[:, 1] == train_session)
    test_mask  = (sess_ids[:, 1] == test_session)

    if verbose:
        print(f"[CIL split] Dog {target_dog}: train session={train_session} ({train_mask.sum()} windows), "
              f"test session={test_session} ({test_mask.sum()} windows)")

    return train_mask, test_mask, train_session, test_session

def check_CIL_validity(dog_data, target_dog, all_behaviors, behavior_to_idx, verbose):
    
    """
    Returns True iff this dog has >0 train AND >0 test windows for every behavior
    when splitting by session (TestNum). You can force which two sessions to use,
    otherwise it picks the first two by TestNum.
    """
    y = dog_data[target_dog]['y'].numpy()
    try:
        train_mask, test_mask, tr_sess, te_sess = split_by_testnum_for_CIL(dog_data, target_dog, verbose=True)
    except ValueError as e:
        if verbose:
            print(f"[CIL validity] Dog {target_dog}: {e}")
        return False

    y_tr = y[train_mask]
    y_te = y[test_mask]

    for cls in all_behaviors:
        gid = behavior_to_idx[cls]
        trn = int(np.sum(y_tr == gid))
        tst = int(np.sum(y_te == gid))
        if trn == 0 or tst == 0:
            if verbose:
                print(f"[CIL validity] Dog {target_dog}: class '{cls}' has train={trn}, test={tst} (session split {tr_sess}/{te_sess})")
            return False

    if verbose:
        print(f"[CIL validity] Dog {target_dog} is VALID with sessions {tr_sess}/{te_sess}.")
    return True

def load_CIL_data(dog_data, target_dog, all_behaviors, behavior_to_idx, train_mean, train_std, seed, verbose):
    """
    Build X_train/X_test from two different sessions of the SAME dog.
    Normalizes with provided train_mean/std from backbone training.
    """
    X = dog_data[target_dog]['X'].numpy()     #(N, C, L)
    y = dog_data[target_dog]['y'].numpy().copy()
    #segment_ids = dog_data[target_dog]['segment_ids'] # not needed here
    train_mask, test_mask, tr_sess, te_sess = split_by_testnum_for_CIL(
        dog_data, target_dog, verbose=verbose
    )

    X_train_np, X_test_np = X[train_mask], X[test_mask]
    y_train_np, y_test_np = y[train_mask], y[test_mask]

    if verbose:
        print(f"Dog {target_dog}: session {tr_sess}  train ({len(X_train_np)} windows), "
              f"session {te_sess}  test ({len(X_test_np)} windows)")
        print(f"\n--- TARGET DOG {target_dog} CLASS DISTRIBUTION (session-split) ---")
        for cls in all_behaviors:
            gid = behavior_to_idx[cls]
            trn = int(np.sum(y_train_np == gid))
            tst = int(np.sum(y_test_np  == gid))
            print(f"  {cls}: {trn} train, {tst} test")

    #normalize using backbone stats
    X_train_np = normalize_features(X_train_np, train_mean, train_std)
    X_test_np  = normalize_features(X_test_np,  train_mean, train_std)

    #back to tensors
    X_train = torch.from_numpy(X_train_np).float()
    y_train = torch.from_numpy(y_train_np).long()
    X_test  = torch.from_numpy(X_test_np).float()
    y_test  = torch.from_numpy(y_test_np).long()

    return X_train, y_train, X_test, y_test

def train_with_simplified_tdm_no_replay(model, cfg, registry, task_classes, teacher, prev_num,
                                        train_loader, behavior_to_idx, optimizer, criterion, device,
                                        seen_global_ids, seen_names, task_idx, epoch):
    """Modified training function for no-replay ablation that uses evaluation mapping."""
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0

    #For no-replay, we use forward_eval7 to get logits for all classes
    #This ensures the model can predict the new class at the correct evaluation local ID
    for step, (batch_x, batch_y_local) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y_local = batch_y_local.to(device)

        #---- forward: use eval7 to get logits for all classes
        logits_live = model.forward_eval7(batch_x, registry)
        live_loss = criterion(logits_live, batch_y_local)

        #---- backward
        optimizer.zero_grad()
        live_loss.backward()
        optimizer.step()

        #---- metrics
        epoch_loss += live_loss.item()
        with torch.no_grad():
            pred = logits_live.argmax(dim=1)
            epoch_correct += (pred == batch_y_local).sum().item()
            epoch_total += batch_y_local.size(0)

    return epoch_loss / len(train_loader), 100.0 * epoch_correct / epoch_total

def train_with_simplified_tdm(model, cfg, registry, task_classes, teacher, prev_num,
                              train_loader, behavior_to_idx, optimizer, criterion, device,
                              replay_buffer_q, seen_global_ids, seen_names, task_idx, epoch):
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0

    kd_weight = cfg['ddr_kd_weight']
    buffer_size = cfg['buffer_size']
    alpha = cfg['cwi_alpha']
    beta  = cfg['cwi_beta']
    ema   = cfg['cwi_ema']

    #Once per task computer
    rows_seen = torch.tensor(registry.rows_for_all_known(), device=device)
    map_seen, _ = make_global_to_local_map(registry.seen_classes(),
                                       behavior_to_idx,
                                       device=device)
    _, task_gids = make_global_to_local_map(task_classes, behavior_to_idx, device=device)
    
    W = model.head.linear.weight
    for step, (batch_x, batch_y_local) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y_local = batch_y_local.to(device)

        #---- forward: live
        logits_live = model.forward_task(batch_x, registry, task_classes)
        live_loss = criterion(logits_live, batch_y_local)

        #---- forward: replay
        replay_loss = None
        if len(replay_buffer_q) > cfg['replay_batch_size']:
            replay_batch_size = cfg['replay_batch_size']
            replay_feats, replay_targets_global = replay_buffer_q.sample_q(replay_batch_size, seen_global_ids, device=device)
            if replay_feats is not None:
                logits_replay = model.head.forward_rows(replay_feats, rows_seen)
                replay_local = map_seen[replay_targets_global]  #shape: (B,)
                replay_loss = criterion(logits_replay, replay_local)
        
        #Combine loss on replay and live
        loss = live_loss + replay_loss if replay_loss is not None else live_loss
        
        with torch.no_grad():
            feats_batch = model._features(batch_x).detach().cpu()
        
        #------ KD / DDR on previous classes
        if teacher is not None and prev_num > 0:
            logits_prev = model.head.forward_rows(feats_batch, list(range(prev_num)))
            with torch.no_grad():
                teacher_logits = teacher(feats_batch)
            loss += kd_loss_ce(logits_prev, teacher_logits, T=cfg['kd_temp'], weight=kd_weight)


        #--------- CWI Updating
        #calculate grads w.r.t. head weights for each component
        g_curr = torch.autograd.grad(live_loss, W, retain_graph=True, allow_unused=True)[0]
        g_mem  = torch.autograd.grad(replay_loss, W, retain_graph=True, allow_unused=True)[0] if replay_loss is not None else None

        model.head.update_cwi_from_grads(g_curr, g_mem, alpha, beta, ema)

        #Step
        optimizer.zero_grad()
        loss.backward()
        
        with torch.no_grad(): #keep pruned weights from updating
            if W.grad is not None:
                W.grad.mul_(model.head.mask)
        optimizer.step()

        #apply mask every step! just to enforce
        with torch.no_grad():           
            W.mul_(model.head.mask)
        
        #------ add to replay
        global_targets = task_gids[batch_y_local]  #vectorized: (B,) global IDs
        if cfg['replay']:
            replay_buffer_q.add_batch(feats_batch, global_targets) #feats batch in # (B, D) FP32

        with torch.no_grad():
            pred = logits_live.argmax(dim=1)
            epoch_total += batch_y_local.size(0)
            epoch_correct += (pred == batch_y_local).sum().item()
            epoch_loss += float(loss.item())

    train_acc = 100 * epoch_correct / epoch_total
    train_loss = epoch_loss / len(train_loader)
    print(f"epoch {epoch:02d} | train_acc={train_acc:.1f} | loss={train_loss:.3f}")
    
    return train_loss, train_acc

        
def CIL_post_task_eval(seen_classes, test_loader, device, all_behaviors, model, registry):
    
        with torch.no_grad():
            total, correct = 0, 0
            class_correct = {c: 0 for c in seen_classes}
            class_total   = {c: 0 for c in seen_classes}

            for xb, yb_local_full in test_loader:
                xb = xb.to(device)
                #Make a local label vector for seen_classes 0..len(seen)-1
                yb_local = torch.zeros_like(yb_local_full)
                for i, cls in enumerate(seen_classes):
                    yb_local[yb_local_full == all_behaviors.index(cls)] = i
                yb_local = yb_local.to(device)

                logits_seen = model.forward_task(xb, registry, seen_classes)
                pred = logits_seen.argmax(dim=1)

                total += yb_local.size(0)
                correct += (pred == yb_local).sum().item()

                #per-class
                for i, cls in enumerate(seen_classes):
                    mask = (yb_local == i)
                    class_total[cls] += mask.sum().item()
                    class_correct[cls] += (pred[mask] == yb_local[mask]).sum().item()

        overall_acc = 100.0 * correct / max(total, 1)
        class_acc = {cls: 100.0 * class_correct[cls] / max(class_total[cls], 1) for cls in seen_classes}
        
        return class_acc, overall_acc
    
def save_CIL_progression_to_csv(cfg, backbone_class_acc, accuracy_history, plot_dir, backbone_overall_acc=None):
    """
    Save CIL progression data to CSV for multi-seed analysis.
    """
    #Get seed from config
    seed = cfg.get('random_seed', 'unknown')
    
    #Prepare data for CSV
    csv_data = []
    
    #Add backbone data
    for behavior, accuracy in backbone_class_acc.items():
        csv_data.append({
            'seed': seed,
            'stage': 'Backbone',
            'behavior': behavior,
            'accuracy': accuracy,
            'overall_accuracy': backbone_overall_acc if backbone_overall_acc is not None else sum(backbone_class_acc.values()) / len(backbone_class_acc)
        })
    
    #Add CIL task data
    for task_idx, task_metrics in enumerate(accuracy_history):
        stage = f'Task_{task_idx + 1}'
        
        #Handle new data structure with both class_acc and overall_acc
        if isinstance(task_metrics, dict):
            class_acc = task_metrics['class_acc']
            overall_acc = task_metrics['overall_acc']
        else:
            #Backward compatibility for old format
            class_acc = task_metrics
            overall_acc = sum(class_acc.values()) / len(class_acc)
        
        for behavior, accuracy in class_acc.items():
            csv_data.append({
                'seed': seed,
                'stage': stage,
                'behavior': behavior,
                'accuracy': accuracy,
                'overall_accuracy': overall_acc
            })
    
    #Create DataFrame and save
    df = pd.DataFrame(csv_data)
    
    #Append to existing file or create new one
    csv_file = os.path.join(plot_dir, 'cil_progression_data.csv')
    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
        try:
            existing_df = pd.read_csv(csv_file)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            #File exists but is empty or corrupted, start fresh
            combined_df = df
    else:
        combined_df = df
    
    combined_df.to_csv(csv_file, index=False)
    print(f"CIL progression data saved to {csv_file}")

def save_plotting_data_to_csv(cfg, backbone_class_acc, accuracy_history, backbone_overall_acc=None):
    """
    Save plotting data to CSV files for the two intelligent plots.
    This allows for faster plot iteration without re-running training.
    """
    plot_dir = cfg["plot_dir"]
    
    stages = ['Backbone'] + [f'Task {i+1}' for i in range(len(accuracy_history))]
    
    #Get all unique behaviors across all stages
    all_behaviors = set(backbone_class_acc.keys())
    for stage_data in accuracy_history:
        if isinstance(stage_data, dict):
            all_behaviors.update(stage_data['class_acc'].keys())
        else:
            all_behaviors.update(stage_data.keys())
    
    #Prepare per-class plotting data
    plot_data = []
    for stage_idx, stage in enumerate(stages):
        if stage == 'Backbone':
            current_acc_dict = backbone_class_acc
        else:
            if isinstance(accuracy_history[stage_idx - 1], dict):
                current_acc_dict = accuracy_history[stage_idx - 1]['class_acc']
            else:
                current_acc_dict = accuracy_history[stage_idx - 1]
        
        for behavior in all_behaviors:
            accuracy = current_acc_dict.get(behavior, 0.0)
            plot_data.append({
                'Stage': stage,
                'Behavior': behavior,
                'Accuracy': accuracy
            })
    
    #Save per-class data
    per_class_df = pd.DataFrame(plot_data)
    per_class_csv = os.path.join(plot_dir, 'intelligent_plotting_per_class.csv')
    per_class_df.to_csv(per_class_csv, index=False)
    print(f"Per-class plotting data saved to {per_class_csv}")
    
    #Prepare overall accuracy data
    overall_data = []
    for stage_idx, stage in enumerate(stages):
        if stage == 'Backbone':
            overall_acc = backbone_overall_acc if backbone_overall_acc is not None else 0.0
        else:
            overall_acc = accuracy_history[stage_idx - 1]['overall_acc']
        overall_data.append({
            'Stage': stage,
            'Overall_Accuracy': overall_acc
        })
    
    #Save overall data
    overall_df = pd.DataFrame(overall_data)
    overall_csv = os.path.join(plot_dir, 'intelligent_plotting_overall.csv')
    overall_df.to_csv(overall_csv, index=False)
    print(f"Overall plotting data saved to {overall_csv}")

def plot_from_csv(cfg):
    """
    Load plotting data from CSV files and create the two intelligent plots.
    This allows for faster plot iteration without re-running training.
    """
    plot_dir = cfg["plot_dir"]
    
    #Load per-class data
    per_class_csv = os.path.join(plot_dir, 'ablations/intelligent_plotting_per_class_withreplay.csv')
    overall_csv = os.path.join(plot_dir, 'ablations/intelligent_plotting_overall_withreplay.csv')
    
    if not os.path.exists(per_class_csv) or not os.path.exists(overall_csv):
        print(f"Error: CSV files not found. Please run training first to generate plotting data.")
        print(f"Expected files: {per_class_csv}, {overall_csv}")
        return
    
    #Load data
    per_class_df = pd.read_csv(per_class_csv)
    overall_df = pd.read_csv(overall_csv)
    
    #Set LaTeX-style fonts to match document
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 15,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 17,
        'figure.titlesize': 17,
        'text.usetex': False,  #Set to True if have LaTeX installed
        'mathtext.fontset': 'cm'  #Use Computer Modern for math text
    })
    
    #Set seaborn style to match other plots
    plt.style.use('seaborn-v0_8-deep')
    sns.set_palette("Set1")  #This will affect all subsequent plots

    stages = overall_df['Stage'].tolist()
    
    #Create per-class plot
    plt.figure(figsize=(14, 8))
    
    #Plot each behavior separately to match the overall plot approach
    for behavior in per_class_df['Behavior'].unique():
        behavior_data = per_class_df[per_class_df['Behavior'] == behavior]
        #Sort by stage order to ensure correct plotting
        behavior_data = behavior_data.sort_values('Stage', key=lambda x: x.map({stage: i for i, stage in enumerate(stages)}))
        plt.plot(range(len(stages)), behavior_data['Accuracy'], 
                marker='o', linewidth=2.5, markersize=8, label=behavior)
    
    #Set x-axis ticks to match the overall plot approach
    plt.xticks(range(len(stages)), stages, rotation=45)
    
    #Customize the plot
    plt.xlabel('Training Stage', fontweight='bold')
    plt.ylabel('Accuracy (%)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Behavior Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/intelligent_tdm_per_class_progression.pdf', bbox_inches='tight')
    plt.close()
    
    #Create overall accuracy plot
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(x=stages, y=overall_df['Overall_Accuracy'], marker='o', linewidth=3, markersize=10)
    
    #Customize
    plt.xlabel('Training Stage', fontweight='bold')
    plt.ylabel('Overall Accuracy (%)', fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    #Add value annotations
    for i, (stage, acc) in enumerate(zip(stages, overall_df['Overall_Accuracy'])):
        plt.annotate(f'{acc:.1f}%', 
                    xy=(i, acc), 
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/intelligent_tdm_overall_progression.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Plots generated from CSV data and saved to {plot_dir}")

def make_CIL_plots(cfg, backbone_class_acc, accuracy_history, backbone_overall_acc=None):
    
    #Set LaTeX-style fonts to match document
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 15,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 25,
        'ytick.labelsize': 20,
        'legend.fontsize': 17,
        'figure.titlesize': 17,
        'text.usetex': False,  #Set to True if have LaTeX installed
        'mathtext.fontset': 'cm'  #Use Computer Modern for math text
    })
    
    #Set seaborn style to match other plots
    plt.style.use('seaborn-v0_8-deep')
    
    #Prepare data for plotting
    stages = ['Backbone'] + [f'Task {i+1}' for i in range(len(accuracy_history))]
    
    #Get all unique behaviors across all stages
    all_behaviors = set(backbone_class_acc.keys())
    for stage_data in accuracy_history:
        if isinstance(stage_data, dict):
            all_behaviors.update(stage_data['class_acc'].keys())
        else:
            all_behaviors.update(stage_data.keys())
    
    #Create DataFrame for seaborn
    plot_data = []
    for stage_idx, stage in enumerate(stages):
        if stage == 'Backbone':
            #For backbone stage, only show backbone classes
            current_acc_dict = backbone_class_acc
        else:
            #For CIL stages, show all classes introduced up to this point
            if isinstance(accuracy_history[stage_idx - 1], dict):
                current_acc_dict = accuracy_history[stage_idx - 1]['class_acc']
            else:
                current_acc_dict = accuracy_history[stage_idx - 1]
        
        #For each behavior, either show its accuracy or 0% if not introduced yet
        for behavior in all_behaviors:
            if behavior in current_acc_dict:
                accuracy = current_acc_dict[behavior]
            else:
                accuracy = 0.0  #Not introduced yet
            
            plot_data.append({
                'Stage': stage,
                'Behavior': behavior,
                'Accuracy': accuracy
            })
    
    df = pd.DataFrame(plot_data)
    
    #Create the main plot
    plt.figure(figsize=(14, 8))
    
    #Plot per-class accuracy progression
    ax = sns.lineplot(data=df, x='Stage', y='Accuracy', hue='Behavior', 
                     marker='o', linewidth=2.5, markersize=8)
    
    #Set x-axis ticks to match the overall plot approach
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages)
    
    #Customize the plot
    #plt.title('Per-Class Accuracy Progression: Backbone CIL Tasks',
    #fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Training Stage', fontweight='bold')
    plt.ylabel('Accuracy (%)', fontweight='bold')
    
    #Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    #Add grid and customize legend
    plt.grid(True, alpha=0.3)
    plt.legend(title='Behavior Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    #Add value annotations on points with better positioning to avoid overlap
    for stage_idx, stage in enumerate(stages):
        if stage == 'Backbone':
            acc_dict = backbone_class_acc
        else:
            #Handle new data structure with both class_acc and overall_acc
            if isinstance(accuracy_history[stage_idx - 1], dict):
                acc_dict = accuracy_history[stage_idx - 1]['class_acc']
            else:
                #Backward compatibility for old format
                acc_dict = accuracy_history[stage_idx - 1]
        
        #Sort behaviors by accuracy to avoid text overlap
        #Ensure acc_dict is a dictionary of {behavior: accuracy} pairs
        #if isinstance(acc_dict, dict):
        #sorted_behaviors = sorted(acc_dict.items(), key=lambda x: x[1], reverse=True)
            
        #for i, (behavior, accuracy) in enumerate(sorted_behaviors):
        ## Offset text based on position to avoid overlap
        #y_offset = 10 if i % 2 == 0 else -15
        #x_offset = 5 if i % 2 == 0 else -5
                
        #plt.annotate(f'{accuracy:.1f}%',
        #xy=(stage_idx, accuracy),
        #xytext=(x_offset, y_offset), textcoords='offset points',
        #fontsize=9, fontweight='bold', alpha=0.8,
        #bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    #Adjust layout to prevent label cutoff
    plt.tight_layout()
    plot_dir = cfg['plot_dir']
    #Save the plot
    plt.savefig(f'{plot_dir}/intelligent_tdm_per_class_progression.pdf', bbox_inches='tight')
    #plt.show()
    
    #Create a second plot showing overall accuracy progression
    plt.figure(figsize=(10, 6))
    
    #Use the overall accuracies directly (no calculations needed)
    overall_accuracies = []
    for stage_idx, stage in enumerate(stages):
        if stage == 'Backbone':
            #Use the passed backbone overall accuracy
            overall_acc = backbone_overall_acc
        else:
            #Use the overall accuracy from accuracy_history (already calculated)
            overall_acc = accuracy_history[stage_idx - 1]['overall_acc']
        overall_accuracies.append(overall_acc)
    
    #Plot overall accuracy
    ax = sns.lineplot(x=stages, y=overall_accuracies, marker='o', linewidth=3, markersize=10)
    
    #Customize
    #plt.title('Overall Accuracy Progression: Backbone CIL Tasks',
    #fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Training Stage', fontweight='bold')
    plt.ylabel('Overall Accuracy (%)', fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    #Add value annotations
    for i, (stage, acc) in enumerate(zip(stages, overall_accuracies)):
        plt.annotate(f'{acc:.1f}%', 
                    xy=(i, acc), 
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/intelligent_tdm_overall_progression.pdf', bbox_inches='tight')
    #plt.show()
    
    #Save CSV data for multi-seed analysis
    save_CIL_progression_to_csv(cfg, backbone_class_acc, accuracy_history, plot_dir, backbone_overall_acc)
    
    #Save plotting data for fast plot iteration
    save_plotting_data_to_csv(cfg, backbone_class_acc, accuracy_history, backbone_overall_acc)
    
    print(f"\nResults saved to:")
    print(f"  - plots/ intelligent_tdm_per_class_progression.png")
    print(f"  - plots/ intelligent_tdm_overall_progression.png")
    print(f"  - plots/ cil_progression_data.csv")
    #print(f"TinyML metrics saved to plots/tdm_intelligent_metrics.csv")
