# cnn_CIL_training_utils.py

# from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import torch
from utils import normalize_features, kd_loss_ce, make_global_to_local_map
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random 
from collections import defaultdict
from math import ceil

def split_by_testnum_for_CIL(dog_data, target_dog, verbose=True):
    
    """
    Put ALL windows from one TestNum into train, ALL from another TestNum into test.
    By default, chooses the LARGER session (more windows) as train, the smaller as test.
    """

    sess_ids = dog_data[target_dog]['session_ids']   # (N, 2) [DogID, TestNum]
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
    train_session = sorted_sessions[0][0]  # session with most windows
    test_session  = sorted_sessions[1][0]  # session with second-most windows

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
    X = dog_data[target_dog]['X'].numpy()     # (N, C, L)
    y = dog_data[target_dog]['y'].numpy().copy()
    # segment_ids = dog_data[target_dog]['segment_ids']  # not needed here
    train_mask, test_mask, tr_sess, te_sess = split_by_testnum_for_CIL(
        dog_data, target_dog, verbose=verbose
    )

    X_train_np, X_test_np = X[train_mask], X[test_mask]
    y_train_np, y_test_np = y[train_mask], y[test_mask]

    if verbose:
        print(f"Dog {target_dog}: session {tr_sess} → train ({len(X_train_np)} windows), "
              f"session {te_sess} → test ({len(X_test_np)} windows)")
        print(f"\n--- TARGET DOG {target_dog} CLASS DISTRIBUTION (session-split) ---")
        for cls in all_behaviors:
            gid = behavior_to_idx[cls]
            trn = int(np.sum(y_train_np == gid))
            tst = int(np.sum(y_test_np  == gid))
            print(f"  {cls}: {trn} train, {tst} test")

    # normalize using backbone stats
    X_train_np = normalize_features(X_train_np, train_mean, train_std)
    X_test_np  = normalize_features(X_test_np,  train_mean, train_std)

    # back to tensors
    X_train = torch.from_numpy(X_train_np).float()
    y_train = torch.from_numpy(y_train_np).long()
    X_test  = torch.from_numpy(X_test_np).float()
    y_test  = torch.from_numpy(y_test_np).long()

    return X_train, y_train, X_test, y_test

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

    # Once per task computer
    rows_seen = torch.tensor(registry.rows_for_all_known(), device=device)
    map_seen, _ = make_global_to_local_map(registry.seen_classes(),
                                       behavior_to_idx,
                                       device=device)
    _, task_gids = make_global_to_local_map(task_classes, behavior_to_idx, device=device)
    
    W = model.head.linear.weight
    for step, (batch_x, batch_y_local) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y_local = batch_y_local.to(device)

        # ---- forward: live
        logits_live = model.forward_task(batch_x, registry, task_classes)
        live_loss = criterion(logits_live, batch_y_local)

        # ---- forward: replay
        replay_loss = None
        if len(replay_buffer_q) > 0 and task_idx > 0:
            replay_batch_size = cfg['replay_batch_size']
            replay_feats, replay_targets_global = replay_buffer_q.sample_q(replay_batch_size, seen_global_ids, device=device)
            if replay_feats is not None:
                logits_replay = model.head.forward_rows(replay_feats, rows_seen)
                replay_local = map_seen[replay_targets_global]  # shape: (B,)
                replay_loss = criterion(logits_replay, replay_local)
        
        # Combine loss on replay and live        
        loss = live_loss + replay_loss if replay_loss is not None else live_loss
        
        #------ KD / DDR on previous classes
        if teacher is not None and prev_num > 0:
            with torch.no_grad():
                feats_batch = model._features(batch_x).detach().cpu()
            logits_prev = model.head.forward_rows(feats_batch, list(range(prev_num)))
            with torch.no_grad():
                teacher_logits = teacher(feats_batch)
            loss += kd_loss_ce(logits_prev, teacher_logits, T=2.0, weight=kd_weight)


        #--------- CWI Updating
        # calculate grads w.r.t. head weights for each component 
        g_curr = torch.autograd.grad(live_loss, W, retain_graph=True, allow_unused=True)[0]
        g_mem  = torch.autograd.grad(replay_loss, W, retain_graph=True, allow_unused=True)[0] if replay_loss is not None else None

        model.head.update_cwi_from_grads(g_curr, g_mem, alpha, beta, ema)

        # Step
        optimizer.zero_grad()
        loss.backward()
        
        with torch.no_grad(): # keep pruned weights from updating
            if W.grad is not None:
                W.grad.mul_(model.head.mask)
        optimizer.step()

        # apply mask every step! just to enforce
        with torch.no_grad():           
            W.mul_(model.head.mask)
        
        #------ add to replay
        global_targets = task_gids[batch_y_local]  # vectorized: (B,) global IDs
        replay_buffer_q.add_batch(feats_batch, global_targets) # feats batch in # (B, D) FP32    

        # stats
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
                # Make a local label vector for seen_classes 0..len(seen)-1
                yb_local = torch.zeros_like(yb_local_full)
                for i, cls in enumerate(seen_classes):
                    yb_local[yb_local_full == all_behaviors.index(cls)] = i
                yb_local = yb_local.to(device)

                logits_seen = model.forward_task(xb, registry, seen_classes)
                pred = logits_seen.argmax(dim=1)

                total += yb_local.size(0)
                correct += (pred == yb_local).sum().item()

                # per-class
                for i, cls in enumerate(seen_classes):
                    mask = (yb_local == i)
                    class_total[cls] += mask.sum().item()
                    class_correct[cls] += (pred[mask] == yb_local[mask]).sum().item()

        overall_acc = 100.0 * correct / max(total, 1)
        class_acc = {cls: 100.0 * class_correct[cls] / max(class_total[cls], 1) for cls in seen_classes}
        
        return class_acc, overall_acc
    
def make_CIL_plots(cfg, backbone_class_acc, accuracy_history):
    
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Prepare data for plotting
    stages = ['Backbone'] + [f'Task {i+1}' for i in range(len(accuracy_history))]
    
    # Create DataFrame for seaborn
    plot_data = []
    for stage_idx, stage in enumerate(stages):
        if stage == 'Backbone':
            acc_dict = backbone_class_acc
        else:
            acc_dict = accuracy_history[stage_idx - 1]
        
        for behavior, accuracy in acc_dict.items():
            plot_data.append({
                'Stage': stage,
                'Behavior': behavior,
                'Accuracy': accuracy
            })
    
    df = pd.DataFrame(plot_data)
    
    # Create the main plot
    plt.figure(figsize=(14, 8))
    
    # Plot per-class accuracy progression
    ax = sns.lineplot(data=df, x='Stage', y='Accuracy', hue='Behavior', 
                     marker='o', linewidth=2.5, markersize=8)
    
    # Customize the plot
    plt.title('Per-Class Accuracy Progression: Backbone → CIL Tasks', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Training Stage', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add grid and customize legend
    plt.grid(True, alpha=0.3)
    plt.legend(title='Behavior Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add value annotations on points
    for stage_idx, stage in enumerate(stages):
        if stage == 'Backbone':
            acc_dict = backbone_class_acc
        else:
            acc_dict = accuracy_history[stage_idx - 1]
        
        for behavior, accuracy in acc_dict.items():
            plt.annotate(f'{accuracy:.1f}%', 
                        xy=(stage_idx, accuracy), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plot_dir = cfg['plot_dir']
    # Save the plot
    plt.savefig(f'{plot_dir}/intelligent_tdm_per_class_progression.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # Create a second plot showing overall accuracy progression
    plt.figure(figsize=(10, 6))
    
    # Calculate overall accuracy for each stage
    overall_accuracies = []
    for stage_idx, stage in enumerate(stages):
        if stage == 'Backbone':
            acc_dict = backbone_class_acc
        else:
            acc_dict = accuracy_history[stage_idx - 1]
        overall_acc = sum(acc_dict.values()) / len(acc_dict)
        overall_accuracies.append(overall_acc)
    
    # Plot overall accuracy
    ax = sns.lineplot(x=stages, y=overall_accuracies, marker='o', linewidth=3, markersize=10)
    
    # Customize
    plt.title('Overall Accuracy Progression: Backbone → CIL Tasks', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Training Stage', fontsize=12, fontweight='bold')
    plt.ylabel('Overall Accuracy (%)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Add value annotations
    for i, (stage, acc) in enumerate(zip(stages, overall_accuracies)):
        plt.annotate(f'{acc:.1f}%', 
                    xy=(i, acc), 
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/intelligent_tdm_overall_progression.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    print(f"\nResults saved to:")
    print(f"  - plots/ intelligent_tdm_per_class_progression.png")
    print(f"  - plots/ intelligent_tdm_overall_progression.png")
    # print(f"TinyML metrics saved to plots/tdm_intelligent_metrics.csv")
