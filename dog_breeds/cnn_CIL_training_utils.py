# cnn_CIL_training_utils.py

import numpy as np
import torch
from utils import kd_loss_ce, make_global_to_local_map_gids
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from utils import build_tensors

def load_CIL_data(CIL_dog_data, train_tf, val_tf, gid2breed=None, local_to_gid=None):

    # Build (X_train, y_train), (X_val, y_val) tensors for the provided subset of df.
    # Expects columns: ['img_path', 'local', 'split'] in CIL_dog_data.

    df_train= CIL_dog_data[CIL_dog_data["split"]=="train"].copy()
    df_val  = CIL_dog_data[CIL_dog_data["split"]=="test"].copy()
    
    X_train, y_train = build_tensors(df_train, train_tf)
    X_val, y_val = build_tensors(df_val, val_tf)

    # Print CIL class distribution (add back in in I change seeds)
    
    # vc_tr = pd.Series(y_train.cpu().numpy()).value_counts().sort_index()
    # vc_te = pd.Series(y_val.cpu().numpy()).value_counts().sort_index()
    # print("\n--- CIL LOCAL CLASS DISTRIBUTION ---")
    # for i in range(max(vc_tr.index.max() if len(vc_tr) else -1,
    #                    vc_te.index.max() if len(vc_te) else -1) + 1):
    #     trn = int(vc_tr.get(i, 0))
    #     tst = int(vc_te.get(i, 0))
    #     # create readable prints
    #     gid = int(local_to_gid[i].item())
    #     name = gid2breed.get(str(gid), f"gid_{gid}")
    #     print(f"  local[{i}] ({name}): {trn} train, {tst} test")
       
    return X_train, y_train, X_val, y_val

def seen_and_new(schedule, t):
    seen = []
    for i in range(t+1):
        seen += schedule[i]
    # keep order unique
    seen = sorted(set(int(g) for g in seen))
    new  = [int(g) for g in schedule[t]]
    return seen, new

def train_with_simplified_tdm(model, cfg, registry, teacher, prev_num,
                              train_loader, optimizer, criterion, device,
                              replay_buffer_q, seen_global_ids, local_to_gid, task_idx, epoch, gid2breed=None):
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0

    kd_weight = cfg['ddr_kd_weight']
    buffer_size = cfg['buffer_size']
    alpha = cfg['cwi_alpha']
    beta  = cfg['cwi_beta']
    ema   = cfg['cwi_ema']

    # Once per task compute
    # rows in the SAME order as local_to_gid so logits match y_local
    rows_seen = torch.tensor(registry.rows_for_gids(local_to_gid.tolist()), device=device)
    gid_to_local = {int(g): i for i, g in enumerate(local_to_gid.tolist())}
    
    W = model.head.linear.weight
    for step, (batch_x, batch_y_local) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y_local = batch_y_local.to(device)
        
        # ---- compute features 
        feats_batch = model._features(batch_x)

        # ---- forward: live
        # check should this be model.head or model.forward_task ???/
        logits_live = model.head.forward_rows(feats_batch, rows_seen)
        live_loss = criterion(logits_live, batch_y_local)

        # ---- forward: replay
        replay_loss = None
        if len(replay_buffer_q) > 0 and task_idx > 0:
            replay_batch_size = cfg['replay_batch_size']
            replay_feats, replay_targets_global = replay_buffer_q.sample_q(replay_batch_size, seen_global_ids, device=device)
            if replay_feats is not None:
                logits_replay = model.head.forward_rows(replay_feats, rows_seen)
                replay_local = torch.tensor(
                    [gid_to_local[int(g.item())] for g in replay_targets_global],
                    dtype=torch.long, device=device
                )
                replay_loss = criterion(logits_replay, replay_local)
        
        # Combine loss on replay and live        
        loss = live_loss + replay_loss if replay_loss is not None else live_loss
        
        #------ KD / DDR on previous classes
        if teacher is not None and prev_num > 0:
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
        global_targets = local_to_gid.to(device)[batch_y_local]  # vectorized: (B,) global IDs
        replay_buffer_q.add_batch(feats_batch, global_targets) # feats batch in # (B, D) FP32    

        # stats
        with torch.no_grad():
            pred = logits_live.argmax(dim=1)
            epoch_total += batch_y_local.size(0)
            epoch_correct += (pred == batch_y_local).sum().item()
            epoch_loss += float(loss.item())

    print(f"epoch {epoch:02d} | train_acc={100*epoch_correct/epoch_total:.1f} | loss={epoch_loss/len(train_loader):.3f}")

        
def CIL_post_task_eval(test_loader, device, model, registry, local_to_gid, gid2breed=None):
    
        with torch.no_grad():
            total, correct = 0, 0
            num_local = len(local_to_gid)
            class_correct = {c: 0 for c in range(num_local)}
            class_total   = {c: 0 for c in range(num_local)}
            
            rows_seen = torch.tensor(registry.rows_for_gids(local_to_gid.tolist()), device=device)
            
            for xb, yb_local in test_loader:
                xb, yb_local = xb.to(device), yb_local.to(device)
                # Make a local label vector for seen_classes 0..len(seen)-1
                feats = model._features(xb)
                logits_seen = model.head.forward_rows(feats, rows_seen)
                pred = logits_seen.argmax(dim=1)

                total += yb_local.size(0)
                correct += (pred == yb_local).sum().item()

                # per-class
                for c in range(num_local):
                    mask = (yb_local == c)
                    class_total[c] += mask.sum().item()
                    class_correct[c] += (pred[mask] == yb_local[mask]).sum().item()

        overall_acc = 100.0 * correct / max(total, 1)
        class_acc = {int(c): (100.0 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0) for c in range(num_local)}
        
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
        
        for c, accuracy in acc_dict.items():
            plot_data.append({
                'Stage': stage,
                'Class': str(c),
                'Accuracy': accuracy
            })
    
    df = pd.DataFrame(plot_data)
    
    # Create the main plot
    plt.figure(figsize=(14, 8))
    
    # Plot per-class accuracy progression
        # Per-class accuracy progression
    plt.figure(figsize=(14, 8))
    ax = sns.lineplot(data=df, x='Stage', y='Accuracy', hue='Class',
                      marker='o', linewidth=2.5, markersize=8)
    plt.title('Per-Class Accuracy Progression: Backbone → CIL Tasks',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Training Stage', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_dir = cfg['plot_dir']
    plt.savefig(f'{plot_dir}/dog_breed_per_class_progression.png', dpi=300, bbox_inches='tight')

    # Overall accuracy progression
    plt.figure(figsize=(10, 6))
    overall_accuracies = []
    for stage_idx, stage in enumerate(stages):
        if stage == 'Backbone':
            acc_dict = backbone_class_acc
        else:
            acc_dict = accuracy_history[stage_idx - 1]
        overall_acc = sum(acc_dict.values()) / max(1, len(acc_dict))
        overall_accuracies.append(overall_acc)

    ax = sns.lineplot(x=stages, y=overall_accuracies, marker='o', linewidth=3, markersize=10)
    plt.title('Overall Accuracy Progression: Backbone → CIL Tasks',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Training Stage', fontsize=12, fontweight='bold')
    plt.ylabel('Overall Accuracy (%)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    for i, (stage, acc) in enumerate(zip(stages, overall_accuracies)):
        plt.annotate(f'{acc:.1f}%', xy=(i, acc), xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/dog_breed_overall_progression.png', dpi=300, bbox_inches='tight')

    print(f"\nResults saved to:")
    print(f"  - {plot_dir}/dog_breed_per_class_progression.png")
    print(f"  - {plot_dir}/dog_breed_overall_progression.png")