# cnn_CIL_training_utils.py

from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import torch
from utils import normalize_features, kd_loss_ce, make_global_to_local_map
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def load_CIL_data(dog_data, target_dog, all_behaviors, behavior_to_idx, train_mean, train_std):

    # convert to np for sklearn (need for splitting)
    X_target = dog_data[target_dog]['X'].numpy()
    y_target = dog_data[target_dog]['y'].numpy().copy()
    segment_ids = dog_data[target_dog]['segment_ids']

    # GroupShuffleSplit to prevent temporal leakage (n_splits return one for each dog)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X_target, y_target, groups=segment_ids))

        
    X_train_np, X_test_np = X_target[train_idx], X_target[test_idx]
    y_train_np, y_test_np = y_target[train_idx], y_target[test_idx]

    print(f"Segment-aware split: {len(X_train_np)} train samples, {len(X_test_np)} test samples")
    print(f"Train segments: {len(np.unique(segment_ids[train_idx]))}, Test segments: {len(np.unique(segment_ids[test_idx]))}")
    print(f"Ensuring no segment overlap: {len(set(segment_ids[train_idx]) & set(segment_ids[test_idx])) == 0}")

    # Print target dog class distribution
    # class distribution (using global 0..6 labels)
    print(f"\n--- TARGET DOG {target_dog} CLASS DISTRIBUTION ---")
    for i, cls in enumerate(all_behaviors):
        trn = int(np.sum(y_train_np == behavior_to_idx[cls]))
        tst = int(np.sum(y_test_np  == behavior_to_idx[cls]))
        print(f"  {cls}: {trn} train, {tst} test")

    # Normalize target dog data using SAME backbone training statistics
    X_train_np = normalize_features(X_train_np, train_mean, train_std)
    X_test_np  = normalize_features(X_test_np,  train_mean, train_std)

    # Back to tensors for loaders/models
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

    print(f"epoch {epoch:02d} | train_acc={100*epoch_correct/epoch_total:.1f} | loss={epoch_loss/len(train_loader):.3f}")

        
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
