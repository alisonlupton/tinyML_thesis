#cnn_CIL_training_utils.py

import numpy as np
import torch
from utils import kd_loss_ce, make_global_to_local_map_gids
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from utils import build_tensors
import torch.nn.functional as F

def load_CIL_data(CIL_dog_data, train_tf, val_tf, gid2breed=None, local_to_gid=None):

    #Build (X_train, y_train), (X_val, y_val) tensors for the provided subset of df.
    #Expects columns: ['img_path', 'local', 'split'] in CIL_dog_data.

    df_train= CIL_dog_data[CIL_dog_data["split"]=="train"].copy()
    df_val  = CIL_dog_data[CIL_dog_data["split"]=="test"].copy()
    
    X_train, y_train = build_tensors(df_train, train_tf)
    X_val, y_val = build_tensors(df_val, val_tf)

    #Print CIL class distribution (add back in in I change seeds)
    
    #vc_tr = pd.Series(y_train.cpu().numpy()).value_counts().sort_index()
    #vc_te = pd.Series(y_val.cpu().numpy()).value_counts().sort_index()
    #print("\n--- CIL LOCAL CLASS DISTRIBUTION ---")
    #for i in range(max(vc_tr.index.max() if len(vc_tr) else -1,
    #vc_te.index.max() if len(vc_te) else -1) + 1):
    #trn = int(vc_tr.get(i, 0))
    #tst = int(vc_te.get(i, 0))
    ## create readable prints
    #gid = int(local_to_gid[i].item())
    #name = gid2breed.get(str(gid), f"gid_{gid}")
    #print(f" local[{i}] ({name}): {trn} train, {tst} test")
       
    return X_train, y_train, X_val, y_val

def seen_and_new(schedule, t):
    seen = []
    for i in range(t+1):
        seen += schedule[i]
    #keep order unique
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
    lam_r = cfg.get("lambda_replay", 1.0)


    #Once per task compute
    #rows in the SAME order as local_to_gid so logits match y_local
    rows_seen = torch.tensor(registry.rows_for_gids(local_to_gid.tolist()), device=device)
    gid_to_local = {int(g): i for i, g in enumerate(local_to_gid.tolist())}
    
    W = model.head.linear.weight
    
    #Add tqdm progress bar for training
    from tqdm import tqdm
    batch_iter = tqdm(train_loader, desc=f"Task {task_idx+1}, Epoch {epoch+1}")
    
    for step, (batch_x, batch_y_local) in enumerate(batch_iter):
        #Debug: Print training data information (only for first batch of first epoch)
        if step == 0 and epoch == 0:
            print(f"DEBUG: Training batch contains {len(batch_y_local)} samples")
            unique_batch_classes = torch.unique(batch_y_local).tolist()
            print(f"DEBUG: Unique classes in this batch: {unique_batch_classes}")
            if gid2breed:
                print(f"DEBUG: Batch classes: {[gid2breed.get(str(local_to_gid[i].item()), f'local_{i}') for i in unique_batch_classes]}")
        batch_x = batch_x.to(device)
        batch_y_local = batch_y_local.to(device)
        
        #---- compute features
        #Ensure backbone and proj are in eval mode to avoid BatchNorm issues
        model.backbone.eval()
        model.proj.eval()
        with torch.no_grad():
            backbone_feats = model._features(batch_x)
            proj_feats = model.proj(backbone_feats)  #Project to correct dimension
        #Set backbone back to eval mode (it should stay frozen anyway)
        model.backbone.eval()
        model.proj.eval()
            
        #---- forward: live
        #Use projected features for head
        logits_live = model.head.forward_rows(proj_feats, rows_seen)
        live_loss = criterion(logits_live, batch_y_local)

        #---- forward: replay
        replay_loss = None
        if replay_buffer_q is not None and len(replay_buffer_q) >  cfg['replay_batch_size']:
            replay_batch_size = cfg['replay_batch_size']
            replay_backbone_feats, replay_targets_global = replay_buffer_q.sample_q(replay_batch_size, seen_global_ids, device=device)
            if replay_backbone_feats is not None:
                #Debug: Print replay information (only for first batch of first epoch)
                if step == 0 and epoch == 0:
                    unique_replay_classes = torch.unique(replay_targets_global).tolist()
                    print(f"DEBUG: Replay batch contains {len(replay_targets_global)} samples")
                    print(f"DEBUG: Unique classes in replay batch: {unique_replay_classes}")
                    if gid2breed:
                        print(f"DEBUG: Replay classes: {[gid2breed.get(str(g), f'gid_{g}') for g in unique_replay_classes]}")
                
                replay_proj_feats = model.proj(replay_backbone_feats)  #Project replay features
                logits_replay = model.head.forward_rows(replay_proj_feats, rows_seen)
                replay_local = torch.tensor(
                    [gid_to_local[int(g.item())] for g in replay_targets_global],
                    dtype=torch.long, device=device
                )
                replay_loss = criterion(logits_replay, replay_local)
        elif step == 0 and epoch == 0:
            print(f"DEBUG: No replay - replay_buffer_q is {'None' if replay_buffer_q is None else f'length {len(replay_buffer_q)}'}")
                
                
        #---- forward: importance metrics
        with torch.no_grad():
            probs = torch.softmax(logits_live, dim=1)
            top2 = torch.topk(probs, k=2, dim=1).values   #[B, 2]
            margin = (top2[:,0] - top2[:,1])              #lower = harder
            entropy = -(probs * (probs.clamp_min(1e-8)).log()).sum(dim=1)
            ce_i = F.cross_entropy(logits_live, batch_y_local, reduction='none')  #[B] # implement CE per-sample
            hardness = ce_i + (1.0 - margin) + 0.5 * entropy        #tune weights
        
        #Combine loss on replay and live
        loss = live_loss + (lam_r * replay_loss if replay_loss is not None else 0.0)
        
        #------ KD / DDR on previous classes
        if teacher is not None and prev_num > 0:
            logits_prev = model.head.forward_rows(proj_feats, list(range(prev_num)))
            with torch.no_grad():
                teacher_logits = teacher(proj_feats)  #Teacher expects projected features
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
        global_targets = local_to_gid.to(device)[batch_y_local]  #vectorized: (B,) global IDs
        if cfg['replay']:
            #Debug: Print what's being added to replay buffer (only for first batch of first epoch)
            if step == 0 and epoch == 0:
                unique_add_classes = torch.unique(global_targets).tolist()
                print(f"DEBUG: Adding {len(global_targets)} samples to replay buffer")
                print(f"DEBUG: Unique classes being added: {unique_add_classes}")
                if gid2breed:
                    print(f"DEBUG: Classes being added: {[gid2breed.get(str(g), f'gid_{g}') for g in unique_add_classes]}")
            
            replay_buffer_q.add_batch(backbone_feats, global_targets, raw_scores=hardness) #Store backbone features for replay, hardness metric

        #stats
        with torch.no_grad():
            pred = logits_live.argmax(dim=1)
            epoch_total += batch_y_local.size(0)
            epoch_correct += (pred == batch_y_local).sum().item()
            epoch_loss += float(loss.item())
            
            #Update progress bar with current accuracy
            current_acc = 100 * epoch_correct / epoch_total
            current_loss = epoch_loss / (step + 1)
            batch_iter.set_postfix({
                'acc': f'{current_acc:.1f}%',
                'loss': f'{current_loss:.3f}'
            })
        if (step + 1) % 200 == 0 and cfg['replay']:
            replay_buffer_q.rebalance_to_cap()
        #if (step + 1) % 1000 == 0:
        #replay_buffer_q.debug_summary(title=f"Task {task_idx+1}, step {step+1}")


    print(f"epoch {epoch:02d} | train_acc={100*epoch_correct/epoch_total:.1f} | loss={epoch_loss/len(train_loader):.3f}")
    if cfg['replay']:
        replay_buffer_q.rebalance_to_cap()
    
    #Return training metrics for plotting
    avg_loss = epoch_loss / len(train_loader)
    train_acc = 100 * epoch_correct / epoch_total
    return avg_loss, train_acc



        
def CIL_post_task_eval(test_loader, device, model, registry, local_to_gid, gid2breed=None):
    
        with torch.no_grad():
            total, correct = 0, 0
            num_local = len(local_to_gid)
            class_correct = {c: 0 for c in range(num_local)}
            class_total   = {c: 0 for c in range(num_local)}
            
            rows_seen = torch.tensor(registry.rows_for_gids(local_to_gid.tolist()), device=device)
            
            for xb, yb_local in test_loader:
                xb, yb_local = xb.to(device), yb_local.to(device)
                #Make a local label vector for seen_classes 0..len(seen)-1
                #Ensure backbone and proj are in eval mode
                model.backbone.eval()
                model.proj.eval()
                backbone_feats = model._features(xb)
                proj_feats = model.proj(backbone_feats)
                logits_seen = model.head.forward_rows(proj_feats, rows_seen)
                pred = logits_seen.argmax(dim=1)

                total += yb_local.size(0)
                correct += (pred == yb_local).sum().item()

                #per-class
                for c in range(num_local):
                    mask = (yb_local == c)
                    class_total[c] += mask.sum().item()
                    class_correct[c] += (pred[mask] == yb_local[mask]).sum().item()

        overall_acc = 100.0 * correct / max(total, 1)
        class_acc = {int(c): (100.0 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0) for c in range(num_local)}
        #Debugging
        #print(f"[Debug] totals per class: {class_total}")
        
        return class_acc, overall_acc
    
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def clean_breed_name(breed_id: str) -> str:
    """
    Convert breed ID like 'n02097209-standard_schnauzer' to 'Standard Schnauzer'
    """
    if '-' in breed_id:
        #Extract the part after the last dash
        breed_part = breed_id.split('-')[-1]
        #Replace underscores with spaces and title case
        return breed_part.replace('_', ' ').title()
    else:
        #If no dash, just clean up underscores and title case
        return breed_id.replace('_', ' ').title()

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def _seen_history_to_df(hist, gid2breed):
    """
    hist: list of {"stage": str, "per_class": {local_idx: acc}, "local_to_gid": LongTensor}
    Returns tidy DF with columns: Stage, gid, ClassName, Accuracy
    (local indices are converted to global gid using local_to_gid for that stage)
    """
    rows = []
    for entry in hist:
        stage = entry["stage"]
        per_class = entry["per_class"]
        l2g = entry["local_to_gid"].tolist()  #local idx -> gid (list)
        for local_idx, acc in per_class.items():
            gid = int(l2g[int(local_idx)])
            cname = gid2breed.get(str(gid), f"gid_{gid}")
            rows.append({"Stage": stage, "gid": gid, "ClassName": cname, "Accuracy": float(acc)})
    if not rows:
        return pd.DataFrame(columns=["Stage","gid","ClassName","Accuracy"])
    #keep stage order numeric
    df = pd.DataFrame(rows)
    df["StageIdx"] = df["Stage"].str.extract(r"(\d+)").fillna(0).astype(int)
    df = df.sort_values(["StageIdx", "ClassName"]).drop(columns=["StageIdx"])
    return df

def _fixed13_to_df(hist, target_13_gids, gid2breed):
    """
    hist: list of {"stage": str, "overall": float, "per_class": {0..12: acc}}
    target_13_gids: list of gids in the same order used to build eval13 tensors
    Returns two DFs:
      - df_overall: Stage, Overall
      - df_pc: Stage, gid, ClassName, Accuracy
    """
    rows_overall, rows_pc = [], []
    for entry in hist:
        stage = entry["stage"]
        rows_overall.append({"Stage": stage, "Overall": float(entry.get("overall", np.nan))})
        pc = entry.get("per_class", {})
        for i, gid in enumerate(target_13_gids):
            #pc is a dict with GID keys, not position keys
            acc = float(pc.get(int(gid), np.nan))
            cname = gid2breed.get(str(int(gid)), f"gid_{int(gid)}")
            #Clean the breed name for better display
            clean_cname = clean_breed_name(cname)
            rows_pc.append({"Stage": stage, "gid": int(gid), "ClassName": clean_cname, "Accuracy": acc})
    df_overall = pd.DataFrame(rows_overall) if rows_overall else pd.DataFrame(columns=["Stage","Overall"])
    df_pc = pd.DataFrame(rows_pc) if rows_pc else pd.DataFrame(columns=["Stage","gid","ClassName","Accuracy"])
    #stage order - handle "Backbone" and "Task X" properly
    if not df_overall.empty:
        def get_stage_order(stage):
            if stage == "Backbone":
                return 0
            elif stage.startswith("Task "):
                return int(stage.split(" ")[1])
            else:
                return 999  #fallback for unknown stages
        
        df_overall["StageIdx"] = df_overall["Stage"].apply(get_stage_order)
        df_overall = df_overall.sort_values("StageIdx").drop(columns=["StageIdx"])
    
    if not df_pc.empty:
        def get_stage_order(stage):
            if stage == "Backbone":
                return 0
            elif stage.startswith("Task "):
                return int(stage.split(" ")[1])
            else:
                return 999  #fallback for unknown stages
        
        df_pc["StageIdx"] = df_pc["Stage"].apply(get_stage_order)
        df_pc = df_pc.sort_values(["StageIdx","ClassName"]).drop(columns=["StageIdx"])
    return df_overall, df_pc

def make_CIL_plots(
    cfg,
    gid2breed,
    seen_pre_hist,          #list of dicts: {"stage","per_class","local_to_gid"}
    seen_post_hist,         #list of dicts: {"stage","per_class","local_to_gid"}
    fixed13_hist,           #list of dicts: {"stage","overall","per_class"}
    target_13_gids,         #list[int] (order used in eval13 tensors)
    highlight_new_class=None #optional: list of gid just added per stage (same length as seen_*_hist)
):
    #Set LaTeX-style fonts to match document
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 12,
        'axes.labelsize': 15,
        'axes.titlesize': 15,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'figure.titlesize': 17,
        'text.usetex': False,  #Set to True if have LaTeX installed
        'mathtext.fontset': 'cm'  #Use Computer Modern for math text
    })
    
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plot_dir = cfg["plot_dir"]
    _ensure_dir(plot_dir)

    #---------- Build tidy frames ----------
    df_seen_pre  = _seen_history_to_df(seen_pre_hist,  gid2breed)
    df_seen_post = _seen_history_to_df(seen_post_hist, gid2breed)
    df13_overall, df13_pc = _fixed13_to_df(fixed13_hist, target_13_gids, gid2breed)

    #---------- (1) Seen-only overall: PRE vs POST ----------
    def _agg_overall(df):
        if df.empty: return pd.DataFrame(columns=["Stage","Overall"])
        tmp = df.groupby("Stage", as_index=False)["Accuracy"].mean().rename(columns={"Accuracy":"Overall"})
        tmp["Kind"] = ""  #will set later
        return tmp

    pre_overall  = _agg_overall(df_seen_pre);  pre_overall["Kind"]  = "Seen PRE"
    post_overall = _agg_overall(df_seen_post); post_overall["Kind"] = "Seen POST"
    df_seen_overall = pd.concat([pre_overall, post_overall], ignore_index=True)

    if not df_seen_overall.empty:
        plt.figure(figsize=(10,6))
        sns.lineplot(data=df_seen_overall, x="Stage", y="Overall", hue="Kind", marker="o")
        plt.title("Seen-only Overall Accuracy (PRE vs POST)")
        plt.ylabel("Accuracy (\%)"); plt.xlabel("Stage"); plt.ylim(0, 100); plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/seen_overall_pre_vs_post.png", dpi=300, bbox_inches="tight")
        plt.close()

    #---------- (2) Seen-only per-class spaghetti (POST by default; overlay PRE faint if want) ----------
    if not df_seen_post.empty:
        plt.figure(figsize=(14,8))
        #POST solid
        ax = sns.lineplot(data=df_seen_post, x="Stage", y="Accuracy", hue="ClassName",
                          marker="o", linewidth=2.2, legend=False)
        #Optionally overlay PRE as faint dashed
        if not df_seen_pre.empty:
            sns.lineplot(data=df_seen_pre, x="Stage", y="Accuracy", hue="ClassName",
                         marker="o", linewidth=1.0, style=True, dashes=True, alpha=0.35, legend=False)
        plt.title("Seen-only Per-Class Accuracy (POST; PRE dashed)")
        plt.ylabel("Accuracy (\%)"); plt.xlabel("Stage"); plt.ylim(0, 100); plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/seen_per_class_spaghetti.png", dpi=300, bbox_inches="tight")
        plt.close()

    #---------- (3) New class PREPOST bars per task ----------
    #If pass highlight_new_class = list of gids (one per task), well draw bars
    if highlight_new_class and len(highlight_new_class) == len(seen_pre_hist) == len(seen_post_hist):
        rows = []
        for i, gid in enumerate(highlight_new_class):
            stage = seen_pre_hist[i]["stage"]
            #find name
            cname = gid2breed.get(str(int(gid)), f"gid_{int(gid)}")
            #map gid to local for that stage
            l2g = seen_pre_hist[i]["local_to_gid"].tolist()
            #inverse map: gid -> local
            inv = {int(g): li for li, g in enumerate(l2g)}
            if int(gid) not in inv:
                continue
            li = inv[int(gid)]
            pre_acc  = float(seen_pre_hist[i]["per_class"].get(li, np.nan))
            post_acc = float(seen_post_hist[i]["per_class"].get(li, np.nan))
            rows.append({"Stage": stage, "ClassName": cname, "Kind": "PRE",  "Accuracy": pre_acc})
            rows.append({"Stage": stage, "ClassName": cname, "Kind": "POST", "Accuracy": post_acc})
        df_new = pd.DataFrame(rows)
        if not df_new.empty:
            plt.figure(figsize=(12,6))
            sns.barplot(data=df_new, x="Stage", y="Accuracy", hue="Kind")
            for i, txt in enumerate(df_new["ClassName"].unique()):
                plt.text(i, 99, txt, ha="center", va="top", fontsize=9, alpha=0.8, rotation=45)
            plt.title("New Class Improvement per Task (PRE  POST)")
            plt.ylim(0, 100); plt.ylabel("Accuracy (\%)"); plt.xlabel("Stage"); plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/new_class_pre_post_bars.png", dpi=300, bbox_inches="tight")
            plt.close()

    #---------- (4) Fixed-13 overall trend ----------
    if not df13_overall.empty:
        plt.figure(figsize=(10,6))
        ax = sns.lineplot(data=df13_overall, x="Stage", y="Overall", marker="o", linewidth=3, markersize=10)
        plt.title('Overall Accuracy Progression: Backbone  CIL Tasks', 
              fontweight='bold', pad=20)
        plt.xlabel('Training Stage', fontweight='bold')
        plt.ylabel('Overall Accuracy (\%)', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        #Add value annotations
        for i, (stage, acc) in enumerate(zip(df13_overall["Stage"], df13_overall["Overall"])):
            plt.annotate(f'{acc:.1f}\%', 
                        xy=(i, acc), 
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/fixed13_overall.png", dpi=300, bbox_inches="tight")
        plt.close()

    #---------- (5) Fixed-13 per-class line plot with legend ----------
    if not df13_pc.empty:
        #Filter out breeds with all NaN or 0 values
        breed_means = df13_pc.groupby('ClassName')['Accuracy'].mean()
        valid_breeds = breed_means[breed_means > 0].index
        df13_pc_filtered = df13_pc[df13_pc['ClassName'].isin(valid_breeds)]
        
        print(f"Fixed-13 per-class: {len(valid_breeds)} breeds with data out of {len(breed_means)} total")
        print(f"Breeds with data: {list(valid_breeds)}")
        
        plt.figure(figsize=(14, 8))
        
        #Plot per-class accuracy progression with legend
        ax = sns.lineplot(data=df13_pc_filtered, x="Stage", y="Accuracy", hue="ClassName", 
                         marker="o", linewidth=2.5, markersize=8)
        
        #Customize the plot
        plt.title('Per-Class Accuracy Progression: Backbone  CIL Tasks', fontweight='bold', pad=20)
        plt.xlabel('Training Stage', fontweight='bold')
        plt.ylabel('Accuracy (\%)', fontweight='bold')
        plt.ylim(0, 100)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        #Add legend with dog breed names
        plt.legend(title='Dog Breeds', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/fixed13_per_class_progression.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("\nSaved plots to:", os.path.abspath(plot_dir))

def save_plotting_data_to_csv(cfg, backbone_class_acc, accuracy_history, backbone_overall_acc=None):
    """
    Save plotting data to CSV files for the two intelligent plots.
    This allows for faster plot iteration without re-running training.
    """
    plot_dir = cfg["plot_dir"]
    _ensure_dir(plot_dir)
    
    #accuracy_history includes backbone + CIL tasks, so we need to subtract 1 for CIL task count
    cil_task_count = len(accuracy_history) - 1  #Subtract backbone
    stages = ['Backbone'] + [f'Task {i+1}' for i in range(cil_task_count)]
    
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
            #For CIL tasks, use accuracy_history[stage_idx] (not stage_idx - 1)
            #because accuracy_history[0] is backbone, accuracy_history[1] is Task 1, etc.
            if isinstance(accuracy_history[stage_idx], dict):
                current_acc_dict = accuracy_history[stage_idx]['class_acc']
            else:
                current_acc_dict = accuracy_history[stage_idx]
        
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
            #For CIL tasks, use accuracy_history[stage_idx] (not stage_idx - 1)
            overall_acc = accuracy_history[stage_idx]['overall_acc']
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
        'xtick.labelsize': 25,
        'ytick.labelsize': 20,
        'legend.fontsize': 17,
        'figure.titlesize': 17,
        'text.usetex': False,  #Set to True if have LaTeX installed
        'mathtext.fontset': 'cm'  #Use Computer Modern for math text
    })
    
    #Set seaborn style to match other plots
    plt.style.use('seaborn-v0_8-deep')
    
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
    plt.legend(title='Dog Breeds', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/dog2_intelligent_tdm_per_class_progression.pdf', bbox_inches='tight')
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
    plt.savefig(f'{plot_dir}/dog2_intelligent_tdm_overall_progression.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Plots generated from CSV data and saved to {plot_dir}")
    
    
    
