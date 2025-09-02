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
    
    # Add tqdm progress bar for training
    from tqdm import tqdm
    batch_iter = tqdm(train_loader, desc=f"Task {task_idx+1}, Epoch {epoch+1}")
    
    for step, (batch_x, batch_y_local) in enumerate(batch_iter):
        batch_x = batch_x.to(device)
        batch_y_local = batch_y_local.to(device)
        
        # ---- compute features 
        # Ensure backbone and proj are in eval mode to avoid BatchNorm issues
        model.backbone.eval()
        model.proj.eval()
        with torch.no_grad():
            backbone_feats = model._features(batch_x)
            proj_feats = model.proj(backbone_feats)  # Project to correct dimension
        # Set backbone back to eval mode (it should stay frozen anyway)
        model.backbone.eval()
        model.proj.eval()

        # ---- forward: live
        # Use projected features for head
        logits_live = model.head.forward_rows(proj_feats, rows_seen)
        live_loss = criterion(logits_live, batch_y_local)

        # ---- forward: replay
        replay_loss = None
        if len(replay_buffer_q) >  cfg['replay_batch_size']:
            replay_batch_size = cfg['replay_batch_size']
            replay_backbone_feats, replay_targets_global = replay_buffer_q.sample_q(replay_batch_size, seen_global_ids, device=device)
            if replay_backbone_feats is not None:
                replay_proj_feats = model.proj(replay_backbone_feats)  # Project replay features
                logits_replay = model.head.forward_rows(replay_proj_feats, rows_seen)
                replay_local = torch.tensor(
                    [gid_to_local[int(g.item())] for g in replay_targets_global],
                    dtype=torch.long, device=device
                )
                replay_loss = criterion(logits_replay, replay_local)
        
        # Combine loss on replay and live        
        loss = live_loss + replay_loss if replay_loss is not None else live_loss
        
        #------ KD / DDR on previous classes
        if teacher is not None and prev_num > 0:
            logits_prev = model.head.forward_rows(proj_feats, list(range(prev_num)))
            with torch.no_grad():
                teacher_logits = teacher(proj_feats)  # Teacher expects projected features
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
        replay_buffer_q.add_batch(backbone_feats, global_targets) # Store backbone features for replay    

        # stats
        with torch.no_grad():
            pred = logits_live.argmax(dim=1)
            epoch_total += batch_y_local.size(0)
            epoch_correct += (pred == batch_y_local).sum().item()
            epoch_loss += float(loss.item())
            
            # Update progress bar with current accuracy
            current_acc = 100 * epoch_correct / epoch_total
            current_loss = epoch_loss / (step + 1)
            batch_iter.set_postfix({
                'acc': f'{current_acc:.1f}%',
                'loss': f'{current_loss:.3f}'
            })

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
                # Ensure backbone and proj are in eval mode
                model.backbone.eval()
                model.proj.eval()
                backbone_feats = model._features(xb)
                proj_feats = model.proj(backbone_feats)
                logits_seen = model.head.forward_rows(proj_feats, rows_seen)
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
        # Debugging 
        print(f"[Debug] totals per class: {class_total}")
        
        return class_acc, overall_acc
    
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
        l2g = entry["local_to_gid"].tolist()  # local idx -> gid (list)
        for local_idx, acc in per_class.items():
            gid = int(l2g[int(local_idx)])
            cname = gid2breed.get(str(gid), f"gid_{gid}")
            rows.append({"Stage": stage, "gid": gid, "ClassName": cname, "Accuracy": float(acc)})
    if not rows:
        return pd.DataFrame(columns=["Stage","gid","ClassName","Accuracy"])
    # keep stage order numeric
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
            acc = float(pc.get(i, np.nan))
            cname = gid2breed.get(str(int(gid)), f"gid_{int(gid)}")
            rows_pc.append({"Stage": stage, "gid": int(gid), "ClassName": cname, "Accuracy": acc})
    df_overall = pd.DataFrame(rows_overall) if rows_overall else pd.DataFrame(columns=["Stage","Overall"])
    df_pc = pd.DataFrame(rows_pc) if rows_pc else pd.DataFrame(columns=["Stage","gid","ClassName","Accuracy"])
    # stage order
    if not df_overall.empty:
        df_overall["StageIdx"] = df_overall["Stage"].str.extract(r"(\d+)").fillna(0).astype(int)
        df_overall = df_overall.sort_values("StageIdx").drop(columns=["StageIdx"])
    if not df_pc.empty:
        df_pc["StageIdx"] = df_pc["Stage"].str.extract(r"(\d+)").fillna(0).astype(int)
        df_pc = df_pc.sort_values(["StageIdx","ClassName"]).drop(columns=["StageIdx"])
    return df_overall, df_pc

def make_CIL_plots(
    cfg,
    gid2breed,
    seen_pre_hist,          # list of dicts: {"stage","per_class","local_to_gid"}
    seen_post_hist,         # list of dicts: {"stage","per_class","local_to_gid"}
    fixed13_hist,           # list of dicts: {"stage","overall","per_class"}
    target_13_gids,         # list[int] (order used in eval13 tensors)
    highlight_new_class=None # optional: list of gid just added per stage (same length as seen_*_hist)
):
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plot_dir = cfg["plot_dir"]
    _ensure_dir(plot_dir)

    # ---------- Build tidy frames ----------
    df_seen_pre  = _seen_history_to_df(seen_pre_hist,  gid2breed)
    df_seen_post = _seen_history_to_df(seen_post_hist, gid2breed)
    df13_overall, df13_pc = _fixed13_to_df(fixed13_hist, target_13_gids, gid2breed)

    # ---------- (1) Seen-only overall: PRE vs POST ----------
    def _agg_overall(df):
        if df.empty: return pd.DataFrame(columns=["Stage","Overall"])
        tmp = df.groupby("Stage", as_index=False)["Accuracy"].mean().rename(columns={"Accuracy":"Overall"})
        tmp["Kind"] = ""  # will set later
        return tmp

    pre_overall  = _agg_overall(df_seen_pre);  pre_overall["Kind"]  = "Seen PRE"
    post_overall = _agg_overall(df_seen_post); post_overall["Kind"] = "Seen POST"
    df_seen_overall = pd.concat([pre_overall, post_overall], ignore_index=True)

    if not df_seen_overall.empty:
        plt.figure(figsize=(10,6))
        sns.lineplot(data=df_seen_overall, x="Stage", y="Overall", hue="Kind", marker="o")
        plt.title("Seen-only Overall Accuracy (PRE vs POST)", fontsize=15)
        plt.ylabel("Accuracy (%)"); plt.xlabel("Stage"); plt.ylim(0, 100); plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/seen_overall_pre_vs_post.png", dpi=300, bbox_inches="tight")
        plt.close()

    # ---------- (2) Seen-only per-class spaghetti (POST by default; overlay PRE faint if you want) ----------
    if not df_seen_post.empty:
        plt.figure(figsize=(14,8))
        # POST solid
        ax = sns.lineplot(data=df_seen_post, x="Stage", y="Accuracy", hue="ClassName",
                          marker="o", linewidth=2.2, legend=False)
        # Optionally overlay PRE as faint dashed
        if not df_seen_pre.empty:
            sns.lineplot(data=df_seen_pre, x="Stage", y="Accuracy", hue="ClassName",
                         marker="o", linewidth=1.0, style=True, dashes=True, alpha=0.35, legend=False)
        plt.title("Seen-only Per-Class Accuracy (POST; PRE dashed)", fontsize=15)
        plt.ylabel("Accuracy (%)"); plt.xlabel("Stage"); plt.ylim(0, 100); plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/seen_per_class_spaghetti.png", dpi=300, bbox_inches="tight")
        plt.close()

    # ---------- (3) New class PRE→POST bars per task ----------
    # If you pass highlight_new_class = list of gids (one per task), we’ll draw bars
    if highlight_new_class and len(highlight_new_class) == len(seen_pre_hist) == len(seen_post_hist):
        rows = []
        for i, gid in enumerate(highlight_new_class):
            stage = seen_pre_hist[i]["stage"]
            # find name
            cname = gid2breed.get(str(int(gid)), f"gid_{int(gid)}")
            # map gid to local for that stage
            l2g = seen_pre_hist[i]["local_to_gid"].tolist()
            # inverse map: gid -> local
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
            plt.title("New Class Improvement per Task (PRE → POST)", fontsize=15)
            plt.ylim(0, 100); plt.ylabel("Accuracy (%)"); plt.xlabel("Stage"); plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/new_class_pre_post_bars.png", dpi=300, bbox_inches="tight")
            plt.close()

    # ---------- (4) Fixed-13 overall trend ----------
    if not df13_overall.empty:
        plt.figure(figsize=(10,6))
        sns.lineplot(data=df13_overall, x="Stage", y="Overall", marker="o")
        plt.title("Fixed-13 Overall Accuracy", fontsize=15)
        plt.ylabel("Accuracy (%)"); plt.xlabel("Stage"); plt.ylim(0, 100); plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/fixed13_overall.png", dpi=300, bbox_inches="tight")
        plt.close()

    # ---------- (5) Fixed-13 per-class heatmap ----------
    if not df13_pc.empty:
        # pivot Stage x ClassName
        pivot = df13_pc.pivot_table(index="ClassName", columns="Stage", values="Accuracy", aggfunc="mean")
        plt.figure(figsize=(1.2*len(pivot.columns)+4, 0.38*len(pivot.index)+3))
        sns.heatmap(pivot, annot=False, vmin=0, vmax=100, cmap="viridis", cbar_kws={"label": "Acc (%)"})
        plt.title("Fixed-13 Per-Class Accuracy (Heatmap)", fontsize=15)
        plt.xlabel("Stage"); plt.ylabel("Class")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/fixed13_per_class_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("\nSaved plots to:", os.path.abspath(plot_dir))