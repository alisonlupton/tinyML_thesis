import torch

torch.backends.quantized.engine = "qnnpack"
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import csv
import numpy as np
from collections import deque
from tqdm import tqdm
from models.quicknet import QuickNet
from models.quicknet_BNN import BinarizedQuickNet
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import yaml
import wandb
from utils import load_dataset_custom, make_split_dataset_loaders, update_replay_reservoir, tail_forward, replay_ratio_schedule, sample_replay
from metrics import continual_learning_metrics_extended, tiny_ML_metrics
from collections import Counter
import torch.nn.functional as F
from cwr_head import CWRHead, CosineClassifier, MLPClassifier


#### CITE https://github.com/vlomonaco/ar1-pytorch/blob/master/ar1star_lat_replay.py


###################################################################################################################################################
############################################################### CONTINUAL LEARNING Functions ######################################################
###################################################################################################################################################



def debug_logger(
    task,
    epoch,
    step,
    loss,
    logits,
    live_feats,
    grad,
    save_path="logs/debug_log.csv",
    write_header=False,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    row = {
        "task": task,
        "epoch": epoch,
        "step": step,
        "loss": f"{loss:.6f}",
        "logits_min": f"{logits.min().item():.4f}",
        "logits_max": f"{logits.max().item():.4f}",
        "logits_mean": f"{logits.mean().item():.4f}",
        "live_feats_mean": f"{live_feats.mean().item():.6f}",
        "live_feats_std": f"{live_feats.std().item():.6f}",
        "live_feats_min": f"{live_feats.min().item():.4f}",
        "live_feats_max": f"{live_feats.max().item():.4f}",
        "grad_norm": f"{grad.norm().item():.6f}" if grad is not None else "NaN",
        "grad_mean": f"{grad.mean().item():.6f}" if grad is not None else "NaN",
    }

    write_header = write_header or not os.path.exists(save_path)

    with open(save_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def restrict_to_seen(logits, seen):
    mask = torch.full_like(logits, float('-inf'))
    mask[:, seen] = logits[:, seen]
    return mask

@torch.no_grad()
def get_lr(prefix, x):
    h = prefix(x)
    # map to {-1,+1}
    return (h >= 0).to(h.dtype).mul_(2).sub_(1)

@torch.no_grad()
def dump_cwr_stats(head, k=10, tag=""):
    cw = head.cw        # [C, D]
    cb = head.cb        # [C]
    hc = head.hist_count  # [C]
    norms = cw.norm(dim=1).cpu()
    print(f"[CWR {tag}] hist_count (first {k}):", hc[:k].cpu().tolist())
    # larger norm often means that class has larger-magnitude logits → more likely to be predicted.
    # goal would be that norms for old vs new classes should be in the same ballpark
    print(f"[CWR {tag}] cw norms   (first {k}):", norms[:k].tolist())
    
    # positive bias shifts logits toward that class regardless of features.
    # if cb is not None:
    #     print(f"[CWR {tag}] cb bias   (first {k}):", cb[:k].cpu().tolist())

    # Optional: top/bottom classes by norm
    top = torch.topk(norms, k=min(k, norms.numel())).indices.tolist()
    bot = torch.topk(-norms, k=min(k, norms.numel())).indices.tolist()
    print(f"[CWR {tag}] top-norm classes:", top)
    print(f"[CWR {tag}] low-norm classes:", bot)
    
@torch.no_grad()
def snapshot_hist(head):
    return head.hist_count.clone().cpu()
# -----  Continual learning
def train_with_latent_replay(
    frozen_model,
    prefix,
    tail,
    optimizer,
    loss_fn,
    task_loaders_train,
    task_loaders_test,
    device,
    replay_size,
    live_batch,
    replay_batch,
    epochs,
    full_test_dataset,
    scheduler=None,
):

    replay_size_per_class = replay_size // frozen_model.classifier.num_classes
    train_exp, eval_exp, accs, forgettings, overall, overall_full = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    classwise_feats = defaultdict(
        lambda: deque(maxlen=replay_size_per_class)
    )  # dict of classwise deques
    scale = 30                       # try 20–30 later on


    # Helpers to extract latents and run forward, and replay settup

    # Loop over tasks
    first_acc = {}
    


    if config["wandb_activate"]:
        print("Using WandB!")
        run = wandb.init(
            project=config["project"],
            name=f"{config['backbone']}_{config['dataset']}_{config['live_batch']}_{config['replay_batch']}",
            config=config,
            group=config.get("group", "default"),
        )
    else:
        print("Not using WandB!")
        run = None
        
    if config['CWR']:
        print("Using CWR*")
    else:
        print("CWR* turned off!")
        

    for t, train_loader in enumerate(task_loaders_train):
        print(f"\n=== Training on Task {t} ===")
        present_counter = Counter() # reset each task
        if config['CWR']:
            start_hist = snapshot_hist(frozen_model.classifier)


        # Get task classes
        task_classes = set()
        for _, y in task_loaders_train[t]:
            task_classes.update(y.numpy().tolist())
        task_classes = list(sorted(task_classes))
        
        if t > 0:
            # Get classes from the directly previous task
            prev_task_classes = set()
            for _, y in task_loaders_train[t-1]:
                prev_task_classes.update(y.numpy().tolist())
            prev_task_classes = list(prev_task_classes)
            
            
        # Track seen classes across all tasks
        prev_seen_classes = set()
        for prev_task in range(t):
            for _, y in task_loaders_train[prev_task]:
                prev_seen_classes.update(y.numpy().tolist())
        
        # Initialize classifier weights for new task classes
        for c in task_classes:
            if c not in prev_seen_classes:
                # Initialize new class weights with small random values
                nn.init.normal_(frozen_model.classifier.weight[c], mean=0.0, std=0.01)
                nn.init.zeros_(frozen_model.classifier.bias[c])
        
        print(f"CLASSES FOR TASK {t}: {task_classes}")

        # Training on this task
        frozen_model.train()
        # Non CWR initialisation 
        if not config["CWR"]:
            # Standard initialization for new classes
            with torch.no_grad():
                for c in task_classes:
                    nn.init.normal_(frozen_model.classifier.weight[c], std=0.01)
                    if frozen_model.classifier.bias is not None:
                        frozen_model.classifier.bias[c].zero_()

        # # accumulating weights at the end of each epoch, instead of every minibatch (trying to improve smoothing)
        # if config['CWR']:
        #     cw_epoch_accumulator = torch.zeros_like(frozen_model.classifier.cw)
        #     count_epoch_accumulator = torch.zeros(config['num_classes'], dtype=torch.float, device=device)
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"Task {t}", unit="batch")
            for step, (x, y) in enumerate(pbar):
                x, y = x.to(device), y.to(device)
                
                # Live batch
                x_live, y_live = x[:live_batch], y[:live_batch]
                live_feats = get_lr(prefix, x_live).to(device).float() # B_live × C × H × W

                # Replay batch
                selected_feats = []
                selected_labels = []

                # Sample all seen classes, S samples per class
                available_classes = [c for c in classwise_feats if len(classwise_feats[c]) > 0]

                # Compute replay ratio & effective batch
                replay_ratio = replay_ratio_schedule(t, epoch, step, len(train_loader))      # warm-up by epoch, grow by task
                # if step == 0:
                #     print(f"REPLAY RATIO: {replay_ratio}")
                effective_replay_batch = max(live_batch, int(replay_batch * replay_ratio))


                # Pick replay samples (int8 CPU tensors)
                prev_list = prev_task_classes if t > 0 else None
                selected_feats, selected_labels = sample_replay(
                    classwise_feats=classwise_feats,
                    available_classes=available_classes,
                    total_k=effective_replay_batch,
                    prev_task_classes=prev_list,
                    prev_frac=0.4,
                    rng=random
                )

                
                # if step == 0:
                #     print(f"[replay] wanted {effective_replay_batch}, got {len(selected_feats)} "
                #         f"from {len(available_classes)} classes")
                # ---- Get replay batch
                if selected_feats:
                    # Stack keeps int8; cast to float on device right before forward
                    replay_feats_int8 = torch.stack(selected_feats)            # (R, C, H, W) int8, CPU
                    replay_lbls = torch.tensor(selected_labels, device=device) # (R,)
                    replay_feats = replay_feats_int8.to(device=device, dtype=torch.float32) 
                else:
                    replay_feats = None
                
                
                #------ Combine live and replay
                optimizer.zero_grad()
                # Replay Buffer has elements
                if replay_feats is not None:
                    x_feats = torch.cat([live_feats, replay_feats], dim=0)              # (B_live + B_rep, C,H,W) at LR level
                    y_all   = torch.cat([y_live, replay_lbls],  dim=0)                  # (B_live + B_rep,)
                else:
                    x_feats = live_feats
                    y_all   = y_live
                
                
                present_idx = torch.unique(y_all)
                present_classes = present_idx.tolist()
                present = present_idx.tolist()
                
                # Map original labels -> [0..len(present)-1]
                map_vec = torch.full((config["num_classes"],), -1, device=device, dtype=torch.long)
                map_vec[present_idx] = torch.arange(len(present_idx), device=device)
                y_all_remap = map_vec[y_all]  
                
                # ---- CWR*: preload consolidated rows for classes in this minibatch
                if config['CWR']:
                    # assert len(present_classes) > 0
                    # assert all(0 <= c < config["num_classes"] for c in present_classes)
                    
                    present_counter.update(present_classes)
                    frozen_model.classifier.preload_cw(present_classes)  # load cw rows -> tw; zero tw for others
                    
                # with torch.no_grad():
                #     if step % 100 == 0:
                #         tb_pre = frozen_model.classifier.bias.index_select(0, present_idx).clone()
                #         cb_pre = frozen_model.classifier.cb.index_select(0, present_idx).clone()
                #         print(f"[mini pre] tb (pre-step): {tb_pre.tolist()}  cb (pre): {cb_pre.tolist()}")

                # Forward on the tail + head (combine live and replay)                
                # trying cosine reweighting 
                # Cosine head forward (use logits_all, then slice to present)
                z = tail(x_feats).flatten(1)                       # (B, D)
                z = F.normalize(z, dim=1)
                W = frozen_model.classifier.weight                 # (C, D) — tw during training
                Wn = F.normalize(W, dim=1)
                
                cos_all = z @ Wn.t()                               # unscaled cosine similarities, (B, C)
                cos_present = cos_all[:, present_idx]              # (B, |present|)

                # logits_all = scale * (z @ Wn.t())    # no bias
                
                # CosFace margin on the *target* logit only
                # inside the loop
                target_m = 0.2 # maybe try pushing to 0.25 if necessary 
                warm_frac = 0.4
                if epoch == 0:
                    progress = min(1.0, step / max(1, int(warm_frac * len(train_loader))))
                else:
                    progress = 1.0  # full margin after epoch 0
                m = target_m * progress
                
                # subtract m from the target cosine *before* scaling
                cos_margin = cos_present.clone()
                cos_margin[torch.arange(cos_margin.size(0)), y_all_remap] -= m
                logits_present = scale * cos_margin                # (B, |present|)

                
                # Debugging step (want these to remain small)
                with torch.no_grad():
                    pos = logits_present.gather(1, y_all_remap.unsqueeze(1)).abs().mean().item()
                    # average magnitude of non-target logits
                    onehot = torch.zeros_like(logits_present).scatter_(1, y_all_remap.unsqueeze(1), 1.0)
                    neg = (logits_present * (1 - onehot)).abs().sum(dim=1) / (logits_present.size(1) - 1 + 1e-9)
                    #print(f"[logit mag] pos|neg ~ {pos:.1f} | {neg.mean().item():.1f}")

                # Per-class mean (strict balancing)
                binc = torch.bincount(y_all_remap, minlength=len(present)).float()
                w = 1.0 / (binc[y_all_remap] + 1e-6)               # inverse frequency
                w = w / w.mean()                                   # keep scale stable

                loss_vec = loss_fn(logits_present, y_all_remap)    # reduction='none'
                loss_all = (loss_vec * w).mean()
                loss_all.backward()
                
                optimizer.step()
                with torch.no_grad():
                    rows = present_idx  # tensor of class ids in this minibatch
                    w = frozen_model.classifier.weight.index_select(0, rows)
                    w = F.normalize(w, dim=1)
                    frozen_model.classifier.weight.index_copy_(0, rows, w)
                
                # if step % 100 == 0:
                #     with torch.no_grad():
                #         idx = present_idx  # tensor of class ids in this minibatch
                #         tw_norms = frozen_model.classifier.weight.index_select(0, idx).norm(dim=1).cpu() # ephemeral norms
                #         cw_norms = frozen_model.classifier.cw.index_select(0, idx).norm(dim=1).cpu()
                        
                        
                #         eps = 1e-12
                #         tw_proportional_norms = (tw_norms / tw_norms.sum().clamp_min(eps)).cpu().tolist()
                #         cw_proportional_norms = (cw_norms / cw_norms.sum().clamp_min(eps)).cpu().tolist()
                        
                #         print(f"[mini] tw norms %  : {[round(x*100,2) for x in tw_proportional_norms]}")
                #         print(f"[mini] cw norms %  : {[round(x*100,2) for x in cw_proportional_norms]}")
                        
                #         # tb_post = frozen_model.classifier.bias.index_select(0, present_idx).clone()
                #         # print(f"[mini post-step] tb (post-step): {tb_post.tolist()}") # ephemeral bias
                
                # ---- CWR*: consolidate for *this minibatch*
                if config['CWR']:
                    num_classes = config["num_classes"]
                    binc = torch.bincount(y_all, minlength=config["num_classes"])
                    
                    # CWR* uses these numbers for weight consolidation
                    if step % 100 == 0:
                        print(f"[count/batch t={t} e={epoch} s={step}]",
                            {c:int(binc[c]) for c in range(num_classes) if binc[c] > 0})
                        
                    counts_present = binc[present_idx].float()
                    counts_present.fill_(1.0) # equal contribution this batch
                    # assert len(present) == counts_present.numel()
                    
                    
                    frozen_model.classifier.consolidate(present_classes, counts_present)
                    # per epoch consolidation     
                    # with torch.no_grad():
                    #     tw_present = frozen_model.classifier.weight.index_select(0, present_idx)
                    #     cw_epoch_accumulator.index_add_(0, present_idx, tw_present)
                    #     count_epoch_accumulator.index_add_(0, present_idx, counts_present)
                    # if step % 100 == 0:
                    #     with torch.no_grad():
                    #         cb_post = frozen_model.classifier.cb.index_select(0, present_idx).clone()
                    #         print(f"[mini post-cons] cb (post-cons): {cb_post.tolist()}")
 
                pbar.set_postfix(pos=f"{pos:.2f}", neg=f"{neg.mean().item():.2f}")


            # per epoch consolidation
            # if config['CWR']:
            #     frozen_model.classifier.cw += cw_epoch_accumulator
            #     frozen_model.classifier.hist_count += count_epoch_accumulator.to(torch.long)
                
                # Norm alignment of CWR head
                # with torch.no_grad():
                #     cw_norms = frozen_model.classifier.cw.norm(dim=1)
                #     tw_norms = frozen_model.classifier.weight.norm(dim=1)

                #     ratio = (tw_norms + 1e-8) / (cw_norms + 1e-8)
                #     ratio = ratio.clamp(min=0.1, max=10.0)  # prevent extreme scaling

                #     frozen_model.classifier.cw.mul_(ratio.unsqueeze(1))  # scale row-wise

            # Step the scheduler after each epoch
            if scheduler:
                scheduler.step()

        
        
        # Add current task classes to replay buffer AFTER full training
        with torch.no_grad():
            pairs = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                lr = get_lr(prefix, xb.to(device)).cpu()     # {-1,+1} float
                # assert torch.all((lr == 1) | (lr == -1)), "LR not binary!"
                lr_q = lr.to(torch.int8)                     # store as int8 for now
                # assert torch.all((lr_q == 1) | (lr_q == -1)), "Quantized LR not binary!"
                pairs.extend([(feat, int(lbl.item())) for feat, lbl in zip(lr_q, yb.cpu())])
            update_replay_reservoir(classwise_feats, pairs, replay_size_per_class)


            

        # TODO: set seed!
        if config['CWR']:
            frozen_model.classifier.use_consolidated()
            


        frozen_model.eval(); tail.eval()
        
        # all seen classes, including curr task
        task_classes = set(task_classes)  # turn back to a set
        seen_so_far = sorted(prev_seen_classes | task_classes) # union, then sort for stable indexing
        # print(f"[t={t}] prev_seen={sorted(prev_seen_classes)} "
        # f"task={sorted(task_classes)} seen_so_far={seen_so_far[:10]}... (len={len(seen_so_far)})")


        
        print(f"\n=== Evaluating on Task {t} ===")
        
        #should be balanced for each class
        print(f"[Task {t}] present class hits:", dict(present_counter)) # how many minibatches contained each class in the current task’s training loop (live + replay combined).
        
        if config['CWR']:
            end_hist = snapshot_hist(frozen_model.classifier)
            # How much the CWR* bank’s running sample counters increased during Task t for each class.
            # should be somewhat balanced, although new classes can be larger
            print(f"[Task {t}] hist_count delta:", (end_hist - start_hist).tolist()[:10])

        for u in range(t + 1):
            loader = task_loaders_test[u]
            corr = 0
            tot  = 0
            pred_hist = []
            gt_hist   = []
            # optional: quick confusion matrix; works for <= num_classes you use
            num_classes = config["num_classes"]
            conf = torch.zeros(num_classes, num_classes, dtype=torch.int64)

            with torch.no_grad():
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    
                    lr = get_lr(prefix, xb).to(device).float()
                    # new trying cosine 
                    z = tail(lr).flatten(1)
                    z = F.normalize(z, dim=1)
                    if config["CWR"]:
                        Wc = F.normalize(frozen_model.classifier.cw, dim=1)
                        logits = scale * (z @ Wc.t())        # no bias
                    else:
                        W = F.normalize(frozen_model.classifier.weight, dim=1)
                        logits = scale * (z @ W.t())  # no bias

                    logits = restrict_to_seen(logits, seen_so_far)
                    
                    preds = logits.argmax(1)

                    # accumulate stats
                    corr += (preds == yb).sum().item()
                    tot  += yb.size(0)

                    p = preds.cpu() # p is a tensor
                    g = yb.cpu()
                    pred_hist.extend(p.tolist()) # convert tensor to integer list, then use extend to add to list
                    gt_hist.extend(g.tolist())
                    conf.index_put_((g, p), torch.ones_like(g, dtype=torch.int64), accumulate=True)

            acc = corr / max(1, tot)
            pred_counts = Counter(pred_hist)
            gt_counts   = Counter(gt_hist) # ground truth

            print(f"[Task {u}] acc: {acc:.3f}")
            print(f"[Task {u}] Prediction Labels: {dict(sorted(pred_counts.items()))}")
            print(f"[Task {u}] True Labels: {dict(sorted(gt_counts.items()))}")

            # Optional: per-class accuracy over seen labels in this split
            seen_labels = sorted(gt_counts.keys())
            per_class_acc = {}
            for c in seen_labels:
                denom = conf[c].sum().item()
                per_class_acc[c] = (conf[c, c].item() / denom) if denom > 0 else float('nan')
            print(f"[Task {u}] per-class acc: {per_class_acc}")

            # keep your metrics bookkeeping
            train_exp.append(t); eval_exp.append(u); accs.append(acc)
            if u not in first_acc: first_acc[u] = acc; forgettings.append(0.0)
            else: forgettings.append(first_acc[u] - acc)
            
        if config['CWR']:
            dump_cwr_stats(frozen_model.classifier, tag=f"after task {t}")



    if config['CWR']:
        norms = torch.norm(frozen_model.classifier.cw, dim=1).cpu().numpy()
        plt.plot(norms)
        plt.title("L2 Norm of CWR Bank per Class")
    if run:
        run.summary["final_accuracy"] = overall_full[-1][1]
        run.summary["avg_accuracy"] = np.mean([a for _, a in overall])
        run.finish()
    return train_exp, eval_exp, accs, forgettings, overall, overall_full


########################################################################################################
############# MAIN FUNCTION LOOP ##########
########################################################################################################


def main():

    retrain = (
        False  # won't retrain if there is existing model, unless this is set to True
    )

    # Define Model paths
    os.makedirs(config["model_folder"], exist_ok=True)
    BACKBONE_PATH = config["backbone_path"]
    model_paths = {"Backbone": BACKBONE_PATH}

    # Begin Pipeline

    # ----- 0. Data loading
    device = torch.device(config["device"])
    total_classes = config["num_classes"]
    batch_size_train = config["batch_size"]["train"]
    batch_size_test = config["batch_size"]["test"]

    # Transforms for CIFAR both datasets
    train_transform = transforms.Compose(
        [
            transforms.Pad(4),  # zero-pad 4 pixels each side
            transforms.RandomCrop(32),  # random 32×32 crop
            transforms.RandomHorizontalFlip(),  # random flip
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),  # scale to [–1, +1]
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )

    if config["dataset"] == "CIFAR10":
        full_train_dataset = load_dataset_custom(
            datasets.CIFAR10, True, train_transform
        )
        full_test_dataset = load_dataset_custom(datasets.CIFAR10, False, test_transform)
        full_train_loader = DataLoader(
            full_train_dataset, batch_size=batch_size_train, shuffle=True
        )
        full_test_loader = DataLoader(
            full_test_dataset, batch_size=batch_size_test, shuffle=False
        )

    elif config["dataset"] == "CIFAR100":
        full_train_dataset = load_dataset_custom(
            datasets.CIFAR100, True, train_transform
        )
        full_test_dataset = load_dataset_custom(
            datasets.CIFAR100, False, test_transform
        )
        full_train_loader = DataLoader(
            full_train_dataset, batch_size=batch_size_train, shuffle=True
        )
        full_test_loader = DataLoader(
            full_test_dataset, batch_size=batch_size_test, shuffle=False
        )
    else:
        raise ValueError("No valid dataset provided")

    # ----- 1. Load base model in FP32, trained on tinyimagenet
    if os.path.exists(BACKBONE_PATH):
        FP_backbone = QuickNet(num_classes=200).to(device)
        FP_backbone.load_state_dict(torch.load(BACKBONE_PATH, map_location=device))
        print("FP32 model loaded")
        
        # FP_backbone.eval()
        # with torch.no_grad():
        #     x = torch.randn(1, 3, 32, 32).to(device)
        #     x = FP_backbone.stem(x)
        #     for i, m in enumerate(FP_backbone.features):
        #         x = m(x)
        #         if isinstance(m, nn.BatchNorm2d):
        #             print(f"[FP32 QuickNet] After block {i}: {tuple(x.shape)}")
    else:
        raise FileNotFoundError("Couldn't find trained model. Please check.")

    # Quick check of backbone
    # eval_quicknet_backbone(FP_backbone, BACKBONE_PATH, device)

    # ----- 2. Load fp32 backbone into BNN and add classifier
    backbone = BinarizedQuickNet(num_classes=200, first_layer_fp32=True).to(device) # adding another modifiable layer
    
    # change to match CIFAR10 classes
    backbone.classifier = CWRHead(backbone.classifier.in_features, total_classes)
    
    if config['CWR']:
        with torch.no_grad():
            nn.init.normal_(backbone.classifier.weight, mean=0.0, std=0.01)
            if backbone.classifier.bias is not None:
                backbone.classifier.bias.zero_()
            backbone.classifier.cw.zero_()
            backbone.classifier.cb.zero_()
            backbone.classifier.hist_count.zero_()
 
    print(f"Added classifier head with a {config['dataset']} head")


    # ----- 3. Freeze backbone
    
    # [Feature map] After block 1: (1, 64, 16, 16)
    # [Feature map] After block 4: (1, 64, 16, 16)
    # [Feature map] After block 7: (1, 128, 8, 8)
    # [Feature map] After block 10: (1, 128, 8, 8)
    # [Feature map] After block 13: (1, 256, 4, 4)
    # [Feature map] After block 16: (1, 256, 4, 4)


    # with torch.no_grad():
    #     x = torch.randn(1, 3, 32, 32).to(device)  # or 64x64 for TinyImageNet
    #     x = backbone.stem(x)
    #     for i, m in enumerate(backbone.features):
    #         x = m(x)
    #         if isinstance(m, nn.BatchNorm2d):  # BN comes at end of each block
    #             print(f"[Feature map] After block {i}: {tuple(x.shape)}")
                
    # choose how many binary blocks to keep trainable
    
    k_tail = config["k_layers"]  # trainable conv layers
    n = len(backbone.features) # should be 18 
    lr_idx =n - 3 * k_tail   # each block ~ (conv, bn, act), 3 layers per block

    prefix = nn.Sequential(backbone.stem, backbone.features[:lr_idx]).to(device)
    tail   = nn.Sequential(backbone.features[lr_idx:], backbone.global_pool).to(device)

    prefix.requires_grad_(False)
    prefix.eval()            # BN/Dropout in inference mode

    tail.requires_grad_(True)  # usually already True; keeps grads flowing
    tail.train()               # BN/Dropout in training mode

    backbone.classifier.requires_grad_(True)  # just to be safe
    params = [
    {'params': tail.parameters(), 'lr': config['learning_rate'], 'weight_decay': 1e-4},
    {'params': backbone.classifier.parameters(), 'lr': 5*config['learning_rate'], 'weight_decay': 0.0},
]
    optimizer = torch.optim.SGD(params, momentum=0.9)
 
    def load_fp_weights_into_bnn(fp_model, bnn_model):
        fp_dict = fp_model.state_dict()
        bnn_dict = bnn_model.state_dict()
        for name in bnn_dict:
            if name in fp_dict and bnn_dict[name].shape == fp_dict[name].shape:
                bnn_dict[name] = fp_dict[name]
        bnn_model.load_state_dict(bnn_dict, strict=False) 

    load_fp_weights_into_bnn(FP_backbone, backbone)
    
    print("Loaded weights into BNN")
    # TODO: aim would be to have classifier in 16 or 8 quant
    
    backbone = backbone.to(device)
    print("Backbone configured")

    # TODO: fix quantization of classifier! (based on BNN paper)

    # ----- 4. Prepare continual learning
    print("Preparing CL experiences!")
    # TODO: for later on , batch size should be 1-2 !
    # TODO: make replay buffer size a func of epochs 
    replay_buffer_size = config[
        "replay_buffer_size"
    ]  # overall size of my replay buffer
    live_batch = config[
        "live_batch"
    ]  # How many new (or “live”) samples to take from each minibatch to train on right now (before mixing in replay)
    replay_batch = config[
        "replay_batch"
    ]  # How many old samples to pull from the LR buffer of stored activations.
    epochs = config["epochs"]  # Number of training epochs for the CL

    num_tasks = config["num_tasks"]  # number of experiences
    task_loaders_train = make_split_dataset_loaders(
        full_train_dataset,
        n_splits=num_tasks,
        train=True,
        batch_size=live_batch,  # don't load more samples than we need
    )
    task_loaders_test = make_split_dataset_loaders(
        full_test_dataset,
        n_splits=num_tasks,
        train=False,
        batch_size=live_batch + replay_batch,  # TODO: check this
    )

    
    # Use label smoothing for better training stability
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='none')

    
    # Simple learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

    # ----- 5. Perform CL

    train_exp, eval_exp, accs, forgettings, overall, overall_full = (
        train_with_latent_replay(
            frozen_model=backbone,
            prefix = prefix,
            tail = tail,
            optimizer=optimizer,
            loss_fn=loss_fn,
            task_loaders_train=task_loaders_train,
            task_loaders_test=task_loaders_test,
            device=device,
            replay_size=replay_buffer_size,
            live_batch=live_batch,
            replay_batch=replay_batch,
            epochs=epochs,
            full_test_dataset=full_test_dataset,
            scheduler=scheduler,
        )
    )
    breakpoint()

    tasks, accuracy = zip(*overall_full)
    plt.plot(tasks, accuracy, marker="o")
    plt.xlabel("Task")
    plt.ylabel(f'Overall {config["dataset"]} accuracy')
    plt.grid(True)
    plt.show()


    # train_exp, eval_exp, accs, forgettings, overall_accs = training_results['outs']
    print("finished CL experiences")

    # First we compute standard accuracy for each class in a df and print out results
    # 1) Extended CL table:
    with open("metrics/cl_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eval_exp", "training_exp", "eval_accuracy", "forgetting"])
        for te, ee, ac, fg in zip(train_exp, eval_exp, accs, forgettings):
            writer.writerow([ee, te, f"{ac:.4f}", f"{fg:.4f}"])

    # 2) Overall curve:
    with open("metrics/overall_acc.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "overall_acc"])
        for t, acc in overall:
            writer.writerow([t, f"{acc:.4f}"])

    # Now we compute extended metrics and save
    continual_learning_metrics_extended(
        eval_exp, train_exp, accs, forgettings, model_name="Hybrid"
    )
    print("CL extended metrics saved to CL_metrics_extended.csv")

    # ----- 7. Standard Metrics
    models = {"Backbone": backbone}
    tiny_ML_metrics(models, model_paths, full_test_loader, device)
    print("Standard tinyML metrics saved to tinyml_metrics_summary.csv")


if __name__ == "__main__":
    with open("CL_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Loaded config file")
    assert config is not None, "Config file failed to load!"
    main()
    print("Completed successfully!")

# STEPS:
# debugging thing to wandb --> 
# plot / print Conv layers (output)
# tsne 
# try adding a strong head to see how it works 