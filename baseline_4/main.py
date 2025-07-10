import torch
torch.backends.quantized.engine = "qnnpack"
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as tq
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import os
from thop import profile as thop_profile
import time
import torch.nn.utils.prune as prune
import csv
import numpy as np 
import psutil
import threading
import math
from collections import deque
from torch import Tensor
from typing import Optional
from tqdm import tqdm
import gc
import pandas as pd 

#### CITE https://github.com/vlomonaco/ar1-pytorch/blob/master/ar1star_lat_replay.py


###################################################################################################################################################
############################################################### Define Models ##################################################################
###################################################################################################################################################

# -----  Backbone CNN Model 
class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1) # kernel size 3, input channels 1, output channels 16
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # kernel size 3, input channels 16, output channels 32
        self.bn2   = nn.BatchNorm2d(32)
        self.fc1   = nn.Linear(32*7*7, 128)
        self.fc2   = nn.Linear(128, num_classes) # will output 10 for MNIST
        # quantization stubs
        self.quant   = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def fuse_model(self):
        # fuse only conv+bn
        # fuse conv1→bn1 into a single op and fuse conv2→bn2
        tq.fuse_modules(self, [["conv1","bn1"], ["conv2","bn2"]], inplace=True)

    def forward(self, x):
        x = self.quant(x)
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2)) # size 28 --> 14 (from the max pooling)
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), 2)) # size 14 --> 7 (from the max pooling)
        x = x.flatten(start_dim=1) # flattens to (batch, 32*7*7) so we can input to fc layers
        x = F.relu(self.fc1(x)) # --> (batch, 128)
        x = self.fc2(x) # --> (batch, num_classes)
        return self.dequant(x)
    
# -----  Base CL CNN Model 
class TinyMLContinualModel(nn.Module):
    def __init__(self, backbone_int8: nn.Module, adapter: nn.Module):
        super().__init__()
        # frozen, quantized backbone up to fc1
        self.backbone = backbone_int8.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # small float adapter head
        self.adapter = adapter

    def extract(self, x: Tensor) -> Tensor:
        """Run backbone up to just before adapter, return D-dim feature."""
        # you can factor out your quant→conv→flatten→fc1→dequant→relu
        with torch.no_grad():
            z = self.backbone.quant(x)
            z = F.relu(F.max_pool2d(self.backbone.bn1(self.backbone.conv1(z)), 2))
            z = F.relu(F.max_pool2d(self.backbone.bn2(self.backbone.conv2(z)), 2))
            z = z.flatten(1)
            z = self.backbone.fc1(z)
            z = self.backbone.dequant(z)
            return F.relu(z)

    def forward(self, x: Tensor, latent_input: Optional[Tensor]=None) -> Tensor:
        """
        x:       your new-task batch of size B×…
        latent_input:  R×D tensor of replayed features (or None)
        returns logits for a combined batch
        """
        live_feats = self.extract(x)
        if latent_input is not None:
            # cat along batch-dim
            feats = torch.cat([live_feats, latent_input], dim=0)
        else:
            feats = live_feats
        return self.adapter(feats)

###################################################################################################################################################
############################################################### Define Functions ##################################################################
###################################################################################################################################################

def make_split_dataset_loaders(
    dataset_cls,
    root,
    n_splits=5,
    train: bool = False,
    batch_size: int = 1000,
    normalize=True,
):
    """
    Partition any torchvision dataset (e.g. KMNIST) into `n_splits` tasks,
    each containing consecutive labels.  E.g. with n_splits=5:
      split 0 -> labels {0,1}, split 1 -> {2,3}, …, split 4 -> {8,9}.
    Returns a list of DataLoaders over those subsets.
    """
    tf = [transforms.ToTensor()]
    if normalize:
        tf.append(transforms.Normalize((0.5,), (0.5,)))
    full_ds = dataset_cls(root, train=train, download=True,
                          transform=transforms.Compose(tf))

    per_split = 10 // n_splits
    loaders = []
    for split_id in range(n_splits):
        lo = split_id * per_split
        hi = lo + per_split
        idxs = [i for i, lbl in enumerate(full_ds.targets) if lo <= lbl < hi]
        sub = Subset(full_ds, idxs)
        loaders.append(DataLoader(sub, batch_size=batch_size, shuffle=not train))
    return loaders

def train_on_loader(model, loader, optimizer, criterion, device, epochs):   
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=False)
        for x,y in pbar:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
# ----- Helper for Dataloaders
def concat_loaders(loaders):
    datasets = [loader.dataset for loader in loaders]
    return DataLoader(
        ConcatDataset(datasets),
        batch_size=loaders[0].batch_size,
        shuffle=False
    )
    
# -----  Eval helper for the no learning backbone testing
def eval_model_naive(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            probs = torch.softmax(model(x), dim=1)
            preds = torch.multinomial(probs, num_samples=1).squeeze()
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct/total    
            
def eval_model(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct/total      
          
def prune_and_finetune(model, train_loader, device, sparsity, finetune_epochs):
    # Collect all conv and fc layers
    to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            to_prune.append((module, 'weight'))

    # Apply global unstructured pruning (using torch pruning)
    prune.global_unstructured(
        to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )

    # Remove pruning reparam so weights are truly zero
    for module, name in to_prune:
        prune.remove(module, name)

    # Fine-tune training
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    train_on_loader(model, train_loader, optimizer, criterion, device, finetune_epochs)
    
    return model

def quantize(model, test_loader, device):
    model.eval()
    model.fuse_model()
    model.qconfig = tq.get_default_qconfig("qnnpack")
    tq.prepare(model, inplace=True)
    with torch.no_grad():
        for i,(x,_) in enumerate(test_loader):
            model(x.to(device))
            if i>=10: break
    return tq.convert(model.eval(), inplace=False)

# -----  Helper to measure zero-shot baseline accuracy of the frozen, quantized backbone on a new dataset
def backbone_zero_shot_baseline(backbone_model: nn.Module, device, test_loader,
                                num_splits: int = 5
                                ):
    """
    Build a zero‐trained head on top of your frozen backbone, and
    evaluate it on each of the "SplitMNIST" splits defined above.
    """
    # head that predicts uniformly at random (logits all equal)
    head = nn.Linear(in_features=128, out_features=10, bias=True)
    head.weight.data.zero_()
    head.bias.data.fill_(math.log(1/10))
    head.requires_grad_(False)

    model = TinyMLContinualModel(backbone_model, head).to(device)
    model.eval()

    test_loaders = test_loader

    

    for task_id, loader in enumerate(test_loaders):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                # your eval_model_naive just samples uniformly
                probs = torch.softmax(model(x), dim=1)
                preds = torch.multinomial(probs, num_samples=1).squeeze()
                correct += (preds == y).sum().item()
                total   += y.size(0)
        print(f"Zero‐shot on split {task_id}: {correct/total*100:.2f}%")

# -----  Metrics helper
def peak_ram_delta(model, loader, process, device):
            # Record baseline before any inference
            base = process.memory_info().rss

            peak = base
            model.eval()
            with torch.no_grad():
                for x, _ in loader:
                    _ = model(x.to(device))
                    mem = process.memory_info().rss
                    peak = max(peak, mem)

            # Return the extra memory your model+activations used, in KB
            return (peak - base) / 1024
# -----  Metrics helper        
def host_latency(model, iters=100):
    xs = torch.randn(1,1,28,28)
    # warm-up
    for _ in range(10): model(xs)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(xs)
        times.append(time.perf_counter()-t0)
    return (sum(times)/iters)*1e3  # ms
# -----  Metrics helper
def latency(m, iters=50):
            xs = torch.randn(1,1,28,28)
            # warm-up
            for _ in range(10):
                m(xs)
            times = []
            for _ in range(iters):
                t0 = time.perf_counter()
                m(xs)
                times.append((time.perf_counter() - t0) * 1e3)
            return sum(times)/len(times), torch.std(torch.tensor(times))
     
# -----  Metrics 
def tiny_ML_metrics(models, model_paths, train_loader, test_loader, device):  
    proc = psutil.Process(os.getpid())
    results = {}
    for name, m in models.items():
        acc = eval_model(m, test_loader, device)
        
        # Volatile memory the MCU uses at runtime for: Model activations, Weight buffers, Stack space, Heap
        ram = peak_ram_delta(m, test_loader, proc, device) # measured in KB
        
        # Non-volatile storage where I place my program binary and model weights, e.g. quantized .pth
        flash = os.path.getsize(model_paths[name]) / 1024
        

        # MACs & Params (remember Multiply–Accumulate operations --> quantify the compute cost of a NN, FLOPS is about 2x MACS)
        dummy = torch.randn(1,1,28,28)
        macs, params = thop_profile(m, (dummy,)) 
        macs_m = macs / 1e6 # measured in millions/kilo
        params_k = params / 1e3 # measured in millions/kilo
        
        lat = host_latency(m, iters=100)
        
        results[name] = {
        "Acc":        acc,
        "Flash (KB)": flash,
        "RAM (KB)":   ram,
        "MACs (M)":   macs_m,
        "Params (K)": params_k,
        "Latency":    lat
        }
        
        
    # convert to CSV
    rows = []
    for model_name, m_dict in results.items():
        row = {"model": model_name}
        row.update(m_dict)
        rows.append(row)

    # get CSV names
    fieldnames = ["model"] + list(next(iter(results.values())).keys())
    os.makedirs("metrics", exist_ok=True)
    with open(f"metrics/tinyml_metrics_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return 


def flatten_results_to_long(results, output_path="metrics/avalanche_long_metrics.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []

    for exp_idx, res_dict in enumerate(results):
        for full_key, value in res_dict.items():
            parts = full_key.split('/')
            metric_name = parts[0]
            phase       = parts[1] if len(parts) > 1 else ''
            stream      = parts[2] if len(parts) > 2 else ''
            task_label  = parts[3] if len(parts) > 3 else ''
            sub_id      = parts[4] if len(parts) > 4 else ''

            rows.append({
                "Task":        task_label.replace("Task", ""),
                "Experience":  exp_idx,
                "Phase":       phase.replace("_phase", ""),
                "Stream":      stream.replace("_stream", ""),
                "Metric":      metric_name,
                "SubID":       sub_id.replace("Exp", ""),
                "Value":       value
            })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Task", "Experience", "Phase", "Stream", "Metric", "SubID", "Value"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved long-format metrics to {output_path}")
    return

# -----  Continual learning
# now the training loop with latent replay
def train_with_latent_replay(model: TinyMLContinualModel,
                             adapter_opt,
                             loss_fn,
                             task_loaders_train,
                             task_loaders_test,
                             device,
                             replay_size,
                             live_B,
                             replay_R):
    
    
    # accuracy collections
    train_exp_list, eval_exp_list = [], []
    accs_list, forgettings_list = [], []
    overall_accs = []
    first_seen_accuracy = {}
    
    feat_buf  = deque(maxlen=replay_size)
    label_buf = deque(maxlen=replay_size)

    for task_id, train_loader in enumerate(task_loaders_train):
        
        # 1) train on this new experience
        print(f"\n=== Training on Task {task_id} ===")
        model.train()
        for epoch in range(1):
            pbar = tqdm(train_loader, desc=f"Task {task_id}", unit="batch", leave=False)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)

                # sample up to B replay patterns, prepare replay
                if task_id > 0 and len(feat_buf) >= replay_R:
                    idxs = torch.randperm(len(feat_buf))[:replay_R]
                    replay_feats = torch.stack([feat_buf[i] for i in idxs]).to(device)
                    replay_lbls  = torch.tensor([label_buf[i] for i in idxs],
                                                dtype=torch.long,
                                                device=device)

                else:
                    replay_feats = None
                    replay_lbls  = None

                # forward on (live + replay), split!
                x_live, y_live = x[:live_B], y[:live_B]
                
                # 2) compute live logits via backbone→adapter
                live_feats   = model.extract(x_live)        # B×D
                logits_live  = model.adapter(live_feats)    # B×C

                if replay_feats is not None:
                    replay_logits = model.adapter(replay_feats)  # R×C
                    logits = torch.cat([logits_live, replay_logits], dim=0)
                    y_all   = torch.cat([y_live, replay_lbls], dim=0)
                else:
                    logits, y_all = logits_live, y_live

                loss = loss_fn(logits, y_all)
                adapter_opt.zero_grad()
                loss.backward()
                adapter_opt.step()
                pbar.set_postfix(loss=loss.item())

                # compute new features & stash them (on CPU)
                with torch.no_grad():
                    new_feats_live = model.extract(x_live).cpu()
                for feat_vec, lbl in zip(new_feats_live, y_live.cpu()):
                    feat_buf.append(feat_vec)
                    label_buf.append(int(lbl))
                    
        ################# Task t done training --> EVALUATE ################
        # 1) Overall accuracy on all seen classes
        eval_bs = live_B + replay_R
        seen_dataset = ConcatDataset([ld.dataset for ld in task_loaders_test[: task_id+1]])
        seen_loader  = DataLoader(seen_dataset, batch_size=eval_bs)
        overall = eval_model(model, seen_loader, device)
        overall_accs.append((task_id, overall))
        print(f"After Task {task_id}: overall accuracy = {overall:.4f}")
        
        # 2) Per-split accuracy and forgetting
        for j in range(task_id+1):
            split_loader = task_loaders_test[j]
            acc_j = eval_model(model, split_loader, device)
            train_exp_list.append(task_id)
            eval_exp_list.append(j)
            accs_list.append(acc_j)

            if j not in first_seen_accuracy:
                first_seen_accuracy[j] = acc_j
                forgettings_list.append(0.0)
            else:
                forgettings_list.append(first_seen_accuracy[j] - acc_j)

   
    print("Done training over all tasks.")
    return train_exp_list, eval_exp_list, accs_list, forgettings_list, overall_accs




# -----  RAM metrics helper for continual learning
def measure_sram_peak(fn, interval=0.00001):
    """
    Runs fn() in a background thread, samples the current process RSS
    every `interval` seconds, and returns (peak - baseline) in MiB.
    """
    proc = psutil.Process(os.getpid())
    # collect a clean baseline
    gc.collect()
    baseline = proc.memory_info().rss

    # help us know when fn() has finished
    done = threading.Event()

    def wrapper():
        try:
            fn()
        finally:
            done.set()

    t = threading.Thread(target=wrapper)
    t.start()

    peak = baseline
    while not done.is_set():
        rss = proc.memory_info().rss
        if rss > peak:
            peak = rss
        time.sleep(interval)

    # one more sample after completion
    rss = proc.memory_info().rss
    if rss > peak:
        peak = rss

    return (peak - baseline) / (1024 * 1024)  # MiB

# -----  SRAM metrics helper for saving data
def calculate_sram(mem_delta):
    # Save RAM usage to my own CSV
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/peak_SRAM_usage.csv", "w", newline="") as fp:
        writer = csv.writer(fp)
        # header
        writer.writerow(["strategy", "mem_delta_MiB"])
        # data rows
        writer.writerow(["SRAM", mem_delta])
    return


# -----  Calculated metrics for accuracy per class CL 
def compute_per_class_accuracy(model, loader, device):
    model.eval()
    class_correct = {}
    class_total = {}
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            for p, t in zip(preds.cpu(), y.cpu()):
                class_correct.setdefault(int(t), 0)
                class_total.setdefault(int(t), 0)
                if int(p) == int(t):
                    class_correct[int(t)] += 1
                class_total[int(t)] += 1
    return {c: class_correct[c] / class_total[c] 
            for c in sorted(class_total) if class_total[c] > 0}
        
# -----  Calculated metrics for continual learning inspired by Towards Lifelong Deep Learning 
def continual_learning_metrics_extended(eval_exp, train_exp, acc, forgetting, model_name = 'model'):

    
    eval_exp = np.array(eval_exp, dtype=int)
    train_exp = np.array(train_exp, dtype=int)
    acc = np.array(acc, dtype=float)
    forgetting = np.array(forgetting, dtype=float)

    # number of experiences
    K = max(eval_exp) + 1
    
    # create KxK matrix as the paper suggests 
    R = np.full((K, K), np.nan, dtype=float)
    
    # populate R (only for valid entries where data has been seen before)
    # i  <-- number of experiences seen  =>  training_exp (rows)
    # j  <-- which task we eval         =>  eval_exp (cols)
    valid_acc = (train_exp >= eval_exp)
    R[train_exp[valid_acc ], eval_exp[valid_acc ] ] = acc[valid_acc ]

    
    final_mask = (train_exp == (K-1)) # get indices for only the accuracies after all data has been seen 
    final_accs = acc[final_mask] 
    avg_acc = np.nanmean(final_accs)

    valid_forg = (train_exp > eval_exp)
    avg_forg = np.nanmean(forgetting[valid_forg])
    
    # --- Average Incremental Accuracy A_K = 2/(K*(K+1)) * sum_{i>=j} R[i,j]
    # mask lower triangle including diagonal
    mask = np.tril(np.ones((K, K), dtype=bool))
    sum_lower = np.nansum(R[mask])
    avg_inc_acc = (2.0 / (K*(K+1))) * sum_lower
    
    

    # --- Backward Transfer BWT = avg_{j < K} [ R[K-1,j] - R[j,j] ]
    last_row = R[K-1, :K-1]          # R[K-1, j] for j=0..K-2
    diagonal = np.diag(R)[:K-1]      # R[j,j] for j=0..K-2
    bwt = np.nanmean(last_row - diagonal)
    
    # --- Forward Transfer FWT = avg_{i < j} [ R[i,j] - R[0,j] ]
    # Here R[0,j] is the “initial” accuracy before training on any task
    # We compare performance of model after task i on unseen task j>i vs that initial baseline
    baseline = R[0, :]               # shape (K,)
    # get indices i<j
    tri_i, tri_j = np.triu_indices(K, k=1)
    fwt_values   = R[tri_i, tri_j] - baseline[tri_j]
    fwt = np.nanmean(fwt_values)

    # STILL NEEd TO CALCULATE:
    # modified_bwt = 
    # modified_fwt = 
    #model_size_efficiency = 
    # sample_storage_size =
        
    CL_dic = {
        "avg_acc": avg_acc,
        "avg_forg": avg_forg,
        "avg_inc_acc": avg_inc_acc,
        "fwt": fwt, 
        "bwt": bwt,
    }
    
    # Convert to CSV
    row = {"model": model_name, **CL_dic}

    # get CSV names
    fieldnames = ["model"] + list(CL_dic.keys())
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/CL_metrics_extended.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
        
    return



# Classifier head for ODL, adds down projection, relu, up projection to get some CL involved. 
# Gets added to my quantised base model 
class AdapterHead(nn.Module):
    def __init__(self, in_features=128, num_classes=10, bottleneck=32):
        super().__init__()
        # a low-rank adapter: down-project → nonlinearity → up-project
        self.down = nn.Linear(in_features, bottleneck)
        self.relu = nn.ReLU(inplace=True)
        self.up   = nn.Linear(bottleneck, num_classes)
    def forward(self, x):
        # x is shape (batch, 128)
        x = self.down(x)
        x = self.relu(x)
        return self.up(x)

########################################################################################################
                            ############# MAIN FUNCTION LOOP ##########
########################################################################################################
def main():
    
    retrain = False # won't retrain if there is existing model, unless this is set to True

    # Define model paths
    os.makedirs("models", exist_ok=True)
    MODEL_PATH           = "models/kmnist_fp32.pth"
    PRUNED_PATH          = "models/kmnist_pruned.pth"
    BACKBONE     = "models/vkmnist_backbone_int8.pth"
    HYBRID = "models/k_and_mnist_hybrid.pth"
    model_paths = {
        "FP32":           MODEL_PATH,
        "Pruned":         PRUNED_PATH,
        "Backbone":    BACKBONE,
        "Hybrid": HYBRID
    }
    

    # Begin Pipeline
    
    # ----- 0. Data loading
    device = torch.device("cpu")
    base_train_ds = datasets.KMNIST("../data", train=True, download=True,
                               transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,))]))
    base_test_ds  = datasets.KMNIST("../data", train=False, download=True,
                               transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,))]))
    
    # These below will be used later for CL testing 
    full_test_ds = datasets.MNIST("../data", train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))]))
    full_test_loader = DataLoader(full_test_ds, batch_size=1000, shuffle=False)
    
    base_train_loader = DataLoader(base_train_ds, batch_size=64, shuffle=True)
    base_test_loader  = DataLoader(base_test_ds,  batch_size=1000, shuffle=False)
    # ----- 1. Base CNN Model 
    model_fp32 = TinyCNN().to(device)
    
    # ----- 2 Train FP32 and save
    if not retrain:
        if os.path.exists(MODEL_PATH):
            print(f"Found existing model at {MODEL_PATH}, skipping training.")
            model_fp32.load_state_dict(torch.load(MODEL_PATH))
        else:
            # --- Train from scratch
            print(f"Training new model: {MODEL_PATH}")

            train_on_loader(model_fp32, base_train_loader,
                        torch.optim.SGD(model_fp32.parameters(),lr=0.01),
                        nn.CrossEntropyLoss(), device, 5)
            torch.save(model_fp32.state_dict(), MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")
    else:
            # --- Train from scratch
            print(f"Training new model: {MODEL_PATH}")

            train_on_loader(model_fp32, base_train_loader,
                        torch.optim.SGD(model_fp32.parameters(),lr=0.01),
                        nn.CrossEntropyLoss(), device, 5)
            torch.save(model_fp32.state_dict(), MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")
    
    # ----- 3. Prune FP32 and retrain (currently applying unstructured pruning so I won't see benefits)
    model_pruned = TinyCNN().to(device)
    if not retrain:
        if os.path.exists(PRUNED_PATH):
            print(f"Found existing model at {PRUNED_PATH}, skipping training.")
            model_pruned.load_state_dict(torch.load(PRUNED_PATH))
        else:
            # --- Prune and tune from scratch
            print(f"Training new model: {PRUNED_PATH}")
            model_pruned = prune_and_finetune(model_fp32, base_train_loader, device,
                                    sparsity=0.5, finetune_epochs=3)
            torch.save(model_pruned.state_dict(), PRUNED_PATH)
            print(f"Model saved to {PRUNED_PATH}")
    else:
            # --- Prune and tune from scratch
            print(f"Training new model: {PRUNED_PATH}")
            model_pruned = prune_and_finetune(model_fp32, base_train_loader, device,
                                    sparsity=0.5, finetune_epochs=3)
            torch.save(model_pruned.state_dict(), PRUNED_PATH)
            print(f"Model saved to {PRUNED_PATH}")

    # ----- 4. PTQ (from 32 --> 8)
    model_quant = quantize(model_pruned, base_train_loader, device)
    torch.save(model_quant.state_dict(), BACKBONE)
    print(f"Model saved to {BACKBONE}")
    
    
    # ----- 4.1 TEST PURE QUANTISED
    n_splits = 5
    # get split to simulate CL 
    test_loaders = make_split_dataset_loaders(
    datasets.MNIST, "../data",
    n_splits=n_splits,
    train=False,
    batch_size=1000
    )
    print('Testing only quantised backbone')
    backbone_zero_shot_baseline(backbone_model= model_quant, device = device, test_loader = test_loaders)
    
    # ----- 5. Combine PQT + classifier head
    backbone = model_quant.cpu()
    
    # initialise an adapter head 
    adapter = AdapterHead(in_features=128, num_classes=10, bottleneck=32).to(device)
    
    hybrid_model = TinyMLContinualModel(backbone, adapter).to(device)
    torch.save(hybrid_model.state_dict(), HYBRID)
    print(f"Model saved to {HYBRID}")
 
    # ----- 6. Continual learning

    # Hyperparams
    replay_size = 200
    batch_size     = 32
    live_B   = 1
    replay_R = 200    
    # Load dataset splits 
    task_loaders_train = make_split_dataset_loaders(datasets.MNIST, "../data", 
                                            n_splits=n_splits,
                                            train = True,
                                            batch_size = batch_size)
    task_loaders_test = make_split_dataset_loaders(datasets.MNIST, "../data", 
                                            n_splits=n_splits,
                                            train = False,
                                            batch_size = 1000)
    

    train_exp, eval_exp, accs, forgettings, overall_accs = train_with_latent_replay(
        hybrid_model,
        torch.optim.SGD(adapter.parameters(), lr=1e-2),
        nn.CrossEntropyLoss(),
        task_loaders_train,
        task_loaders_test,
        device,
        replay_size=replay_size,
        live_B=live_B,
        replay_R= replay_R)
    print("finished CL experiences")
    
    # FIX BELOW SRAM MEASURMENT!!!
    
    print("Starting CL experiences")
    # def do_cl():
    #     train_with_latent_replay(
    #     hybrid_model,
    #     torch.optim.SGD(adapter.parameters(), lr=1e-2),
    #     nn.CrossEntropyLoss(),
    #     task_loaders_train,
    #     task_loaders_test,
    #     device,
    #     replay_size=rm_sz,
    #     batch_size=B
    #     )
    # mem_delta = measure_sram_peak(do_cl, interval=0.01)
    

            
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
        for t, acc in overall_accs:
            writer.writerow([t, f"{acc:.4f}"])
    
    # Now we compute extended metrics and save
    continual_learning_metrics_extended(eval_exp, train_exp, accs, forgettings, model_name="Hybrid")
      
    ### FIX BELOW!!!
    # print(f"Replay training RAM delta on host: {mem_delta:.1f} MiB")
    #calculate_sram(mem_delta)
    #print("Saved SRAM usage to peak_SRAM_usage.csv")
    

    # ----- 7. Standard Metrics 
    models = {
        "FP32":        model_fp32,
        "Pruned":      model_pruned,
        "Backbone": model_quant,
        "Hybrid": hybrid_model
    }
    tiny_ML_metrics(models, model_paths, base_train_loader, base_test_loader, device)
    print("Standard tinyML metrics saved to tinyml_metrics_summary.csv")
    
if __name__ == "__main__":
    main()
    print("Completed successfully!")
    