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
from typing import Optional, Tuple
from tqdm import tqdm
import gc
import tracemalloc
#### CITE https://github.com/vlomonaco/ar1-pytorch/blob/master/ar1star_lat_replay.py


###################################################################################################################################################
############################################################### Define Models ##################################################################
###################################################################################################################################################

# -----  Backbone CNN Model 
class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) # kernel size 3, input channels 3, output channels 16
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # kernel size 3, input channels 16, output channels 32
        self.bn2   = nn.BatchNorm2d(32)
        self.fc1   = nn.Linear(32*8*8, 128)
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
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2)) # size 32 --> 16 (from the max pooling)
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), 2)) # size 16 --> 8 (from the max pooling)
        x = x.flatten(start_dim=1) # flattens to (batch, 32*8*8) so we can input to fc layers
        x = F.relu(self.fc1(x)) # --> (batch, 128) # 128 comes from my linear def. 
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
            # Access the quantized modules properly
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
        # Before calling the quantized adapter
        feats = feats.to('cpu')
        self.adapter = self.adapter.to('cpu')
        return self.adapter(feats)

###################################################################################################################################################
############################################################### Define Functions ##################################################################
###################################################################################################################################################
def load_dataset(dataset_cls, training, transform):
    # 1) Load full set
    full_ds = dataset_cls(
        "../data",
        train=training,
        download=True,
        transform=transform
    )

    
    return full_ds

def make_split_dataset_loaders(
    full_ds,
    n_splits=5,
    train: bool = False,
    batch_size: int = 1000,
):
    """
    Partition any torchvision dataset (e.g. KMNIST) into `n_splits` tasks,
    each containing consecutive labels.  E.g. with n_splits=5:
      split 0 -> labels {0,1}, split 1 -> {2,3}, …, split 4 -> {8,9}.
    Returns a list of DataLoaders over those subsets.
    """

    per_split = len(set(full_ds.targets))  // n_splits
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
            
# -----  Evaluator for CL model where classes must be filtered
def eval_continual_model(model, loader, seen_labels, device):
    model = model.to(device).eval()
    correct = total = 0
    seen_labels = torch.tensor(seen_labels, device=device)

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)                         # shape (B,100)
            # create a -inf mask, then fill seen positions from logits
            mask = torch.full_like(logits, float("-inf"))
            mask[:, seen_labels] = logits[:, seen_labels]
            preds = mask.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)

    return correct / total 
# ----- Evaluator for standard backbone model
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
def backbone_zero_shot_baseline(backbone_model: nn.Module, device, test_loader, full_test_dataset,
                                num_splits
                                ):
    """
    Build a zero‐trained head on top of your frozen backbone, and
    evaluate it on each of the "SplitMNIST" splits defined above.
    """
    # head that predicts uniformly at random (logits all equal)
    # full 100-way uniform head
    num_classes = len(full_test_dataset.classes)
    head = nn.Linear(128, num_classes, bias=True)
    head.weight.data.zero_()
    head.bias.data.fill_(math.log(1/num_classes))
    head.requires_grad_(False)

    cl_model = TinyMLContinualModel(backbone_model, head).to(device).eval()

    for task_id, loader in enumerate(test_loader):
        # compute which labels belong to this split
        per_split = num_classes // len(test_loader)
        lo = task_id * per_split
        hi = lo + per_split
        mask = torch.full((num_classes,), float("-inf"), device=device)
        mask[lo:hi] = 0.0

        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = cl_model(x)          # shape [B,100]
                masked = logits + mask        # unseen classes stay at -inf
                preds = masked.argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += y.size(0)

        print(f"Zero‐shot on split {task_id}: {100*correct/total:.2f}%")
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
    xs = torch.randn(1,3,32,32)
    # warm-up
    for _ in range(10): model(xs)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(xs)
        times.append(time.perf_counter()-t0)
    return (sum(times)/iters)*1e3  # ms
# -----  Metrics 
def tiny_ML_metrics(models, model_paths, test_loader, device):  
    proc = psutil.Process(os.getpid())
    results = {}
    for name, m in models.items():
        acc = eval_model(m, test_loader, device)
        
        # Volatile memory the MCU uses at runtime for: Model activations, Weight buffers, Stack space, Heap
        ram = peak_ram_delta(m, test_loader, proc, device) # measured in KB
        
        # Non-volatile storage where I place my program binary and model weights, e.g. quantized .pth
        flash = os.path.getsize(model_paths[name]) / 1024
        

        # MACs & Params (remember Multiply–Accumulate operations --> quantify the compute cost of a NN, FLOPS is about 2x MACS)
        dummy = torch.randn(1,3, 32, 32)
        profile_result = thop_profile(m, (dummy,))
        if len(profile_result) == 2:
            macs, params = profile_result
        else:
            macs, params, _ = profile_result  # Handle case where thop returns 3 values
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


# -----  Continual learning
def calibrate_feature_range(backbone: nn.Module,
                             quant_stub: tq.QuantStub,
                             dequant_stub: tq.DeQuantStub,
                             calib_loader: DataLoader,
                            device: torch.device,
                            num_batches: int = 10):
    """
   "Runs a few batches through the backbone up to `fc1` to
   find global min/max of features.
    """
    backbone.eval()
    feat_min, feat_max = float("inf"), float("-inf")

    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            x = x.to(device)
            z = quant_stub(x)
            z = F.relu(F.max_pool2d(backbone.bn1(backbone.conv1(z)), 2))
            z = F.relu(F.max_pool2d(backbone.bn2(backbone.conv2(z)), 2))
            z = z.flatten(1)
            z = backbone.fc1(z)
            feats = dequant_stub(z)
            b_min, b_max = feats.min().item(), feats.max().item()
            feat_min = min(feat_min, b_min)
            feat_max = max(feat_max, b_max)
            if i + 1 >= num_batches:
                break
            
    print(f"Calibrated feature range: [{feat_min:.4f}, {feat_max:.4f}]")

    return feat_min, feat_max


def train_with_latent_replay(model: TinyMLContinualModel,
                             adapter_opt,
                             loss_fn,
                             task_loaders_train,
                             task_loaders_test,
                             device,
                             replay_size,
                             live_B,
                             replay_R,
                             float_adapter,
                             full_test_ds,
                             quant_params: Tuple[float,int]):
    
    # Unpack pre computed quant values 
    scale, zero_point = quant_params


    # accuracy and buffer collections
    train_exp_list, eval_exp_list = [], []
    accs_list, forgettings_list = [], []
    overall_accs = []
    first_seen_accuracy = {}
    # store 8bit latent feature vectors + labels (for replay)
    feat_buf  = deque(maxlen=replay_size)
    label_buf = deque(maxlen=replay_size)

    for task_id, train_loader in enumerate(task_loaders_train):
        
        # 1) train on this new experience
        print(f"\n=== Training on Task {task_id} ===")
        model.train()
        for epoch in range(15):
            pbar = tqdm(train_loader, desc=f"Task {task_id}", unit="batch", leave=False)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)

                # REPLAY!!!
                # random latent replay sampling (first check we've processed at least one task)
                if task_id > 0 and len(feat_buf) >= replay_R:
                    idxs = torch.randperm(len(feat_buf))[:replay_R]
                    replay_feats_q = torch.stack([feat_buf[i] for i in idxs], dim=0).to(device)
                    replay_lbls    = torch.tensor([label_buf[i] for i in idxs], device=device)

                if task_id > 0 and len(feat_buf) >= replay_R:
                    idxs = torch.randperm(len(feat_buf))[:replay_R]
                    replay_feats_q = torch.stack([feat_buf[i] for i in idxs], dim=0).to(device)
                    replay_lbls    = torch.tensor([label_buf[i] for i in idxs], device=device)

                else:
                    replay_feats_q = None
                    replay_lbls  = None

                # LIVE TRAINING!!!
                
                # extract live batch from current minibatch 
                x_live, y_live = x[:live_B], y[:live_B]
                
                # compute live features & quantize them on-the-fly
                live_feats_f = model.extract(x_live)  # float32 B×D
                # quantize → then immediately dequantize for float training
                live_feats_q = torch.quantize_per_tensor(live_feats_f.cpu(), 
                            scale=scale,
                            zero_point=zero_point, dtype=torch.qint8).to(device)
                live_feats_f_train = live_feats_q.dequantize().to(device)
                    
                # run the float adapter on the dequantized features
                logits_live = float_adapter(live_feats_f_train)
                
                # if there's replay data, run that through adapter also
                if replay_feats_q is not None:
                    replay_feats_f_train = replay_feats_q.dequantize().to(device)
                    logits_replay = float_adapter(replay_feats_f_train)
                    logits = torch.cat([logits_live, logits_replay], dim=0)
                    y_all = torch.cat([y_live, replay_lbls], dim=0)
                else:
                    logits, y_all = logits_live, y_live

                    
                # loss + step (combine replay and live)
                loss = loss_fn(logits, y_all)
                adapter_opt.zero_grad()
                loss.backward()
                adapter_opt.step()
                pbar.set_postfix(loss=loss.item())

                # re‐extract live features & quantize them for buffer
                with torch.no_grad():
                    for qfeat, lbl in zip(live_feats_q, y_live.cpu()):
                        feat_buf.append(qfeat)
                        label_buf.append(int(lbl))

                    
        ################# Task t done training --> EVALUATE ################
        # 1) Overall accuracy on all seen classes
        eval_bs = live_B + replay_R
        # build a single dataset containing all test splits up to task_id
        seen_dataset = ConcatDataset([ld.dataset for ld in task_loaders_test[: task_id+1]])
        seen_loader  = DataLoader(seen_dataset, batch_size=eval_bs, shuffle=False)
        
        #extract the list of labels in that dataset
        all_targets = full_test_ds.targets  # the full 100-class list
        seen_indices = []
        for loader in task_loaders_test[: task_id+1]:
            seen_indices.extend(loader.dataset.indices)
        seen_labels = sorted({ all_targets[i] for i in seen_indices })

        overall = eval_continual_model(model, seen_loader, seen_labels, device)
        overall_accs.append((task_id, overall))
        print(f"After Task {task_id}: overall accuracy on seen labels = {overall:.4f}")
        
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
def record_sram_to_csv(mem_delta):
    # Save RAM usage to my own CSV
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/peak_SRAM_usage.csv", "w", newline="") as fp:
        writer = csv.writer(fp)
        # header
        writer.writerow(["strategy", "mem_delta_MiB"])
        # data rows
        writer.writerow(["SRAM", mem_delta])
    return

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
    # AND FIX FWT
        
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
    MODEL_PATH           = "models/cifar10.pth"
    PRUNED_PATH          = "models/cifar10_fp32.pth"
    BACKBONE_PATH     = "models/cifar10_backboneint8.pth"
    HYBRID_PATH = "models/cifar100_hybrid.pth"
    model_paths = {
        "FP32":           MODEL_PATH,
        "Pruned":         PRUNED_PATH,
        "Backbone":    BACKBONE_PATH,
        "Hybrid": HYBRID_PATH
    }
    

    # Begin Pipeline
    
    # ----- 0. Data loading
    device = torch.device("cpu")
    
    # transforms
    train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2470, 0.2435, 0.2616))
    ])

    # backbone training datasets
    base_train_ds = load_dataset(datasets.CIFAR10, True, train_transform)
    base_test_ds  = load_dataset(datasets.CIFAR10, False, test_transform)
    base_train_loader = DataLoader(base_train_ds, batch_size=64, shuffle=True)
    base_test_loader  = DataLoader(base_test_ds,  batch_size=1000, shuffle=False)

    
    # These below will be used later for CL testing 
    full_train_ds = load_dataset(datasets.CIFAR100, True, train_transform)
    full_test_ds = load_dataset(datasets.CIFAR100, False, test_transform)
    full_test_loader = DataLoader(full_test_ds, batch_size=1000, shuffle=False)
    
    
    # ----- 1. Base CNN Model 
    model_fp32 = TinyCNN(num_classes=10).to(device)
    
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
    torch.save(model_quant.state_dict(), BACKBONE_PATH)
    print(f"Model saved to {BACKBONE_PATH}")
    test = eval_model(model_quant, base_test_loader, device)
    print(f"BACKBONE ACC on CIFAR10; {test}")

    
    
    # ----- 4.1 TEST PURE QUANTISED
    n_splits = 5
    # get split to simulate CL (make_split_dataset_loaders takes in a dataset)
    task_loaders_test = make_split_dataset_loaders(
    full_test_ds,
    n_splits=n_splits,
    train=False,
    batch_size=1000
    )
    
    print('Testing only quantised backbone')
    backbone_zero_shot_baseline(backbone_model= model_quant, device = device, test_loader = task_loaders_test, full_test_dataset=full_test_ds,num_splits= n_splits)
    
    # ----- 5. Prepare quantization and adapter
    
    # Load dataset splits for CL model
    # Hyperparams
    replay_size = 1000 #size of the circular buffer storing past quantized feature vectors
    live_B = 16 # how many live samples per batch you actually learn on
    replay_R = 100 # how many replayed features you sample (up to) each step    
    task_loaders_train = make_split_dataset_loaders(
        full_train_ds,
        n_splits=n_splits,
        train=True,
        batch_size=32   
        )
    # This always stays frozen 
    backbone = model_quant.cpu() # rename
    
    #  Calibrate the backbone’s output range once using a small subset of CORE50 task 1
    calib_loader = DataLoader(
         Subset(task_loaders_train[0].dataset, list(range(200))),  # first 200 samples
            batch_size=32, shuffle=False
    )
    feat_min, feat_max = calibrate_feature_range(
    backbone, backbone.quant, backbone.dequant, calib_loader, device, num_batches=20)
    
    # Compute the 8-bit quant params for float adapter
    qmin, qmax = -128, 127
    scale      = (feat_max - feat_min) / float(qmax - qmin)
    zero_point = int(qmin - feat_min/scale)

    # Build tiny FP32 adapter and its optimizer (trainable part)
    float_adapter = AdapterHead(in_features=128, bottleneck=32, num_classes=100).to(device)
    optimizer     = torch.optim.SGD(float_adapter.parameters(), lr=1e-2)
    loss_fn       = nn.CrossEntropyLoss()

 
    # ----- 6. Continual learning
    hybrid_model = TinyMLContinualModel(backbone, float_adapter)
    # Save combined model to be able to measure later
    torch.save(hybrid_model.state_dict(), HYBRID_PATH)
    print(f"Model saved to {HYBRID_PATH}")
    

    training_results = {}
    print("Starting CL experiences")
    def do_cl():
        training_results['outs'] = train_with_latent_replay(
        hybrid_model,  # backbone+float head
        adapter_opt=optimizer,
        loss_fn=loss_fn,
        task_loaders_train=task_loaders_train,
        task_loaders_test=task_loaders_test,
        device=device,
        replay_size=replay_size,
        live_B= live_B,
        replay_R=replay_R,
        # here we inject the scale & zero_point so your function
        # can quantize+dequantize on the fly
        float_adapter=float_adapter,
        full_test_ds = full_test_ds,
        quant_params=(scale, zero_point)
    )

    mem_delta = measure_sram_peak(do_cl, interval=0.01)
          
    print(f"Replay training RAM delta on host: {mem_delta:.1f} MiB")
    record_sram_to_csv(mem_delta)
    print("Saved SRAM usage to peak_SRAM_usage.csv")
    
    train_exp, eval_exp, accs, forgettings, overall_accs = training_results['outs']
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
        for t, acc in overall_accs:
            writer.writerow([t, f"{acc:.4f}"])
    
    # Now we compute extended metrics and save
    continual_learning_metrics_extended(eval_exp, train_exp, accs, forgettings, model_name="Hybrid")
    print("CL extended metrics saved to CL_metrics_extended.csv")

    # ----- 7. Standard Metrics 
    models = {
        "FP32":        model_fp32,
        "Pruned":      model_pruned,
        "Backbone": model_quant,
        "Hybrid": hybrid_model
        
    }
    tiny_ML_metrics(models, model_paths, full_test_loader, device)
    print("Standard tinyML metrics saved to tinyml_metrics_summary.csv")
if __name__ == "__main__":
    main()
    print("Completed successfully!")
    