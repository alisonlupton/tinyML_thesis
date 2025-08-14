
import torch
torch.backends.quantized.engine = "qnnpack"
import os
from thop import profile as thop_profile
import time
import csv
import numpy as np 
import psutil
import threading
import gc
# from utils import eval_model
import torch.nn as nn
########################################################################################################
                            ############# HELPERS ##########
########################################################################################################
#TODO: metrics to implement and record: FLOPs, take all metrics from my 3 main papers and implement, also all metrics below + ones from CL paper
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

# -----  Metrics helper for calculating RAM
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

# -----  Metrics helper for saving SRAM data
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

########################################################################################################
                            ############# FUNCTIONS ##########
########################################################################################################


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

# -----  Calculated metrics for memory, flash, etc.
# def tiny_ML_metrics(models, model_paths, test_loader, device):  
#     proc = psutil.Process(os.getpid())
#     results = {}
#     for name, m in models.items():
#         acc = eval_model(m, test_loader, device)
        
#         # Volatile memory the MCU uses at runtime for: Model activations, Weight buffers, Stack space, Heap
#         ram = peak_ram_delta(m, test_loader, proc, device) # measured in KB
        
#         # Non-volatile storage where I place my program binary and model weights, e.g. quantized .pth
#         flash = os.path.getsize(model_paths[name]) / 1024
        

#         # MACs & Params (remember Multiply–Accumulate operations --> quantify the compute cost of a NN, FLOPS is about 2x MACS)
#         dummy = torch.randn(1,3, 32, 32)
#         profile_result = thop_profile(m, (dummy,))
#         if len(profile_result) == 2:
#             macs, params = profile_result
#         else:
#             macs, params, _ = profile_result  # Handle case where thop returns 3 values
#         macs_m = macs / 1e6 # measured in millions/kilo
#         params_k = params / 1e3 # measured in millions/kilo
        
#         lat = host_latency(m, iters=100)
        
#         results[name] = {
#         "Acc":        acc,
#         "Flash (KB)": flash,
#         "RAM (KB)":   ram,
#         "MACs (M)":   macs_m,
#         "Params (K)": params_k,
#         "Latency":    lat
#         }
        
        
#     # convert to CSV
#     rows = []
#     for model_name, m_dict in results.items():
#         row = {"model": model_name}
#         row.update(m_dict)
#         rows.append(row)

#     # get CSV names
#     fieldnames = ["model"] + list(next(iter(results.values())).keys())
#     os.makedirs("metrics", exist_ok=True)
#     with open(f"metrics/tinyml_metrics_summary.csv", "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(rows)
#     return 
import math
from contextlib import contextmanager

def _bytes_str(n):
    if n < 1024: return f"{n} B"
    if n < 1024**2: return f"{n/1024:.2f} KB"
    if n < 1024**3: return f"{n/1024**2:.2f} MB"
    return f"{n/1024**3:.2f} GB"

def _bitbytes(bits):
    return bits // 8

def _count_params(module):
    n = 0
    for p in module.parameters(recurse=True):
        n += p.numel()
    return n

def _count_params_trainable(module):
    n = 0
    for p in module.parameters(recurse=True):
        if p.requires_grad:
            n += p.numel()
    return n

@torch.no_grad()
def _infer_feature_dim(model, C, L, device):
    dummy = torch.zeros(1, C, L, device=device)
    feats = model._features(dummy)  # (1, F)
    return feats.shape[1]

@contextmanager
def _capture_peak_activations(modules, device, C, L, dtype=torch.float32):
    peak = {'bytes': 0}
    hooks = []
    def hook(_, __, out):
        if isinstance(out, (tuple, list)):
            outs = out
        else:
            outs = [out]
        s = 0
        for o in outs:
            if torch.is_tensor(o):
                s += o.numel() * o.element_size()
        if s > peak['bytes']:
            peak['bytes'] = s

    # register on backbone’s leaf modules (conv/bn/relu/dropout/pool)
    for m in modules.modules():
        if isinstance(m, (nn.Conv1d, nn.BatchNorm1d, nn.ReLU, nn.Dropout, nn.AdaptiveAvgPool1d)):
            hooks.append(m.register_forward_hook(hook))

    # run one forward
    x = torch.zeros(1, C, L, dtype=dtype, device=device)
    modules.eval()
    _ = modules(x)

    for h in hooks:
        h.remove()
    yield peak['bytes']  # single-sample peak

def profile_cil_memory(
    cil_model: nn.Module,
    C: int,
    L: int,
    *,
    batch_size_train: int = 16,
    batch_size_infer: int = 1,
    replay_size: int = 5000,
    optimizer_kind: str = "adam",          # "adam" or "sgd"
    deployed_bits_backbone: int = 8,       # how you’ll store backbone in flash
    deployed_bits_classifier: int = 8,     # how you’ll store classifier in flash
    training_bits_classifier: int = 32,    # online training dtype, usually 32
    replay_bits: int = 8,                  # store replay features as int8 if you can
    activation_bits_infer: int = 8,        # if you run int8 inference, set to 8
    activation_bits_train: int = 32,       # during training, activations are FP32
    safety_factor: float = 1.2             # account for framework/workspace overhead
):
    device = next(cil_model.parameters()).device

    # Split modules
    backbone = cil_model.backbone
    classifier = cil_model.head

    # Parameter counts
    n_backbone = _count_params(backbone)
    n_classifier = _count_params(classifier)
    n_classifier_trainable = _count_params_trainable(classifier)  # should be all of it

    # Feature dim (from backbone head)
    F = _infer_feature_dim(cil_model, C, L, device)
    num_classes = cil_model.head.out_features

    # TDM mask memory (actual dtype)
    mask_bytes = 0
    if hasattr(cil_model, "mask") and torch.is_tensor(cil_model.mask):
        mask_bytes = cil_model.mask.numel() * cil_model.mask.element_size()

    # ---- FLASH (deployed) ----
    # Typically store weights quantized; biases often remain 8/16/32 depending on toolchain.
    bb_flash = n_backbone * _bitbytes(deployed_bits_backbone)
    clf_flash = n_classifier * _bitbytes(deployed_bits_classifier)
    mask_flash = mask_bytes  # if you persist mask; if not stored, set 0
    flash_total = bb_flash + clf_flash + mask_flash

    # ---- SRAM: peak inference ----
    # Peak activations in the backbone (per sample) under inference dtype
    with _capture_peak_activations(backbone, device, C, L, dtype=torch.float32) as peak_act_1x:
        pass
    # Scale by batch size and bytes per activation element
    act_bytes_infer = math.ceil(peak_act_1x * batch_size_infer * (_bitbytes(activation_bits_infer) / 4.0))
    # Classifier activations are tiny (B x F and B x num_classes); include them too
    clf_act_infer = batch_size_infer * (F + num_classes) * _bitbytes(activation_bits_infer)
    sram_infer_peak = int(safety_factor * (act_bytes_infer + clf_act_infer))

    # ---- SRAM: peak during online training (classifier only) ----
    # 1) Backbone activations (computed in FP32 for feature extraction)
    act_bytes_train = math.ceil(peak_act_1x * batch_size_train * (_bitbytes(activation_bits_train) / 4.0))
    feats_bytes_train = batch_size_train * F * _bitbytes(training_bits_classifier)  # features fed into classifier
    # 2) Classifier params + grads + optimizer states (kept in RAM for training)
    clf_param_bytes = n_classifier * _bitbytes(training_bits_classifier)
    clf_grad_bytes  = n_classifier * 4  # gradients are FP32
    if optimizer_kind.lower() == "adam":
        opt_state_bytes = n_classifier * 2 * 4  # m and v, FP32 each
    else:  # SGD (+momentum)
        opt_state_bytes = n_classifier * 4      # one momentum buffer

    # 3) Replay buffer (store features, not raw windows)
    replay_bytes = replay_size * F * _bitbytes(replay_bits)

    # 4) TDM mask is resident in RAM too (depends on your implementation)
    mask_sram = mask_bytes

    sram_train_peak = int(safety_factor * (act_bytes_train + feats_bytes_train +
                                           clf_param_bytes + clf_grad_bytes + opt_state_bytes +
                                           replay_bytes + mask_sram))

    report = {
        "feature_dim": F,
        "num_classes": num_classes,
        "params": {
            "backbone_params": n_backbone,
            "classifier_params": n_classifier,
        },
        "flash": {
            "backbone_flash_bytes": bb_flash,
            "classifier_flash_bytes": clf_flash,
            "mask_flash_bytes": mask_flash,
            "total_flash_bytes": flash_total,
        },
        "sram": {
            "inference_peak_bytes": sram_infer_peak,
            "training_peak_bytes": sram_train_peak,
            "breakdown_training": {
                "backbone_activations_bytes": act_bytes_train,
                "classifier_forward_feats_bytes": feats_bytes_train,
                "classifier_param_bytes": clf_param_bytes,
                "classifier_grad_bytes": clf_grad_bytes,
                "optimizer_state_bytes": opt_state_bytes,
                "replay_buffer_bytes": replay_bytes,
                "mask_bytes": mask_sram,
                "safety_factor": safety_factor,
            },
            "breakdown_inference": {
                "backbone_peak_activations_bytes": act_bytes_infer,
                "classifier_activations_bytes": clf_act_infer,
                "safety_factor": safety_factor,
            }
        },
        "notes": [
            "Flash is computed at deployed bitwidths (deployed_bits_*).",
            "Training SRAM assumes classifier trained in FP32 (training_bits_classifier).",
            "Adam uses 2 FP32 states per param; SGD+momentum uses 1.",
            "Backbone is frozen; we do not allocate backbone grads or optimizer states.",
            "Replay buffer is in feature space (size = replay_size x feature_dim).",
        ]
    }
    return report

def print_cil_memory_report(report):
    print("\n=== CIL Memory Report ===")
    print(f"Feature dim (F): {report['feature_dim']}")
    print(f"Num classes:      {report['num_classes']}")
    print("\nParameters:")
    print(f"  Backbone:   {report['params']['backbone_params']:,}")
    print(f"  Classifier: {report['params']['classifier_params']:,}")
    print("\nFlash (deployed):")
    f = report['flash']
    print(f"  Backbone:   {_bytes_str(f['backbone_flash_bytes'])}")
    print(f"  Classifier: {_bytes_str(f['classifier_flash_bytes'])}")
    if f['mask_flash_bytes'] > 0:
        print(f"  TDM mask:   {_bytes_str(f['mask_flash_bytes'])}")
    print(f"  TOTAL:      {_bytes_str(f['total_flash_bytes'])}")
    print("\nSRAM (peak) — inference:")
    s = report['sram']['breakdown_inference']
    print(f"  Backbone peak activations: {_bytes_str(s['backbone_peak_activations_bytes'])}")
    print(f"  Classifier activations:    {_bytes_str(s['classifier_activations_bytes'])}")
    print(f"  Safety factor:             x{s['safety_factor']}")
    print(f"  ==> PEAK: {_bytes_str(report['sram']['inference_peak_bytes'])}")
    print("\nSRAM (peak) — online training (classifier only):")
    t = report['sram']['breakdown_training']
    print(f"  Backbone activations:   {_bytes_str(t['backbone_activations_bytes'])}")
    print(f"  Classifier feats (B×F): {_bytes_str(t['classifier_forward_feats_bytes'])}")
    print(f"  Classifier params:      {_bytes_str(t['classifier_param_bytes'])}")
    print(f"  Classifier grads:       {_bytes_str(t['classifier_grad_bytes'])}")
    print(f"  Optimizer state:        {_bytes_str(t['optimizer_state_bytes'])}")
    print(f"  Replay buffer:          {_bytes_str(t['replay_buffer_bytes'])}")
    if t['mask_bytes'] > 0:
        print(f"  TDM mask:               {_bytes_str(t['mask_bytes'])}")
    print(f"  Safety factor:          x{t['safety_factor']}")
    print(f"  ==> PEAK: {_bytes_str(report['sram']['training_peak_bytes'])}")