
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
from utils import eval_model
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
