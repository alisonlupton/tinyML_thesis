import torch
torch.backends.quantized.engine = "qnnpack"
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as tq
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from thop import profile as thop_profile
from memory_profiler import memory_usage    
import time
import torch.nn.utils.prune as prune
from avalanche.benchmarks import SplitMNIST
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.supervised import Naive
from avalanche.training import Replay
from avalanche.logging.csv_logger import CSVLogger
import csv
import numpy as np 
import pandas as pd
import psutil
import threading


# -----  Base CNN Model 
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

# Define Functions 
def get_data():
        train_ds = datasets.MNIST("./data", train=True, download=True,
                                transform=transforms.ToTensor())
        test_ds  = datasets.MNIST("./data", train=False, download=True,
                                transform=transforms.ToTensor())
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader  = DataLoader(test_ds, batch_size=1000)
        scenario = SplitMNIST(n_experiences=5)
        return train_loader, test_loader, scenario

def train_on_loader(model, loader, optimizer, criterion, device, epochs):
    model.train()
    for _ in range(epochs):
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            
def eval_model(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds==y).sum().item()
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
        
        metric_dict = results
            
        # convert to CSV
        rows = []
        for model_name, m_dict in results.items():
            row = {"model": model_name}
            row.update(m_dict)
            rows.append(row)

        # get CSV names
        fieldnames = ["model"] + list(next(iter(results.values())).keys())
        with open(f"tinyml_metrics_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return 

# -----  Continual learning
def run_continual(model: nn.Module, trainable_params, scenario, device, csv_logger, replay: bool=False, mem_size: int=200):
    
    evaluator = EvaluationPlugin(
    accuracy_metrics(experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=[csv_logger])
    
    StrategyCls = Replay if replay else Naive
    strategy_args = {
        "model":       model,
        "optimizer":   torch.optim.SGD(trainable_params, lr=1e-2),
        "criterion":   nn.CrossEntropyLoss(),
        "train_mb_size": 2,
        "train_epochs": 1,
        "eval_mb_size": 1000,
        "device":       device,
        "evaluator":    evaluator
    }
    
    if replay:
        strategy_args["mem_size"] = mem_size
    strategy = StrategyCls(**strategy_args)
    
    for exp in scenario.train_stream:
        strategy.train(exp)
        strategy.eval(scenario.test_stream)

    return evaluator


# -----  RAM metrics helper for continual learning
def measure_sram_peak(fn, interval=0.00001):
    """
    Runs fn() in a background thread, samples the current process RSS
    every `interval` seconds, and returns (peak - baseline) in MiB.
    """
    proc = psutil.Process(os.getpid())
    # collect a clean baseline
    import gc; gc.collect()
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
    with open("peak_SRAM_usage.csv", "w", newline="") as fp:
        writer = csv.writer(fp)
        # header
        writer.writerow(["strategy", "mem_delta_MiB"])
        # data rows
        writer.writerow(["SRAM", mem_delta])
    return

# -----  Calculated metrics for continual learning inspired by Towards Lifelong Deep Learning 
def continual_learning_metrics_extended(eval_exp, train_exp, acc, forgetting, model_name = 'model'):
    # breakpoint()
    
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
    with open("CL_metrics_extended.csv", "w", newline="") as f:
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
    
class TinyMLContinualModel(nn.Module):
    def __init__(self, backbone_int8: nn.Module, adapter: nn.Module):
        super().__init__()
        self.backbone = backbone_int8
        # make sure backbone is in eval and frozen
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.adapter = adapter  # only this is trainable

    def forward(self, x):
        # run INT8 backbone up to penultimate layer
        # assume backbone returns logits at fc2, so we'd need to hook earlier.
        # For simplicity let's reimplement forward:
        with torch.no_grad():
            x = self.backbone.quant(x)
            x = F.relu(F.max_pool2d(self.backbone.bn1(self.backbone.conv1(x)), 2))
            x = F.relu(F.max_pool2d(self.backbone.bn2(self.backbone.conv2(x)), 2))
            x = x.flatten(start_dim=1)
            x = self.backbone.fc1(x)
            # DEQUANT here  
            x = self.backbone.dequant(x)
            x = F.relu(x)
        # now pass through adapter head (float)
        return self.adapter(x)

########################################################################################################
                            ############# MAIN FUNCTION LOOP ##########
########################################################################################################
def main():
    
    retrain = False # won't retrain if there is existing model, unless this is set to True

    # Define model paths
    MODEL_PATH           = "mnist_fp32.pth"
    PRUNED_PATH          = "mnist_pruned.pth"
    BACKBONE     = "mnist_backbone_int8.pth"
    HYBRID = "mnist_hybrid.pth"
    model_paths = {
        "FP32":           MODEL_PATH,
        "Pruned":         PRUNED_PATH,
        "Backbone":    BACKBONE,
        "Hybrid": HYBRID
    }
    

    # Begin Pipeline
    
    # ----- 0. Data loading
    device = torch.device("cpu")
    train_loader, test_loader, scenario = get_data()
    
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

            train_on_loader(model_fp32, train_loader,
                        torch.optim.SGD(model_fp32.parameters(),lr=0.01),
                        nn.CrossEntropyLoss(), device, 5)
            torch.save(model_fp32.state_dict(), MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")
    else:
            # --- Train from scratch
            print(f"Training new model: {MODEL_PATH}")

            train_on_loader(model_fp32, train_loader,
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
            model_pruned = prune_and_finetune(model_fp32, train_loader, device,
                                    sparsity=0.5, finetune_epochs=3)
            torch.save(model_pruned.state_dict(), PRUNED_PATH)
            print(f"Model saved to {PRUNED_PATH}")
    else:
            # --- Prune and tune from scratch
            print(f"Training new model: {PRUNED_PATH}")
            model_pruned = prune_and_finetune(model_fp32, train_loader, device,
                                    sparsity=0.5, finetune_epochs=3)
            torch.save(model_pruned.state_dict(), PRUNED_PATH)
            print(f"Model saved to {PRUNED_PATH}")

    # ----- 4. PTQ (from 32 --> 8)
    model_quant = quantize(model_pruned, train_loader, device)
    torch.save(model_quant.state_dict(), BACKBONE)
    print(f"Model saved to {BACKBONE}")
    
    
    # ----- 5. Combine PQT + classifier head
    model_quant.cpu()
    
    # initialise an adapter head 
    adapter = AdapterHead(in_features=128, num_classes=10, bottleneck=32).to(device)
    
    hybrid_model = TinyMLContinualModel(model_quant, adapter).to(device)
    torch.save(hybrid_model.state_dict(), HYBRID)
    print(f"Model saved to {HYBRID}")
 
    # ----- 6. Continual learning
    scenario = SplitMNIST(n_experiences=5)
    csv_logger = CSVLogger(log_folder="avalanche_logs")
       
    # Run Avalanche, which saves basic metrics
    mem_delta = measure_sram_peak(
        lambda: run_continual(
            model=hybrid_model,
            trainable_params=adapter.parameters(),
            scenario=scenario,
            device=device,
            csv_logger=csv_logger,
            replay=True,
            mem_size=200
        ), interval = 0.01 )
    print(f"Replay training RAM delta on host: {mem_delta:.1f} MiB")
    
    calculate_sram(mem_delta)
    print("Saved SRAM usage to peak_SRAM_usage.csv")
    
    # Calculate my own metrics from the basic avalanche logs
    print("Calculating extended CL metrics")
    
    df = pd.read_csv("avalanche_logs/eval_results.csv")
    eval_exp      = df["eval_exp"].to_numpy(dtype=int)
    train_exp     = df["training_exp"].to_numpy(dtype=int)
    acc           = df["eval_accuracy"].to_numpy(dtype=float)
    forgetting    = df["forgetting"].to_numpy(dtype=float)
    
    
    continual_learning_metrics_extended(eval_exp, train_exp, acc, forgetting)
    print("Extended CL metrics saved")

    # ----- 7. Standard Metrics 
    models = {
        "FP32":        model_fp32,
        "Pruned":      model_pruned,
        "Backbone": model_quant,
        "Hybrid": hybrid_model
    }
    tiny_ML_metrics(models, model_paths, train_loader, test_loader, device)
    print("Standard tinyML metrics saved to tinyml_metrics_summary.csv")
    
if __name__ == "__main__":
    main()
    print("Completed successfully!")
    
