import torch
torch.backends.quantized.engine = "qnnpack"
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as tq
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import psutil
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
import json
import csv


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
def calculate_metrics(models, model_paths, train_loader, test_loader, device):  
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
    return results

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
        "train_mb_size": 16,
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


# -----  Metrics helper for continual learning
def measure_continual(mem_func, interval=0.1):
    # Gives peak_rss_delta (in MiB)

    trace = memory_usage((mem_func, ), interval=interval, retval=False)
    base = trace[0]
    peak = max(trace)
    return peak - base

# Classifier head for ODL, adds down projection, relu, up projection to get some CL involved. 
# Gets added to my quantised base model ;
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
    os.makedirs('models', exist_ok= True)
    # Define model paths
    MODEL_PATH           = "models/mnist_fp32.pth"
    PRUNED_PATH          = "models/mnist_pruned.pth"
    BACKBONE     = "models/mnist_backbone_int8.pth"
    HYBRID = "vmnist_hybrid.pth"
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
    scenario   = SplitMNIST(n_experiences=5)
    csv_logger = CSVLogger(log_folder="avalanche_logs_adapters")
    csv_logger_replay = CSVLogger(log_folder="avalanche_logs_adapters_replay")

    # Naïve adapter-only
    def profiled_replay_run():
        # this calls exactly what you do for replay‐enabled continual learning
        run_continual(
            model=hybrid_model,
            trainable_params=adapter.parameters(),
            scenario=scenario,
            device=device,
            csv_logger=csv_logger,
            replay=False
        )
    naive_mem_delta = measure_continual(profiled_replay_run, interval=0.05)
    print(f"Naïve training RAM delta on host: {naive_mem_delta:.1f} MiB")


    # Replay adapter-only
    def profiled_replay_run():
        # this calls exactly what you do for replay‐enabled continual learning
        run_continual(
            model=hybrid_model,
            trainable_params=adapter.parameters(),
            scenario=scenario,
            device=device,
            csv_logger=csv_logger_replay,
            replay=True,
            mem_size=200
        )
    replay_mem_delta = measure_continual(profiled_replay_run, interval=0.05)
    print(f"Replay training RAM delta on host: {replay_mem_delta:.1f} MiB")
    
    os.makedirs("metrics", exist_ok= True)
    with open("metrics/continual_memory_usage.csv", "w", newline="") as fp:
        writer = csv.writer(fp)
        # header
        writer.writerow(["strategy", "mem_delta_MiB"])
        # data rows
        writer.writerow(["naive",  f"{naive_mem_delta:.2f}"])
        writer.writerow(["replay", f"{replay_mem_delta:.2f}"])
    
    print("Wrote continual_memory_usage.csv")
    print("Continual metrics saved")
    
    
    
    # ----- 7. Metrics 
    models = {
        "FP32":        model_fp32,
        "Pruned":      model_pruned,
        "Backbone": model_quant,
        "Hybrid": hybrid_model
    }
    metrics = calculate_metrics(models, model_paths, train_loader, test_loader, device)

    quantized_metric_dict = metrics
    
    # Save to JSON
    with open("metrics/quantized_metrics.json", "w") as f:
        json.dump(quantized_metric_dict, f, indent=2)
        
    # convert to CSV
    rows = []
    for model_name, m_dict in metrics.items():
        row = {"model": model_name}
        row.update(m_dict)
        rows.append(row)

    # get CSV names
    fieldnames = ["model"] + list(next(iter(metrics.values())).keys())
    with open(f"metrics/quantized_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Quantized metrics saved to a quantized_metrics.csv")
    # print(f"Static metrics:{metrics}")
    print("Metrics saved sucessfuly")
    
    
    
if __name__ == "__main__":
    main()
    print("Completed successfully!")
    
