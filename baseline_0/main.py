import torch
torch.backends.quantized.engine = "qnnpack"
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as tq
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import psutil
from thop import profile
import time
import torch.nn.utils.prune as prune
from avalanche.benchmarks import SplitMNIST
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.supervised import Naive
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
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total      
          
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
        
        # MACs & Params (remember Multiply–Accumulate operations --> quantify the compute cost of a NN layer)
        macs, params = profile(m, inputs=(torch.randn(1,1,28,28),))
        macs_m = macs / 1e6 # measured in millions/kilo
        params_k = params / 1e3 # measured in millions/kilo
        
        mean, std = latency(m)
        lat_str = f"{mean:.2f}±{std:.2f}"
        
        results[name] = {
        "Acc":        acc,
        "Flash (KB)": flash,
        "RAM (KB)":   ram,
        "MACs (M)":   macs_m,
        "Params (K)": params_k,
        "Latency":    lat_str
        }
    return results
        
def run_continual(model, scenario, device, csv_logger):
    evaluator = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[csv_logger]  
    )
    strategy = Naive(model, torch.optim.SGD(model.parameters(),lr=0.01),
                     nn.CrossEntropyLoss(),
                     train_mb_size=64, train_epochs=1,
                     eval_mb_size=1000, device=device,
                     evaluator=evaluator)
    for exp in scenario.train_stream:
        strategy.train(exp)
        strategy.eval(scenario.test_stream)
    return evaluator

########################################################################################################
                            ############# MAIN FUNCTION LOOP ##########
########################################################################################################
def main():
    
    retrain = False # won't retrain if there is existing model, unless this is set to True

    # Define model paths
    MODEL_PATH           = "mnist_fp32.pth"
    PRUNED_PATH          = "mnist_pruned.pth"
    PRUNED_INT8_PATH     = "mnist_pruned_int8.pth"
    model_paths = {
        "FP32":           MODEL_PATH,
        "Pruned":         PRUNED_PATH,
        "Pruned+INT8":    PRUNED_INT8_PATH
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
            train_on_loader(model_fp32, train_loader,
                        torch.optim.SGD(model_fp32.parameters(),lr=0.01),
                        nn.CrossEntropyLoss(), device, 5)
            torch.save(model_fp32.state_dict(), MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")
    
    # ----- 3 Prune FP32 and retrain (currently applying unstructured pruning so I won't see benefits)
    model_pruned = TinyCNN().to(device)
    if not retrain:
        if os.path.exists(PRUNED_PATH):
            print(f"Found existing model at {PRUNED_PATH}, skipping training.")
            model_pruned.load_state_dict(torch.load(PRUNED_PATH))
        else:
            # --- Prune and tune from scratch
            model_pruned = prune_and_finetune(model_fp32, train_loader, device,
                                    sparsity=0.5, finetune_epochs=3)
            torch.save(model_pruned.state_dict(), PRUNED_PATH)
            print(f"Model saved to {PRUNED_PATH}")

    # ----- 4. PTQ (from 32 --> 8)
    model_quant = quantize(model_pruned, train_loader, device)
    torch.save(model_quant.state_dict(), PRUNED_INT8_PATH)
    print(f"Model saved to {PRUNED_INT8_PATH}")
    
    # ----- 4.1 TEST PURE QUANTISED 
    
    model_quant.eval()
    scenario = SplitMNIST(n_experiences=5)
    for task_id, experience in enumerate(scenario.test_stream):
        loader = DataLoader(experience.dataset, batch_size=1000)
        acc = eval_model(model_quant, loader, device)
        print(f"Split {task_id}: {acc*100:.2f}%")
        
    breakpoint()
    
    # ----- 5. Metrics 
    models = {
        "FP32":        model_fp32,
        "Pruned":      model_pruned,
        "Pruned+INT8": model_quant
    }
    metrics = calculate_metrics(models, model_paths, train_loader, test_loader, device)

    quantized_metric_dict = metrics
    
    # Save to JSON
    with open("quantized_metrics.json", "w") as f:
        json.dump(quantized_metric_dict, f, indent=2)
        
    # convert to CSV
    rows = []
    for model_name, m_dict in metrics.items():
        row = {"model": model_name}
        row.update(m_dict)
        rows.append(row)

    # get CSV names
    fieldnames = ["model"] + list(next(iter(metrics.values())).keys())
    with open(f"quantized_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Quantized metrics saved to a quantized_metrics.csv")
    # print(f"Static metrics:{metrics}")
    print("Metrics saved sucessfuly")
    
    # ----- 6. Continual learning
    csv_logger = CSVLogger(
        log_folder="avalanche_logs"
    )
    run_continual(model_pruned, scenario, device, csv_logger)
    print("Continual metrics saved")
    
    
if __name__ == "__main__":
    main()
    print("Completed successfully!")
    
