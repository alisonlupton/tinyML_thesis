import torch
torch.backends.quantized.engine = "qnnpack"
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
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
from utils import load_dataset_custom, make_split_dataset_loaders
from metrics import continual_learning_metrics_extended, tiny_ML_metrics
#### CITE https://github.com/vlomonaco/ar1-pytorch/blob/master/ar1star_lat_replay.py


###################################################################################################################################################
############################################################### CONTINUAL LEARNING Functions ######################################################
###################################################################################################################################################

        

# -----  Continual learning
def train_with_latent_replay(frozen_model,
                             optimizer,
                             loss_fn,
                             task_loaders_train,
                             task_loaders_test,
                             device,
                             replay_size,
                             live_batch,
                             replay_batch,
                             epochs,
                             full_test_dataset
                             ):
                

    replay_size_per_class = replay_size // frozen_model.classifier.out_features
    train_exp, eval_exp, accs, forgettings, overall, overall_full = [], [], [], [], [], []
    classwise_feats = defaultdict(lambda: deque(maxlen=replay_size_per_class)) # dict of classwise deques

    # Helpers to extract latents and run head
    def extract_latent(x):
        # runs through stem/features/global_pool → (B, D)
        with torch.no_grad():
            h = frozen_model.stem(x)
            h = frozen_model.features(h)
            h = frozen_model.global_pool(h)            # (B, C, 1, 1)
            return h.flatten(1)                 # (B, C)

    def classify(latents):
        # runs only classifier head  → (B, n_classes)
        return frozen_model.classifier(latents)
    
    # Loop over tasks
    class_counts = torch.zeros(frozen_model.classifier.out_features, device=device)
    first_acc     = {}
    
    if config["wandb_activate"]:
        print("Using WandB!")
        run = wandb.init(
            project=config["project"],
            name=f"{config['backbone']}_{config['dataset']}_replay_{config['replay_buffer_size']}",
            config=config,
            group=config.get("group", "default"),
        )
    else:
        print("Not using WandB!")
        run = None
    
    
    for t, train_loader in enumerate(task_loaders_train):
        print(f"\n=== Training on Task {t} ===")
        
        # CWR*: For this task, collect the classes the model will get introduced to
        task_classes = set()
        for _, y in train_loader:
            task_classes.update(y.cpu().numpy().tolist())  # add all labels from this batch to the set
        task_classes = list(sorted(task_classes))
        
        # Training on this task
        frozen_model.train()
        
        # CWR*: Re-init the head for this task's classes 
        with torch.no_grad():
            for c in task_classes:
                nn.init.normal_(frozen_model.classifier.weight[c], std=0.01)
                if frozen_model.classifier.bias is not None:
                    frozen_model.classifier.bias[c].zero_()
            
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"Task {t}", unit="batch")
            for x, y in pbar:
                x, y = x.to(device), y.to(device)

                # Live batch
                x_live, y_live = x[:live_batch], y[:live_batch]
                live_feats     = extract_latent(x_live)     # (live_B, D)
                
                # Replay batch
                selected_feats = []
                selected_labels = []
                
                # Sample K classes, S samples per class
                available_classes = [c for c in classwise_feats if len(classwise_feats[c]) > 0]
                if len(available_classes) > 0:
                    num_classes_to_sample = min(5, len(available_classes))
                    samples_per_class = replay_batch // num_classes_to_sample

                    for c in random.sample(available_classes, num_classes_to_sample):
                        feats = random.sample(classwise_feats[c], k=min(samples_per_class, len(classwise_feats[c])))
                        selected_feats.extend(feats)
                        selected_labels.extend([c] * len(feats))

                    replay_feats = torch.stack(selected_feats).to(device)
                    replay_lbls  = torch.tensor(selected_labels, device=device)
                else:
                    replay_feats = None



                # Combine live and replay
                if replay_feats is not None:
                    feats = torch.cat([live_feats, replay_feats], 0)
                    labels = torch.cat([y_live,   replay_lbls], 0)
                else:
                    feats  = live_feats
                    labels = y_live

                # Forward and backward on classifier only
                #TODO: make this tunable later on 
                optimizer.zero_grad()
                logits = classify(feats)
                loss   = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())
                

                # Store live latents into buffer
                with torch.no_grad():
                    for lf, lbl in zip(live_feats.cpu(), y_live.cpu()):
                        classwise_feats[int(lbl)].append(lf.cpu())
        
        
        # CWR*: Check if the stable weight bank has been initialised, if not, create it
        if not hasattr(frozen_model.classifier, "cwr_bank"):
            frozen_model.classifier.cwr_bank = torch.zeros_like(frozen_model.classifier.weight.data)

        # CWR*: Calculate average weight for the current task classes
        alpha = 1.0 / (t + 1)  # running average
        for c in task_classes:
            frozen_model.classifier.cwr_bank[c] = \
                (1 - alpha) * frozen_model.classifier.cwr_bank[c] + alpha * frozen_model.classifier.weight.data[c]
            print(f"Task {t} - class {c} - alpha {alpha:.3f}")
            print("Head weight:", frozen_model.classifier.weight.data[c][:5])
            print("CWR bank   :", frozen_model.classifier.cwr_bank[c][:5])

        # CWR*: Copy back the averaged weights to the model
        with torch.no_grad():
            for c in task_classes:
                frozen_model.classifier.weight[c].copy_(frozen_model.classifier.cwr_bank[c])
                
        # Convert to CPU and NumPy
        bank_np = frozen_model.classifier.cwr_bank.detach().cpu().numpy()

        # Save to CSV
        np.savetxt(f"models/cwr_bank_task{t}.csv", bank_np, delimiter=",", fmt="%.6f")

        #TODO: set seed!
        # Evaluation after task t on all SEEN data
        frozen_model.eval()
        print(f"Starting Evaluation on task: {t}")
        # overall accuracy on all seen tasks
        seen = ConcatDataset([ld.dataset for ld in task_loaders_test[:t+1]])
        seen_loader = DataLoader(seen, batch_size=live_batch+replay_batch, shuffle=False)
        corr = 0; total = 0
        with torch.no_grad():
            for x, y in tqdm(seen_loader, desc="Evaluating on seen data", unit="batch"):
                x, y = x.to(device), y.to(device)
                feats = extract_latent(x)
                preds = classify(feats).argmax(1)
                corr += (preds==y).sum().item()
                total+= y.size(0)
        overall_acc = corr/total
        overall.append((t, overall_acc))
        print(f"After task {t}, overall acc on seen data: {overall_acc:.3f}")
        

        # per-split acc & forgetting
        for u in range(t+1):
            loader = task_loaders_test[u]
            acc = 0; tot=0
            for x,y in loader:
                x,y = x.to(device), y.to(device)
                feats = extract_latent(x)
                acc  += (classify(feats).argmax(1)==y).sum().item()
                tot  += y.size(0)
            acc = acc/tot
            print(f"  Task {u} → acc: {acc:.3f}")
            train_exp.append(t)
            eval_exp.append(u)
            accs.append(acc)
            if u not in first_acc:
                first_acc[u] = acc
                forgettings.append(0.0)
            else:
                forgettings.append(first_acc[u] - acc)
                
                
                
        # Evaluation after task t on ALL data (seen and unseen)
        print(f"Starting full evaluation on task: {t}")
        # overall accuracy on all seen tasks
        full_loader = DataLoader(full_test_dataset, batch_size=live_batch+replay_batch, shuffle=False)
        corr_full = 0; total_full = 0
        with torch.no_grad():
            for x, y in tqdm(full_loader, desc="Evaluating on full CIFAR10", unit="batch"):
                x, y = x.to(device), y.to(device)
                feats = extract_latent(x)
                preds = classify(feats).argmax(1)
                corr_full += (preds==y).sum().item()
                total_full+= y.size(0)
        full_acc = corr_full/total_full
        overall_full.append((t, full_acc))
        if run:
            run.log({
                "full_accuracy": full_acc,
                "Accuracy on seen data": overall_acc,
                "Average Forgetting": np.mean(forgettings),
                "Average accuracy on seen data": np.mean([a for _, a in overall])
            }, step=t)
        print(f"After task {t}, CIFAR10 full acc: {full_acc:.3f}")
        
    if run:
        run.summary["final_accuracy"] = overall_full[-1][1]
        run.summary["avg_accuracy"] = np.mean([a for _, a in overall])
        run.finish()
    return train_exp, eval_exp, accs, forgettings, overall, overall_full


    
########################################################################################################
                            ############# MAIN FUNCTION LOOP ##########
########################################################################################################

def main():
    
    retrain = False # won't retrain if there is existing model, unless this is set to True

    # Define Model paths
    os.makedirs(config["model_folder"], exist_ok=True)
    BACKBONE_PATH = config["backbone_path"]
    model_paths = {
        "Backbone": BACKBONE_PATH
    }
    

    # Begin Pipeline
    
    # ----- 0. Data loading
    device = torch.device(config['device'])
    total_classes = config["num_classes"]
    batch_size_train = config["batch_size"]["train"]
    batch_size_test  = config["batch_size"]["test"]
    
    # Transforms for CIFAR both datasets
    train_transform = transforms.Compose([
    transforms.Pad(4),                             # zero-pad 4 pixels each side
    transforms.RandomCrop(32),                     # random 32×32 crop
    transforms.RandomHorizontalFlip(),             # random flip
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3,             # scale to [–1, +1]
                std=[0.5]*3),
    ])

    test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3,
                std=[0.5]*3),
    ])
    

    if config["dataset"] == "CIFAR10":    
        full_train_dataset = load_dataset_custom(datasets.CIFAR10, True, train_transform)
        full_test_dataset  = load_dataset_custom(datasets.CIFAR10, False, test_transform)
        full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size_train, shuffle=True)
        full_test_loader  = DataLoader(full_test_dataset,  batch_size=batch_size_test, shuffle=False)
    
    elif config["dataset"] == "CIFAR100":
        full_train_dataset = load_dataset_custom(datasets.CIFAR100, True, train_transform)
        full_test_dataset  = load_dataset_custom(datasets.CIFAR100, False, test_transform)
        full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size_train, shuffle=True)
        full_test_loader  = DataLoader(full_test_dataset,  batch_size=batch_size_test   , shuffle=False)
    else:
        raise ValueError("No valid dataset provided")
    
    # ----- 1. Load base model in FP32, trained on tinyimagenet
    if os.path.exists(BACKBONE_PATH):
        FP_backbone = QuickNet(num_classes=200).to(device)
        FP_backbone.load_state_dict(torch.load(BACKBONE_PATH, map_location=device))
        print("FP32 model loaded")
    else:
        raise FileNotFoundError("Couldn't find trained model. Please check.")

    
    # Quick check of backbone 
    # eval_quicknet_backbone(FP_backbone, BACKBONE_PATH, device)

    # ----- 2. Load fp32 backbone into BNN
    
    backbone = BinarizedQuickNet(num_classes=200).to(device)
    
    def load_fp_weights_into_bnn(fp_model, bnn_model):
        fp_dict = fp_model.state_dict()
        bnn_dict = bnn_model.state_dict()
        for name in bnn_dict:
            if name in fp_dict and bnn_dict[name].shape == fp_dict[name].shape:
                bnn_dict[name] = fp_dict[name]
        bnn_model.load_state_dict(bnn_dict, strict=False)
    
    load_fp_weights_into_bnn(FP_backbone, backbone)
    print("Loaded weights into BNN")
    
    # change to match CIFAR10 classes
    backbone.classifier = nn.Linear(backbone.classifier.in_features, total_classes)        
    print(f"Loaded backbone with {config['dataset']} head")

    
    # ----- 3. Freeze backbone 
    for name, param in backbone.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = False      
    backbone = backbone.to(device)
    print("Backbone frozen")
    
    #TODO: fix quantization of classifier! (based on BNN paper)
    
    # ----- 4. Prepare continual learning
    print("Preparing CL experiences!")
    #TODO: for later on , batch size should be 1-2 !
    
    replay_buffer_size = config["replay_buffer_size"] # overall size of my replay buffer
    live_batch = config["live_batch"] # How many new (or “live”) samples to take from each minibatch to train on right now (before mixing in replay)
    replay_batch = config["replay_batch"] # How many old samples to pull from the LR buffer of stored activations.
    epochs = config["epochs"] # Number of training epochs for the CL

    num_tasks = config["num_tasks"] # number of experiences  
    task_loaders_train = make_split_dataset_loaders(
        full_train_dataset,
        n_splits=num_tasks,
        train=True,
        batch_size=live_batch # don't load more samples than we need   
        )
    task_loaders_test = make_split_dataset_loaders(
        full_test_dataset,
        n_splits=num_tasks,
        train=False,
        batch_size=live_batch+replay_batch #TODO: check this   
        )


    optimizer = torch.optim.SGD(backbone.classifier.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    # ----- 5. Perform CL
    
    train_exp, eval_exp, accs, forgettings, overall, overall_full = train_with_latent_replay(
                             frozen_model = backbone,
                             optimizer = optimizer,
                             loss_fn = loss_fn,
                             task_loaders_train = task_loaders_train,
                             task_loaders_test = task_loaders_test,
                             device = device,
                             replay_size = replay_buffer_size,
                             live_batch = live_batch,
                             replay_batch = replay_batch,
                             epochs = epochs,
                             full_test_dataset = full_test_dataset
                             )
    
    tasks, accuracy = zip(*overall_full)
    plt.plot(tasks, accuracy, marker = 'o')
    plt.xlabel('Task')
    plt.ylabel(f'Overall {config["dataset"]} accuracy')
    plt.grid(True)
    plt.show()
    
    
    # training_results = {}
    # print("Starting CL experiences")
    # def do_cl():
    #     training_results['outs'] = train_with_latent_replay(
    #     backbone,  # 
    #     adapter_opt=optimizer,
    #     loss_fn=loss_fn,
    #     task_loaders_train=task_loaders_train,
    #     task_loaders_test=task_loaders_test,
    #     device=device,
    #     replay_size=replay_buffer_size,
    #     live_B= live_batch,
    #     replay_R=replay_batch,
    #     # here we inject the scale & zero_point so your function
    #     # can quantize+dequantize on the fly
    #     float_adapter=float_adapter,
    #     full_test_ds = full_test_ds,
    #     quant_params=(scale, zero_point)
    # )

    # mem_delta = measure_sram_peak(do_cl, interval=0.01)
          
    # print(f"Replay training RAM delta on host: {mem_delta:.1f} MiB")
    # record_sram_to_csv(mem_delta)
    # print("Saved SRAM usage to peak_SRAM_usage.csv")
    
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
    continual_learning_metrics_extended(eval_exp, train_exp, accs, forgettings, model_name="Hybrid")
    print("CL extended metrics saved to CL_metrics_extended.csv")

    # ----- 7. Standard Metrics 
    models = {
        "Backbone": backbone        
    }
    tiny_ML_metrics(models, model_paths, full_test_loader, device)
    print("Standard tinyML metrics saved to tinyml_metrics_summary.csv")

if __name__ == "__main__":
    with open('CL_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    print("Loaded config file")
    assert config is not None, "Config file failed to load!"
    main()
    print("Completed successfully!")
    