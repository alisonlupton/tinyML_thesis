
# -----  Helper to measure zero-shot baseline accuracy of the frozen, quantized backbone on a new dataset
# def backbone_zero_shot_baseline(backbone_model: nn.Module, device, test_loader, full_test_dataset,
#                                 num_splits
#                                 ):
#     """
#     Build a zero‐trained head on top of your frozen backbone, and
#     evaluate it on each of the "SplitMNIST" splits defined above.
#     """
#     # head that predicts uniformly at random (logits all equal)
#     # full 100-way uniform head
#     num_classes = len(full_test_dataset.classes)
#     head = nn.Linear(128, num_classes, bias=True)
#     head.weight.data.zero_()
#     head.bias.data.fill_(math.log(1/num_classes))
#     head.requires_grad_(False)



#     cl_model = TinyMLContinualModel(backbone_model, head).to(device).eval()

#     for task_id, loader in enumerate(test_loader):
#         # compute which labels belong to this split
#         per_split = num_classes // len(test_loader)
#         lo = task_id * per_split
#         hi = lo + per_split
#         mask = torch.full((num_classes,), float("-inf"), device=device)
#         mask[lo:hi] = 0.0

#         correct = total = 0
#         with torch.no_grad():
#             for x, y in loader:
#                 x, y = x.to(device), y.to(device)
#                 logits = cl_model(x)          # shape [B,100]
#                 masked = logits + mask        # unseen classes stay at -inf
#                 preds = masked.argmax(dim=1)
#                 correct += (preds == y).sum().item()
#                 total   += y.size(0)

#         print(f"Zero‐shot on split {task_id}: {100*correct/total:.2f}%")

# # -----  Evaluator for CL model where classes must be filtered
# def eval_continual_model(model, loader, seen_labels, device):
#     model = model.to(device).eval()
#     correct = total = 0
#     seen_labels = torch.tensor(seen_labels, device=device)

#     with torch.inference_mode():
#         for x, y in loader:
#             x, y = x.to(device), y.to(device)
#             logits = model(x)                         # shape (B,100)
#             # create a -inf mask, then fill seen positions from logits
#             mask = torch.full_like(logits, float("-inf"))
#             mask[:, seen_labels] = logits[:, seen_labels]
#             preds = mask.argmax(dim=1)
#             correct += (preds == y).sum().item()
#             total   += y.size(0)

#     return correct / total 

# def prune_and_finetune(model, train_loader, device, sparsity, finetune_epochs):
#     # Collect all conv and fc layers
#     to_prune = []
#     for module in model.modules():
#         if isinstance(module, (nn.Conv2d, nn.Linear)):
#             to_prune.append((module, 'weight'))

#     # Apply global unstructured pruning (using torch pruning)
#     prune.global_unstructured(
#         to_prune,
#         pruning_method=prune.L1Unstructured,
#         amount=sparsity,
#     )

#     # Remove pruning reparam so weights are truly zero
#     for module, name in to_prune:
#         prune.remove(module, name)

#     # Fine-tune training
#     optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=0.01)
#     criterion = nn.CrossEntropyLoss()

#     train_on_loader(model, train_loader, optimizer, criterion, device, finetune_epochs)
    
#     return model

# def quantize(model, test_loader, device):
#     model.eval()
#     model.fuse_model()
#     model.qconfig = tq.get_default_qconfig("qnnpack")
#     tq.prepare(model, inplace=True)
#     with torch.no_grad():
#         for i,(x,_) in enumerate(test_loader):
#             model(x.to(device))
#             if i>=10: break
#     return tq.convert(model.eval(), inplace=False)
# def calibrate_feature_range(backbone: nn.Module,
#                              quant_stub: tq.QuantStub,
#                              dequant_stub: tq.DeQuantStub,
#                              calib_loader: DataLoader,
#                             device: torch.device,
#                             num_batches: int = 10):
#     """
#    "Runs a few batches through the backbone up to `fc1` to
#    find global min/max of features.
#     """
#     backbone.eval()
#     feat_min, feat_max = float("inf"), float("-inf")

#     with torch.no_grad():
#         for i, (x, _) in enumerate(calib_loader):
#             x = x.to(device)
#             z = quant_stub(x)
#             z = F.relu(F.max_pool2d(backbone.bn1(backbone.conv1(z)), 2))
#             z = F.relu(F.max_pool2d(backbone.bn2(backbone.conv2(z)), 2))
#             z = z.flatten(1)
#             z = backbone.fc1(z)
#             feats = dequant_stub(z)
#             b_min, b_max = feats.min().item(), feats.max().item()
#             feat_min = min(feat_min, b_min)
#             feat_max = max(feat_max, b_max)
#             if i + 1 >= num_batches:
#                 break
            
#     print(f"Calibrated feature range: [{feat_min:.4f}, {feat_max:.4f}]")

#     return feat_min, feat_max

# # Classifier head for ODL, adds down projection, relu, up projection to get some CL involved. 
# # Gets added to my quantised base model 
# class AdapterHead(nn.Module):
#     def __init__(self, in_features=128, num_classes=10, bottleneck=32):
#         super().__init__()
#         # a low-rank adapter: down-project → nonlinearity → up-project
#         self.down = nn.Linear(in_features, bottleneck)
#         self.relu = nn.ReLU(inplace=True)
#         self.up   = nn.Linear(bottleneck, num_classes)
#     def forward(self, x):
#         # x is shape (batch, 128)
#         x = self.down(x)
#         x = self.relu(x)
#         return self.up(x)
    
    
    
#### EWC STUFF ###

# # Add EWC regularization if enabled (enabled in all epochs to protect old knowledge)
#                     if config['EWC'] and len(fisher_info) > 0:
#                         # Balanced EWC strength: moderate protection in early tasks
#                         adaptive_ewc_lambda = config['ewc_lambda'] * (0.1 + 0.9 * t / config['num_tasks'])
#                         ewc_loss_val = ewc_loss(frozen_model, fisher_info, old_params, adaptive_ewc_lambda)
#                         total_loss += ewc_loss_val
                        
                        
        
#         # Update EWC Fisher information and old parameters for next task
#         if config['EWC']:
#             # print(f"Computing Fisher information for task {t}...")
#             # Create a temporary dataloader for Fisher computation
#             temp_dataset = ConcatDataset([ld.dataset for ld in task_loaders_train[:t+1]])
#             temp_loader = DataLoader(temp_dataset, batch_size=32, shuffle=True)
            
#             fisher_info = compute_fisher_information(
#                 frozen_model, temp_loader, device, config['ewc_fisher_samples']
#             )
            
#             # Store current parameters as old parameters for next task
#             old_params = {}
#             for name, param in frozen_model.named_parameters():
#                 if param.requires_grad:
#                     old_params[name] = param.data.clone()
            
#             print(f"EWC Fisher info computed for task {t}")
            

# def compute_fisher_information(model, dataloader, device, num_samples=100):
#     """
#     Compute Fisher information matrix for EWC.
#     Returns diagonal approximation of Fisher matrix.
#     """
#     model.eval()
#     fisher_info = {}
    
#     # Initialize Fisher info with zeros
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             fisher_info[name] = torch.zeros_like(param.data)
    
#     # Sample data to estimate Fisher information
#     samples_processed = 0
#     for batch_idx, (data, target) in enumerate(dataloader):
#         if samples_processed >= num_samples:
#             break
            
#         data, target = data.to(device), target.to(device)
        
#         # Forward pass
#         model.zero_grad()
#         output = model(data)
#         loss = F.cross_entropy(output, target)
#         loss.backward()
        
#         # Accumulate Fisher information (squared gradients)
#         for name, param in model.named_parameters():
#             if param.requires_grad and param.grad is not None:
#                 fisher_info[name] += param.grad.data ** 2
        
#         samples_processed += data.size(0)
    
#     # Average over samples
#     for name in fisher_info:
#         fisher_info[name] /= samples_processed
    
#     return fisher_info
  
# def ewc_loss(model, fisher_info, old_params, lambda_ewc):
#     """
#     Compute EWC loss to prevent forgetting.
#     """
#     ewc_loss_val = 0.0
    
#     for name, param in model.named_parameters():
#         if param.requires_grad and name in fisher_info and name in old_params:
#             fisher_diag = fisher_info[name]
#             old_param = old_params[name]
#             ewc_loss_val += (fisher_diag * (param - old_param) ** 2).sum()
    
#     return lambda_ewc * ewc_loss_val
#     # Initialize EWC storage
#     fisher_info = {}
#     old_params = {}


### AR1 STUFF ###

# def ar1_regularization(model, task_id, ar1_lambda, ar1_alpha):
#     """
#     AR1* adaptive regularization based on task similarity.
#     """
#     ar1_loss = 0.0
    
#     # Simple AR1* implementation: L2 regularization with adaptive strength
#     for name, param in model.named_parameters():
#         if param.requires_grad and 'classifier' in name:
#             # Adaptive regularization strength based on task progress
#             adaptive_lambda = ar1_lambda * (1 + ar1_alpha * task_id)
#             ar1_loss += adaptive_lambda * (param ** 2).sum()
    
#     return ar1_loss
# # Add AR1* regularization if enabled (enabled in all epochs to protect old knowledge)
#                     if config['AR1']:
#                         # Balanced AR1 strength: moderate protection in early tasks
#                         adaptive_ar1_lambda = config['ar1_lambda'] * (0.1 + 0.9 * t / config['num_tasks'])
#                         ar1_loss_val = ar1_regularization(frozen_model, t, adaptive_ar1_lambda, config['ar1_alpha'])
#                         total_loss += ar1_loss_val



### EVALUATION TO PUT BACK IN
            # Evaluation after task t on ALL data (seen and unseen)
            # print(f"Starting full evaluation on task: {t}")
            # # overall accuracy on all seen tasks
            # full_loader = DataLoader(
            #     full_test_dataset, batch_size=live_batch + replay_batch, shuffle=False
            # )
            # corr_full = 0
            # total_full = 0
            # with torch.no_grad():
            #     for x, y in tqdm(
            #         full_loader, desc="Evaluating on full CIFAR10", unit="batch"
            #     ):
            #         x, y = x.to(device), y.to(device)
            #         feats = extract_latent(x)
            #         preds = classify(feats).argmax(1)
            #         corr_full += (preds == y).sum().item()
            #         total_full += y.size(0)
            # full_acc = corr_full / total_full
            # overall_full.append((t, full_acc))
            # if run:
            #     run.log(
            #         {
            #             "full_accuracy": full_acc,
            #             "Accuracy on seen data": overall_acc,
            #             "Average Forgetting": np.mean(forgettings),
            #             "Average accuracy on seen data": np.mean([a for _, a in overall]),
            #         },
            #         step=t,
            #     )
            # print(f"After task {t}, CIFAR10 full acc: {full_acc:.3f}")
            
    # Evaluation after task t on all SEEN data
    # if t > 0:
    #     frozen_model.eval()
    #     print(f"\n{'='*30} STARTING EVALUATION — TASK {t} {'='*30}")
    #     # overall accuracy on all seen tasks
    #     seen = ConcatDataset([ld.dataset for ld in task_loaders_test[: t + 1]])
    #     seen_loader = DataLoader(
    #         seen, batch_size=live_batch + replay_batch, shuffle=False
    #     )
    #     corr = 0
    #     total = 0

    #     pred_counter = Counter()
    #     label_counter = Counter()
    #     with torch.no_grad():
    #         for x, y in tqdm(seen_loader, desc="Evaluating on seen data", unit="batch"):
    #             x, y = x.to(device), y.to(device)
    #             feats = extract_latent(x)
    #             preds = forward_tail_to_logits(tail, backbone, lr_maps).argmax(1)
    #             corr += (preds == y).sum().item()
    #             total += y.size(0)
    #             pred_counter.update(preds.cpu().numpy().tolist())
    #             label_counter.update(y.cpu().numpy().tolist())
    #     print(
    #         f"True label distribution on seen test data for task {t}: {label_counter}"
    #     )
    #     print(f"Predictions on Task {t} test set: {pred_counter}")
    #     overall_acc = corr / total
    #     overall.append((t, overall_acc))
    #     print(f"After task {t}, overall acc on seen data: {overall_acc:.3f}")
    
    
    
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