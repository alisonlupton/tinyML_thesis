
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
    