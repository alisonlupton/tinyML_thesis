import torch.nn as nn
from .cnn_head import ClassRegistry, SimplifiedTDMHead

class SimplifiedTDMModelCNN(nn.Module):
    def __init__(self, in_channels, init_num_classes, device, sparsity_ratio=0.3):
        super().__init__()
        self.device = device

        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = SimplifiedTDMHead(32, init_num_classes, device, sparsity_ratio)

    def _features(self, x):
        h = self.backbone(x)
        return self.gap(h).squeeze(-1)  # (N, 32)

    # for backbone training
    def forward(self, x):
        feats = self._features(x)
        rows = list(range(self.head.out_dim))        # all current rows
        return self.head.forward_rows(feats, rows)
    # task-time forward (only current classes)
    def forward_task(self, x, registry: ClassRegistry, task_classes):
        feats = self._features(x)
        rows = registry.rows_for_task(task_classes)
        return self.head.forward_rows(feats, rows)

    # eval-time forward (7-way assembled on the fly)
    def forward_eval7(self, x, registry: ClassRegistry):
        feats = self._features(x)
        return self.head.forward_eval7(feats, registry)

    def expand_head(self, new_class_names, registry: ClassRegistry):
        old_seen = set(registry.seen_classes())
        registry.add_classes(new_class_names)
        added = [c for c in new_class_names if c not in old_seen]
        if len(added) > 0:
            self.head.expand(len(added))
    # in SimplifiedTDMModelCNN
    def forward_rows(self, x, rows):
        feats = self._features(x)
        return self.head.forward_rows(feats, rows)