# cnn.py
import torch.nn as nn
from .cnn_head import ClassRegistry, SimplifiedTDMHead

class SimplifiedTDMModelCNN(nn.Module):
    def __init__(self, in_channels, init_num_classes, device, sparsity_ratio, feat_dim):
        super().__init__()
        self.device = device

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 24, 3, stride=2, padding=1), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 128,3, stride=2, padding=1), nn.BatchNorm2d(128),nn.ReLU(inplace=True),
            nn.Conv2d(128,256,3, stride=2, padding=1), nn.BatchNorm2d(256),nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1, 1)
        self.proj = nn.Linear(256, feat_dim)      # project to feature dim used by the head

        self.head = SimplifiedTDMHead(feat_dim, init_num_classes, device, sparsity_ratio)

    def _features(self, x):
        ''' 
        x: (N, C=3, H, W)
        returns: (N, feat_dim)
        '''
        h = self.backbone(x)            # (N, 256, H/32, W/32) for img_size≳160
        h = self.gap(h).flatten(1)      # (N, 256)
        z = self.proj(h)                # (N, feat_dim)
        return z

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