# cnn.py
import torch.nn as nn
from .cnn_head import ClassRegistry, SimplifiedTDMHead

class SimplifiedTDMModelCNN(nn.Module):
    def __init__(self, in_channels, init_num_classes, device, sparsity_ratio, feat_dim):
        super().__init__()
        self.device = device
        # note that when set in .eval() the dropout is identity
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128 , 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Dropout2d(0.2),
            nn.Conv2d(128, 192, 3, stride=2, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True), nn.Dropout2d(0.2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(192, feat_dim)
            # nn.BatchNorm1d(feat_dim)
        )
        self.head = SimplifiedTDMHead(feat_dim, init_num_classes, device, sparsity_ratio)

    def _features(self, x):
        ''' 
        x: (N, C=3, H, W)
        Return *pre-projection* backbone features, shape (N, 192).
        '''

        h = self.backbone(x)           # (N, 192, H/16, W/16)
        h = self.gap(h).flatten(1)     # (N, 192)
        return h


    # for backbone training
    def forward(self, x):
        feats_192 = self._features(x)
        z = self.proj(feats_192)       # (N, feat_dim)
        rows = list(range(self.head.out_dim))
        return self.head.forward_rows(z, rows)

    # task-time forward (only current classes)
    def forward_task(self, x, registry: ClassRegistry, task_classes):
        feats = self._features(x)
        z = self.proj(feats)
        rows = registry.rows_for_task(task_classes)
        return self.head.forward_rows(z, rows)


    # eval-time forward (7-way assembled on the fly)
    def forward_eval7(self, x, registry: ClassRegistry):
        feats= self._features(x)
        z = self.proj(feats)
        return self.head.forward_eval7(z, registry)

    def expand_head(self, new_class_names, registry: ClassRegistry):
        old_seen = set(registry.seen_classes())
        registry.add_classes(new_class_names)
        added = [c for c in new_class_names if c not in old_seen]
        if len(added) > 0:
            self.head.expand(len(added))
    # in SimplifiedTDMModelCNN
    def forward_rows(self, x, rows):
        feats = self._features(x)
        z = self.proj(feats)
        return self.head.forward_rows(z, rows)