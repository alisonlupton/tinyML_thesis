# smaller_good_model.py
import torch.nn as nn
from .cnn_head import ClassRegistry, SimplifiedTDMHead

def CBR3(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )

def CBR1(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )

class TunedMCUStudentCNN(nn.Module):
    """
    M3: Old backbone, cheaper compute
    - 1st conv uses stride=2 (drop first MaxPool)
    - Second 3x3 in each block replaced with 1x1
    - Small width bump in the last stage (to 160) to recover accuracy
    """
    def __init__(self, in_channels=3, feat_dim=128, num_classes=10, img_size=160):
        super().__init__()

        # Create backbone as a single Sequential (this is what Colab code expects)
        self.backbone = nn.Sequential(
            # Stage 1: 160 -> 80 (stride=2)  [no pool here]
            CBR3(in_channels, 32, stride=2),  # downsample early
            CBR1(32, 32),                     # 1x1 instead of second 3x3
            
            # Stage 2: 80 -> 40
            CBR3(32, 64, stride=1),
            CBR1(64, 64),
            nn.MaxPool2d(2, 2),
            
            # Stage 3: 40 -> 20
            CBR3(64, 128, stride=1),
            CBR1(128, 128),
            nn.MaxPool2d(2, 2),
            
            # Stage 4: 20 -> 10  (slight width bump to help accuracy)
            CBR3(128, 160, stride=1),
            CBR1(160, 160),
            nn.MaxPool2d(2, 2),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Dropout(0.20), nn.Linear(160, feat_dim))
        self.classifier = nn.Linear(feat_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _features(self, x):
        x = self.backbone(x)
        x = self.gap(x).flatten(1)  # (N, 160)
        return x

    def forward(self, x):
        feats = self._features(x)
        z = self.proj(feats)             # (N, feat_dim)
        logits = self.classifier(z)      # (N, num_classes)
        return logits, z  # Return both logits and projected features for distillation


class TunedMCUCILCNN_M3(nn.Module):
    """
    CIL-compatible wrapper around the TunedMCUStudentCNN.
    - Keeps your pipeline API (backbone/proj/head + helper forwards)
    - Uses the exact same backbone structure as TunedMCUStudentCNN
    """
    def __init__(self, in_channels, init_num_classes, device, sparsity_ratio, feat_dim):
        super().__init__()
        self.device = device

        # ---- Backbone (exact same as TunedMCUStudentCNN) ----
        self.backbone = nn.Sequential(
            # Stage 1: 160 -> 80 (stride=2)  [no pool here]
            CBR3(in_channels, 32, stride=2),  # downsample early
            CBR1(32, 32),                     # 1x1 instead of second 3x3
            
            # Stage 2: 80 -> 40
            CBR3(32, 64, stride=1),
            CBR1(64, 64),
            nn.MaxPool2d(2, 2),
            
            # Stage 3: 40 -> 20
            CBR3(64, 128, stride=1),
            CBR1(128, 128),
            nn.MaxPool2d(2, 2),
            
            # Stage 4: 20 -> 10  (slight width bump to help accuracy)
            CBR3(128, 160, stride=1),
            CBR1(160, 160),
            nn.MaxPool2d(2, 2),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        # Projection (same as TunedMCUStudentCNN)
        self.proj = nn.Sequential(nn.Dropout(0.20), nn.Linear(160, feat_dim))

        # CIL head with TDM
        self.head = SimplifiedTDMHead(feat_dim, init_num_classes, device, sparsity_ratio)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    # ---- shared feature path ----
    def _features(self, x):
        """
        x: (N, C, H, W) with H=W=160 typically
        Return pre-projection backbone features, shape (N, 160).
        Downsample path: 160→80 (stride2) →40→20→10 (×16 total), then GAP.
        """
        h = self.backbone(x)        # (N, 160, 10, 10) for 160x160 input
        h = self.gap(h).flatten(1)  # (N, 160)
        return h

    # ---- standard forwards used in your pipeline ----
    def forward(self, x):
        feats = self._features(x)
        z = self.proj(feats)                  # (N, feat_dim)
        rows = list(range(self.head.out_dim))
        return self.head.forward_rows(z, rows)

    def forward_task(self, x, registry: ClassRegistry, task_classes):
        feats = self._features(x)
        z = self.proj(feats)
        rows = registry.rows_for_task(task_classes)
        return self.head.forward_rows(z, rows)

    def forward_eval7(self, x, registry: ClassRegistry):
        feats = self._features(x)
        z = self.proj(feats)
        return self.head.forward_eval7(z, registry)

    def expand_head(self, new_class_names, registry: ClassRegistry):
        old_seen = set(registry.seen_classes())
        registry.add_classes(new_class_names)
        added = [c for c in new_class_names if c not in old_seen]
        if added:
            self.head.expand(len(added))

    def forward_rows(self, x, rows):
        feats = self._features(x)
        z = self.proj(feats)
        return self.head.forward_rows(z, rows)