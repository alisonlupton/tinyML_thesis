# tuned_mcu_cnn.py
# CIL-compatible version of the tuned MCU student
import torch.nn as nn
from .cnn_head import ClassRegistry, SimplifiedTDMHead

class TunedMCUCILCNN(nn.Module):
    """
    CIL-compatible version of the tuned MCU student
    - Same interface as SimplifiedTDMModelCNN
    - Uses the improved architecture from distillation_mcu_friendly_tuned.py
    - Compatible with CIL training pipeline
    """
    def __init__(self, in_channels, init_num_classes, device, sparsity_ratio, feat_dim):
        super().__init__()
        self.device = device
        
        # Improved backbone with better channel progression (from tuned distillation)
        self.backbone = nn.Sequential(
            # First conv block - more channels for better feature extraction
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Improved projection head with better dropout
        self.proj = nn.Sequential(
            nn.Dropout(0.25),  # Slightly higher dropout for regularization
            nn.Linear(128, feat_dim),  # Map 128-D backbone features to feat_dim
        )
        
        # CIL head (same as SimplifiedTDMModelCNN)
        self.head = SimplifiedTDMHead(feat_dim, init_num_classes, device, sparsity_ratio)
        
        # Better weight initialization
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _features(self, x):
        ''' 
        x: (N, C=3, H, W)
        Return *pre-projection* backbone features, shape (N, 128).
        '''
        h = self.backbone(x)           # (N, 128, H/16, W/16)
        h = self.gap(h).flatten(1)     # (N, 128)
        return h

    # for backbone training
    def forward(self, x):
        feats_128 = self._features(x)
        z = self.proj(feats_128)       # (N, feat_dim)
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
