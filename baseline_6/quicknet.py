import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias
        )
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class QuickNet(nn.Module):
    def __init__(self, num_classes=200, base_filters=32, blocks_per_stage=2, dropout=0.5):
        super().__init__()
        # Entry stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_filters, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.PReLU()
        )
        print(f"hello starting stem basefilters: {base_filters}")

        # Build separable-conv stages: double channels at start of each stage
        # Build separable‐conv stages
        filters = base_filters            # start at 32
        layers = []
        filters = base_filters   # 32
        for stage in range(3):
            # first block in this stage actually upsamples
            layers += [
                DepthwiseSeparableConv(filters, filters*2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(filters*2),
                nn.PReLU()
            ]
            filters *= 2           # 64, then 128, then 256
            for _ in range(blocks_per_stage-1):
                layers += [
                    DepthwiseSeparableConv(filters, filters, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(filters),
                    nn.PReLU()
                ]
            print(f"hello again stem basefilters: {filters}")
        self.features = nn.Sequential(*layers)

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(filters, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
