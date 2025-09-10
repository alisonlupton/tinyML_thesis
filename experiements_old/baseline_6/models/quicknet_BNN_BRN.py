# models/quicknet_BNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(__file__))
from batch_renorm import BatchRenorm2d

class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, g_out):
        (x,) = ctx.saved_tensors
        # Pass gradients only for inputs in [-1, 1]
        mask = (x.abs() <= 1).to(g_out.dtype)
        return g_out * mask

def sign_ste(x):  # convenience
    return SignSTE.apply(x)

class BinaryActivation(nn.Module):
    def forward(self, x):
        return sign_ste(x)

class BinaryConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.real_weights = nn.Parameter(torch.randn(out_ch, in_ch // groups, k, k) * 0.05)
        self.stride, self.padding, self.groups = stride, padding, groups
        self.register_buffer("eps", torch.tensor(1e-8))
        self.bias = None  # keep bias off for binary convs (see next point)


    def forward(self, x):
        w = self.real_weights.clamp(-1, 1)
        w_b = sign_ste(w)
        # per-output-channel scaling (mean abs)
        alpha = w.abs().mean(dim=(1,2,3), keepdim=True)
        # Add minimum alpha to prevent dead weights
        alpha = alpha.clamp(min=0.01)
        w_eff = w_b * alpha
        return F.conv2d(x, w_eff, None, self.stride, self.padding, groups = self.groups)
    
class BinaryDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1, padding=0, bias=False):
        super().__init__()
        self.depthwise = BinaryConv2d(in_ch, in_ch, k, stride=stride, padding=padding, groups=in_ch, bias=bias)
        self.pointwise = BinaryConv2d(in_ch, out_ch, 1, bias=bias)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class BinarizedQuickNetBRN(nn.Module):
    def __init__(self, num_classes=10, base_filters=32, blocks_per_stage=2, dropout=0.5, first_layer_fp32=True):
        super().__init__()
        stem_conv = (
            nn.Conv2d(3, base_filters, kernel_size=5, stride=1, padding=2, bias=False)
            if first_layer_fp32 else
            BinaryConv2d(3, base_filters, k=5, stride=1, padding=0, bias=False)
        )
        self.stem = nn.Sequential(
            stem_conv,
            BatchRenorm2d(base_filters, momentum=0.01),
            nn.ReLU(inplace=True) if first_layer_fp32 else BinaryActivation()
        )

        filters = base_filters
        layers = []
        for _ in range(3):
            layers += [
                BinaryDepthwiseSeparableConv(filters, filters*2, k=3, stride = 2, padding=1, bias=False),
                BatchRenorm2d(filters*2, momentum=0.01),
                BinaryActivation()
            ]
            filters *= 2
            for _ in range(blocks_per_stage - 1):
                layers += [
                    BinaryDepthwiseSeparableConv(filters, filters, k=3, stride = 1,  padding=1, bias=False),
                    BatchRenorm2d(filters, momentum=0.01),
                    BinaryActivation()
                ]
        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        # your CWRHead will replace this at runtime
        self.classifier = nn.Linear(filters, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)