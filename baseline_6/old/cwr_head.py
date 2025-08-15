# cwr_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CWRHead(nn.Module):
    def __init__(self, in_features, num_classes, device):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        # Temporary weights/bias used during *training* forward
        self.weight = nn.Parameter(torch.zeros(num_classes, in_features, device=device))
        self.bias   = nn.Parameter(torch.zeros(num_classes, device=device))
        # Consolidated weights/bias used for *inference*
        self.register_buffer("cw", torch.zeros(num_classes, in_features, device=device))
        self.register_buffer("cb", torch.zeros(num_classes, device=device))
        # Running sample counts per class
        self.register_buffer("hist_count", torch.zeros(num_classes, dtype=torch.long, device=device))
        self._dbg_printed = False

    def forward(self, x):
        # During training you’ll use tw/tb; during eval you’ll call use_consolidated()
        return F.linear(x, self.weight, self.bias)

    @torch.no_grad()
    def preload_cw(self, present_classes):
        """Reset tw/tb to zero, then preload cw/cb for present classes only."""
        self.weight.zero_()
        self.bias.zero_()
        if len(present_classes) > 0:
            idx = torch.as_tensor(present_classes, device=self.weight.device, dtype=torch.long)
            self.weight.index_copy_(0, idx, self.cw.index_select(0, idx))
            self.bias.index_copy_(0, idx, self.cb.index_select(0, idx))
        if not self._dbg_printed:
            print("[CWR] preload first call, present:", present_classes[:20])
            self._dbg_printed = True

    @torch.no_grad()
    def consolidate(self, present_classes, counts_present):
        """
        Weighted update of consolidated weights for classes seen in this minibatch.
        cw[c] <- (cw[c]*hist + tw[c]*n_batch) / (hist + n_batch)
        """
        if len(present_classes) == 0:
            return
        idx = torch.as_tensor(present_classes, device=self.weight.device, dtype=torch.long)
        n_hist = self.hist_count.index_select(0, idx).to(self.weight.dtype) # how many samples have contributed to each cw row before this batch

        n_new  = counts_present.to(self.weight.dtype) # how many this batch contributes per class 

        # Avoid div by zero
        denom = (n_hist + n_new).clamp_min(1)

            
        cw_old = self.cw.index_select(0, idx)                  # (K, D), consolidated rows before this batch
        tw_now = self.weight.index_select(0, idx)              # tw rows (K, D), temporary rows (what SGD just learned this batch)

        # Weighted average fuse 
        ah = (n_hist / denom).unsqueeze(1)                     # (K, 1), old
        an = (n_new  / denom).unsqueeze(1)                     # (K, 1), new
        cw_new = ah * cw_old + an * tw_now                     # (K, D)

        # If any row is (near) zero, fall back to old row to avoid NaNs on normalize (safety line)
        zero_mask = (cw_new.norm(dim=1) < 1e-12)
        if zero_mask.any():
            cw_new[zero_mask] = cw_old[zero_mask]

        # **Cosine head requirement: unit-norm rows**
        cw_new = F.normalize(cw_new, dim=1)                    # (K, D), safe with eps internally

        # (Optional) If you truly run bias-free cosine head, keep cb at 0 #TODO: fix the bias stuff later
        if self.bias is not None and self.cb is not None:
            cb_old = self.cb.index_select(0, idx)
            bb_now = self.bias.index_select(0, idx)
            cb_new = (cb_old * n_hist + bb_now * n_new) / denom
            # For cosine head, bias usually 0; comment next line if you want to keep cb
            cb_new.zero_()
            self.cb.index_copy_(0, idx, cb_new)

        # Write back & update counts
        self.cw.index_copy_(0, idx, cw_new)
        self.hist_count.index_put_((idx,), (n_hist + n_new).to(self.hist_count.dtype))

        if not hasattr(self, "_dbg_printed2"):
            print("[CWR] consolidate first call: classes:", present_classes[:20],
                "counts:", counts_present[:20].tolist())
            self._dbg_printed2 = True

    @torch.no_grad()
    def use_consolidated(self):
        """Load cw/cb into tw/tb (for evaluation)."""
        self.weight.copy_(self.cw)
        self.bias.copy_(self.cb)
        
        
        
        
## DEBUGGING STRONGER HEADS!!

class CosineClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.kaiming_normal_(self.weight, nonlinearity='linear')
        self.num_classes = num_classes

    def forward(self, x):
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        return F.linear(x, w)
    
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dim=None):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim or in_features

        self.fc1 = nn.Linear(in_features, self.hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.hidden_dim, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    @property
    def weight(self):
        return self.fc2.weight

    @property
    def bias(self):
        return self.fc2.bias