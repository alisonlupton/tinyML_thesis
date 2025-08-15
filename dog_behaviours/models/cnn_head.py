# cnn_head.py
import torch
import torch.nn as nn
import math
from utils import ClassRegistry
# ----- Class Indexing

# ----------------- SPARCL TDM Implementation    
def _init_linear_rows(linear, start_row, end_row):
    nn.init.kaiming_uniform_(linear.weight.data[start_row:end_row], a=math.sqrt(5))
    linear.bias.data[start_row:end_row].zero_()

class SimplifiedTDMHead(nn.Module):
    """A standalone linear head with TDM/CWI that can grow rows dynamically."""
    def __init__(self, in_dim, init_num_classes, device, sparsity_ratio=0.30):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = init_num_classes
        self.device = device
        self.sparsity_ratio = sparsity_ratio

        self.linear = nn.Linear(in_dim, init_num_classes).to(device)
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        nn.init.zeros_(self.linear.bias)

        # TDM mask + CWI
        self.mask = torch.ones_like(self.linear.weight.data, device=device)
        self._apply_initial_sparsity(self.mask, self.sparsity_ratio)
        self.linear.weight.data *= self.mask

        self.cwi = torch.zeros_like(self.linear.weight.data, device=device)  # importance accumulator

    @staticmethod
    def _apply_initial_sparsity(mask, sparsity_ratio):
        with torch.no_grad():
            flat = mask.view(-1)
            k = int(flat.numel() * sparsity_ratio)
            idx = torch.randperm(flat.numel(), device=mask.device)[:k]
            flat[idx] = 0

    def expand(self, n_new):
        """Append n_new rows, init params, mask, and CWI for only new rows."""
        if n_new <= 0: return
        old_out = self.out_dim
        new_out = old_out + n_new

        # grow weight/bias
        new_linear = nn.Linear(self.in_dim, new_out).to(self.device)
        with torch.no_grad():
            new_linear.weight[:old_out].copy_(self.linear.weight)
            new_linear.bias[:old_out].copy_(self.linear.bias)
        _init_linear_rows(new_linear, old_out, new_out)

        # swap
        self.linear = new_linear
        self.out_dim = new_out

        # grow mask and cwi
        new_mask = torch.ones_like(self.linear.weight.data, device=self.device)
        new_mask[:old_out].copy_(self.mask)
        with torch.no_grad():
            # sparsify only the new block
            block = new_mask[old_out:]
            self._apply_initial_sparsity(block, self.sparsity_ratio)
        self.mask = new_mask
        self.linear.weight.data *= self.mask

        new_cwi = torch.zeros_like(self.linear.weight.data, device=self.device)
        new_cwi[:old_out].copy_(self.cwi)
        self.cwi = new_cwi

    def forward_rows(self, feats, rows):
        """Compute logits for a selected set of rows (task subset or all-known)."""
        W = self.linear.weight[rows]
        b = self.linear.bias[rows]
        return torch.addmm(b, feats, W.t())

    def forward_eval7(self, feats, registry: ClassRegistry):
        """Build 7-way logits on the fly. Unknown classes get a large negative logit."""
        rows = registry.gather_rows_for_eval7()   # indices or -1 for unseen
        known_mask = torch.tensor([r >= 0 for r in rows], device=feats.device)
        known_rows = [r for r in rows if r >= 0]

        # compute logits only for known classes
        out = torch.full((feats.size(0), len(rows)), -1e9, device=feats.device)  # -inf for unseen
        if len(known_rows) > 0:
            logits_known = self.forward_rows(feats, known_rows)
            out[:, known_mask] = logits_known
        return out

    # ---- TDM / CWI ----
    def apply_mask(self):
        with torch.no_grad():
            self.linear.weight.data *= self.mask

    def accumulate_cwi(self, weight_grads):
        # CWI += |grad * weight|
        with torch.no_grad():
            self.cwi += (weight_grads * self.linear.weight.grad).abs()

    def update_mask_intra(self, p=0.02):
        """Shrink/grow within currently active rows using CWI."""
        with torch.no_grad():

            flat_imp = self.cwi.view(-1)
            flat_mask = self.mask.view(-1)
            num = int(flat_mask.numel() * p)

            active_idx = torch.where(flat_mask > 0)[0]
            if len(active_idx) > 0 and num > 0:
                least = torch.argsort(flat_imp[active_idx])[:min(num, len(active_idx))]
                flat_mask[active_idx[least]] = 0

            inactive_idx = torch.where(flat_mask == 0)[0]
            if len(inactive_idx) > 0 and num > 0:
                add = torch.randperm(len(inactive_idx), device=flat_imp.device)[:min(num, len(inactive_idx))]
                flat_mask[inactive_idx[add]] = 1

            self.mask = flat_mask.view_as(self.mask)
            self.apply_mask()
            
    def update_mask_inter(self, p=0.05):
        """Warm-up expand: open a fraction of zeros; later shrink by low-CWI."""
        with torch.no_grad():
            
            flat_mask = self.mask.view(-1)
            zeros = torch.where(flat_mask == 0)[0]
            if len(zeros) > 0:
                num = int(flat_mask.numel() * p)
                add = torch.randperm(len(zeros), device=flat_mask.device)[:min(num, len(zeros))]
                flat_mask[zeros[add]] = 1
            self.mask = flat_mask.view_as(self.mask)
            self.apply_mask()
            
            

    def shrink_after_warmup(self, p=0.05):
        with torch.no_grad():
            flat_imp = self.cwi.view(-1)
            flat_mask = self.mask.view(-1)
            num = int(flat_mask.numel() * p)
            active_idx = torch.where(flat_mask > 0)[0]
            if len(active_idx) > 0 and num > 0:
                least = torch.argsort(flat_imp[active_idx])[:min(num, len(active_idx))]
                flat_mask[active_idx[least]] = 0
            self.mask = flat_mask.view_as(self.mask)
            self.apply_mask()
            
