#cnn_head.py
import torch
import torch.nn as nn
import math
from utils import ClassRegistry
#----- Class Indexing

#----------------- SPARCL TDM Implementation
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

        #TDM mask + CWI
        self.mask = torch.ones_like(self.linear.weight, device=device)
        self._apply_initial_sparsity(self.mask, self.sparsity_ratio)
        self.linear.weight.data *= self.mask

        self.cwi = torch.zeros_like(self.linear.weight, device=device)  #total importance (weight)
        self.cwi_curr = torch.zeros_like(self.linear.weight, device=device) #|L_curr| (current task grad)
        self.cwi_mem  = torch.zeros_like(self.linear.weight, device=device) #|L_mem| (replay buffer grad)

        
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

        #grow weight/bias
        new_linear = nn.Linear(self.in_dim, new_out).to(self.device)
        with torch.no_grad():
            new_linear.weight[:old_out].copy_(self.linear.weight)
            new_linear.bias[:old_out].copy_(self.linear.bias)
        _init_linear_rows(new_linear, old_out, new_out)

        #swap
        self.linear = new_linear
        self.out_dim = new_out

        #grow mask and cwi
        new_mask = torch.ones_like(self.linear.weight.data, device=self.device)
        new_mask[:old_out].copy_(self.mask)
        with torch.no_grad():
            #sparsify only the new block
            block = new_mask[old_out:]
            self._apply_initial_sparsity(block, self.sparsity_ratio)
        self.mask = new_mask
        with torch.no_grad():
            self.linear.weight.mul_(self.mask)

        new_cwi = torch.zeros_like(self.linear.weight.data, device=self.device)
        new_cwi_curr = torch.zeros_like(self.linear.weight.data, device=self.device)
        new_cwi_mem = torch.zeros_like(self.linear.weight.data, device=self.device)
        
        new_cwi[:old_out].copy_(self.cwi)
        new_cwi_curr[:old_out].copy_(self.cwi_curr)
        new_cwi_mem[:old_out].copy_(self.cwi_mem)
        self.cwi, self.cwi_curr, self.cwi_mem = new_cwi, new_cwi_curr, new_cwi_mem

    def forward_rows(self, feats, rows):
        """Compute logits for a selected set of rows (task subset or all-known)."""
        W = self.linear.weight[rows]
        b = self.linear.bias[rows]
        return torch.addmm(b, feats, W.t())

    def forward_eval7(self, feats, registry: ClassRegistry):
        """Build 7-way logits on the fly. Unknown classes get a large negative logit."""
        rows = registry.gather_rows_for_eval7()   #indices or -1 for unseen
        known_mask = torch.tensor([r >= 0 for r in rows], device=feats.device)
        known_rows = [r for r in rows if r >= 0]

        #compute logits only for known classes
        out = torch.full((feats.size(0), len(rows)), -1e9, device=feats.device)  #-inf for unseen
        if len(known_rows) > 0:
            logits_known = self.forward_rows(feats, known_rows)
            out[:, known_mask] = logits_known
        return out

    def apply_mask(self):
        with torch.no_grad():
            self.linear.weight.mul_(self.mask)

    def update_cwi_from_grads(self, g_curr, g_mem, alpha, beta, ema):
        W = self.linear.weight
        with torch.no_grad():
            w_l1 = W.abs()
            if g_curr is None: g_curr = torch.zeros_like(W)
            if g_mem  is None: g_mem  = torch.zeros_like(W)
            g_curr = g_curr.abs(); g_mem = g_mem.abs()

            self.cwi_curr.mul_(ema).add_((1 - ema) * g_curr)
            self.cwi_mem.mul_(ema).add_((1 - ema) * g_mem)
            self.cwi.mul_(ema).add_((1 - ema) * (w_l1 + alpha * self.cwi_curr + beta * self.cwi_mem))

    def update_mask_intra(self, p=0.02):
        """ 
        Close (zero) a small fraction of the currently active weights with the lowest CW
        The same number of active weights always stays the same. Happens every epoch % delta_k == 0 
        """
        with torch.no_grad():

            flat_cwi = self.cwi.view(-1)
            flat_mask = self.mask.view(-1)
        
            active_idx = torch.where(flat_mask > 0)[0]
            inactive_idx = torch.where(flat_mask == 0)[0]
            
            if len(active_idx) == 0 or len(inactive_idx) == 0:
                return
            #number of inactive weights to open and active weights to drop (equal!)
            num_swap = max(1, int(len(active_idx) * p))
            num_swap = min(num_swap, len(inactive_idx))  #preserve density
            
            #DDM implementation: drop lowest CWI
            if num_swap > 0:
                least = torch.argsort(flat_cwi[active_idx])[:num_swap]
                flat_mask[active_idx[least]] = 0
            
            #DGM implementation grow highest-CWI zeros
            if num_swap > 0:
                add = torch.argsort(flat_cwi[inactive_idx], descending=True)[:num_swap]
                flat_mask[inactive_idx[add]] = 1

            self.mask = flat_mask.view_as(self.mask)
            self.apply_mask()
            
    def update_mask_inter(self, p=0.05):
        """Warm-up expand (at the start of each new task): open a fraction of zeros; later shrink by low-CWI."""
        with torch.no_grad():
            
            flat_mask = self.mask.view(-1)
            zeros = torch.where(flat_mask == 0)[0] #find current zeros
            num_open = int(len(zeros) * p) #inactive weights to open
            
            if num_open > 0:
                add = torch.randperm(len(zeros), device=flat_mask.device)[:num_open]
                flat_mask[zeros[add]] = 1 #change zeroes --> 1
                
            self.mask = flat_mask.view_as(self.mask)
            self.apply_mask()
            
    def shrink_after_warmup(self, p=0.05):
        with torch.no_grad():
            flat_imp = self.cwi.view(-1)
            flat_mask = self.mask.view(-1)
            
            active_idx = torch.where(flat_mask > 0)[0]
            num_drop = int(len(active_idx) * p)
            
            if num_drop > 0:
                least = torch.argsort(flat_imp[active_idx])[:num_drop]
                flat_mask[active_idx[least]] = 0
                
            self.mask = flat_mask.view_as(self.mask)
            self.apply_mask()
            
