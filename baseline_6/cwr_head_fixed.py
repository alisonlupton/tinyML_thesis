# cwr_head_fixed.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CWRHeadFixed(nn.Module):
    def __init__(self, in_features, num_classes, device, preserve_magnitude=True, preserve_bias=True):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.preserve_magnitude = preserve_magnitude
        self.preserve_bias = preserve_bias
        
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
        # During training you'll use tw/tb; during eval you'll call use_consolidated()
        return F.linear(x, self.weight, self.bias)

    @torch.no_grad()
    def preload_batch(self, present_idx: torch.Tensor):
        """
        CWR* line 7, per-batch:
            tw[j] = cw[j] if class j already seen (hist_count>0)
            tw[j] = 0     otherwise
        We only touch rows for classes in this minibatch.
        """
        if present_idx.numel() == 0:
            return

        # zero tw rows for present classes
        self.weight.index_fill_(0, present_idx, 0.0)
        if self.bias is not None:
            self.bias.index_fill_(0, present_idx, 0.0)

        # copy cw for the subset of present classes that are already seen
        seen_mask = self.hist_count.index_select(0, present_idx) > 0
        if seen_mask.any():
            idx_old = present_idx[seen_mask]
            self.weight.index_copy_(0, idx_old, self.cw.index_select(0, idx_old))
            if self.preserve_bias and self.cb is not None:
                self.bias.index_copy_(0, idx_old, self.cb.index_select(0, idx_old))

    @torch.no_grad()
    def consolidate(self, present_classes, counts_present):
        """
        CWR* consolidation following the paper algorithm exactly:
        wpast_j = sqrt(past_j / cur_j)
        cw[j] = (cw[j] * wpast_j + (tw[j] - avg(tw))) / (wpast_j + 1)
        """
        if len(present_classes) == 0:
            return
        idx = torch.as_tensor(present_classes, device=self.weight.device, dtype=torch.long)
        
        # Get historical counts for present classes
        past_j = self.hist_count.index_select(0, idx).to(self.weight.dtype)  # past_j
        cur_j = counts_present.to(self.weight.dtype)  # cur_j
        
        # Calculate wpast_j = sqrt(past_j / cur_j) as per paper
        wpast_j = torch.sqrt(past_j / (cur_j + 1e-8))  # Add small epsilon to avoid div by zero
        
        # Get current weights
        cw_old = self.cw.index_select(0, idx)  # cw[j]
        tw_now = self.weight.index_select(0, idx)  # tw[j]
        
        # Calculate avg(tw) - average of all training weights for present classes
        avg_tw = tw_now.mean(dim=0, keepdim=True)  # avg(tw)
        
        # Apply CWR* formula: cw[j] = (cw[j] * wpast_j + (tw[j] - avg(tw))) / (wpast_j + 1)
        wpast_j_expanded = wpast_j.unsqueeze(1)  # (K, 1) for broadcasting
        cw_new = (cw_old * wpast_j_expanded + (tw_now - avg_tw)) / (wpast_j_expanded + 1)
        
        # Handle bias if present
        if self.bias is not None and self.cb is not None:
            cb_old = self.cb.index_select(0, idx)
            bb_now = self.bias.index_select(0, idx)
            avg_bb = bb_now.mean()  # Average bias for present classes
            cb_new = (cb_old * wpast_j + (bb_now - avg_bb)) / (wpast_j + 1)
            self.cb.index_copy_(0, idx, cb_new)
        
        # Write back consolidated weights
        self.cw.index_copy_(0, idx, cw_new)
        
        # Update historical counts: past_j = past_j + cur_j
        self.hist_count.index_put_((idx,), (past_j + cur_j).to(self.hist_count.dtype))

        if not hasattr(self, "_dbg_printed2"):
            print("[CWR*] consolidate first call: classes:", present_classes[:20],
                "counts:", counts_present[:20].tolist())
            print("[CWR*] wpast_j values:", wpast_j[:10].tolist())
            print("[CWR*] past_j values:", past_j[:10].tolist())
            print("[CWR*] avg_tw value:", avg_tw.mean().item())
            print("[CWR*] cw_old mean:", cw_old.mean().item())
            print("[CWR*] tw_now mean:", tw_now.mean().item())
            print("[CWR*] cw_new mean:", cw_new.mean().item())
            self._dbg_printed2 = True
            
    @torch.no_grad()
    def consolidate_batch(self, present_idx: torch.Tensor, counts_present: torch.Tensor):
        """
        CWR* lines 11–14, per-batch consolidation:
            wpast_j = sqrt(past_j / cur_j)
            cw[j]   = (cw[j]*wpast_j + (tw[j] - avg(tw))) / (wpast_j + 1)
            past_j += cur_j
        Only over classes in the current minibatch.
        """
        if present_idx.numel() == 0:
            return

        eps = 1e-8
        past_j = self.hist_count.index_select(0, present_idx).to(self.weight.dtype)
        cur_j  = counts_present.to(self.weight.dtype).clamp_min(1.0)  # avoid div/0
        wpast  = torch.sqrt(past_j / (cur_j + eps))                   # (K,)

        tw_now = self.weight.index_select(0, present_idx)             # (K, D)
        avg_tw = tw_now.mean(dim=0, keepdim=True)                     # (1, D)
        cw_old = self.cw.index_select(0, present_idx)                 # (K, D)

        wp = wpast.unsqueeze(1)                                       # (K,1)
        cw_new = (cw_old * wp + (tw_now - avg_tw)) / (wp + 1.0)
        self.cw.index_copy_(0, present_idx, cw_new)

        if self.bias is not None and self.cb is not None:
            bb_now = self.bias.index_select(0, present_idx)           # (K,)
            avg_bb = bb_now.mean()
            cb_old = self.cb.index_select(0, present_idx)
            cb_new = (cb_old * wpast + (bb_now - avg_bb)) / (wpast + 1.0)
            self.cb.index_copy_(0, present_idx, cb_new)

        # update historical counts
        self.hist_count.index_add_(0, present_idx, counts_present.to(self.hist_count.dtype))

    @torch.no_grad()
    def use_consolidated(self):
        """Load cw/cb into tw/tb (for evaluation)."""
        self.weight.copy_(self.cw)
        self.bias.copy_(self.cb) 