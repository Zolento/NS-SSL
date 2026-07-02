import torch
import torch.nn as nn
from utils import fft2d, ifft2d

class data_consistency(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(0.05), requires_grad=True)

    def forward(self, z_k, x0, csm, mask):
        rhs = x0 + self.lam * z_k
        AtA = myAtA(csm, mask, self.lam)
        rec = conjgrad(rhs, csm, mask, self.lam, AtA)
        return rec

class myAtA(nn.Module):
    """
    performs DC step
    """
    def __init__(self, csm, mask, lam):
        super(myAtA, self).__init__()
        self.csm = csm # complex (B x ncoil x nrow x ncol)
        self.mask = mask # complex (B x nrow x ncol)
        self.lam = lam 

    def forward(self, im): #step for batch image
        """
        :im: complex image (B x nrow x nrol)
        """
        im_coil = self.csm * im 
        k_full = fft2d(im_coil) # convert into k-space 
        k_u = k_full * self.mask # undersampling
        im_u_coil = ifft2d(k_u) # convert into image domain
        im_u = torch.sum(im_u_coil * self.csm.conj(), axis=1, keepdim=True)
        return im_u + self.lam * im

def conjgrad(rhs, sens_maps, mask, mu, AtA):
    mu = mu.type(torch.complex64)
    x = torch.zeros_like(rhs)
    i, r, p = 0, rhs, rhs
    rsnot = torch.sum(r.conj() * r).real
    rsold, rsnew = rsnot, rsnot

    for ii in range(10):
        Ap = AtA(p)
        pAp = torch.sum(p.conj() * Ap).real
        alpha = (rsold / pAp)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.sum(r.conj() * r).real
        beta = (rsnew / rsold)
        rsold = rsnew
        p = beta * p + r

    return x

def random_drop_mask(
    mask: torch.Tensor,
    drop_ratio: float,
    batch: int = 1,
    acs_block: tuple = (4, 4)
):
    assert mask.dim() == 4 and mask.shape[0] == 1 and mask.shape[1] == 1, \
        "mask must be of shape (1, 1, H, W)"
    assert 0.0 <= drop_ratio <= 1.0, "drop_ratio must be in [0.0, 1.0]"
    assert batch >= 1, "batch must be at least 1"
    assert len(acs_block) == 2 and all(isinstance(x, int) and x >= 0 for x in acs_block), \
        "acs_block must be a tuple of two non-negative integers"

    H, W = mask.shape[2], mask.shape[3]
    device = mask.device
    dtype = mask.dtype

    center_h = H // 2
    center_w = W // 2
    acs_h, acs_w = acs_block

    acs_mask = torch.zeros((H, W), device=device, dtype=dtype)
    h_start = max(0, center_h - acs_h // 2)
    h_end = min(H, center_h + acs_h // 2 + (acs_h % 2))
    w_start = max(0, center_w - acs_w // 2)
    w_end = min(W, center_w + acs_w // 2 + (acs_w % 2))
    acs_mask[h_start:h_end, w_start:w_end] = 1

    flat_mask = mask.view(-1)  # (H*W,)
    flat_acs = acs_mask.view(-1)  # (H*W,)

    all_ones = flat_mask.nonzero(as_tuple=False).squeeze(-1)  # (N_all,)

    if all_ones.numel() == 0:
        kept = torch.zeros((batch, 1, H, W), device=device, dtype=dtype)
        dropped = torch.zeros((batch, 1, H, W), device=device, dtype=dtype)
        return kept, dropped

    acs_ones = all_ones[flat_acs[all_ones] == 1]  # (N_acs,)
    non_acs_ones = all_ones[flat_acs[all_ones] == 0]  # (N_non_acs,)

    num_non_acs = non_acs_ones.numel()
    num_to_drop = int(drop_ratio * num_non_acs)
    num_to_drop = min(num_to_drop, num_non_acs)

    kept_masks = []
    dropped_masks = []

    for _ in range(batch):
        kept_flat = torch.zeros_like(flat_mask)
        dropped_flat = torch.zeros_like(flat_mask)

        if acs_ones.numel() > 0:
            kept_flat[acs_ones] = 1

        if num_to_drop == 0:
            if non_acs_ones.numel() > 0:
                kept_flat[non_acs_ones] = 1
        else:
            perm = torch.randperm(num_non_acs, device=device)
            drop_idx = non_acs_ones[perm[:num_to_drop]]
            keep_idx = non_acs_ones[perm[num_to_drop:]]

            dropped_flat[drop_idx] = 1
            kept_flat[keep_idx] = 1

        kept = kept_flat.view(1, 1, H, W)
        dropped = dropped_flat.view(1, 1, H, W)
        kept_masks.append(kept)
        dropped_masks.append(dropped)

    kept_stacked = torch.cat(kept_masks, dim=0)   # (batch, 1, H, W)
    dropped_stacked = torch.cat(dropped_masks, dim=0)  # (batch, 1, H, W)

    original_mask_batch = mask.expand(batch, -1, -1, -1)
    reconstructed = kept_stacked + dropped_stacked
    assert torch.equal(reconstructed, original_mask_batch), "Reconstruction failed!"
    assert torch.all((kept_stacked * dropped_stacked) == 0), "Overlap detected!"
    
    return kept_stacked, dropped_stacked
