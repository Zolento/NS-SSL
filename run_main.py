import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import einops
from utils import chan_dim_to_complex, complex_to_chan_dim, MixL1L2Loss, psnr, ssim, EMAclass, save_images, ifft2d_torch, fft2d_torch, DATA_CONFIGS
from models.ssl import random_drop_mask
from models.ResNet import ResNet
from models.ssl import myAtA, conjgrad
from utils import save_images
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from natsort import natsorted


def construct_pixel_bank(kspace, window_size, patch_size, num_similar):
    """
    Constructs a pixel bank for complex img data.
    
    Args:
        kspace (torch.Tensor): Complex img data of shape [coil, height, width].
        args (Namespace): Arguments containing window size, patch size, etc.
                          Must include: args.ws, args.ps, args.nn (num neighbors/sim), 
                          args.save_path (to save output).

    Returns:
        pixel_bank (torch.Tensor): Shape (height, width, coil, num_sim)
    """
    # Handle Complex Numbers
    # Convert [C, H, W] complex -> [1, C, H, W] complex for processing
    input_tensor = kspace.unsqueeze(0)

    _, channels, H, W = input_tensor.shape
    WINDOW_SIZE = window_size
    PATCH_SIZE = patch_size
    NUM_NEIGHBORS = num_similar
    
    # Padding
    pad_sz = WINDOW_SIZE // 2 + PATCH_SIZE // 2
    center_offset = WINDOW_SIZE // 2
    
    img_pad = F.pad(input_tensor, (pad_sz, pad_sz, pad_sz, pad_sz), mode='reflect')

    img_unfold = F.unfold(img_pad, kernel_size=PATCH_SIZE, padding=0, stride=1)
    
    H_new = H + WINDOW_SIZE
    W_new = W + WINDOW_SIZE
    img_unfold = einops.rearrange(img_unfold, 'b c (h w) -> b c h w', h=H_new, w=W_new)

    # Block Processing (to manage memory usage)
    blk_sz = 16 # Size of blocks to process at once
    num_blk_w = int(np.ceil(W / blk_sz))
    num_blk_h = int(np.ceil(H / blk_sz))
    
    is_window_size_even = (WINDOW_SIZE % 2 == 0)
    topk_list_rows = []

    for blk_j in range(num_blk_h): # Loop Height
        topk_list_cols = []
        for blk_i in range(num_blk_w): # Loop Width
            
            # Define current block boundaries
            start_h = blk_j * blk_sz
            end_h = min(start_h + blk_sz + WINDOW_SIZE, H_new)
            
            start_w = blk_i * blk_sz
            end_w = min(start_w + blk_sz + WINDOW_SIZE, W_new)
            
            sub_img_uf = img_unfold[..., start_h:end_h, start_w:end_w]
            sub_img_shape = sub_img_uf.shape

            if is_window_size_even:
                sub_img_uf_inp = sub_img_uf[..., :-1, :-1]
            else:
                sub_img_uf_inp = sub_img_uf

            patch_windows = F.unfold(sub_img_uf_inp, kernel_size=WINDOW_SIZE, padding=0, stride=1)
            
            curr_h = sub_img_uf_inp.shape[-2] - WINDOW_SIZE + 1
            curr_w = sub_img_uf_inp.shape[-1] - WINDOW_SIZE + 1
            patch_windows = einops.rearrange(
                patch_windows,
                'b (c k1 k2 k3 k4) (h w) -> b (c k1 k2) (k3 k4) h w',
                k1=PATCH_SIZE, k2=PATCH_SIZE, k3=WINDOW_SIZE, k4=WINDOW_SIZE,
                h=curr_h, w=curr_w
            )

            img_center = einops.rearrange(
                sub_img_uf,
                'b (c k1 k2) h w -> b (c k1 k2) 1 h w',
                k1=PATCH_SIZE, k2=PATCH_SIZE
            )
            
            # Crop the center part corresponding to the valid block area
            img_center = img_center[..., center_offset:center_offset + curr_h, center_offset:center_offset + curr_w]

            # Similarity Calculation
            img_center_norm = F.normalize(img_center, p=2, dim=1).abs()
            patch_windows_norm = F.normalize(patch_windows, p=2, dim=1).abs()
            cos_sim = torch.sum(img_center_norm * patch_windows_norm, dim=1) # 结果形状: (B, Win2, H, W)
            # Top-K
            _, sort_indices = torch.topk(cos_sim, k=NUM_NEIGHBORS, largest=True, sorted=True, dim=1)

            patch_windows_reshape = einops.rearrange(
                patch_windows,
                'b (c k1 k2) (k3 k4) h w -> b c (k1 k2) (k3 k4) h w',
                k1=PATCH_SIZE, k2=PATCH_SIZE, k3=WINDOW_SIZE, k4=WINDOW_SIZE
            )
            
            patch_center_pixel = patch_windows_reshape[:, :, (PATCH_SIZE*PATCH_SIZE) // 2, ...]
            
            gather_indices = sort_indices.unsqueeze(1).repeat(1, channels, 1, 1, 1)
            
            topk = torch.gather(patch_center_pixel, dim=-3, index=gather_indices)
            
            # Result shape: [B, C, num_sim, H_blk, W_blk]
            topk_list_cols.append(topk.cpu())

        # Concatenate columns (Width)
        topk_row = torch.cat(topk_list_cols, dim=-1)
        topk_list_rows.append(topk_row)

    # Final Merge and Reshape
    pixel_bank = torch.cat(topk_list_rows, dim=-2)
    
    pixel_bank = pixel_bank.squeeze(0) # [C, num_sim, H, W]
    pixel_bank = pixel_bank.permute(2, 3, 0, 1)

    return pixel_bank

class ssl(nn.Module):
    def __init__(self):
        super().__init__()
        self.regularizer = ResNet('cuda', in_ch=2, num_of_resblocks=8)
        self.lam = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        
    def forward(self, cur_ks, csm, acs_mask, mask, x0, ks_acq):   
        input_x0 = (ifft2d_torch(cur_ks) * csm.conj()).sum(1, keepdim=True)
        
        for _ in range(10):
            x = (ifft2d_torch(cur_ks) * csm.conj()).sum(1, keepdim=True)
            z = chan_dim_to_complex(self.regularizer(complex_to_chan_dim(x)))
            x = self.data_consistency(z, input_x0, csm, mask)
            cur_ks = fft2d_torch(x * csm)
        return cur_ks, x

    def data_consistency(self, z, x0, csm, mask):
        rhs = x0 + self.lam * z
        AtA = myAtA(csm, mask, self.lam)
        rec = conjgrad(rhs, csm, mask, self.lam, AtA)
        return rec

def generate_cartesian_mask_exclusive(shape, acc=4, acs_lines=28, exclude_indices=None):
    num_cols = shape[-1]
    mask_1d = np.zeros(num_cols, dtype=np.float32)
    
    center = num_cols // 2
    acs_start = center - acs_lines // 2
    acs_end = center + acs_lines // 2
    mask_1d[acs_start:acs_end] = 1.0
    
    outer_indices = np.setdiff1d(np.arange(num_cols), np.arange(acs_start, acs_end))
    
    if exclude_indices is not None:
        outer_indices = np.setdiff1d(outer_indices, exclude_indices)
    
    total_lines_needed = int(num_cols / acc)
    lines_to_add = total_lines_needed - acs_lines
    
    if lines_to_add > 0:
        num_to_sample = min(lines_to_add, len(outer_indices))
        selected_indices = np.random.choice(
            outer_indices, 
            size=num_to_sample, 
            replace=False
        )
        mask_1d[selected_indices] = 1.0
    
    mask_tensor = torch.from_numpy(mask_1d)
    return mask_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 1, shape[-2], 1)

def shuffle_bank_hwck(pixel_bank):
    K = pixel_bank.shape[-1]
    idx = torch.randperm(K, device=pixel_bank.device)
    return pixel_bank[..., idx]
    
class HybridConsistencyBank:
    def __init__(self, kspace_shape, kernel_size=5, device='cuda'):
        self.device = device
        self.kernel_size = kernel_size
        self.k_bank = torch.zeros(kspace_shape, dtype=torch.complex64, device=device)
        
        self.score_bank = torch.full(kspace_shape, float('inf'), device=device)
        
        self.kernels = None
        self.initialized = False
        self.alpha = 1.0
        self.beta = 1.0
        self.is_weights_set = False
        
    def calibrate_kernel(self, kspace_acs, steps=50, lr=0.01):
        B, C, H, W = kspace_acs.shape
        pad = self.kernel_size // 2
        
        weights = torch.randn(B, C, C, self.kernel_size, self.kernel_size, dtype=torch.complex64, device=self.device)
        weights = nn.Parameter(weights)
        
        optimizer = torch.optim.Adam([weights], lr=lr)
        
        mask = torch.ones_like(weights, dtype=torch.bool)
        center = self.kernel_size // 2
        mask[:, :, :, center, center] = 0
        
        kspace_acs = kspace_acs.detach()
        
        for _ in range(steps):
            optimizer.zero_grad()
            
            masked_weights = weights * mask
            
            loss = 0
            for b in range(B):
                w = masked_weights[b] # (C, C, K, K)
                inp = kspace_acs[b:b+1] # (1, C, H, W)
                
                out_real = F.conv2d(inp.real, w.real, padding=pad) - F.conv2d(inp.imag, w.imag, padding=pad)
                out_imag = F.conv2d(inp.real, w.imag, padding=pad) + F.conv2d(inp.imag, w.real, padding=pad)
                out = torch.complex(out_real, out_imag)
                
                loss += F.mse_loss(torch.view_as_real(out), torch.view_as_real(inp))
                # print(f"Calibration Loss: {loss.item():.12f}")
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            self.kernels = weights * mask
            print(f"SPIRiT Kernel Calibrated. Final Loss: {loss.item():.12f}")

    def compute_hybrid_score(self, k_pred, csm):
        B, C, H, W = k_pred.shape
        
        img_domain = (ifft2d_torch(k_pred) * csm.conj()).sum(dim=1, keepdim=True)
        k_sense_proj = fft2d_torch(img_domain * csm)
        
        score_sense = (k_pred - k_sense_proj).abs()# (1, C, H, W) #.sum(dim=1, keepdim=True)
        return score_sense

    def update(self, k_pred, csm, mask_sampled):
        with torch.no_grad():
            mask_sampled = mask_sampled.repeat(1, k_pred.shape[1], 1, 1)
            total_score = self.compute_hybrid_score(k_pred, csm)
            
            better_mask = (total_score < self.score_bank) & (mask_sampled == 0)
            self.k_bank = torch.where(better_mask, k_pred, self.k_bank)
            self.score_bank = torch.where(better_mask, total_score, self.score_bank)
            self.initialized = True
            full_kspace = k_pred * mask + self.k_bank * (1 - mask)
            return full_kspace
    
def generate_frequency_weight_mask(shape, cutoff_percent=0.3, slope=10.0, device='cuda'):
    H, W = shape
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    radius = torch.sqrt(xx**2 + yy**2)
    
    weight_mask = 1 / (1 + torch.exp(-slope * (radius - cutoff_percent)))
    mask_ = weight_mask.detach().cpu().numpy()
    plt.imsave(
        WORKSPACE+'/weight_mask1.jpg',
        mask_,
        cmap=plt.cm.gray
        )
    return weight_mask.unsqueeze(0).unsqueeze(0)

def train(epoch, model, bank, point_bank, output_ori, final_kspace, y_pred_ts, tr_mask, val_mask, mask, raw_outer_indices, sens_maps, kspace_data_tr, device, loss_fun, optimizer, ema, best_psnr, best_ssim):
    loss_avg = 0
    running_score = defaultdict(int)
    if epoch > 0:
        with torch.no_grad():
            out_real = F.conv2d(final_kspace.real, bank.kernels.real.squeeze(), padding=bank.kernel_size // 2) - \
            F.conv2d(final_kspace.imag, bank.kernels.imag.squeeze(), padding=bank.kernel_size // 2)
            out_imag = F.conv2d(final_kspace.real, bank.kernels.imag.squeeze(), padding=bank.kernel_size // 2) + \
            F.conv2d(final_kspace.imag, bank.kernels.real.squeeze(), padding=bank.kernel_size // 2)
            k_spirit_pred = torch.complex(out_real, out_imag)
        _, freq_mask = random_drop_mask(mask, 0.5, 1, (0, 0)) # spirit ratio
        k_spirit_pred = freq_mask * k_spirit_pred + (1 - freq_mask) * final_kspace
        pixel_bank_in = ((ifft2d_torch(k_spirit_pred) * sens_maps.conj()).sum(1, keepdim=True))
        with torch.no_grad():
            point_bank = construct_pixel_bank(
                pixel_bank_in.squeeze(0),
                window_size=40,
                patch_size=7,
                num_similar=NUM_SAMPLES,
            ).to(device)
        point_bank = shuffle_bank_hwck(point_bank)
    for i in range(NUM_SAMPLES):
        kspace_in = kspace_data_tr
        train_mask = tr_mask[i]
        loss_mask = val_mask[i]
        output_ori, _ = model(kspace_in * train_mask, sens_maps, None, train_mask, None, None)
        loss_phy = loss_fun(kspace_in * loss_mask, output_ori * loss_mask)

        if point_bank is not None:
            gt_pseudo = point_bank.permute(3, 2, 0, 1)[i:i+1, :, :, :]
            mask_pseudo = generate_cartesian_mask_exclusive(
                gt_pseudo.shape,
                acc=ACC,
                acs_lines=ACS_NUM,
                exclude_indices=raw_outer_indices
            ).to(gt_pseudo.device)
            
            kspace_pseudo = fft2d_torch(gt_pseudo * sens_maps)
            output_pseudo, _ = model(kspace_pseudo * mask_pseudo, sens_maps, None, mask_pseudo, None, None)

            loss_reg = LAMBDA * (loss_fun(kspace_pseudo, output_pseudo))
        else:
            loss_reg = torch.tensor(0.0, device=device)
            
        loss = loss_phy + loss_reg
        loss_avg += loss.item() / (NUM_SAMPLES)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            ema.update()
            ks_pred_ts, y_pred_ts = model(kspace_data_tr * mask, sens_maps, None, mask, None, None)
            y_pred_ts_ = y_pred_ts[0, 0, :, :].abs().detach().cpu().numpy()
            final_kspace = bank.update(ks_pred_ts, sens_maps, mask)
            
        factor = np.max(gt)
        y_img = gt
        y_img = (y_img / factor)
        y_pred_ts_ = (y_pred_ts_ / factor)
        p_, s_ = psnr(y_img, y_pred_ts_), ssim(y_img, y_pred_ts_)
        
        # save best scores
        if p_ > best_psnr:
            best_psnr = p_
        if s_ > best_ssim:
            best_ssim = s_
        running_score['PSNR'] += p_ / (NUM_SAMPLES)
        running_score['SSIM'] += s_ / (NUM_SAMPLES)
        
    return running_score, loss_avg, best_psnr, best_ssim, output_ori.detach(), y_pred_ts.detach(), final_kspace.detach()

def val(model, ema, tr_mask, val_mask, mask, sens_maps, kspace_data_tr, final_kspace, loss_fun):
    loss_avg = 0
    # ema.apply_shadow()
    with torch.no_grad():
        # for i in range(NUM_SAMPLES):
        kspace_in = kspace_data_tr
        output, _ = model(kspace_in * tr_mask, sens_maps, mask, tr_mask, None, kspace_in)
        loss = loss_fun(kspace_in * val_mask, output * val_mask)   
        loss_avg += loss.item()# / (NUM_SAMPLES)
    # ema.restore()
    
    return loss_avg

def test(model, ema, mask, sens_maps, kspace_data_tr, final_kspace):
    ema.apply_shadow()
    with torch.no_grad():
        ks_pred_ts, y_pred_ts = model(kspace_data_tr * mask, sens_maps, None, mask, None, None)
        y_pred_ts = y_pred_ts[0, 0, :, :].abs().detach().cpu().numpy()
        ks_pred_ts = ks_pred_ts[0, 0, :, :].abs().detach().cpu().numpy()
        plt.imsave(
            WORKSPACE+'/x_pred_final.jpg',
            y_pred_ts,
            cmap=plt.cm.gray
        )
        plt.imsave(
            WORKSPACE+'/ks_pred_final.jpg',
            np.log(ks_pred_ts + 1e-5),
            cmap=plt.cm.gray
        )

    factor = np.max(gt)
    y_img = gt
    y_img = (y_img / factor)
    y_pred_ts_metric = (y_pred_ts / factor)
    p_, s_ = psnr(y_img, y_pred_ts_metric), ssim(y_img, y_pred_ts_metric)
    ema.restore()
    
    return p_, s_, y_pred_ts

def save_training_curves(logs_path, output_path):
    data = np.load(logs_path)
    psnrs = data['psnrs']
    ssims = data['ssims']
    losses = data['losses']
    losses_val = data['losses_val']
    n_epochs = len(psnrs)
    if not (len(ssims) == len(losses) == n_epochs):
        raise ValueError("PSNR, SSIM, and Loss arrays must have the same length.")

    epochs = np.arange(0, n_epochs)

    fig, axs = plt.subplots(2, 2, figsize=(8, 9), sharex=True)
    fig.suptitle('Training Metrics vs Epoch')

    axs = axs.ravel()
    
    axs[0].plot(epochs, psnrs, color='tab:blue')
    axs[0].set_ylabel('PSNR (dB)')
    axs[0].grid(True)

    axs[1].plot(epochs, ssims, color='tab:orange')
    axs[1].set_ylabel('SSIM')
    axs[1].grid(True)

    axs[2].plot(epochs, losses, color='tab:red')
    axs[2].set_ylabel('Loss train')
    axs[2].set_xlabel('Epoch')
    axs[2].grid(True)

    axs[3].plot(epochs, losses_val, color='tab:red')
    axs[3].set_ylabel('Loss val')
    axs[3].set_xlabel('Epoch')
    axs[3].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Training curves saved to: {output_path}")

def evaluate_results(results_path, files):

    all_psnrs = []
    all_ssims = []
    img_savepath = results_path+'/img'

    print(f"Starting evaluation on {len(files)} image pairs...")
    
    for idx, f in enumerate(files):
        f = f[:-4]
        pred_path = os.path.join(results_path + '/ks', f + '.npy')
        gt_path = os.path.join(results_path, f"{f}_gt.npy")

        if not os.path.exists(gt_path):
            print(f"Warning: GT file not found for {f}, skipping.")
            continue

        y_pred = np.load(pred_path)
        y_gt = np.load(gt_path)
        factor = np.max(y_gt)
        
        y_gt = y_gt / factor
        y_pred = y_pred / factor
        p_score = psnr(y_gt, y_pred)
        s_score = ssim(y_gt, y_pred)
        print(f, str(idx), p_score, s_score)
        save_images(y_gt, y_pred, img_savepath, None, f, save_pred=False)
        
        all_psnrs.append(p_score)
        all_ssims.append(s_score)

    log_save_path = os.path.join(results_path, 'logs_all.npz')
    np.savez(log_save_path, psnrs=all_psnrs, ssims=all_ssims)

    if len(all_psnrs) > 0:
        avg_psnr = sum(all_psnrs) / len(all_psnrs)
        avg_ssim = sum(all_ssims) / len(all_ssims)
        print(f"Evaluation Finished.")
        print(f"Saved to: {log_save_path}")
        print(f"Final avg score - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    else:
        print("No valid pairs found for evaluation.")

    return all_psnrs, all_ssims

def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument("--num_samples", type=int, default=25,
                        help="Number of samples (default: 25)")
    parser.add_argument("--num_banksamples", type=int, default=None,
                        help="Number of bank samples (default: same as num_samples)")
    parser.add_argument("--stop_training_steps", type=int, default=25,
                        help="Steps to stop training (default: 25)")
    parser.add_argument("--num_epochs", type=int, default=1000,
                        help="Number of training epochs (default: 1000)")
    parser.add_argument("--lam", type=float, default=0.1, dest="lambda_",
                        help="Distillation loss weight (default: 0.1)")
    parser.add_argument("--lam_con", type=float, default=0.1, dest="lambda_con",
                        help="Consistency loss weight (default: 0.1)")
    parser.add_argument("--eva", type=bool, default=False, dest="eva",
                    help="Do evaluation")
    parser.add_argument('--data_key', type=str, default='sslT1_uni_4x', 
                        help=f'available{list(DATA_CONFIGS.keys())}')
    parser.add_argument("--expname", type=str, default="", dest="expname",
                    help="Experiment name")
    parser.add_argument("--lr", type=float, default=5e-4, dest="lr",
                    help="Learning rate (default: 5e-4)")
    parser.add_argument("--results_path", type=str, default="", dest="results_path",
                    help="Path to results for evaluation")
    args = parser.parse_args()
    
    if args.num_banksamples is None:
        args.num_banksamples = args.num_samples
    
    return args

if __name__ == "__main__":
    args = parse_args()
    print(f"NUM_SAMPLES = {args.num_samples}")
    print(f"NUM_BANKSAMPLES = {args.num_banksamples}")
    print(f"STOP_TRAINING_STEPS = {args.stop_training_steps}")
    print(f"NUM_EPOCHS = {args.num_epochs}")
    print(f"LAMBDA = {args.lambda_}")
    print(f"LAMBDA_CON = {args.lambda_con}")
    print(f"LR = {args.lr}")
    
    device = 'cuda'
    if args.data_key in DATA_CONFIGS:
        config = DATA_CONFIGS[args.data_key]
        
        root_path = config['root_path']
        mask_f0 = config['mask_f0']
        if '4x' in mask_f0:
            ACS_NUM = 28
            ACC = 4.0
        elif '6x' in mask_f0:
            ACS_NUM = 22
            ACC = 6.0
        elif '8x' in mask_f0:
            ACS_NUM = 16
            ACC = 8.0
        print(f"--- : {args.data_key} ---")
        print(f"Data path: {root_path}")
        print(f"Mask path: {mask_f0}")
    else:
        raise ValueError(f"Key '{args.data_key}' not defined in DATA_CONFIGS")
    
    WORKSPACE = 'ssltest'
    if args.expname == '':
        args.expname = args.data_key
    results_path = WORKSPACE+'/' + args.expname + '/'
    
    NUM_SAMPLES = args.num_samples
    NUM_BANKSAMPLES = args.num_banksamples
    STOP_TRAINING_STEPS = args.stop_training_steps
    NUM_EPOCHS = args.num_epochs
    LAMBDA = args.lambda_
    LAMBDA_CON = args.lambda_con
    UPDATE_INTERVAL = 10
    evaluate_only = args.eva
    
    os.makedirs(WORKSPACE, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(results_path+'/img', exist_ok=True)
    os.makedirs(results_path + '/ks', exist_ok=True)
    path = Path(os.path.join(root_path, "ksslicecom/"))
    files = natsorted([file.name for file in path.rglob("*.npy")])

    print('file nums: ', len(files))
    all_psnrs, all_ssims = [], []
    all_psnrs_repo, all_ssims_repo = [], []
    if evaluate_only:
        results_path = args.results_path
        sens_maps = []
        for index, f in enumerate(files):
            sens_map = np.load(root_path + 'csmslice/' + f.replace('ks', 'csm'))
            sens_maps.append(sens_map[:,sens_map.shape[1]//2-160:sens_map.shape[1]//2+160,:])
        evaluate_results(results_path=results_path, files=files)
    else:
        for index, f in enumerate(files):
            mask, sens_maps, kspace_data = np.load(mask_f0), \
            np.load(root_path + 'csmslice/' + f.replace('ks', 'csm')), \
            np.load(root_path + 'ksslicecom/' + f)
            mask = torch.tensor(mask).to(device)
            sens_maps = torch.tensor(sens_maps).to(device)[:,sens_maps.shape[1]//2-160:sens_maps.shape[1]//2+160,:]
            kspace_data = torch.tensor(kspace_data).to(device)
            kspace_data = fft2d_torch(ifft2d_torch(kspace_data)[:,kspace_data.shape[1]//2-160:kspace_data.shape[1]//2+160,:])
            kspace_data = kspace_data / kspace_data.abs().max()
            kspace_data_tr = (kspace_data * mask)
            
            h, w = kspace_data_tr.shape[-2], kspace_data_tr.shape[-1]
            
            global gt
            gt = (ifft2d_torch(kspace_data) * sens_maps.conj()).sum(0).abs().cpu().numpy()
            np.save(results_path + f[:-4] + '_gt.npy', gt)

            loss_fun = MixL1L2Loss()
            best_psnr, best_ssim, loss_val_min = 0, 0, 1e10
            psnrs, ssims, losses, losses_val = [], [], [], []
            model = ssl().to(device)
            ema = EMAclass(model, 0.999)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)
            scaler = None
            point_bank = None
            final_kspace = None
            y_pred_ts = None
            kspace_data_tr = kspace_data_tr.unsqueeze(0)
            
            mask = mask.repeat(1,1,h,1)
            mask1, maskvali = random_drop_mask(mask, 0.2, 1)
            masktr, maskval = random_drop_mask(mask1, 0.4, NUM_SAMPLES)
            
            epoch_iterator = tqdm(range(NUM_EPOCHS), desc="Training", unit="epoch")
            output_img_ori = None
            mask_1d_raw = mask[0, 0, 0, :].cpu().numpy()
            center = mask_1d_raw.shape[0] // 2
            acs_start, acs_end = center - ACS_NUM // 2, center + ACS_NUM // 2
            
            B, C, H, W = kspace_data_tr.shape
            bank = HybridConsistencyBank((B, C, H, W), kernel_size=11, device=device)
            mask_acs = torch.zeros_like(mask)
            mask_acs[:, :, :, acs_start:acs_end] = 1
            bank.calibrate_kernel(kspace_data_tr * mask_acs, steps=200)

            raw_sampling_indices = np.where(mask_1d_raw == 1.0)[0]
            raw_outer_indices = np.setdiff1d(raw_sampling_indices, np.arange(acs_start, acs_end))
            for epoch in epoch_iterator:
                running_score, loss_avg, best_psnr, best_ssim, output_img_ori, y_pred_ts, final_kspace = train(epoch, 
                                                                                                        model,
                                                                                                        bank,
                                                                                                        point_bank,
                                                                                                        output_img_ori,
                                                                                                        final_kspace,
                                                                                                        y_pred_ts,
                                                                                                        masktr,
                                                                                                        maskval,
                                                                                                        mask,
                                                                                                        raw_outer_indices,
                                                                                                        sens_maps,
                                                                                                        kspace_data_tr, 
                                                                                                        device, 
                                                                                                        loss_fun, 
                                                                                                        optimizer, 
                                                                                                        ema, 
                                                                                                        best_psnr, 
                                                                                                        best_ssim)
                loss_val = val(model, ema,
                                mask1,
                                maskvali,
                                mask, 
                                sens_maps, 
                                kspace_data_tr, 
                                final_kspace, loss_fun)
                if loss_val <= loss_val_min:
                    loss_val_min = loss_val
                    val_loss_tracker = 0
                else:
                    val_loss_tracker += 1
                epoch_iterator.set_postfix(
                    PSNR=f"{running_score['PSNR']:.4f}",
                    SSIM=f"{running_score['SSIM']:.4f}",
                    Loss=f"{loss_avg:.8f}",
                    best_score=f"{best_psnr:.4f}, {best_ssim:.4f}",
                    val_track=f"{val_loss_tracker:d}",
                    val_loss=f"{loss_val:.6f}"
                )
                psnrs.append(running_score['PSNR'])
                ssims.append(running_score['SSIM'])
                losses.append(loss_avg)
                losses_val.append(loss_val)
                if val_loss_tracker >= STOP_TRAINING_STEPS:
                    break
            psnr_, ssim_, y_pred_ts = test(model, ema,
                                        mask, 
                                        sens_maps, 
                                        kspace_data_tr,
                                        final_kspace)
            print('current score: ', str(index), psnr_, ssim_)

            np.save(results_path + '/ks/' + f[:-4] + '.npy', y_pred_ts)
            all_psnrs.append(psnr_)
            all_ssims.append(ssim_)

            torch.cuda.empty_cache()
            
        np.savez(results_path +'/logs_all.npz',
                psnrs=all_psnrs,
                ssims=all_ssims)
        print('final avg score: ', sum(all_psnrs) / len(all_psnrs), sum(all_ssims) / len(all_ssims))

