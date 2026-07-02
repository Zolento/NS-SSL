import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from piq import SSIMLoss
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from typing import Union, Type, List, Tuple
from timm.layers import trunc_normal_
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
matplotlib.use('Agg')

def save_images(gt, pred, output_path, metrics_path, filename, save_pred=True):
    #gt = normalize(gt)
    #pred = normalize(pred)
    diff = np.abs(gt - pred)
    psnr_value = psnr(gt, pred, data_range=1.0)
    ssim_value = ssim(gt, pred, data_range=1.0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(gt, cmap='gray')
    axes[0].set_title('gt')
    axes[0].axis('off')
    
    axes[1].imshow(pred, cmap='gray')
    axes[1].set_title('pred')
    axes[1].axis('off')
    
    cax = axes[2].imshow(diff, cmap='viridis')
    axes[2].set_title(f'diff (PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f})')
    axes[2].axis('off')
    
    fig.colorbar(cax, ax=axes[2])
    
    fig.suptitle(f'Filename: {filename}, PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}', fontsize=16)
    
    plt.savefig(output_path+'/'+filename+'.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    if save_pred:
        np.save(output_path+'/'+filename+'.npy', pred)
    
    if metrics_path is not None:
        with open(metrics_path, 'a+') as f:
            f.write(filename + ': \n')
            f.write(f'PSNR: {psnr_value:.2f}')
            f.write(f'SSIM: {ssim_value:.4f}\n')
        
    return psnr_value, ssim_value
def psnr(y, y_pred):
    return peak_signal_noise_ratio(y, y_pred, data_range=1.0)

def ssim(y, y_pred):
    return structural_similarity(y, y_pred, data_range=1.0)

def mse(y, y_pred):
    return np.mean((y-y_pred)**2)

def rmse(y, y_pred):
    return math.sqrt(mse(y, y_pred))

class MixL1L2Loss(nn.Module):
    def __init__(self, eps=1e-6,scalar=1/2):
        super().__init__()
        #self.mse = nn.MSELoss()
        self.eps = eps
        self.scalar=scalar
    def forward(self, y, yhat):

        loss = self.scalar*(torch.norm(yhat-y) / torch.norm(y)) + self.scalar*(torch.norm(yhat-y,p=1) / torch.norm(y, p=1))
        
        return loss

class EMAclass:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """validation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def complex_to_chan_dim(x: torch.Tensor, dim=1) -> torch.Tensor:
    assert torch.is_complex(x)
    return torch.cat([x.real, x.imag], dim=dim)

def chan_dim_to_complex(x: torch.Tensor) -> torch.Tensor:
    assert not torch.is_complex(x)
    if len(x.shape) == 4:
        _, c, _, _ = x.shape
    elif len(x.shape) == 5:
        _, c, _, _, _ = x.shape
    assert c % 2 == 0
    c = c // 2
    return torch.complex(x[:,:c], x[:,c:])

def ifft2d_np(data):
    data = np.fft.ifftshift(data, axes=[-2, -1])
    data = np.fft.ifftn(data, axes=[-2, -1], norm='ortho')
    data = np.fft.fftshift(data, axes=[-2, -1])
    return data

def fft2d_np(data):
    data = np.fft.ifftshift(data, axes=[-2, -1])
    data = np.fft.fftn(data, axes=[-2, -1], norm='ortho')
    data = np.fft.fftshift(data, axes=[-2, -1])
    return data

def ifft2d_torch(data: torch.Tensor) -> torch.Tensor:
    data = torch.fft.ifftshift(data, dim=[-2, -1])
    data = torch.fft.ifftn(data, dim=[-2, -1], norm='ortho')
    data = torch.fft.fftshift(data, dim=[-2, -1])
    return data

def fft2d_torch(data: torch.Tensor) -> torch.Tensor:
    data = torch.fft.ifftshift(data, dim=[-2, -1])
    data = torch.fft.fftn(data, dim=[-2, -1], norm='ortho')
    data = torch.fft.fftshift(data, dim=[-2, -1])
    return data

DATA_CONFIGS = {
}