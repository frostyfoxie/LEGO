"""
LEGO Annotated Multi-Stage Training Engine (v5.0)

Key Engineering Decisions:
1. Multi-Stage Schedule:
   - Stage 1: DIV2K pixel pre-training using Charbonnier, FFT, and edge gradient losses.
   - Stage 2: Streaming COCO + Flickr30k mixed with 50% DIV2K replay and L2-SP anchoring to prevent catastrophic forgetting.
   - Stage 3: GAN and Perceptual fine-tuning (Real-ESRGAN style).
2. DistributedDataParallel (DDP): One independent training process per GPU with NCCL all-reduce gradient synchronization.
3. Memory Guard: malloc_trim arena caps, DataLoader worker recreation every epoch, and process recycling to eliminate COW leaks.
"""

import os
import gc
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models.lego_net import DeepSpatialRestorationNet


# ============================================================================
# LOSS FORMULATIONS
# ============================================================================

class PixelLoss(nn.Module):
    """
    Multiscale Loss combining:
    1. Charbonnier Penalty: Smooth L1 surrogate robust to outlier gradients.
    2. Frequency FFT Loss: Penalizes Fourier phase/magnitude discrepancies.
    3. Spatial Gradient Loss: Preserves crisp structural boundaries.
    """
    def __init__(self, fft_w: float = 0.1, grad_w: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.fft_w = fft_w
        self.grad_w = grad_w
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        p, t = pred.float(), target.float()
        
        # 1. Charbonnier
        charbonnier = torch.sqrt((p - t) ** 2 + self.eps).mean()
        total = charbonnier

        # 2. Fourier Space (Frequency Domain Matching)
        if self.fft_w > 0:
            fft_p = torch.fft.rfft2(p, norm="ortho")
            fft_t = torch.fft.rfft2(t, norm="ortho")
            total += self.fft_w * (fft_p - fft_t).abs().mean()

        # 3. Spatial Gradient / Edge Preservation
        if self.grad_w > 0:
            gx = (p[..., :, 1:] - p[..., :, :-1]) - (t[..., :, 1:] - t[..., :, :-1])
            gy = (p[..., 1:, :] - p[..., :-1, :]) - (t[..., 1:, :] - t[..., :-1, :])
            total += self.grad_w * (gx.abs().mean() + gy.abs().mean())

        return total, charbonnier.detach()


def l2sp_regularization(current_model: nn.Module, anchor_weights: dict) -> torch.Tensor:
    """
    L2-SP Penalty (Li et al.):
    Pulls parameter tensors back toward the Stage 1 solution.
    Crucial in Stage 2 to prevent catastrophic forgetting while adapting to streamed web images.
    """
    penalty = torch.tensor(0.0, device=next(current_model.parameters()).device)
    for name, param in current_model.named_parameters():
        if name in anchor_weights:
            penalty += ((param - anchor_weights[name]) ** 2).sum()
    return penalty


# ============================================================================
# EXPONENTIAL MOVING AVERAGE (EMA)
# ============================================================================

class ModelEMA:
    """Maintains an exponential moving average of parameters for smoother inference checkpoints."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        import copy
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int):
        d = min(self.decay, (1 + step) / (10 + step))
        ema_params = list(self.module.parameters())
        model_params = [p.detach() for p in model.parameters()]
        torch._foreach_mul_(ema_params, d)
        torch._foreach_add_(ema_params, model_params, alpha=1.0 - d)


# ============================================================================
# LR SCHEDULING WITH WARM RESTARTS
# ============================================================================

def get_cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float = 1e-6) -> float:
    """Linear warmup followed by cosine annealing terminating exactly at min_lr."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
