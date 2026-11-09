"""
Instance-segmentation head for RT-DETRv4 (custom addition, not part of upstream).

Query-based masks in the style of MaskFormer / Mask DINO / RF-DETR-Seg:

    mask_features = MaskFeatureHead(encoder S3)          # (B, D, H/4, W/4)
    mask_embed    = MLP(decoder query hidden state)      # (B, Q, D)
    mask_logits   = einsum('bqd,bdhw->bqhw')             # (B, Q, H/4, W/4)

The features come from the highest-resolution output of the hybrid encoder (stride 8
for the S/M/L/X configs) and are upsampled once, to stride 4, which is the grid the
DeepStream-Yolo-Seg parser uses (netW / 4). Mask logits cover the whole network input;
they are only ever materialised for the queries that need them (matched queries in the
loss, top-k queries in the post-processor), never for all Q at once.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation


class ConvBNAct(nn.Sequential):
    def __init__(self, ch_in, ch_out, kernel_size=3, act='silu'):
        super().__init__(
            nn.Conv2d(ch_in, ch_out, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(ch_out),
            get_activation(act),
        )


class MaskFeatureHead(nn.Module):
    """Encoder S3 feature -> per-pixel mask features at twice its resolution."""

    def __init__(self, in_channels, mask_dim=256, act='silu'):
        super().__init__()
        self.reduce = ConvBNAct(in_channels, mask_dim, 3, act)
        self.refine = ConvBNAct(mask_dim, mask_dim, 3, act)
        self.proj = nn.Conv2d(mask_dim, mask_dim, 1)

    def forward(self, x):
        x = self.reduce(x)
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.refine(x)
        return self.proj(x)


def compute_mask_logits(mask_embed, mask_features):
    """(B, K, D) x (B, D, h, w) -> (B, K, h, w)."""
    return torch.einsum('bkd,bdhw->bkhw', mask_embed, mask_features)


def sigmoid_bce_loss(logits, targets):
    """Mean BCE per mask -> (N,)."""
    return F.binary_cross_entropy_with_logits(logits, targets, reduction='none').flatten(1).mean(1)


def dice_loss(logits, targets, eps=1.0):
    """Soft dice per mask -> (N,)."""
    probs = logits.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (probs * targets).sum(1)
    denominator = probs.sum(1) + targets.sum(1)
    return 1 - (numerator + eps) / (denominator + eps)
