"""Learned context aggregation with an intact F8 localization bypass.

Aggregation positions describe feature support, not correspondence coordinates.
Pool covariance is spatial support dispersion, not matching uncertainty.
"""
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .positional_encoding import GRMPositionEncoding2D


@dataclass
class HRAOutput:
    features: Tensor
    valid: Tensor
    positions: Tensor
    covariance: Tensor
    weights: Tensor


def _windows(value):
    """B,C,H,W -> B,ceil(H/2),ceil(W/2),4,C; order TL/TR/BL/BR."""
    b, c, h, w = value.shape
    value = F.pad(value, (0, w % 2, 0, h % 2))
    return F.unfold(value, 2, stride=2).reshape(b, c, 4, (h+1)//2, (w+1)//2).permute(0, 3, 4, 2, 1)


class HierarchicalReliabilityAggregation(nn.Module):
    """Masked 2x2 feature weighting. Weights express aggregation importance."""
    def __init__(self, channels, hidden=32, input_stride=8):
        super().__init__()
        self.input_stride = input_stride
        self.score = nn.Sequential(nn.Linear(2*channels, hidden), nn.GELU(), nn.Linear(hidden, 1))
        # Start as a masked mean; the scorer learns from correspondence supervision.
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        self.norm = nn.LayerNorm(channels)

    def forward(self, features, valid, positions=None, covariance=None):
        b, _, h, w = features.shape
        if valid.shape != (b, h, w) or valid.dtype != torch.bool:
            raise ValueError("HRA mask must be bool [B,H,W]")
        # Remove padded content before scoring, so even invalid NaN values cannot leak.
        features = features.masked_fill(~valid[:, None], 0)
        tokens = _windows(features)
        mask = _windows(valid[:, None].float()).squeeze(-1).bool()
        mean = (tokens * mask[..., None]).sum(-2) / mask.sum(-1, keepdim=True).clamp_min(1)
        logits = self.score(torch.cat((tokens, tokens-mean[..., None, :]), -1)).squeeze(-1).float()
        weights = logits.masked_fill(~mask, torch.finfo(logits.dtype).min).softmax(-1) * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        parent_valid = mask.any(-1)
        pooled = (tokens * weights[..., None]).sum(-2)
        pooled = self.norm(pooled).masked_fill(~parent_valid[..., None], 0)
        if positions is None:
            y, x = torch.meshgrid(torch.arange(h, device=features.device),
                                  torch.arange(w, device=features.device), indexing="ij")
            positions = ((torch.stack((x, y), -1).float()+.5)*self.input_stride-.5)[None].expand(b, -1, -1, -1)
        positions = positions.masked_fill(~valid[..., None], 0)
        child_positions = _windows(positions.permute(0, 3, 1, 2))
        center = (weights[..., None] * child_positions).sum(-2)
        if covariance is None:
            covariance = positions.new_zeros(b, h, w, 2, 2)
        covariance = covariance.masked_fill(~valid[..., None, None], 0)
        child_covariance = _windows(covariance.reshape(b, h, w, 4).permute(0, 3, 1, 2))
        child_covariance = child_covariance.reshape(*weights.shape, 2, 2)
        delta = child_positions-center[..., None, :]
        total_covariance = (weights[..., None, None] * (
            child_covariance + delta[..., :, None]*delta[..., None, :])).sum(-3)
        return HRAOutput(pooled.permute(0, 3, 1, 2), parent_valid, center, total_covariance, weights)


class GRMHRAContextTransformer(nn.Module):
    """F8 -> learned F16/F32 context -> attention -> gated F8 residual.

The final correlation is still F8 all-pairs; this module only compresses
attention. No hierarchy-based candidate search is claimed here.
"""
    def __init__(self, config, context):
        super().__init__()
        self.pool16 = HierarchicalReliabilityAggregation(config.coarse_dim, config.hra_hidden_dim, 8)
        self.pool32 = HierarchicalReliabilityAggregation(config.coarse_dim, config.hra_hidden_dim, 16)
        self.context = context
        self.position = GRMPositionEncoding2D(config.coarse_dim, temperature=config.positional_temperature)
        self.output_norm = nn.LayerNorm(config.coarse_dim)
        self.gain = nn.Parameter(torch.tensor(-2.0))
        self.max_tokens = config.hra_max_tokens
        self.last_token_counts = None

    def aggregate(self, features, valid):
        if valid is None:
            valid = torch.ones(features.shape[0], *features.shape[-2:], device=features.device, dtype=torch.bool)
        first = self.pool16(features, valid)
        second = self.pool32(first.features, first.valid, first.positions, first.covariance)
        if second.valid.shape[-2]*second.valid.shape[-1] > self.max_tokens:
            raise ValueError("HRA global token budget exceeded; reduce input size or explicitly raise hra_max_tokens")
        return first, second

    def forward(self, a, b, mask_a=None, mask_b=None):
        first_a, low_a = self.aggregate(a, mask_a)
        _, low_b = self.aggregate(b, mask_b)
        self.last_token_counts = (a.shape[-2]*a.shape[-1], first_a.valid.shape[-2]*first_a.valid.shape[-1],
                                  low_a.valid.shape[-2]*low_a.valid.shape[-1])
        ta, tb = self.context(low_a.features, low_b.features, low_a.valid, low_b.valid)
        outputs = []
        for original, tokens, low, mask in ((a, ta, low_a, mask_a), (b, tb, low_b, mask_b)):
            context = self.output_norm(tokens).transpose(1, 2).reshape_as(low.features)
            context = context.masked_fill(~low.valid[:, None], 0)
            context = F.interpolate(context, size=original.shape[-2:], mode="bilinear", align_corners=False)
            output = self.position(original) + self.gain.sigmoid()*context
            if mask is not None:
                output = output.masked_fill(~mask[:, None], 0)
            outputs.append(output.flatten(2).transpose(1, 2))
        return tuple(outputs)
