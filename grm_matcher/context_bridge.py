"""Transfer pair-conditioned coarse context to localization features."""
import torch
from torch import nn
import torch.nn.functional as F


class GRMContextBridge(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.coarse_dim)
        self.mid = nn.Conv2d(config.coarse_dim, config.mid_dim, 1)
        self.fine = nn.Conv2d(config.coarse_dim, config.fine_dim, 1)
        # Small nonzero gates retain the original fine texture while allowing
        # both gate and projection gradients on the very first update.
        self.gains = nn.Parameter(torch.full((2,), -3.0))

    def forward(self, features, tokens):
        b, c, h, w = features["F_coarse"].shape
        context = self.norm(tokens).transpose(1, 2).reshape(b, c, h, w)
        result = dict(features)
        for i, (key, projection) in enumerate((("F_mid", self.mid), ("F_fine", self.fine))):
            residual = F.interpolate(projection(context), size=features[key].shape[-2:],
                                     mode="bilinear", align_corners=False)
            result[key] = features[key] + self.gains[i].sigmoid() * residual
        return result
