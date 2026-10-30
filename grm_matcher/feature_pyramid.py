"""Single-pass residual feature pyramid for GRM-Matcher."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone import GRMConvNormAct, GRMResidualConvBlock
from .config import GRMMatcherConfig


class GRMResidualFPN(nn.Module):
    """Produce fine/mid/coarse features without repeated bidirectional fusion.

    The fine level receives one restricted high-level contribution controlled
    by a learnable sigmoid ``fine_alpha`` initialized to approximately 0.25.
    """

    required_channels = {"c2": 64, "c3": 128, "c4": 192}

    def __init__(self, config: GRMMatcherConfig) -> None:
        super().__init__()
        self.coarse_lateral = GRMConvNormAct(
            192, config.coarse_dim, kernel_size=1, activation=False
        )
        self.coarse_output = GRMResidualConvBlock(config.coarse_dim, config.coarse_dim)

        self.mid_lateral = GRMConvNormAct(
            128, config.mid_dim, kernel_size=1, activation=False
        )
        self.coarse_to_mid = GRMConvNormAct(
            config.coarse_dim, config.mid_dim, kernel_size=1, activation=False
        )
        self.mid_output = GRMResidualConvBlock(config.mid_dim, config.mid_dim)

        self.fine_lateral = GRMConvNormAct(
            64, config.fine_dim, kernel_size=1, activation=False
        )
        self.mid_to_fine = GRMConvNormAct(
            config.mid_dim, config.fine_dim, kernel_size=1, activation=False
        )
        self.fine_output = GRMResidualConvBlock(config.fine_dim, config.fine_dim)

        alpha = torch.tensor(config.fpn_fine_alpha_init, dtype=torch.float32)
        # Use logit to ensure that the parameter is unconstrained during optimization.
        self._fine_alpha_logit = nn.Parameter(torch.logit(alpha.clamp(1.0e-4, 1.0 - 1.0e-4)))

    @property
    def fine_alpha(self) -> Tensor:
        return torch.sigmoid(self._fine_alpha_logit)

    def forward(self, features: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        self._validate_features(features)

        coarse = self.coarse_output(self.coarse_lateral(features["c4"]))

        coarse_up = F.interpolate(
            self.coarse_to_mid(coarse),
            size=features["c3"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        mid = self.mid_output(self.mid_lateral(features["c3"]) + coarse_up)

        restricted_high = F.interpolate(
            self.mid_to_fine(mid),
            size=features["c2"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fine = self.fine_output(
            self.fine_lateral(features["c2"]) + self.fine_alpha * restricted_high
        )
        return {"F_fine": fine, "F_mid": mid, "F_coarse": coarse}

    def _validate_features(self, features: Mapping[str, Tensor]) -> None:
        # Validate that the input feature dictionary has the expected keys and shapes.
        missing = sorted(set(self.required_channels) - set(features))
        if missing:
            raise ValueError(f"Backbone feature dictionary is missing keys: {missing}.")
        batch = features["c2"].shape[0]
        for name, channels in self.required_channels.items():
            value = features[name]
            if value.ndim != 4 or value.shape[0] != batch or value.shape[1] != channels:
                raise ValueError(
                    f"{name} must be [B, {channels}, H, W], got {tuple(value.shape)}."
                )
        h2, w2 = features["c2"].shape[-2:]
        h3, w3 = features["c3"].shape[-2:]
        h4, w4 = features["c4"].shape[-2:]
        if (h2, w2) != (2 * h3, 2 * w3) or (h3, w3) != (2 * h4, 2 * w4):
            raise ValueError("Expected c2/c3/c4 spatial strides to form a 4/8/16 pyramid.")

