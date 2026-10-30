"""Dynamic two-dimensional sine/cosine positional encoding."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class GRMPositionEncoding2D(nn.Module):
    """Add deterministic 2D encoding to ``[B, C, H, W]`` features."""

    def __init__(self, dim: int, temperature: float = 10_000.0) -> None:
        super().__init__()
        if dim <= 0 or dim % 4 != 0:
            raise ValueError(f"dim must be positive and divisible by four, got {dim}.")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        self.dim = dim
        self.temperature = temperature

    def forward(self, feature: Tensor) -> Tensor:
        if feature.ndim != 4 or feature.shape[1] != self.dim:
            raise ValueError(
                f"Expected [B, {self.dim}, H, W], got {tuple(feature.shape)}."
            )
        _, _, height, width = feature.shape
        encoding = self.make_encoding(
            height,
            width,
            device=feature.device,
            dtype=feature.dtype,
        )
        return feature + encoding.unsqueeze(0)

    def make_encoding(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return a channel-first ``[C, H, W]`` encoding."""

        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive.")
        axis_dim = self.dim // 2
        frequency_index = torch.arange(0, axis_dim, 2, device=device, dtype=torch.float32)
        frequencies = torch.exp(
            -math.log(self.temperature) * frequency_index / float(axis_dim)
        )
        x = torch.arange(width, device=device, dtype=torch.float32)[:, None] * frequencies[None]
        y = torch.arange(height, device=device, dtype=torch.float32)[:, None] * frequencies[None]
        x_encoding = torch.stack((x.sin(), x.cos()), dim=-1).flatten(1)
        y_encoding = torch.stack((y.sin(), y.cos()), dim=-1).flatten(1)
        x_map = x_encoding.transpose(0, 1).unsqueeze(1).expand(-1, height, -1)
        y_map = y_encoding.transpose(0, 1).unsqueeze(2).expand(-1, -1, width)
        return torch.cat((x_map, y_map), dim=0).to(dtype=dtype)

