"""Independent convolutional and hybrid backbones for GRM-Matcher."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import Tensor, nn

from .config import GRMMatcherConfig
from .coordinates import validate_image_size


GRMFeatureDict = Dict[str, Tensor]


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class GRMConvNormAct(nn.Sequential):
    """Local convolution followed by batch-size-agnostic normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class GRMResidualConvBlock(nn.Module):
    """Two-convolution residual block with optional spatial downsampling."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = GRMConvNormAct(
            in_channels, out_channels, kernel_size=3, stride=stride
        )
        self.conv2 = GRMConvNormAct(
            out_channels, out_channels, kernel_size=3, activation=False
        )
        self.skip: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.skip = GRMConvNormAct(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                activation=False,
            )
        else:
            self.skip = nn.Identity()
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.conv2(self.conv1(x)) + self.skip(x))


def _make_stage(
    in_channels: int,
    out_channels: int,
    block_count: int,
) -> nn.Sequential:
    blocks: list[nn.Module] = [GRMResidualConvBlock(in_channels, out_channels, stride=2)]
    blocks.extend(GRMResidualConvBlock(out_channels, out_channels) for _ in range(block_count - 1))
    return nn.Sequential(*blocks)


class _GRMBackboneBase(nn.Module):
    output_channels = {"c2": 64, "c3": 128, "c4": 192}

    @staticmethod
    def _validate_input(image: Tensor) -> None:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected image [B, 3, H, W], got {tuple(image.shape)}.")
        if not image.is_floating_point():
            raise TypeError("Backbone input must be floating point.")
        validate_image_size(int(image.shape[-2]), int(image.shape[-1]), divisor=16)


class GRMConvBackbone(_GRMBackboneBase):
    """Stable fully local convolutional backbone producing strides 4/8/16."""

    def __init__(self, config: GRMMatcherConfig) -> None:
        super().__init__()
        stem = config.backbone_stem_dim
        c2_blocks, c3_blocks, c4_blocks = config.backbone_stage_blocks
        self.stem = GRMConvNormAct(3, stem, stride=2)
        self.stage_c2 = _make_stage(stem, 64, c2_blocks)
        self.stage_c3 = _make_stage(64, 128, c3_blocks)
        self.stage_c4 = _make_stage(128, 192, c4_blocks)
        self.apply(_initialize_backbone_module)

    def forward(self, image: Tensor) -> GRMFeatureDict:
        self._validate_input(image)
        x = self.stem(image)
        c2 = self.stage_c2(x)
        c3 = self.stage_c3(c2)
        c4 = self.stage_c4(c3)
        return {"c2": c2, "c3": c3, "c4": c4}


class GRMLinearSelfAttention(nn.Module):
    """MobileViTv2-style separable self-attention over stride-16 tokens."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, 1, bias=True)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=True)

    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: [B, Hc*Wc, C]; attention is used only at stride 16.
        context_weights = torch.softmax(self.query(tokens).float(), dim=1).to(tokens.dtype)
        context = torch.sum(self.key(tokens) * context_weights, dim=1, keepdim=True)
        return self.output(self.value(tokens) * context)


class GRMMobileViTv2TokenBlock(nn.Module):
    """Pre-norm linear-attention token block."""

    def __init__(self, dim: int, ffn_multiplier: float) -> None:
        super().__init__()
        hidden_dim = max(dim, int(round(dim * ffn_multiplier)))
        self.norm1 = nn.LayerNorm(dim)
        self.attention = GRMLinearSelfAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = tokens + self.attention(self.norm1(tokens))
        return tokens + self.ffn(self.norm2(tokens))


class GRMMobileViTv2Block(nn.Module):
    """Local representation plus linear global mixing at stride 16 only."""

    def __init__(self, channels: int, depth: int, ffn_multiplier: float) -> None:
        super().__init__()
        self.local_representation = nn.Sequential(
            GRMConvNormAct(channels, channels, groups=channels),
            GRMConvNormAct(channels, channels, kernel_size=1),
        )
        self.token_blocks = nn.ModuleList(
            GRMMobileViTv2TokenBlock(channels, ffn_multiplier) for _ in range(depth)
        )
        self.fusion = GRMConvNormAct(channels * 2, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        local = self.local_representation(x)
        batch, channels, height, width = local.shape
        tokens = local.flatten(2).transpose(1, 2)
        for block in self.token_blocks:
            tokens = block(tokens)
        global_features = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return self.fusion(torch.cat((x, global_features), dim=1))


class GRMHybridMobileViTBackbone(_GRMBackboneBase):
    """Residual c2/c3 stages with MobileViTv2 mixing only in c4."""

    def __init__(self, config: GRMMatcherConfig) -> None:
        super().__init__()
        stem = config.backbone_stem_dim
        c2_blocks, c3_blocks, c4_blocks = config.backbone_stage_blocks
        self.stem = GRMConvNormAct(3, stem, stride=2)
        self.stage_c2 = _make_stage(stem, 64, c2_blocks)
        self.stage_c3 = _make_stage(64, 128, c3_blocks)
        c4_conv_blocks: list[nn.Module] = [GRMResidualConvBlock(128, 192, stride=2)]
        c4_conv_blocks.extend(
            GRMResidualConvBlock(192, 192) for _ in range(c4_blocks - 1)
        )
        self.stage_c4_local = nn.Sequential(*c4_conv_blocks)
        self.stage_c4_mobilevit = GRMMobileViTv2Block(
            192,
            depth=config.mobilevit_depth,
            ffn_multiplier=config.mobilevit_ffn_multiplier,
        )
        self.apply(_initialize_backbone_module)

    def forward(self, image: Tensor) -> GRMFeatureDict:
        self._validate_input(image)
        x = self.stem(image)
        c2 = self.stage_c2(x)
        c3 = self.stage_c3(c2)
        c4 = self.stage_c4_mobilevit(self.stage_c4_local(c3))
        return {"c2": c2, "c3": c3, "c4": c4}


def _initialize_backbone_module(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_grm_backbone(config: GRMMatcherConfig) -> _GRMBackboneBase:
    """Build one of the two namespaced backbone implementations."""

    if config.backbone_name == "conv":
        return GRMConvBackbone(config)
    if config.backbone_name == "hybrid_mobilevit":
        return GRMHybridMobileViTBackbone(config)
    raise ValueError(f"Unsupported GRM backbone: {config.backbone_name!r}.")

