"""LoFTR-style coarse context interaction without coordinate regression."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .config import GRMMatcherConfig
from .positional_encoding import GRMPositionEncoding2D


class GRMPreNormAttentionBlock(nn.Module):
    """Pre-norm attention plus feed-forward residual update."""

    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: Tensor,
        context: Optional[Tensor],
        query_valid_mask: Tensor,
        context_valid_mask: Optional[Tensor],
    ) -> Tensor:
        is_self_attention = context is None
        normalized_query = self.query_norm(query)
        if is_self_attention:
            normalized_context = normalized_query
            context_valid = query_valid_mask
        else:
            assert context is not None and context_valid_mask is not None
            normalized_context = self.context_norm(context)
            context_valid = context_valid_mask

        key_padding_mask = ~context_valid
        all_context_invalid = key_padding_mask.all(dim=1)
        if all_context_invalid.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_context_invalid, 0] = False
            normalized_context = normalized_context.clone()
            normalized_context[all_context_invalid] = 0.0

        attended, _ = self.attention(
            normalized_query,
            normalized_context,
            normalized_context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        if all_context_invalid.any():
            attended = attended.masked_fill(all_context_invalid[:, None, None], 0.0)
        output = query + self.attention_dropout(attended)
        output = output + self.ffn(self.ffn_norm(output))
        return output.masked_fill(~query_valid_mask[..., None], 0.0)


class GRMCoarseContextTransformer(nn.Module):
    """Shared bilateral ``self/cross/self/cross`` coarse interaction."""

    layer_types = ("self", "cross", "self", "cross")

    def __init__(self, config: GRMMatcherConfig) -> None:
        super().__init__()
        if config.transformer_layers != len(self.layer_types):
            raise ValueError("GRM v1 requires self/cross/self/cross (four layers).")
        self.dim = config.coarse_dim
        self.position = GRMPositionEncoding2D(
            config.coarse_dim,
            temperature=config.positional_temperature,
        )
        self.layers = nn.ModuleList(
            GRMPreNormAttentionBlock(
                config.coarse_dim,
                config.transformer_heads,
                config.transformer_ffn_dim,
                config.transformer_dropout,
            )
            for _ in self.layer_types
        )

    def forward(
        self,
        feature_a: Tensor,
        feature_b: Tensor,
        valid_mask_a: Optional[Tensor] = None,
        valid_mask_b: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return enhanced flattened tokens ``[B, Hc*Wc, C]`` for A and B."""

        self._validate_feature_pair(feature_a, feature_b)
        batch = feature_a.shape[0]
        mask_a = self._prepare_mask(valid_mask_a, feature_a, batch)
        mask_b = self._prepare_mask(valid_mask_b, feature_b, batch)

        tokens_a = self.position(feature_a).flatten(2).transpose(1, 2)
        tokens_b = self.position(feature_b).flatten(2).transpose(1, 2)
        flat_mask_a = mask_a.flatten(1)
        flat_mask_b = mask_b.flatten(1)

        for layer_type, layer in zip(self.layer_types, self.layers):
            if layer_type == "self":
                tokens_a = layer(tokens_a, None, flat_mask_a, None)
                tokens_b = layer(tokens_b, None, flat_mask_b, None)
            else:
                previous_a, previous_b = tokens_a, tokens_b
                tokens_a = layer(previous_a, previous_b, flat_mask_a, flat_mask_b)
                tokens_b = layer(previous_b, previous_a, flat_mask_b, flat_mask_a)
        return tokens_a, tokens_b

    def _validate_feature_pair(self, feature_a: Tensor, feature_b: Tensor) -> None:
        for name, feature in (("feature_a", feature_a), ("feature_b", feature_b)):
            if feature.ndim != 4 or feature.shape[1] != self.dim:
                raise ValueError(
                    f"{name} must be [B, {self.dim}, Hc, Wc], got {tuple(feature.shape)}."
                )
        if feature_a.shape[0] != feature_b.shape[0]:
            raise ValueError("feature_a and feature_b must share the batch dimension.")

    @staticmethod
    def _prepare_mask(mask: Optional[Tensor], feature: Tensor, batch: int) -> Tensor:
        expected = (batch, feature.shape[-2], feature.shape[-1])
        if mask is None:
            return torch.ones(expected, device=feature.device, dtype=torch.bool)
        if mask.shape != expected or mask.dtype != torch.bool:
            raise ValueError(f"Valid mask must be bool {expected}, got {tuple(mask.shape)}.")
        return mask.to(device=feature.device)

