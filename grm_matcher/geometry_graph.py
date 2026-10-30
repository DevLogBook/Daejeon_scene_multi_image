"""Candidate token construction and sparse local geometry graph attention."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .config import GRMMatcherConfig


@dataclass
class GRMGeometryTokenInputs:
    feature_a: Tensor
    feature_b: Tensor
    coarse_probability: Tensor
    mid_probability: Tensor
    fine_probability: Tensor
    candidate_entropy: Tensor
    top1_top2_margin: Tensor
    matchability_a: Tensor
    matchability_b: Tensor
    src_xy: Tensor
    dst_xy: Tensor
    seed_residual_xy: Tensor
    symmetric_transfer_error: Tensor
    cycle_error: Tensor
    seed_valid: Tensor
    fine_covariance: Tensor
    valid_mask: Tensor
    image_height: int
    image_width: int


class GRMCandidateTokenEncoder(nn.Module):
    """Encode appearance, probability, coordinates, geometry, and uncertainty."""

    def __init__(self, config: GRMMatcherConfig) -> None:
        super().__init__()
        appearance_dim = config.role_appearance_dim
        hidden_dim = config.graph_hidden_dim
        self.feature_projection = nn.Sequential(
            nn.Linear(config.fine_dim, appearance_dim),
            nn.LayerNorm(appearance_dim),
            nn.GELU(),
        )
        # 4 appearance blocks + 7 probabilities + 6 coordinates/displacement
        # + 5 seed/cycle geometry values + 3 covariance values.
        input_dim = 4 * appearance_dim + 7 + 6 + 5 + 3
        self.token_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, inputs: GRMGeometryTokenInputs) -> Tensor:
        self._validate(inputs)
        projected_a = self.feature_projection(inputs.feature_a.float())
        projected_b = self.feature_projection(inputs.feature_b.float())
        appearance = torch.cat(
            (
                projected_a,
                projected_b,
                projected_a - projected_b,
                projected_a * projected_b,
            ),
            dim=-1,
        )
        probabilities = torch.stack(
            (
                inputs.coarse_probability,
                inputs.mid_probability,
                inputs.fine_probability,
                inputs.candidate_entropy,
                inputs.top1_top2_margin,
                inputs.matchability_a,
                inputs.matchability_b,
            ),
            dim=-1,
        ).float()

        scale = inputs.src_xy.new_tensor(
            (max(inputs.image_width - 1, 1), max(inputs.image_height - 1, 1))
        )
        source_normalized = 2.0 * inputs.src_xy.float() / scale - 1.0
        target_normalized = 2.0 * inputs.dst_xy.float() / scale - 1.0
        displacement_normalized = 2.0 * (
            inputs.dst_xy.float() - inputs.src_xy.float()
        ) / scale
        coordinates = torch.cat(
            (source_normalized, target_normalized, displacement_normalized), dim=-1
        )

        diagonal = math.hypot(inputs.image_width, inputs.image_height)
        geometry = torch.cat(
            (
                inputs.seed_residual_xy.float() / diagonal,
                (inputs.symmetric_transfer_error.float() / diagonal)[..., None],
                (inputs.cycle_error.float() / diagonal)[..., None],
                inputs.seed_valid.float()[..., None],
            ),
            dim=-1,
        )
        covariance = inputs.fine_covariance.float() / (diagonal * diagonal)
        uncertainty = torch.stack(
            (covariance[..., 0, 0], covariance[..., 1, 1], covariance[..., 0, 1]),
            dim=-1,
        )
        token = self.token_projection(
            torch.cat((appearance, probabilities, coordinates, geometry, uncertainty), dim=-1)
        )
        return token * inputs.valid_mask[..., None].float()

    @staticmethod
    def _validate(inputs: GRMGeometryTokenInputs) -> None:
        if inputs.src_xy.ndim != 3 or inputs.src_xy.shape[-1] != 2:
            raise ValueError("src_xy must be [B, N, 2].")
        batch, count, _ = inputs.src_xy.shape
        if inputs.dst_xy.shape != (batch, count, 2):
            raise ValueError("dst_xy must match src_xy.")
        if inputs.feature_a.shape[:2] != (batch, count):
            raise ValueError("feature_a must be [B, N, C].")
        if inputs.feature_b.shape != inputs.feature_a.shape:
            raise ValueError("feature_b must match feature_a.")
        for name in (
            "coarse_probability",
            "mid_probability",
            "fine_probability",
            "candidate_entropy",
            "top1_top2_margin",
            "matchability_a",
            "matchability_b",
            "symmetric_transfer_error",
            "cycle_error",
            "seed_valid",
            "valid_mask",
        ):
            if getattr(inputs, name).shape != (batch, count):
                raise ValueError(f"{name} must be [B, N].")
        if inputs.seed_residual_xy.shape != (batch, count, 2):
            raise ValueError("seed_residual_xy must be [B, N, 2].")
        if inputs.fine_covariance.shape != (batch, count, 2, 2):
            raise ValueError("fine_covariance must be [B, N, 2, 2].")
        if inputs.valid_mask.dtype != torch.bool or inputs.seed_valid.dtype != torch.bool:
            raise TypeError("valid_mask and seed_valid must be boolean.")
        if inputs.image_height <= 0 or inputs.image_width <= 0:
            raise ValueError("Image dimensions must be positive.")


class GRMLocalGraphAttentionLayer(nn.Module):
    """Multi-head attention over k gathered neighbors and encoded edges."""

    def __init__(self, hidden_dim: int, heads: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_key = nn.Linear(edge_dim, hidden_dim)
        self.edge_value = nn.Linear(edge_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: Tensor,
        neighbor_indices: Tensor,
        edge_features: Tensor,
        neighbor_valid: Tensor,
        node_valid: Tensor,
    ) -> Tensor:
        batch, count, hidden = tokens.shape
        neighbors = neighbor_indices.shape[-1]
        normalized = self.norm1(tokens)
        batch_indices = torch.arange(batch, device=tokens.device)[:, None, None]
        neighbor_tokens = normalized[batch_indices, neighbor_indices]
        query = self.query(normalized).reshape(batch, count, self.heads, self.head_dim)
        key = (
            self.key(neighbor_tokens) + self.edge_key(edge_features)
        ).reshape(batch, count, neighbors, self.heads, self.head_dim)
        value = (
            self.value(neighbor_tokens) + self.edge_value(edge_features)
        ).reshape(batch, count, neighbors, self.heads, self.head_dim)
        logits = torch.einsum("bnhd,bnkhd->bnkh", query, key) / math.sqrt(self.head_dim)
        logits = logits.masked_fill(
            ~neighbor_valid[..., None], torch.finfo(logits.dtype).min
        )
        attention = torch.softmax(logits, dim=2) * neighbor_valid[..., None].float()
        attention = attention / attention.sum(dim=2, keepdim=True).clamp_min(1.0e-8)
        update = torch.einsum("bnkh,bnkhd->bnhd", attention, value).reshape(
            batch, count, hidden
        )
        tokens = tokens + self.dropout(self.output(update))
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens * node_valid[..., None].float()


class GRMGeometryGraph(nn.Module):
    """Two-layer sparse kNN geometry graph over candidate matches."""

    edge_dim = 9

    def __init__(self, config: GRMMatcherConfig) -> None:
        super().__init__()
        self.hidden_dim = config.graph_hidden_dim
        self.neighbors = config.graph_neighbors
        self.use_target_neighbors = config.graph_use_target_neighbors
        self.layers = nn.ModuleList(
            GRMLocalGraphAttentionLayer(
                config.graph_hidden_dim,
                config.graph_heads,
                self.edge_dim,
                config.graph_dropout,
            )
            for _ in range(config.graph_layers)
        )
        self.last_memory_estimate_bytes = 0

    def forward(
        self,
        tokens: Tensor,
        src_xy: Tensor,
        dst_xy: Tensor,
        seed_error: Tensor,
        valid_mask: Tensor,
        *,
        image_height: int,
        image_width: int,
    ) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_dim:
            raise ValueError("tokens must be [B, N, graph_hidden_dim].")
        batch, count, _ = tokens.shape
        if count == 0:
            raise ValueError("Geometry graph requires a padded non-empty node dimension.")
        if src_xy.shape != (batch, count, 2) or dst_xy.shape != (batch, count, 2):
            raise ValueError("src_xy and dst_xy must be [B, N, 2].")
        if seed_error.shape != (batch, count) or valid_mask.shape != (batch, count):
            raise ValueError("seed_error and valid_mask must be [B, N].")

        scale = src_xy.new_tensor(
            (max(image_width - 1, 1), max(image_height - 1, 1))
        )
        src = src_xy.float() / scale
        dst = dst_xy.float() / scale
        distance = torch.cdist(src, src)
        if self.use_target_neighbors:
            distance = distance + torch.cdist(dst, dst)
        pair_valid = valid_mask[:, :, None] & valid_mask[:, None, :]
        identity = torch.eye(count, device=tokens.device, dtype=torch.bool)[None]
        pair_valid = pair_valid & ~identity
        distance = distance.masked_fill(~pair_valid, torch.finfo(distance.dtype).max)
        neighbor_count = min(self.neighbors, max(count - 1, 1))
        neighbor_indices = torch.topk(
            distance, k=neighbor_count, dim=-1, largest=False, sorted=True
        ).indices
        batch_indices = torch.arange(batch, device=tokens.device)[:, None, None]
        neighbor_valid = valid_mask[batch_indices, neighbor_indices]
        neighbor_valid = neighbor_valid & valid_mask[..., None]
        neighbor_valid = neighbor_valid & (
            neighbor_indices
            != torch.arange(count, device=tokens.device)[None, :, None]
        )

        neighbor_src = src[batch_indices, neighbor_indices]
        neighbor_dst = dst[batch_indices, neighbor_indices]
        neighbor_seed = seed_error.float()[batch_indices, neighbor_indices]
        delta_src = neighbor_src - src[:, :, None]
        delta_dst = neighbor_dst - dst[:, :, None]
        displacement = dst - src
        neighbor_displacement = neighbor_dst - neighbor_src
        delta_displacement = neighbor_displacement - displacement[:, :, None]
        diagonal = math.hypot(image_width, image_height)
        delta_seed = (neighbor_seed - seed_error.float()[:, :, None]) / diagonal
        edge_features = torch.cat(
            (
                delta_src,
                delta_dst,
                delta_displacement,
                delta_seed[..., None],
                torch.linalg.vector_norm(delta_src, dim=-1, keepdim=True),
                torch.linalg.vector_norm(delta_dst, dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        self.last_memory_estimate_bytes = self.estimate_memory_bytes(
            batch, count, neighbor_count, tokens.element_size()
        )
        output = tokens
        for layer in self.layers:
            output = layer(
                output,
                neighbor_indices,
                edge_features,
                neighbor_valid,
                valid_mask,
            )
        return output

    def estimate_memory_bytes(
        self, batch: int, nodes: int, neighbors: int, element_size: int = 4
    ) -> int:
        """Approximate edge-attention working memory, excluding parameters."""

        per_edge_values = 3 * self.hidden_dim + self.edge_dim + len(self.layers)
        return int(batch * nodes * neighbors * per_edge_values * element_size)

