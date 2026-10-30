"""Within-query candidate reranking and three-way geometry role prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import GRMMatcherConfig


@dataclass
class GRMRerankOutput:
    logits: Tensor
    probabilities: Tensor
    selected_indices: Tensor


@dataclass
class GRMRoleOutput:
    logits: Tensor
    probabilities: Tensor


class GRMCandidateRerankHead(nn.Module):
    """Rerank K candidates using graph context plus explicit appearance prior."""

    def __init__(self, config: GRMMatcherConfig) -> None:
        super().__init__()
        hidden = config.graph_hidden_dim
        self.prior_weight = config.rerank_prior_weight
        self.eps = config.probability_eps
        self.score = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        graph_tokens: Tensor,
        candidate_prior: Tensor,
        candidate_valid_mask: Tensor,
    ) -> GRMRerankOutput:
        if graph_tokens.ndim != 4:
            raise ValueError("graph_tokens must be [B, M, K, C].")
        if candidate_prior.shape != graph_tokens.shape[:3]:
            raise ValueError("candidate_prior must be [B, M, K].")
        if candidate_valid_mask.shape != candidate_prior.shape:
            raise ValueError("candidate_valid_mask must match candidate_prior.")
        logits = self.score(graph_tokens).squeeze(-1)
        logits = logits + self.prior_weight * torch.log(
            candidate_prior.float().clamp_min(self.eps)
        )
        masked_logits = logits.masked_fill(
            ~candidate_valid_mask, torch.finfo(logits.dtype).min
        )
        probabilities = torch.softmax(masked_logits, dim=-1)
        probabilities = probabilities * candidate_valid_mask.float()
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
            self.eps
        )
        return GRMRerankOutput(
            logits=logits,
            probabilities=probabilities,
            selected_indices=probabilities.argmax(dim=-1),
        )


class GRMGeometryRoleHead(nn.Module):
    """Predict Global/Local/False from graph-contextualized match tokens."""

    def __init__(self, config: GRMMatcherConfig) -> None:
        super().__init__()
        hidden = config.graph_hidden_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, selected_tokens: Tensor, valid_mask: Tensor) -> GRMRoleOutput:
        if selected_tokens.ndim != 3 or valid_mask.shape != selected_tokens.shape[:2]:
            raise ValueError("selected_tokens must be [B, N, C] with valid_mask [B, N].")
        logits = self.classifier(selected_tokens)
        probabilities = torch.softmax(logits.float(), dim=-1)
        false_fallback = probabilities.new_tensor((0.0, 0.0, 1.0))
        probabilities = torch.where(
            valid_mask[..., None], probabilities, false_fallback
        )
        return GRMRoleOutput(logits=logits, probabilities=probabilities)

