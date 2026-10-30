"""Stride-8 GRM local refinement."""

from __future__ import annotations

from torch import Tensor

from .config import GRMMatcherConfig
from .local_correlation import GRMLocalRefinerCore, GRMRefinementOutput


class GRMMidRefiner(GRMLocalRefinerCore):
    """Refine coarse target hypotheses in a stride-8 search window."""

    def __init__(self, config: GRMMatcherConfig) -> None:
        super().__init__(
            channels=config.mid_dim,
            stride=config.mid_stride,
            source_patch_size=config.mid_source_patch_size,
            target_window_size=config.mid_window_size,
            temperature=config.local_temperature,
            eps=config.probability_eps,
        )

