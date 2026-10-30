"""Stride-4 subpixel GRM local refinement."""

from __future__ import annotations

from .config import GRMMatcherConfig
from .local_correlation import GRMLocalRefinerCore


class GRMFineRefiner(GRMLocalRefinerCore):
    """Refine mid-scale targets and expose probability-derived covariance."""

    def __init__(self, config: GRMMatcherConfig) -> None:
        super().__init__(
            channels=config.fine_dim,
            stride=config.fine_stride,
            source_patch_size=config.fine_source_patch_size,
            target_window_size=config.fine_window_size,
            temperature=config.local_temperature,
            eps=config.probability_eps,
        )
        self.correlation.peak_localization = config.architecture_version == "v2"

