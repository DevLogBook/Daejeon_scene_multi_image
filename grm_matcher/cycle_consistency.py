"""Continuous bidirectional matching consistency for GRM-Matcher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class GRMCycleConsistencyOutput:
    returned_src_xy: Tensor
    backward_indices: Tensor
    cycle_error: Tensor
    cycle_score: Tensor
    valid_mask: Tensor


def compute_cycle_consistency(
    src_xy: Tensor,
    dst_xy: Tensor,
    backward_src_xy: Tensor,
    backward_dst_xy: Tensor,
    *,
    sigma: float,
    forward_valid_mask: Optional[Tensor] = None,
    backward_valid_mask: Optional[Tensor] = None,
) -> GRMCycleConsistencyOutput:
    """Close ``A_i -> B_j -> A_i'`` using nearest backward source points.

    The function emits a continuous score and does not hard-filter large cycle
    errors. Invalid cycles receive finite error ``1e6`` and score zero.
    """

    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    for name, points in (
        ("src_xy", src_xy),
        ("dst_xy", dst_xy),
        ("backward_src_xy", backward_src_xy),
        ("backward_dst_xy", backward_dst_xy),
    ):
        if points.ndim != 3 or points.shape[-1] != 2 or not points.is_floating_point():
            raise ValueError(f"{name} must be a floating [B, N, 2] tensor.")
    if src_xy.shape != dst_xy.shape:
        raise ValueError("Forward source and destination shapes must match.")
    if backward_src_xy.shape != backward_dst_xy.shape:
        raise ValueError("Backward source and destination shapes must match.")
    if src_xy.shape[0] != backward_src_xy.shape[0]:
        raise ValueError("Forward and backward batches must match.")

    batch, forward_count, _ = src_xy.shape
    backward_count = backward_src_xy.shape[1]
    if backward_count == 0:
        raise ValueError("Backward match dimension must be non-empty; use a padded invalid row.")
    if forward_valid_mask is None:
        forward_valid_mask = torch.ones(
            batch, forward_count, device=src_xy.device, dtype=torch.bool
        )
    if backward_valid_mask is None:
        backward_valid_mask = torch.ones(
            batch, backward_count, device=src_xy.device, dtype=torch.bool
        )
    if forward_valid_mask.shape != (batch, forward_count):
        raise ValueError("forward_valid_mask has the wrong shape.")
    if backward_valid_mask.shape != (batch, backward_count):
        raise ValueError("backward_valid_mask has the wrong shape.")

    distance = torch.cdist(dst_xy.float(), backward_src_xy.float())
    distance = distance.masked_fill(
        ~backward_valid_mask[:, None, :], torch.finfo(distance.dtype).max
    )
    backward_indices = distance.argmin(dim=-1)
    returned_src = torch.gather(
        backward_dst_xy.float(),
        dim=1,
        index=backward_indices[..., None].expand(-1, -1, 2),
    )
    has_backward = backward_valid_mask.any(dim=1, keepdim=True)
    valid = forward_valid_mask & has_backward
    raw_error = torch.linalg.vector_norm(src_xy.float() - returned_src, dim=-1)
    invalid_error = raw_error.new_full(raw_error.shape, 1.0e6)
    cycle_error = torch.where(valid, raw_error, invalid_error)
    cycle_score = torch.exp(-cycle_error / sigma) * valid.float()
    returned_src = torch.where(valid[..., None], returned_src, src_xy.float())
    return GRMCycleConsistencyOutput(
        returned_src_xy=returned_src,
        backward_indices=backward_indices,
        cycle_error=cycle_error,
        cycle_score=cycle_score,
        valid_mask=valid,
    )

