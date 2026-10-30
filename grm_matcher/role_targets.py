"""Single source of truth for GRM geometry-role supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from .coordinates import image_xy_to_normalized_grid


@dataclass(frozen=True)
class GRMGeometryRoleTargets:
    """Role labels sampled at public matcher source coordinates.

    A correspondence is first judged against the exact dense target mapping.
    Only a correct correspondence may be labelled Global or Local.  The
    Global/Local split is read from the dataset's residual-to-``H_ab`` masks;
    it is deliberately independent from the correspondence correctness
    threshold.
    """

    labels: Tensor
    gt_dst_xy: Tensor
    correspondence_error: Tensor
    correct_mask: Tensor
    overlap_mask: Tensor
    global_mask: Tensor
    local_mask: Tensor


def build_geometry_role_targets(
    src_xy: Tensor,
    predicted_dst_xy: Tensor,
    targets: Mapping[str, Tensor],
    *,
    correspondence_correct_threshold: float,
) -> GRMGeometryRoleTargets:
    """Build ``Global=0, Local=1, False=2`` labels for predicted matches.

    ``correspondence_correct_threshold`` measures error to the exact dense
    correspondence.  It must not be replaced by ``local_residual_threshold``:
    that second threshold measures deviation of a *correct* match from the
    global Homography and therefore answers a different question.
    """

    if correspondence_correct_threshold <= 0.0:
        raise ValueError("correspondence_correct_threshold must be positive.")
    if src_xy.shape != predicted_dst_xy.shape or src_xy.shape[-1] != 2:
        raise ValueError("src_xy and predicted_dst_xy must have matching [B,N,2] shapes.")
    required = {
        "gt_dense_mapping",
        "gt_overlap_a",
        "gt_global_mask",
        "gt_local_mask",
    }
    missing = sorted(required - set(targets))
    if missing:
        raise ValueError(f"Role targets are missing keys: {missing}.")

    gt_dst_xy = sample_role_field(targets["gt_dense_mapping"], src_xy)
    overlap = sample_role_field(
        targets["gt_overlap_a"].float(), src_xy
    )[..., 0] > 0.5
    global_mask = sample_role_field(
        targets["gt_global_mask"].float(), src_xy
    )[..., 0] > 0.5
    local_mask = sample_role_field(
        targets["gt_local_mask"].float(), src_xy
    )[..., 0] > 0.5
    correspondence_error = torch.linalg.vector_norm(
        predicted_dst_xy - gt_dst_xy, dim=-1
    )
    correct = overlap & (
        correspondence_error <= float(correspondence_correct_threshold)
    )

    labels = torch.full(
        src_xy.shape[:2],
        2,
        dtype=torch.long,
        device=src_xy.device,
    )
    labels[correct & global_mask] = 0
    labels[correct & local_mask & ~global_mask] = 1
    return GRMGeometryRoleTargets(
        labels=labels,
        gt_dst_xy=gt_dst_xy,
        correspondence_error=correspondence_error,
        correct_mask=correct,
        overlap_mask=overlap,
        global_mask=global_mask,
        local_mask=local_mask,
    )


def sample_role_field(field: Tensor, points_xy: Tensor) -> Tensor:
    """Bilinearly sample ``[B,C,H,W]`` fields at image-pixel ``(x,y)`` points."""

    if field.ndim != 4 or points_xy.ndim != 3 or points_xy.shape[-1] != 2:
        raise ValueError("field must be [B,C,H,W] and points_xy must be [B,N,2].")
    if field.shape[0] != points_xy.shape[0]:
        raise ValueError("field and points_xy batch sizes must match.")
    grid = image_xy_to_normalized_grid(
        points_xy.float(), field.shape[-2], field.shape[-1]
    ).unsqueeze(2)
    return F.grid_sample(
        field.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(-1).transpose(1, 2)
