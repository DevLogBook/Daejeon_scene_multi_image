"""Utilities for safe Stage-1/2/3 joint fine-tuning.

The mixed loader keeps every batch resolution-homogeneous by interleaving
whole Homography and controlled-Role batches rather than concatenating samples
with potentially different spatial sizes.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Iterator, Mapping

import torch
from torch import Tensor


JOINT_SOURCE_HOMOGRAPHY = 0
JOINT_SOURCE_ROLE_AUGMENTED = 1


def joint_source_name(batch: Mapping[str, object]) -> str | None:
    """Return the homogeneous Joint source annotation for one batch."""

    value = batch.get("joint_source_id")
    if value is None:
        return None
    if not isinstance(value, Tensor):
        raise TypeError("joint_source_id must be a scalar Tensor.")
    unique = torch.unique(value.detach().cpu().long())
    if unique.numel() != 1:
        raise ValueError("A Joint batch must contain exactly one data source.")
    source_id = int(unique.item())
    if source_id == JOINT_SOURCE_HOMOGRAPHY:
        return "homography"
    if source_id == JOINT_SOURCE_ROLE_AUGMENTED:
        return "role_augmented"
    raise ValueError(f"Unknown joint_source_id={source_id}.")


class GRMJointMixedBatchLoader(
    Iterable[Dict[str, object]]
):
    """Interleave full batches from exact-H and controlled-Role loaders.

    Every Homography batch is consumed exactly once per epoch.  The controlled
    Role loader is cycled when necessary to achieve the requested batch ratio.
    """

    def __init__(
        self,
        homography_loader: Iterable[Mapping[str, object]],
        role_loader: Iterable[Mapping[str, object]],
        *,
        homography_batches: int = 2,
        role_batches: int = 1,
    ) -> None:
        if homography_batches <= 0 or role_batches <= 0:
            raise ValueError("Joint batch ratios must be positive.")
        if not hasattr(homography_loader, "__len__") or not hasattr(role_loader, "__len__"):
            raise TypeError("Joint component loaders must define __len__.")
        if len(homography_loader) <= 0 or len(role_loader) <= 0:  # type: ignore[arg-type]
            raise ValueError("Joint component loaders must both be non-empty.")
        self.homography_loader = homography_loader
        self.role_loader = role_loader
        self.homography_batches = int(homography_batches)
        self.role_batches = int(role_batches)
        self.role_batch_count = (
            math.ceil(len(homography_loader) / self.homography_batches)  # type: ignore[arg-type]
            * self.role_batches
        )

    def __len__(self) -> int:
        return len(self.homography_loader) + self.role_batch_count  # type: ignore[arg-type]

    def set_epoch(self, epoch: int) -> None:
        """Propagate deterministic epoch state to both bucket samplers."""

        for loader in (self.homography_loader, self.role_loader):
            sampler = getattr(loader, "batch_sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

    @staticmethod
    def _tag(
        batch: Mapping[str, object],
        source_id: int,
    ) -> Dict[str, object]:
        tagged = dict(batch)
        tagged["joint_source_id"] = torch.tensor(source_id, dtype=torch.long)
        return tagged

    def __iter__(self) -> Iterator[Dict[str, object]]:
        role_iterator = iter(self.role_loader)
        role_emitted = 0

        def next_role() -> Mapping[str, object]:
            nonlocal role_iterator
            try:
                return next(role_iterator)
            except StopIteration:
                role_iterator = iter(self.role_loader)
                return next(role_iterator)

        for homography_index, batch in enumerate(self.homography_loader, start=1):
            yield self._tag(batch, JOINT_SOURCE_HOMOGRAPHY)
            if homography_index % self.homography_batches == 0:
                for _ in range(self.role_batches):
                    if role_emitted >= self.role_batch_count:
                        break
                    yield self._tag(next_role(), JOINT_SOURCE_ROLE_AUGMENTED)
                    role_emitted += 1

        while role_emitted < self.role_batch_count:
            yield self._tag(next_role(), JOINT_SOURCE_ROLE_AUGMENTED)
            role_emitted += 1


def merge_joint_validation_metrics(
    homography: Mapping[str, float],
    role_augmented: Mapping[str, float],
) -> Dict[str, float]:
    """Merge two validation domains and expose canonical checkpoint metrics."""

    merged = {
        **{f"homography_{name}": float(value) for name, value in homography.items()},
        **{
            f"role_augmented_{name}": float(value)
            for name, value in role_augmented.items()
        },
    }
    required_homography = {
        "candidate_recall",
        "pixel_error",
        "fine_top1_pck_3",
    }
    missing = sorted(required_homography - set(homography))
    if missing:
        raise KeyError(f"Homography Joint validation is missing metrics: {missing}.")
    if "role_f1" not in role_augmented:
        raise KeyError("Role-augmented Joint validation is missing role_f1.")
    merged["candidate_recall"] = float(homography["candidate_recall"])
    merged["pixel_error"] = float(homography["pixel_error"])
    merged["fine_top1_pck_3"] = float(homography["fine_top1_pck_3"])
    merged["role_f1"] = float(role_augmented["role_f1"])
    return merged


def apply_joint_checkpoint_gate(
    metrics: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    maximum_recall_drop: float,
    maximum_pixel_error_ratio: float,
    maximum_pck3_drop: float,
) -> Dict[str, float]:
    """Attach an eligibility flag and constraint-aware Joint score."""

    if maximum_recall_drop < 0.0 or maximum_pck3_drop < 0.0:
        raise ValueError("Joint metric-drop tolerances must be nonnegative.")
    if maximum_pixel_error_ratio < 1.0:
        raise ValueError("maximum_pixel_error_ratio must be at least one.")
    required = {"candidate_recall", "pixel_error", "fine_top1_pck_3", "role_f1"}
    missing_metrics = sorted(required - set(metrics))
    missing_baseline = sorted(required - set(baseline))
    if missing_metrics or missing_baseline:
        raise KeyError(
            "Joint gate is missing metrics; "
            f"current={missing_metrics}, baseline={missing_baseline}."
        )
    pixel_limit = max(
        float(baseline["pixel_error"]) * maximum_pixel_error_ratio,
        float(baseline["pixel_error"]) + 1.0e-6,
    )
    eligible = (
        float(metrics["candidate_recall"])
        >= float(baseline["candidate_recall"]) - maximum_recall_drop
        and float(metrics["pixel_error"]) <= pixel_limit
        and float(metrics["fine_top1_pck_3"])
        >= float(baseline["fine_top1_pck_3"]) - maximum_pck3_drop
    )
    result = dict(metrics)
    result["joint_eligible"] = float(eligible)
    # Keep the score finite for JSON/TensorBoard.  Checkpoint selection uses
    # joint_eligible as a hard gate before comparing this Role score.
    result["joint_score"] = float(metrics["role_f1"])
    result["joint_recall_floor"] = (
        float(baseline["candidate_recall"]) - maximum_recall_drop
    )
    result["joint_pixel_error_ceiling"] = pixel_limit
    result["joint_pck3_floor"] = (
        float(baseline["fine_top1_pck_3"]) - maximum_pck3_drop
    )
    return result
