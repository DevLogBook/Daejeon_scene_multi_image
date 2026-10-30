"""Robust A-to-B Seed Homography used only as detached geometric context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .config import GRMMatcherConfig

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    cv2 = None


@dataclass
class GRMSeedGeometryOutput:
    homography_ab: Tensor
    homography_valid: Tensor
    selected_mask: Tensor
    inlier_mask: Tensor
    residual_xy: Tensor
    forward_residual: Tensor
    backward_residual: Tensor
    symmetric_transfer_error: Tensor
    failure_reasons: Tuple[str, ...]


def apply_homography(points_xy: Tensor, homography: Tensor, eps: float = 1.0e-8) -> Tensor:
    """Apply ``H_ab`` to A points, returning B ``(x, y)`` pixels."""

    if points_xy.ndim != 3 or points_xy.shape[-1] != 2:
        raise ValueError("points_xy must be [B, N, 2].")
    if homography.shape != (points_xy.shape[0], 3, 3):
        raise ValueError("homography must be [B, 3, 3].")
    ones = torch.ones_like(points_xy[..., :1])
    homogeneous = torch.cat((points_xy, ones), dim=-1)
    projected = torch.einsum("bij,bnj->bni", homography, homogeneous)
    denominator = projected[..., 2:3]
    safe_sign = torch.where(denominator >= 0.0, 1.0, -1.0)
    safe_denominator = torch.where(
        denominator.abs() >= eps, denominator, safe_sign * eps
    )
    return projected[..., :2] / safe_denominator


def estimate_dlt_homography(src_xy: Tensor, dst_xy: Tensor) -> Tensor:
    """Differentiable normalized-scale DLT for already selected inliers."""

    if src_xy.ndim != 2 or src_xy.shape[-1] != 2 or src_xy.shape != dst_xy.shape:
        raise ValueError("src_xy and dst_xy must both be [N, 2].")
    if src_xy.shape[0] < 4:
        raise ValueError("At least four correspondences are required for DLT.")
    src = src_xy.float()
    dst = dst_xy.float()
    zeros = torch.zeros_like(src[:, 0])
    ones = torch.ones_like(src[:, 0])
    x, y = src[:, 0], src[:, 1]
    u, v = dst[:, 0], dst[:, 1]
    row_u = torch.stack((-x, -y, -ones, zeros, zeros, zeros, u * x, u * y, u), dim=-1)
    row_v = torch.stack((zeros, zeros, zeros, -x, -y, -ones, v * x, v * y, v), dim=-1)
    matrix = torch.stack((row_u, row_v), dim=1).reshape(-1, 9)
    if src_xy.shape[0] == 4:
        # Reduced SVD of an 8x9 matrix omits its nullspace. Fix h33=1
        # and solve the nondegenerate four-point system, preserving gradients.
        coefficients = torch.linalg.solve(matrix[:, :8], -matrix[:, 8])
        return torch.cat((coefficients, coefficients.new_ones(1))).reshape(3, 3)
    _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    homography = vh[-1].reshape(3, 3)
    denominator = homography[2, 2]
    safe_denominator = torch.where(
        denominator.abs() >= 1.0e-8,
        denominator,
        torch.where(denominator >= 0.0, denominator.new_tensor(1.0e-8), denominator.new_tensor(-1.0e-8)),
    )
    return homography / safe_denominator


class GRMSeedHomographyEstimator(nn.Module):
    """Select reliable seeds, estimate robust H_ab, and compute residuals."""

    def __init__(self, config: GRMMatcherConfig) -> None:
        super().__init__()
        self.enabled = config.use_seed_homography
        self.detach_homography = config.detach_seed_homography
        self.min_matches = config.seed_min_matches
        self.max_matches = config.seed_max_matches
        self.ransac_threshold = config.seed_ransac_threshold
        self.confidence = config.seed_confidence
        self.max_iterations = config.seed_max_iterations
        self.max_condition_number = config.seed_max_condition_number
        self.cycle_sigma = config.cycle_sigma
        self.eps = config.probability_eps

    def forward(
        self,
        src_xy: Tensor,
        dst_xy: Tensor,
        p_match: Tensor,
        candidate_entropy: Tensor,
        cycle_error: Tensor,
        valid_mask: Tensor,
        mutual_top1: Optional[Tensor] = None,
    ) -> GRMSeedGeometryOutput:
        self._validate_inputs(
            src_xy,
            dst_xy,
            p_match,
            candidate_entropy,
            cycle_error,
            valid_mask,
            mutual_top1,
        )
        batch, count, _ = src_xy.shape
        if mutual_top1 is None:
            mutual_top1 = torch.ones_like(valid_mask)

        reliability = (
            p_match.float().clamp(0.0, 1.0)
            * torch.exp(-candidate_entropy.float().clamp_min(0.0))
            * torch.exp(-cycle_error.float().clamp_min(0.0) / self.cycle_sigma)
            * torch.where(mutual_top1, 1.0, 0.5)
        )
        reliability = reliability * valid_mask.float()

        homographies: list[Tensor] = []
        valid_h: list[bool] = []
        selected_masks: list[Tensor] = []
        inlier_masks: list[Tensor] = []
        reasons: list[str] = []

        for batch_index in range(batch):
            identity = torch.eye(3, device=src_xy.device, dtype=torch.float32)
            valid_indices = torch.nonzero(valid_mask[batch_index], as_tuple=False).squeeze(-1)
            selected_mask = torch.zeros(count, device=src_xy.device, dtype=torch.bool)
            inlier_mask = torch.zeros(count, device=src_xy.device, dtype=torch.bool)
            if not self.enabled:
                homographies.append(identity)
                valid_h.append(False)
                selected_masks.append(selected_mask)
                inlier_masks.append(inlier_mask)
                reasons.append("seed homography disabled")
                continue
            if valid_indices.numel() < self.min_matches:
                homographies.append(identity)
                valid_h.append(False)
                selected_masks.append(selected_mask)
                inlier_masks.append(inlier_mask)
                reasons.append(
                    f"only {int(valid_indices.numel())} valid matches; need {self.min_matches}"
                )
                continue

            selection_count = min(self.max_matches, int(valid_indices.numel()))
            ordering = torch.topk(
                reliability[batch_index, valid_indices], k=selection_count, sorted=True
            ).indices
            selected = valid_indices[ordering]
            selected_mask[selected] = True
            h_cpu, local_inliers, reason = self._estimate_cpu(
                src_xy[batch_index, selected], dst_xy[batch_index, selected]
            )
            if h_cpu is None:
                homographies.append(identity)
                valid_h.append(False)
                selected_masks.append(selected_mask)
                inlier_masks.append(inlier_mask)
                reasons.append(reason)
                continue

            local_inliers_device = local_inliers.to(device=src_xy.device)
            inlier_indices = selected[local_inliers_device]
            inlier_mask[inlier_indices] = True
            if not self.detach_homography and inlier_indices.numel() >= 4:
                homography = estimate_dlt_homography(
                    src_xy[batch_index, inlier_indices],
                    dst_xy[batch_index, inlier_indices],
                )
            else:
                homography = h_cpu.to(device=src_xy.device, dtype=torch.float32)
            if not self._is_valid_homography(homography.detach()):
                homographies.append(identity)
                valid_h.append(False)
                selected_masks.append(selected_mask)
                inlier_masks.append(torch.zeros_like(inlier_mask))
                reasons.append("estimated matrix is non-finite, singular, or ill-conditioned")
                continue
            homographies.append(homography / homography[2, 2])
            valid_h.append(True)
            selected_masks.append(selected_mask)
            inlier_masks.append(inlier_mask)
            reasons.append("")

        homography_ab = torch.stack(homographies)
        if self.detach_homography:
            homography_ab = homography_ab.detach()
        homography_valid = torch.tensor(valid_h, device=src_xy.device, dtype=torch.bool)
        selected_mask_tensor = torch.stack(selected_masks)
        inlier_mask_tensor = torch.stack(inlier_masks)

        projected_dst = apply_homography(src_xy.float(), homography_ab.float(), self.eps)
        residual_xy = projected_dst - dst_xy.float()
        forward_residual = torch.linalg.vector_norm(residual_xy, dim=-1)
        safe_h = homography_ab.float().clone()
        safe_h[~homography_valid] = torch.eye(3, device=safe_h.device)
        inverse_h = torch.linalg.inv(safe_h)
        projected_src = apply_homography(dst_xy.float(), inverse_h, self.eps)
        backward_residual = torch.linalg.vector_norm(projected_src - src_xy.float(), dim=-1)
        symmetric = torch.sqrt(
            forward_residual.square() + backward_residual.square() + self.eps
        )
        geometry_mask = homography_valid[:, None] & valid_mask
        residual_xy = residual_xy * geometry_mask[..., None].float()
        forward_residual = forward_residual * geometry_mask.float()
        backward_residual = backward_residual * geometry_mask.float()
        symmetric = symmetric * geometry_mask.float()
        return GRMSeedGeometryOutput(
            homography_ab=homography_ab,
            homography_valid=homography_valid,
            selected_mask=selected_mask_tensor,
            inlier_mask=inlier_mask_tensor,
            residual_xy=residual_xy,
            forward_residual=forward_residual,
            backward_residual=backward_residual,
            symmetric_transfer_error=symmetric,
            failure_reasons=tuple(reasons),
        )

    def _estimate_cpu(
        self, src_xy: Tensor, dst_xy: Tensor
    ) -> tuple[Optional[Tensor], Tensor, str]:
        count = src_xy.shape[0]
        if cv2 is None:
            try:
                h = estimate_dlt_homography(src_xy.detach().cpu(), dst_xy.detach().cpu())
            except (RuntimeError, ValueError) as error:
                return None, torch.zeros(count, dtype=torch.bool), f"DLT fallback failed: {error}"
            return h.detach(), torch.ones(count, dtype=torch.bool), ""
        source_numpy = src_xy.detach().float().cpu().numpy()
        target_numpy = dst_xy.detach().float().cpu().numpy()
        try:
            matrix, inliers = cv2.findHomography(
                source_numpy,
                target_numpy,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=self.ransac_threshold,
                maxIters=self.max_iterations,
                confidence=self.confidence,
            )
        except cv2.error as error:
            return None, torch.zeros(count, dtype=torch.bool), f"OpenCV USAC/MAGSAC error: {error}"
        if matrix is None or inliers is None:
            return None, torch.zeros(count, dtype=torch.bool), "OpenCV USAC/MAGSAC returned no model"
        h = torch.from_numpy(matrix).to(dtype=torch.float32)
        inlier_mask = torch.from_numpy(inliers.reshape(-1).astype("bool"))
        if int(inlier_mask.sum()) < self.min_matches:
            return None, inlier_mask, "robust model has fewer than four inliers"
        return h, inlier_mask, ""

    def _is_valid_homography(self, homography: Tensor) -> bool:
        if homography.shape != (3, 3) or not torch.isfinite(homography).all():
            return False
        if homography[2, 2].abs() <= self.eps:
            return False
        try:
            determinant = torch.linalg.det(homography)
            condition = torch.linalg.cond(homography)
        except RuntimeError:
            return False
        return bool(
            determinant.abs() > self.eps
            and torch.isfinite(condition)
            and condition <= self.max_condition_number
        )

    @staticmethod
    def _validate_inputs(
        src_xy: Tensor,
        dst_xy: Tensor,
        p_match: Tensor,
        candidate_entropy: Tensor,
        cycle_error: Tensor,
        valid_mask: Tensor,
        mutual_top1: Optional[Tensor],
    ) -> None:
        if src_xy.ndim != 3 or src_xy.shape[-1] != 2 or src_xy.shape != dst_xy.shape:
            raise ValueError("src_xy and dst_xy must both be [B, N, 2].")
        expected = src_xy.shape[:2]
        for name, value in (
            ("p_match", p_match),
            ("candidate_entropy", candidate_entropy),
            ("cycle_error", cycle_error),
            ("valid_mask", valid_mask),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must be {expected}.")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be boolean.")
        if mutual_top1 is not None:
            if mutual_top1.shape != expected or mutual_top1.dtype != torch.bool:
                raise ValueError("mutual_top1 must be a bool [B, N] tensor.")
