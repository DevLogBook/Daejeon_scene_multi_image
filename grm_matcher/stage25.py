"""Fixed/EMA Teacher-Student primitives for real-pair Stage 2.5 adaptation."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import GRMMatcherConfig
from .coordinates import (
    flat_indices_to_image_xy,
    image_xy_to_feature_xy,
    image_xy_to_normalized_grid,
)
from .cycle_consistency import compute_cycle_consistency
from .local_correlation import GRMRefinementOutput
from .matcher import GRMMatcher


@dataclass
class GRMStage25Config:
    """Configuration shared by real-pair refinement/coarse/role adaptation."""

    ema_decay: float = 0.999
    min_candidate_probability: float = 0.0
    max_candidate_normalized_entropy: float = 0.80
    min_mid_peak_probability: float = 0.03
    min_fine_peak_probability: float = 0.06
    max_cycle_error: float = 3.0
    max_fine_std: float = 6.0
    min_pseudo_matches: int = 8
    require_mutual_top1_for_refinement: bool = True

    # Stage 2.5-A-v2: retain an absolute notion of real-batch reliability.
    use_absolute_batch_reliability: bool = True
    absolute_confidence_reference: float = 0.01
    reliable_match_count_reference: int = 32
    reliable_region_count_reference: int = 8
    pseudo_loss_weight: float = 0.05
    homography_loss_weight: float = 0.50
    source_replay_weight: float = 1.0
    max_homography_cycle_error: float = 2.0
    homography_supervision_pair_buckets: tuple[str, ...] = ("global",)

    # Stage 2.5-B: strict real-domain coarse adaptation.
    coarse_min_pseudo_matches: int = 16
    coarse_min_covered_regions: int = 4
    coarse_region_size_cells: int = 4
    coarse_require_mutual_top1: bool = True
    coarse_pseudo_ce_weight: float = 1.0
    coarse_bidirectional_weight: float = 1.0
    matchability_distillation_weight: float = 0.2

    # Stage 3.5: geometry-role pseudo-label rules. Ambiguous points are ignored.
    global_residual_threshold: float = 3.0
    local_residual_threshold: float = 24.0
    local_neighbor_count: int = 6
    local_flow_consistency_threshold: float = 6.0
    false_cycle_error: float = 8.0
    false_normalized_entropy: float = 0.90
    false_fine_std: float = 10.0
    false_fine_peak_probability: float = 0.03
    geometry_source_alignment_radius: float = 8.0
    geometry_rerank_target_radius: float = 5.0
    rerank_pseudo_weight: float = 1.0
    role_pseudo_weight: float = 0.5
    calibration_pseudo_weight: float = 0.1

    crop_scale_min: float = 0.85
    horizontal_flip_probability: float = 0.5
    brightness_jitter: float = 0.20
    contrast_jitter: float = 0.20
    saturation_jitter: float = 0.15
    gamma_jitter: float = 0.15
    noise_std: float = 0.02
    blur_probability: float = 0.20

    mid_coordinate_weight: float = 0.5
    fine_coordinate_weight: float = 1.0
    mid_nll_weight: float = 0.10
    fine_nll_weight: float = 0.10
    coordinate_loss_beta: float = 1.0
    confidence_clip: float = 5.0
    probability_eps: float = 1.0e-8

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in (
            "ema_decay",
            "max_candidate_normalized_entropy",
            "horizontal_flip_probability",
            "blur_probability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        if self.ema_decay <= 0.0:
            raise ValueError("ema_decay must be positive.")
        if not 0.0 < self.crop_scale_min <= 1.0:
            raise ValueError("crop_scale_min must lie in (0, 1].")
        if self.min_pseudo_matches <= 0 or self.coarse_min_pseudo_matches <= 0:
            raise ValueError("Pseudo-label minimum counts must be positive.")
        if (
            self.reliable_match_count_reference <= 0
            or self.reliable_region_count_reference <= 0
        ):
            raise ValueError("Stage-2.5 reliability count references must be positive.")
        if not self.homography_supervision_pair_buckets:
            raise ValueError("homography_supervision_pair_buckets cannot be empty.")
        self.homography_supervision_pair_buckets = tuple(
            str(value) for value in self.homography_supervision_pair_buckets
        )
        if self.coarse_min_covered_regions <= 0 or self.coarse_region_size_cells <= 0:
            raise ValueError("Coarse pseudo-label region settings must be positive.")
        if self.local_neighbor_count <= 0:
            raise ValueError("local_neighbor_count must be positive.")
        positive = (
            "max_cycle_error",
            "max_fine_std",
            "coordinate_loss_beta",
            "confidence_clip",
            "probability_eps",
            "global_residual_threshold",
            "local_residual_threshold",
            "local_flow_consistency_threshold",
            "false_cycle_error",
            "false_fine_std",
            "geometry_source_alignment_radius",
            "geometry_rerank_target_radius",
            "absolute_confidence_reference",
            "max_homography_cycle_error",
        )
        if any(float(getattr(self, name)) <= 0.0 for name in positive):
            raise ValueError(f"Stage-2.5 positive parameters invalid: {positive}.")
        nonnegative = (
            "min_candidate_probability",
            "min_mid_peak_probability",
            "min_fine_peak_probability",
            "brightness_jitter",
            "contrast_jitter",
            "saturation_jitter",
            "gamma_jitter",
            "noise_std",
            "mid_coordinate_weight",
            "fine_coordinate_weight",
            "mid_nll_weight",
            "fine_nll_weight",
            "coarse_pseudo_ce_weight",
            "coarse_bidirectional_weight",
            "matchability_distillation_weight",
            "false_normalized_entropy",
            "false_fine_peak_probability",
            "rerank_pseudo_weight",
            "role_pseudo_weight",
            "calibration_pseudo_weight",
            "pseudo_loss_weight",
            "homography_loss_weight",
            "source_replay_weight",
        )
        if any(float(getattr(self, name)) < 0.0 for name in nonnegative):
            raise ValueError(f"Stage-2.5 nonnegative parameters invalid: {nonnegative}.")
        if self.mid_coordinate_weight + self.fine_coordinate_weight <= 0.0:
            raise ValueError("At least one coordinate consistency weight must be positive.")
        if self.local_residual_threshold <= self.global_residual_threshold:
            raise ValueError("local_residual_threshold must exceed global_residual_threshold.")
        if not 0.0 <= self.false_normalized_entropy <= 1.0:
            raise ValueError("false_normalized_entropy must lie in [0, 1].")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class GRMStage25AugmentedBatch:
    image_a: Tensor
    image_b: Tensor
    transform_a: Tensor  # [B, 3, 3], weak/original pixels -> student pixels
    transform_b: Tensor


@dataclass
class GRMStage25PseudoLabels:
    src_xy: Tensor
    coarse_dst_xy: Tensor
    mid_dst_xy: Tensor
    fine_dst_xy: Tensor
    confidence: Tensor
    cycle_error: Tensor
    fine_std: Tensor
    valid_mask: Tensor
    # Optional diagnostics preserve construction compatibility with older tests/code.
    normalized_entropy: Optional[Tensor] = None
    mid_peak_probability: Optional[Tensor] = None
    fine_peak_probability: Optional[Tensor] = None
    mutual_top1: Optional[Tensor] = None
    base_valid_mask: Optional[Tensor] = None
    query_indices: Optional[Tensor] = None
    target_indices: Optional[Tensor] = None
    region_ids: Optional[Tensor] = None


@dataclass
class GRMStage25BidirectionalLabels:
    """One Teacher forward pass worth of A->B and B->A pseudo labels."""

    forward: GRMStage25PseudoLabels
    backward: GRMStage25PseudoLabels
    teacher_matchability_a: Tensor
    teacher_matchability_b: Tensor
    coarse_height: int
    coarse_width: int


@dataclass
class GRMStage25CoarseOutput:
    similarity: Tensor
    matchability_a: Tensor
    matchability_b: Tensor
    coarse_height: int
    coarse_width: int


@dataclass
class GRMStage25StudentOutput:
    mid: GRMRefinementOutput
    fine: GRMRefinementOutput


def make_ema_teacher(student: GRMMatcher) -> GRMMatcher:
    """Clone a student into a frozen evaluation-only EMA teacher."""

    teacher = copy.deepcopy(student)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_ema_teacher(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    """Update matching parameters/buffers without relying on zip ordering."""

    if not 0.0 <= decay <= 1.0:
        raise ValueError("EMA decay must lie in [0, 1].")
    teacher_parameters = dict(teacher.named_parameters())
    student_parameters = dict(student.named_parameters())
    if teacher_parameters.keys() != student_parameters.keys():
        raise ValueError("Teacher and student parameter names differ.")
    for name, teacher_value in teacher_parameters.items():
        student_value = student_parameters[name].detach()
        teacher_value.mul_(decay).add_(student_value, alpha=1.0 - decay)
    teacher_buffers = dict(teacher.named_buffers())
    student_buffers = dict(student.named_buffers())
    if teacher_buffers.keys() != student_buffers.keys():
        raise ValueError("Teacher and student buffer names differ.")
    for name, teacher_value in teacher_buffers.items():
        student_value = student_buffers[name].detach()
        if teacher_value.is_floating_point():
            teacher_value.mul_(decay).add_(student_value, alpha=1.0 - decay)
        else:
            teacher_value.copy_(student_value)


def transform_points(points_xy: Tensor, transforms: Tensor) -> Tensor:
    """Apply batched weak-to-student homographies to ``[B,N,2]`` points."""

    if points_xy.ndim != 3 or points_xy.shape[-1] != 2:
        raise ValueError("points_xy must be [B, N, 2].")
    if transforms.shape != (points_xy.shape[0], 3, 3):
        raise ValueError("transforms must be [B, 3, 3].")
    homogeneous = torch.cat((points_xy.float(), torch.ones_like(points_xy[..., :1])), -1)
    projected = torch.einsum("bij,bnj->bni", transforms.float(), homogeneous)
    denominator = projected[..., 2:3]
    safe = torch.where(
        denominator.abs() >= 1.0e-8,
        denominator,
        torch.where(denominator < 0.0, -torch.ones_like(denominator), torch.ones_like(denominator))
        * 1.0e-8,
    )
    return projected[..., :2] / safe


def make_student_augmentation(
    image_a: Tensor,
    image_b: Tensor,
    matcher_config: GRMMatcherConfig,
    stage25_config: GRMStage25Config,
) -> GRMStage25AugmentedBatch:
    """Create a strong view and explicit original-to-strong coordinate maps."""

    if image_a.shape != image_b.shape or image_a.ndim != 4:
        raise ValueError("Stage-2.5 augmentation expects equal [B,3,H,W] images.")
    batch, _, height, width = image_a.shape
    device = image_a.device
    dtype = torch.float32
    scale = torch.empty(batch, device=device, dtype=dtype).uniform_(
        stage25_config.crop_scale_min, 1.0
    )
    crop_height = torch.clamp((scale * height).round(), min=2.0, max=float(height))
    crop_width = torch.clamp((scale * width).round(), min=2.0, max=float(width))
    top = torch.rand(batch, device=device) * (height - crop_height).clamp_min(0.0)
    left = torch.rand(batch, device=device) * (width - crop_width).clamp_min(0.0)
    scale_x = float(width) / crop_width
    scale_y = float(height) / crop_height
    transforms = torch.zeros(batch, 3, 3, device=device, dtype=dtype)
    transforms[:, 0, 0] = scale_x
    transforms[:, 1, 1] = scale_y
    transforms[:, 0, 2] = scale_x * (0.5 - left) - 0.5
    transforms[:, 1, 2] = scale_y * (0.5 - top) - 0.5
    transforms[:, 2, 2] = 1.0
    flips = torch.rand(batch, device=device) < stage25_config.horizontal_flip_probability
    flip_matrix = torch.eye(3, device=device, dtype=dtype)[None].repeat(batch, 1, 1)
    flip_matrix[flips, 0, 0] = -1.0
    flip_matrix[flips, 0, 2] = float(width - 1)
    transforms = torch.bmm(flip_matrix, transforms)

    zero_one_a = _to_zero_one(image_a, matcher_config)
    zero_one_b = _to_zero_one(image_b, matcher_config)
    strong_a = _warp_image(zero_one_a, transforms)
    strong_b = _warp_image(zero_one_b, transforms)
    strong_a = _strong_photometric(strong_a, stage25_config)
    strong_b = _strong_photometric(strong_b, stage25_config)
    strong_a = _from_zero_one(strong_a, matcher_config)
    strong_b = _from_zero_one(strong_b, matcher_config)
    return GRMStage25AugmentedBatch(strong_a, strong_b, transforms, transforms.clone())


@torch.no_grad()
def generate_teacher_pseudo_labels(
    teacher: GRMMatcher,
    image_a: Tensor,
    image_b: Tensor,
    stage25_config: GRMStage25Config,
    valid_a: Optional[Tensor] = None,
    valid_b: Optional[Tensor] = None,
    global_step: Optional[int] = None,
) -> GRMStage25PseudoLabels:
    """Compatibility wrapper returning the A->B half of one Teacher pass."""

    return generate_teacher_bidirectional_pseudo_labels(
        teacher,
        image_a,
        image_b,
        stage25_config,
        valid_a,
        valid_b,
        global_step,
    ).forward


@torch.no_grad()
def generate_teacher_bidirectional_pseudo_labels(
    teacher: GRMMatcher,
    image_a: Tensor,
    image_b: Tensor,
    stage25_config: GRMStage25Config,
    valid_a: Optional[Tensor] = None,
    valid_b: Optional[Tensor] = None,
    global_step: Optional[int] = None,
) -> GRMStage25BidirectionalLabels:
    """Run the coarse path once, then refine and filter both directions.

    This path intentionally stops before seed geometry, graph, reranking, and
    role prediction. It is therefore suitable as the Teacher for Stage 2.5-A
    and Stage 2.5-B even when those Stage-3 modules are untrained.
    """

    features_a, features_b, tokens_a, tokens_b = teacher.encode_pair(image_a, image_b, valid_a, valid_b)
    correlation = teacher.correlation(tokens_a, tokens_b,
        features_a["valid_coarse"].flatten(1), features_b["valid_coarse"].flatten(1))
    coarse_height, coarse_width = features_a["F_coarse"].shape[-2:]
    forward = _teacher_direction(
        teacher,
        correlation.probability,
        correlation.matchability_a,
        features_a,
        features_b,
        coarse_height,
        coarse_width,
        global_step,
    )
    backward = _teacher_direction(
        teacher,
        correlation.probability.transpose(1, 2),
        correlation.matchability_b,
        features_b,
        features_a,
        coarse_height,
        coarse_width,
        global_step,
    )
    forward_cycle = compute_cycle_consistency(
        forward["src_xy"],
        forward["fine"].dst_xy,
        backward["src_xy"],
        backward["fine"].dst_xy,
        sigma=teacher.config.cycle_sigma,
        forward_valid_mask=forward["valid"],
        backward_valid_mask=backward["valid"],
    )
    backward_cycle = compute_cycle_consistency(
        backward["src_xy"],
        backward["fine"].dst_xy,
        forward["src_xy"],
        forward["fine"].dst_xy,
        sigma=teacher.config.cycle_sigma,
        forward_valid_mask=backward["valid"],
        backward_valid_mask=forward["valid"],
    )

    if teacher.config.architecture_version == "v2":
        forward_cycle = teacher.direct_cycle(features_a, features_b, correlation.probability,
            forward["src_xy"], forward["fine"].dst_xy, forward["valid"])
        backward_cycle = teacher.direct_cycle(features_b, features_a, correlation.probability.transpose(1, 2),
            backward["src_xy"], backward["fine"].dst_xy, backward["valid"])

    all_forward_top1 = teacher.candidate_sampler.topk_candidates(
        correlation.probability
    ).indices[..., 0]
    all_backward_top1 = teacher.candidate_sampler.topk_candidates(
        correlation.probability.transpose(1, 2)
    ).indices[..., 0]
    forward_mutual = _mutual_top1(forward, all_backward_top1)
    backward_mutual = _mutual_top1(backward, all_forward_top1)
    forward_labels = _build_teacher_labels(
        forward,
        forward_cycle,
        forward_mutual,
        stage25_config,
        valid_a,
        valid_b,
    )
    backward_labels = _build_teacher_labels(
        backward,
        backward_cycle,
        backward_mutual,
        stage25_config,
        valid_b,
        valid_a,
    )
    return GRMStage25BidirectionalLabels(
        forward=forward_labels,
        backward=backward_labels,
        teacher_matchability_a=correlation.matchability_a.float(),
        teacher_matchability_b=correlation.matchability_b.float(),
        coarse_height=coarse_height,
        coarse_width=coarse_width,
    )


def _mutual_top1(direction: Dict[str, object], reverse_all_top1: Tensor) -> Tensor:
    sampled = direction["sampled"]
    target_indices = sampled.candidate_indices[..., 0]
    reverse_source = torch.gather(reverse_all_top1, 1, target_indices)
    return reverse_source == sampled.query_indices


def _build_teacher_labels(
    direction: Dict[str, object],
    cycle: object,
    mutual_top1: Tensor,
    config: GRMStage25Config,
    source_valid_image: Optional[Tensor],
    target_valid_image: Optional[Tensor],
) -> GRMStage25PseudoLabels:
    sampled = direction["sampled"]
    mid = direction["mid"]
    fine = direction["fine"]
    fine_covariance = fine.covariance_image.float()
    fine_std = torch.sqrt(
        torch.diagonal(fine_covariance, dim1=-2, dim2=-1).sum(-1).clamp_min(0.0)
    )
    base_valid = direction["valid"] & cycle.valid_mask
    if source_valid_image is not None:
        base_valid &= _sample_valid_mask(source_valid_image, direction["src_xy"])
    if target_valid_image is not None:
        base_valid &= _sample_valid_mask(target_valid_image, fine.dst_xy)
    valid = (
        base_valid
        & (sampled.candidate_global_probs[..., 0] >= config.min_candidate_probability)
        & (sampled.normalized_entropy <= config.max_candidate_normalized_entropy)
        & (mid.peak_probability >= config.min_mid_peak_probability)
        & (fine.peak_probability >= config.min_fine_peak_probability)
        & (cycle.cycle_error <= config.max_cycle_error)
        & (fine_std <= config.max_fine_std)
    )
    if config.require_mutual_top1_for_refinement:
        valid &= mutual_top1
    enough = valid.sum(dim=1) >= config.min_pseudo_matches
    valid &= enough[:, None]
    local_strength = torch.sqrt(
        (mid.peak_probability * fine.peak_probability).clamp_min(0.0)
    )
    uncertainty_score = torch.exp(-fine_std / config.max_fine_std)
    confidence = (
        sampled.query_scores.float()
        * local_strength.float()
        * cycle.cycle_score.float()
        * uncertainty_score
    ).clamp_min(0.0) * valid.float()
    return GRMStage25PseudoLabels(
        src_xy=direction["src_xy"].float(),
        coarse_dst_xy=direction["coarse_dst_xy"].float(),
        mid_dst_xy=mid.dst_xy.float(),
        fine_dst_xy=fine.dst_xy.float(),
        confidence=confidence,
        cycle_error=cycle.cycle_error.float(),
        fine_std=fine_std,
        valid_mask=valid,
        normalized_entropy=sampled.normalized_entropy.float(),
        mid_peak_probability=mid.peak_probability.float(),
        fine_peak_probability=fine.peak_probability.float(),
        mutual_top1=mutual_top1,
        base_valid_mask=base_valid,
        query_indices=sampled.query_indices,
        target_indices=sampled.candidate_indices[..., 0],
        region_ids=sampled.region_ids,
    )


def transform_pseudo_labels(
    labels: GRMStage25PseudoLabels,
    transform_a: Tensor,
    transform_b: Tensor,
    height: int,
    width: int,
    matcher_config: GRMMatcherConfig,
    stage25_config: GRMStage25Config,
) -> GRMStage25PseudoLabels:
    """Map Teacher coordinates into Student view and remove unreachable labels."""

    src = transform_points(labels.src_xy, transform_a)
    coarse = transform_points(labels.coarse_dst_xy, transform_b)
    mid = transform_points(labels.mid_dst_xy, transform_b)
    fine = transform_points(labels.fine_dst_xy, transform_b)
    valid = labels.valid_mask.clone()
    for points in (src, coarse, mid, fine):
        valid &= _points_inside(points, height, width) & torch.isfinite(points).all(dim=-1)
    mid_radius = float((matcher_config.mid_window_size // 2) * matcher_config.mid_stride)
    fine_radius = float((matcher_config.fine_window_size // 2) * matcher_config.fine_stride)
    valid &= ((mid - coarse).abs() <= mid_radius).all(dim=-1)
    valid &= ((fine - mid).abs() <= fine_radius).all(dim=-1)
    enough = valid.sum(dim=1) >= stage25_config.min_pseudo_matches
    valid &= enough[:, None]
    return GRMStage25PseudoLabels(
        src_xy=src,
        coarse_dst_xy=coarse,
        mid_dst_xy=mid,
        fine_dst_xy=fine,
        confidence=labels.confidence * valid.float(),
        cycle_error=labels.cycle_error,
        fine_std=labels.fine_std,
        valid_mask=valid,
        normalized_entropy=labels.normalized_entropy,
        mid_peak_probability=labels.mid_peak_probability,
        fine_peak_probability=labels.fine_peak_probability,
        mutual_top1=labels.mutual_top1,
        base_valid_mask=labels.base_valid_mask,
        query_indices=labels.query_indices,
        target_indices=labels.target_indices,
        region_ids=labels.region_ids,
    )


def build_homography_supervision_labels(
    labels: GRMStage25PseudoLabels,
    h_ab: Tensor,
    h_ba: Tensor,
    homography_valid: Tensor,
    max_cycle_error: float,
    pair_mask: Optional[Tensor] = None,
) -> GRMStage25PseudoLabels:
    """Replace uncertain Teacher targets with verified A-to-B Homography targets.

    The Teacher is used only to choose spatial source/coarse anchors.  The
    mid/fine target is obtained from ``H_ab`` and is accepted only when the
    supplied reverse transform closes the A->B->A cycle.  Reachability inside
    the local windows is checked later by :func:`transform_pseudo_labels`.
    """

    batch = labels.src_xy.shape[0]
    if h_ab.shape != (batch, 3, 3) or h_ba.shape != (batch, 3, 3):
        raise ValueError("h_ab and h_ba must be [B, 3, 3].")
    if homography_valid.shape != (batch,):
        raise ValueError("homography_valid must be [B].")
    if pair_mask is not None and pair_mask.shape != (batch,):
        raise ValueError("pair_mask must be [B] when provided.")
    if max_cycle_error <= 0.0:
        raise ValueError("max_cycle_error must be positive.")

    target = transform_points(labels.src_xy, h_ab)
    cycled_source = transform_points(target, h_ba)
    cycle_error = torch.linalg.vector_norm(cycled_source - labels.src_xy.float(), dim=-1)
    base_valid = (
        labels.base_valid_mask.clone()
        if labels.base_valid_mask is not None
        else labels.valid_mask.clone()
    )
    sample_valid = homography_valid.bool()
    if pair_mask is not None:
        sample_valid &= pair_mask.bool()
    valid = (
        base_valid
        & sample_valid[:, None]
        & torch.isfinite(target).all(dim=-1)
        & torch.isfinite(cycle_error)
        & (cycle_error <= max_cycle_error)
    )
    return replace(
        labels,
        mid_dst_xy=target,
        fine_dst_xy=target,
        confidence=valid.to(dtype=labels.confidence.dtype),
        cycle_error=cycle_error,
        fine_std=torch.zeros_like(cycle_error),
        valid_mask=valid,
    )


def build_dense_mapping_supervision_labels(
    labels: GRMStage25PseudoLabels,
    dense_mapping_ab: Tensor,
    overlap_a: Tensor,
    overlap_b: Tensor,
) -> GRMStage25PseudoLabels:
    """Replace Teacher targets with exact dense A-to-B correspondence targets.

    ``dense_mapping_ab`` is ``[B, 2, H, W]`` in original-image pixel ``(x, y)``
    coordinates.  It may contain both a global Homography and continuous local
    flow.  Teacher predictions are used only to choose source/coarse anchors;
    the mid/fine supervision itself comes from this exact field.
    """

    batch, point_count = labels.src_xy.shape[:2]
    if dense_mapping_ab.ndim != 4 or dense_mapping_ab.shape[:2] != (batch, 2):
        raise ValueError("dense_mapping_ab must be [B, 2, H, W].")
    height, width = dense_mapping_ab.shape[-2:]
    expected_mask_shape = (batch, 1, height, width)
    if overlap_a.shape != expected_mask_shape or overlap_b.shape != expected_mask_shape:
        raise ValueError(
            "overlap_a and overlap_b must match dense_mapping_ab as [B, 1, H, W]."
        )
    if labels.src_xy.shape != (batch, point_count, 2):
        raise ValueError("labels.src_xy must be [B, N, 2].")

    source_grid = image_xy_to_normalized_grid(
        labels.src_xy.float(), height, width
    ).unsqueeze(2)
    target = F.grid_sample(
        dense_mapping_ab.float(),
        source_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(-1).transpose(1, 2)
    source_overlap = F.grid_sample(
        overlap_a.float(),
        source_grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(1).squeeze(-1) > 0.5
    target_grid = image_xy_to_normalized_grid(target, height, width).unsqueeze(2)
    target_overlap = F.grid_sample(
        overlap_b.float(),
        target_grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(1).squeeze(-1) > 0.5

    base_valid = (
        labels.base_valid_mask.clone()
        if labels.base_valid_mask is not None
        else labels.valid_mask.clone()
    )
    valid = (
        base_valid
        & source_overlap
        & target_overlap
        & torch.isfinite(target).all(dim=-1)
        & _points_inside(target, height, width)
    )
    zeros = torch.zeros_like(labels.cycle_error)
    return replace(
        labels,
        mid_dst_xy=target,
        fine_dst_xy=target,
        confidence=valid.to(dtype=labels.confidence.dtype),
        cycle_error=zeros,
        fine_std=zeros,
        valid_mask=valid,
    )


def merge_refinement_valid_masks(
    primary: GRMStage25PseudoLabels,
    *additional: GRMStage25PseudoLabels,
) -> GRMStage25PseudoLabels:
    """Return primary anchors with the union mask required by all supervision."""

    valid = primary.valid_mask.clone()
    for labels in additional:
        if labels.valid_mask.shape != valid.shape:
            raise ValueError("All refinement label masks must have identical shapes.")
        valid |= labels.valid_mask
    return replace(primary, valid_mask=valid)


def forward_student_refinement(
    student: GRMMatcher,
    image_a: Tensor,
    image_b: Tensor,
    labels: GRMStage25PseudoLabels,
    valid_a: Optional[Tensor] = None,
    valid_b: Optional[Tensor] = None,
) -> GRMStage25StudentOutput:
    """Refine Teacher anchors in the Student view without Stage-3 modules."""

    student._validate_images(image_a, image_b)
    if student.config.architecture_version == "v2":
        features_a, features_b, _, _ = student.encode_pair(image_a,image_b,valid_a,valid_b)
        mid, fine = student.refine_points(features_a,features_b,labels.src_xy,labels.coarse_dst_xy,labels.valid_mask)
        return GRMStage25StudentOutput(mid=mid, fine=fine)
    batch = image_a.shape[0]
    pyramid = student.fpn(student.backbone(torch.cat((image_a, image_b), dim=0)))
    features_a = {name: value[:batch] for name, value in pyramid.items()}
    features_b = {name: value[batch:] for name, value in pyramid.items()}
    mid = student.mid_refiner(
        features_a["F_mid"],
        features_b["F_mid"],
        labels.src_xy,
        labels.coarse_dst_xy,
        labels.valid_mask,
    )
    fine = student.fine_refiner(
        features_a["F_fine"],
        features_b["F_fine"],
        labels.src_xy,
        mid.dst_xy,
        labels.valid_mask,
    )
    return GRMStage25StudentOutput(mid=mid, fine=fine)


class GRMStage25ConsistencyLoss(nn.Module):
    """Confidence-weighted coordinate consistency and local-distribution NLL."""

    def __init__(
        self,
        matcher_config: GRMMatcherConfig,
        stage25_config: GRMStage25Config,
    ) -> None:
        super().__init__()
        self.matcher_config = matcher_config
        self.config = stage25_config

    def forward(
        self,
        student: GRMStage25StudentOutput,
        labels: GRMStage25PseudoLabels,
        *,
        apply_absolute_reliability: Optional[bool] = None,
    ) -> Dict[str, Tensor]:
        mask = labels.valid_mask
        reference = student.fine.dst_xy
        use_absolute = (
            self.config.use_absolute_batch_reliability
            if apply_absolute_reliability is None
            else bool(apply_absolute_reliability)
        )
        if not mask.any():
            zero = reference.sum() * 0.0
            return {
                "total": zero,
                "unscaled_total": zero,
                "mid_coordinate": zero,
                "fine_coordinate": zero,
                "mid_nll": zero,
                "fine_nll": zero,
                "batch_reliability": zero.detach(),
                "mean_absolute_confidence": zero.detach(),
                "covered_region_count": zero.detach(),
                "valid_count": zero.detach(),
            }
        absolute_weights = labels.confidence.float().clamp_min(0.0)
        batch_reliability, mean_confidence, covered_regions = (
            self._absolute_batch_reliability(labels, mask)
        )
        weights = absolute_weights
        mean_weight = weights.masked_select(mask).mean().clamp_min(self.config.probability_eps)
        weights = (weights / mean_weight).clamp_max(self.config.confidence_clip)
        mid_coordinate = _weighted_smooth_l1(
            student.mid.dst_xy.float(), labels.mid_dst_xy, mask, weights,
            self.config.coordinate_loss_beta,
        )
        fine_coordinate = _weighted_smooth_l1(
            student.fine.dst_xy.float(), labels.fine_dst_xy, mask, weights,
            self.config.coordinate_loss_beta,
        )
        mid_nll = _local_probability_nll(
            student.mid.local_probability,
            labels.coarse_dst_xy,
            labels.mid_dst_xy,
            self.matcher_config.mid_stride,
            mask,
            weights,
            self.config.probability_eps,
        )
        fine_nll = _local_probability_nll(
            student.fine.local_probability,
            student.mid.dst_xy.detach(),
            labels.fine_dst_xy,
            self.matcher_config.fine_stride,
            mask,
            weights,
            self.config.probability_eps,
        )
        unscaled_total = (
            self.config.mid_coordinate_weight * mid_coordinate
            + self.config.fine_coordinate_weight * fine_coordinate
            + self.config.mid_nll_weight * mid_nll
            + self.config.fine_nll_weight * fine_nll
        )
        total = unscaled_total * (
            batch_reliability if use_absolute else unscaled_total.new_tensor(1.0)
        )
        return {
            "total": total,
            "unscaled_total": unscaled_total,
            "mid_coordinate": mid_coordinate,
            "fine_coordinate": fine_coordinate,
            "mid_nll": mid_nll,
            "fine_nll": fine_nll,
            "batch_reliability": batch_reliability.detach(),
            "mean_absolute_confidence": mean_confidence.detach(),
            "covered_region_count": covered_regions.detach(),
            "valid_count": mask.sum().to(dtype=total.dtype).detach(),
        }

    def _absolute_batch_reliability(
        self,
        labels: GRMStage25PseudoLabels,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute a bounded reliability gate without batch-mean cancellation."""

        weights = labels.confidence.float().clamp_min(0.0)
        counts = mask.sum(dim=1).float()
        mean_confidence = (weights * mask.float()).sum(dim=1) / counts.clamp_min(1.0)
        confidence_factor = (
            mean_confidence / self.config.absolute_confidence_reference
        ).clamp(0.0, 1.0)
        count_factor = (
            counts / float(self.config.reliable_match_count_reference)
        ).clamp(0.0, 1.0)

        if labels.region_ids is None:
            covered_regions = (counts > 0).float()
            coverage_factor = count_factor
        else:
            if labels.region_ids.shape != mask.shape:
                raise ValueError("region_ids must have the same [B,N] shape as valid_mask.")
            covered_values = []
            for batch_index in range(mask.shape[0]):
                selected = labels.region_ids[batch_index][mask[batch_index]]
                selected = selected[selected >= 0]
                covered_values.append(
                    counts.new_tensor(float(torch.unique(selected).numel()))
                )
            covered_regions = torch.stack(covered_values)
            coverage_factor = (
                covered_regions / float(self.config.reliable_region_count_reference)
            ).clamp(0.0, 1.0)

        per_sample = confidence_factor * torch.sqrt(count_factor * coverage_factor)
        active = counts > 0
        reliability = per_sample[active].mean() if active.any() else per_sample.sum() * 0.0
        return (
            reliability.clamp(0.0, 1.0),
            mean_confidence[active].mean(),
            covered_regions[active].mean(),
        )


def _teacher_direction(
    model: GRMMatcher,
    probability: Tensor,
    matchability: Tensor,
    features_source: Dict[str, Tensor],
    features_target: Dict[str, Tensor],
    coarse_height: int,
    coarse_width: int,
    global_step: Optional[int],
) -> Dict[str, object]:
    sampled = model.candidate_sampler(
        probability,
        matchability,
        coarse_height,
        coarse_width,
        global_step=global_step,
        source_valid_mask=features_source["valid_coarse"].flatten(1),
        target_valid_mask=features_target["valid_coarse"].flatten(1),
    )
    src_xy = flat_indices_to_image_xy(
        sampled.query_indices, coarse_width, model.config.coarse_stride
    )
    coarse_dst = flat_indices_to_image_xy(
        sampled.candidate_indices[..., 0], coarse_width, model.config.coarse_stride
    )
    valid = sampled.valid_mask & (
        sampled.candidate_global_probs[..., 0] > model.config.probability_eps
    )
    mid, fine = model.refine_points(features_source, features_target, src_xy, coarse_dst, valid)
    valid = valid & fine.valid_mask
    return {
        "sampled": sampled,
        "src_xy": src_xy,
        "coarse_dst_xy": coarse_dst,
        "valid": valid,
        "mid": mid,
        "fine": fine,
    }


def _warp_image(image: Tensor, transform: Tensor) -> Tensor:
    batch, _, height, width = image.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=image.device, dtype=torch.float32),
        torch.arange(width, device=image.device, dtype=torch.float32),
        indexing="ij",
    )
    output_points = torch.stack((xx, yy), dim=-1).reshape(1, height * width, 2)
    output_points = output_points.expand(batch, -1, -1)
    source_points = transform_points(output_points, torch.linalg.inv(transform.float()))
    grid = image_xy_to_normalized_grid(source_points, height, width).reshape(
        batch, height, width, 2
    )
    return F.grid_sample(
        image.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


def _strong_photometric(image: Tensor, config: GRMStage25Config) -> Tensor:
    batch = image.shape[0]
    shape = (batch, 1, 1, 1)
    brightness = 1.0 + (torch.rand(shape, device=image.device) * 2.0 - 1.0) * config.brightness_jitter
    contrast = 1.0 + (torch.rand(shape, device=image.device) * 2.0 - 1.0) * config.contrast_jitter
    saturation = 1.0 + (torch.rand(shape, device=image.device) * 2.0 - 1.0) * config.saturation_jitter
    gamma = 1.0 + (torch.rand(shape, device=image.device) * 2.0 - 1.0) * config.gamma_jitter
    value = (image * brightness).clamp(0.0, 1.0)
    channel_mean = value.mean(dim=1, keepdim=True)
    value = ((value - channel_mean) * saturation + channel_mean).clamp(0.0, 1.0)
    spatial_mean = value.mean(dim=(1, 2, 3), keepdim=True)
    value = ((value - spatial_mean) * contrast + spatial_mean).clamp(0.0, 1.0)
    value = value.clamp_min(1.0e-6).pow(gamma).clamp(0.0, 1.0)
    if config.noise_std > 0.0:
        value = (value + torch.randn_like(value) * config.noise_std).clamp(0.0, 1.0)
    if config.blur_probability > 0.0:
        use_blur = (
            torch.rand(batch, 1, 1, 1, device=image.device) < config.blur_probability
        )
        blurred = F.avg_pool2d(value, kernel_size=3, stride=1, padding=1)
        value = torch.where(use_blur, blurred, value)
    return value


def _to_zero_one(image: Tensor, config: GRMMatcherConfig) -> Tensor:
    if config.input_range == "zero_one":
        return image.float().clamp(0.0, 1.0)
    mean = image.new_tensor(config.normalization_mean).view(1, 3, 1, 1)
    std = image.new_tensor(config.normalization_std).view(1, 3, 1, 1)
    return (image.float() * std + mean).clamp(0.0, 1.0)


def _from_zero_one(image: Tensor, config: GRMMatcherConfig) -> Tensor:
    if config.input_range == "zero_one":
        return image.clamp(0.0, 1.0)
    mean = image.new_tensor(config.normalization_mean).view(1, 3, 1, 1)
    std = image.new_tensor(config.normalization_std).view(1, 3, 1, 1)
    return (image - mean) / std


def _points_inside(points: Tensor, height: int, width: int) -> Tensor:
    return (
        (points[..., 0] >= 0.0)
        & (points[..., 0] <= float(width - 1))
        & (points[..., 1] >= 0.0)
        & (points[..., 1] <= float(height - 1))
    )


def _sample_valid_mask(mask: Tensor, points: Tensor) -> Tensor:
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("Validity mask must be [B,1,H,W].")
    grid = image_xy_to_normalized_grid(points.float(), mask.shape[-2], mask.shape[-1])
    sampled = F.grid_sample(
        mask.float(), grid.unsqueeze(2), mode="nearest", padding_mode="zeros",
        align_corners=False,
    ).squeeze(1).squeeze(-1)
    return sampled > 0.5


def _weighted_smooth_l1(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    weights: Tensor,
    beta: float,
) -> Tensor:
    element = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta).sum(-1)
    selected_weights = weights * mask.float()
    return (element * selected_weights).sum() / selected_weights.sum().clamp_min(1.0e-8)


def _local_probability_nll(
    probability: Tensor,
    center_xy: Tensor,
    target_xy: Tensor,
    stride: int,
    mask: Tensor,
    weights: Tensor,
    eps: float,
) -> Tensor:
    batch, count, height, width = probability.shape
    offset = (target_xy.float() - center_xy.float()) / float(stride)
    index_x = offset[..., 0] + (width - 1) * 0.5
    index_y = offset[..., 1] + (height - 1) * 0.5
    in_window = (
        (index_x >= 0.0) & (index_x <= float(width - 1))
        & (index_y >= 0.0) & (index_y <= float(height - 1))
    )
    grid_x = 2.0 * (index_x + 0.5) / float(width) - 1.0
    grid_y = 2.0 * (index_y + 0.5) / float(height) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).reshape(batch * count, 1, 1, 2)
    sampled = F.grid_sample(
        probability.float().reshape(batch * count, 1, height, width),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).reshape(batch, count)
    selected = mask & in_window
    selected_weights = weights * selected.float()
    nll = -torch.log(sampled.clamp_min(eps))
    return (nll * selected_weights).sum() / selected_weights.sum().clamp_min(eps)
