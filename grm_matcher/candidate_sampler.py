"""Top-k retention and adaptive reliability-density candidate sampling."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Union

import torch
from torch import Tensor, nn

from .config import GRMCandidateSamplerConfig, GRMMatcherConfig


@dataclass
class GRMTopKCandidates:
    """Top-k hypotheses for every coarse source cell.

    ``global_probs`` remain probabilities from the full explicit correlation
    matrix. ``conditional_probs`` are separately normalized inside the retained
    top-k set.  The two probability semantics are never interchanged.
    """

    indices: Tensor
    global_probs: Tensor
    conditional_probs: Tensor
    top1_probability: Tensor
    top2_probability: Tensor
    top1_top2_margin: Tensor
    global_top1_top2_margin: Tensor
    entropy: Tensor
    normalized_entropy: Tensor
    valid_mask: Tensor


@dataclass
class GRMSampledCandidates:
    """Fixed-length sampled queries plus allocation diagnostics.

    Existing public fields are retained unchanged. Padding slots use index zero,
    zero-valued probabilities/scores, ``region_ids=-1``, and ``valid_mask=False``.
    """

    query_indices: Tensor
    candidate_indices: Tensor
    candidate_global_probs: Tensor
    candidate_probs: Tensor
    top1_probability: Tensor
    top2_probability: Tensor
    top1_top2_margin: Tensor
    candidate_entropy: Tensor
    normalized_entropy: Tensor
    sampling_score: Tensor
    valid_mask: Tensor

    source_xy_coarse: Tensor
    query_scores: Tensor
    region_ids: Tensor
    region_density: Tensor
    region_valid_count: Tensor
    region_quota: Tensor
    effective_density_power: Tensor
    effective_score_threshold: Tensor
    coverage_budget_clipped: Tensor
    actual_budget: Tensor
    low_quality_sample: Tensor
    diagnostics: Dict[str, Tensor]

    @property
    def source_indices(self) -> Tensor:
        """Alias requested by the adaptive-sampling diagnostic contract."""

        return self.query_indices


class GRMTopKCandidateSampler(nn.Module):
    """Retain hypotheses and adapt query quotas to reliable regional density."""

    def __init__(
        self,
        config: Union[GRMMatcherConfig, GRMCandidateSamplerConfig],
    ) -> None:
        super().__init__()
        if isinstance(config, GRMMatcherConfig):
            self.config = config.candidate_sampler_config()
            self.legacy_grid_rows = config.sampling_grid_rows
            self.legacy_grid_cols = config.sampling_grid_cols
        elif isinstance(config, GRMCandidateSamplerConfig):
            self.config = config
            self.legacy_grid_rows = 8
            self.legacy_grid_cols = 8
        else:
            raise TypeError(
                "config must be GRMMatcherConfig or GRMCandidateSamplerConfig."
            )
        self.topk = self.config.topk_candidates
        self.max_queries = self.config.max_source_queries
        self.eps = self.config.probability_eps

    def topk_candidates(
        self,
        probability: Tensor,
        source_valid_mask: Optional[Tensor] = None,
        target_valid_mask: Optional[Tensor] = None,
        target_width: Optional[int] = None,
    ) -> GRMTopKCandidates:
        """Compute padded top-k hypotheses for every source query."""

        if probability.ndim != 3 or not probability.is_floating_point():
            raise ValueError("probability must be a floating [B, S, T] tensor.")
        batch, source_count, target_count = probability.shape
        if target_count <= 0:
            raise ValueError("The target correlation dimension must be non-empty.")
        source_valid = self._prepare_mask(
            source_valid_mask, batch, source_count, probability.device
        )
        target_valid = self._prepare_mask(
            target_valid_mask, batch, target_count, probability.device
        )

        probability_fp32 = probability.float()
        masked_probability = torch.where(
            torch.isfinite(probability_fp32),
            probability_fp32.clamp_min(0.0),
            torch.zeros_like(probability_fp32),
        )
        masked_probability = masked_probability * source_valid[:, :, None].float()
        masked_probability = masked_probability * target_valid[:, None, :].float()

        effective_k = min(self.topk, target_count)
        global_probs, indices = torch.topk(
            masked_probability, k=effective_k, dim=-1, largest=True, sorted=True
        )
        radius = self.config.candidate_peak_radius
        if radius > 0 and target_width is not None:
            if target_width <= 0 or target_count % target_width:
                raise ValueError("target_width must divide target_count")
            coordinates = torch.arange(target_count, device=probability.device)
            x, y = coordinates % target_width, coordinates // target_width
            remaining = masked_probability
            chosen, weights = [], []
            for _ in range(effective_k):
                value, index = remaining.max(dim=-1)
                chosen.append(index)
                weights.append(value)
                nearby = ((x - (index % target_width)[..., None]).abs() <= radius) & (
                    (y - (index // target_width)[..., None]).abs() <= radius)
                remaining = remaining.masked_fill(nearby, 0.)
            indices = torch.stack(chosen, -1)
            global_probs = torch.stack(weights, -1)
        if effective_k < self.topk:
            pad_count = self.topk - effective_k
            global_probs = torch.cat(
                (global_probs, global_probs.new_zeros(batch, source_count, pad_count)),
                dim=-1,
            )
            indices = torch.cat(
                (
                    indices,
                    torch.zeros(
                        batch,
                        source_count,
                        pad_count,
                        device=indices.device,
                        dtype=torch.long,
                    ),
                ),
                dim=-1,
            )

        retained_mass = global_probs.sum(dim=-1, keepdim=True)
        valid = source_valid & (retained_mass.squeeze(-1) > self.eps)
        conditional_probs = torch.where(
            retained_mass > self.eps,
            global_probs / retained_mass.clamp_min(self.eps),
            torch.zeros_like(global_probs),
        )
        conditional_probs = conditional_probs * valid[..., None].float()
        entropy = -(
            conditional_probs
            * torch.log(conditional_probs.clamp_min(self.eps))
        ).sum(dim=-1)
        if effective_k > 1:
            entropy_denominator = probability.new_tensor(float(effective_k)).log()
            normalized_entropy = (entropy / entropy_denominator).clamp(0.0, 1.0)
        else:
            normalized_entropy = torch.zeros_like(entropy)

        top1_global_probability = global_probs[..., 0]
        top2_global_probability = (
            global_probs[..., 1] if self.topk > 1 else torch.zeros_like(top1_global_probability)
        )
        conditional_top1 = conditional_probs[..., 0]
        conditional_top2 = (
            conditional_probs[..., 1]
            if self.topk > 1
            else torch.zeros_like(conditional_top1)
        )
        return GRMTopKCandidates(
            indices=indices,
            global_probs=global_probs,
            conditional_probs=conditional_probs,
            top1_probability=top1_global_probability,
            top2_probability=top2_global_probability,
            top1_top2_margin=(conditional_top1 - conditional_top2).clamp_min(0.0),
            global_top1_top2_margin=(
                top1_global_probability - top2_global_probability
            ).clamp_min(0.0),
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            valid_mask=valid,
        )

    def forward(
        self,
        probability: Tensor,
        matchability_a: Tensor,
        source_height: int,
        source_width: int,
        source_valid_mask: Optional[Tensor] = None,
        target_valid_mask: Optional[Tensor] = None,
        global_step: Optional[int] = None,
    ) -> GRMSampledCandidates:
        """Sample a fixed maximum length with adaptive density allocation.

        ``global_step`` is explicit by design. When density scheduling is enabled
        and no step is supplied, the conservative warmup exponent is used.
        """

        all_candidates = self.topk_candidates(
            probability, source_valid_mask, target_valid_mask, target_width=source_width
        )
        batch, source_count, _ = probability.shape
        if source_height <= 0 or source_width <= 0:
            raise ValueError("source_height and source_width must be positive.")
        if source_height * source_width != source_count:
            raise ValueError(
                "source_height * source_width must equal the correlation source dimension."
            )
        if matchability_a.shape != (batch, source_count):
            raise ValueError(
                f"matchability_a must be {(batch, source_count)}, got {tuple(matchability_a.shape)}."
            )

        # Ambiguity quality uses conditional top-k probabilities, whereas the
        # multiplicative appearance term remains the full-matrix top-1 value.
        weight_sum = self.config.entropy_weight + self.config.margin_weight
        ambiguity_quality = (
            self.config.entropy_weight * (1.0 - all_candidates.normalized_entropy)
            + self.config.margin_weight * all_candidates.top1_top2_margin
        ) / weight_sum
        top1_global_probability = all_candidates.top1_probability
        effective_score = (
            matchability_a.float().clamp(0.0, 1.0)
            * top1_global_probability
            * ambiguity_quality.clamp(0.0, 1.0)
        )
        effective_score = torch.where(
            all_candidates.valid_mask & torch.isfinite(effective_score),
            effective_score,
            torch.zeros_like(effective_score),
        )
        support_diagnostics = {}
        if self.config.sampling_mode == "overlap_aware":
            threshold_valid, thresholds, support_diagnostics = self._supported_queries(
                probability, matchability_a, all_candidates, effective_score,
                source_valid_mask, target_valid_mask,
            )
        else:
            threshold_valid, thresholds = self._threshold_queries(
                effective_score, all_candidates.valid_mask
            )

        region_ids_all, grid_rows, grid_cols = self._make_region_layout(
            source_height, source_width, probability.device
        )
        region_count = grid_rows * grid_cols
        (
            region_density,
            region_valid_count,
            region_mean_score,
            region_max_score,
        ) = self._region_statistics(
            effective_score, threshold_valid, region_ids_all, region_count
        )
        (
            actual_budgets,
            high_quality_counts,
            mean_valid_scores,
        ) = self._compute_query_budgets(
            effective_score,
            threshold_valid,
            region_valid_count,
        )

        output_indices = torch.zeros(
            batch, self.max_queries, device=probability.device, dtype=torch.long
        )
        output_valid = torch.zeros(
            batch, self.max_queries, device=probability.device, dtype=torch.bool
        )
        region_quota = torch.zeros(
            batch, region_count, device=probability.device, dtype=torch.long
        )
        region_max_quota = torch.zeros_like(region_quota)
        coverage_budget_clipped = torch.zeros(
            batch, device=probability.device, dtype=torch.bool
        )
        effective_power_value = self._effective_density_power(global_step)
        effective_density_power = probability.new_full(
            (batch,), effective_power_value, dtype=torch.float32
        )

        for batch_index in range(batch):
            sample_budget = int(actual_budgets[batch_index].item())
            if self.config.sampling_mode == "overlap_aware":
                selected = self._support_select_single(
                    effective_score[batch_index], threshold_valid[batch_index],
                    source_width, sample_budget,
                )
                actual_quota = torch.bincount(region_ids_all[selected], minlength=region_count)
                # No hard per-region quota in this path; report support capacity.
                maximum_quota = region_valid_count[batch_index].clone()
                clipped = False
            elif self.config.use_adaptive_density_sampling:
                (
                    selected,
                    actual_quota,
                    maximum_quota,
                    clipped,
                ) = self._adaptive_select_single(
                    effective_score[batch_index],
                    threshold_valid[batch_index],
                    region_ids_all,
                    region_density[batch_index],
                    region_valid_count[batch_index],
                    region_mean_score[batch_index],
                    region_max_score[batch_index],
                    source_height,
                    source_width,
                    grid_rows,
                    grid_cols,
                    effective_power_value,
                    sample_budget,
                )
            else:
                selected, actual_quota = self._legacy_select_single(
                    effective_score[batch_index],
                    threshold_valid[batch_index],
                    region_ids_all,
                    region_count,
                    sample_budget,
                )
                maximum_quota = region_valid_count[batch_index].clone()
                clipped = False
            count = min(int(selected.numel()), self.max_queries)
            if count > 0:
                output_indices[batch_index, :count] = selected[:count]
                output_valid[batch_index, :count] = True
            region_quota[batch_index] = actual_quota
            region_max_quota[batch_index] = maximum_quota
            coverage_budget_clipped[batch_index] = clipped

        gathered_candidate_indices = self._gather(
            all_candidates.indices, output_indices
        )
        gathered_global_probs = self._gather(
            all_candidates.global_probs, output_indices
        )
        gathered_conditional_probs = self._gather(
            all_candidates.conditional_probs, output_indices
        )
        gathered_top1 = self._gather(all_candidates.top1_probability, output_indices)
        gathered_top2 = self._gather(all_candidates.top2_probability, output_indices)
        gathered_margin = self._gather(
            all_candidates.top1_top2_margin, output_indices
        )
        gathered_entropy = self._gather(all_candidates.entropy, output_indices)
        gathered_normalized_entropy = self._gather(
            all_candidates.normalized_entropy, output_indices
        )
        gathered_score = self._gather(effective_score, output_indices)

        valid_float = output_valid.float()
        gathered_candidate_indices = torch.where(
            output_valid[..., None],
            gathered_candidate_indices,
            torch.zeros_like(gathered_candidate_indices),
        )
        gathered_global_probs = gathered_global_probs * valid_float[..., None]
        gathered_conditional_probs = gathered_conditional_probs * valid_float[..., None]
        gathered_top1 = gathered_top1 * valid_float
        gathered_top2 = gathered_top2 * valid_float
        gathered_margin = gathered_margin * valid_float
        gathered_entropy = gathered_entropy * valid_float
        gathered_normalized_entropy = gathered_normalized_entropy * valid_float
        gathered_score = gathered_score * valid_float

        x = torch.remainder(output_indices, source_width).float()
        y = torch.div(output_indices, source_width, rounding_mode="floor").float()
        source_xy_coarse = torch.stack((x, y), dim=-1) * valid_float[..., None]
        selected_region_ids = region_ids_all[output_indices]
        selected_region_ids = torch.where(
            output_valid,
            selected_region_ids,
            torch.full_like(selected_region_ids, -1),
        )
        diagnostics = self._build_diagnostics(
            source_xy_coarse,
            gathered_score,
            gathered_entropy,
            output_valid,
            selected_region_ids,
            region_density,
            region_valid_count,
            region_mean_score,
            region_max_score,
            region_quota,
            region_max_quota,
            effective_density_power,
            thresholds,
            coverage_budget_clipped,
            actual_budgets,
            high_quality_counts,
            mean_valid_scores,
            source_height,
            source_width,
            self.config.min_geometry_queries,
            grid_rows,
            grid_cols,
        )
        diagnostics.update(support_diagnostics)
        low_quality_sample = diagnostics["low_quality_sample"]
        return GRMSampledCandidates(
            query_indices=output_indices,
            candidate_indices=gathered_candidate_indices,
            candidate_global_probs=gathered_global_probs,
            candidate_probs=gathered_conditional_probs,
            top1_probability=gathered_top1,
            top2_probability=gathered_top2,
            top1_top2_margin=gathered_margin,
            candidate_entropy=gathered_entropy,
            normalized_entropy=gathered_normalized_entropy,
            sampling_score=gathered_score,
            valid_mask=output_valid,
            source_xy_coarse=source_xy_coarse,
            query_scores=gathered_score,
            region_ids=selected_region_ids,
            region_density=region_density,
            region_valid_count=region_valid_count,
            region_quota=region_quota,
            effective_density_power=effective_density_power,
            effective_score_threshold=thresholds,
            coverage_budget_clipped=coverage_budget_clipped,
            actual_budget=actual_budgets,
            low_quality_sample=low_quality_sample,
            diagnostics=diagnostics,
        )

    def _supported_queries(
        self, probability: Tensor, matchability: Tensor,
        candidates: GRMTopKCandidates, scores: Tensor,
        source_valid_mask: Optional[Tensor], target_valid_mask: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """Estimate reliable support, not GT overlap, without coverage fallback.

        Require at least one reciprocal top-k edge. This permits ambiguous
        multi-hypothesis matches instead of requiring mutual top-1. All gates
        use model predictions; even an all-empty support remains empty.
        """
        reverse = self.topk_candidates(
            probability.transpose(1, 2), target_valid_mask, source_valid_mask
        )
        batch, count = scores.shape
        batch_ids = torch.arange(batch, device=scores.device)[:, None, None]
        back_indices = reverse.indices[batch_ids, candidates.indices]
        back_probs = reverse.global_probs[batch_ids, candidates.indices]
        source_ids = torch.arange(count, device=scores.device)[None, :, None, None]
        reciprocal_edges = (
            (back_indices == source_ids) & (back_probs > self.eps)
            & (candidates.global_probs[..., None] > self.eps)
            & (candidates.global_probs[..., None] >=
               candidates.top1_probability[..., None, None] * self.config.support_relative_score)
        ).any(dim=-1)
        reciprocal = reciprocal_edges.any(dim=-1)
        eligible = (
            candidates.valid_mask & reciprocal & torch.isfinite(matchability)
            & (matchability >= self.config.support_min_matchability)
            & torch.isfinite(scores) & (scores > 0)
        )
        peak = scores.masked_fill(~eligible, 0).amax(dim=1)
        thresholds = torch.maximum(
            peak * self.config.support_relative_score,
            scores.new_full((batch,), self.config.fixed_score_threshold),
        )
        support = eligible & (scores >= thresholds[:, None])
        return support, thresholds, {
            "reciprocal_query_count": reciprocal.sum(dim=1),
            "supported_query_count": support.sum(dim=1),
            "supported_query_fraction": support.float().mean(dim=1),
        }

    def _support_select_single(
        self, scores: Tensor, support: Tensor, width: int, budget: int,
    ) -> Tensor:
        """Quality-first selection with a bounded spatial diversity bonus.

        Distances are normalized within predicted support, not the full image.
        There is no minimum quota, forced out-of-support fill, or region cap
        that can starve a narrow overlap strip of otherwise reliable matches.
        """
        indices = torch.nonzero(support, as_tuple=False).squeeze(-1)
        budget = min(budget, indices.numel())
        if budget == 0:
            return indices[:0]
        if budget == indices.numel():
            return indices[torch.argsort(scores[indices], descending=True, stable=True)]
        xy = torch.stack((indices % width, indices // width), dim=-1).float()
        extent = xy.amax(dim=0) - xy.amin(dim=0)
        diagonal = torch.linalg.vector_norm(extent).clamp_min(1.0)
        nearest = torch.ones_like(scores[indices])
        chosen = torch.zeros_like(indices, dtype=torch.bool)
        selected = torch.empty(budget, device=scores.device, dtype=torch.long)
        for slot in range(budget):
            priority = scores[indices] * (1.0 + self.config.coverage_weight * nearest)
            position = priority.masked_fill(chosen, -torch.inf).argmax()
            selected[slot] = indices[position]
            chosen[position] = True
            distance = torch.linalg.vector_norm(xy - xy[position], dim=-1) / diagonal
            nearest = torch.minimum(nearest, distance.clamp(0.0, 1.0))
        return selected[torch.argsort(scores[selected], descending=True, stable=True)]

    def _threshold_queries(
        self, scores: Tensor, candidate_valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch, source_count = scores.shape
        output = torch.zeros_like(candidate_valid)
        thresholds = scores.new_full(
            (batch,), self.config.fixed_score_threshold, dtype=torch.float32
        )
        for batch_index in range(batch):
            finite_nonzero = (
                candidate_valid[batch_index]
                & torch.isfinite(scores[batch_index])
                & (scores[batch_index] > 0.0)
            )
            absolute_valid = finite_nonzero & (
                scores[batch_index] >= self.config.fixed_score_threshold
            )
            if not absolute_valid.any():
                continue
            if not self.config.use_quantile_threshold:
                output[batch_index] = absolute_valid
                continue
            quantile = torch.quantile(
                scores[batch_index, finite_nonzero], self.config.score_quantile
            )
            threshold = torch.maximum(
                quantile,
                scores.new_tensor(self.config.fixed_score_threshold),
            )
            selected = absolute_valid & (scores[batch_index] >= threshold)
            # Quantile filtering is allowed to underfill.  Fallback only lowers
            # the threshold far enough to provide a small usable query set, not
            # all the way back to max_source_queries.
            if self.config.allow_underfilled_budget:
                desired_count = min(
                    self.config.minimum_query_count,
                    self.max_queries,
                    int(absolute_valid.sum().item()),
                )
            else:
                desired_count = min(
                    self.max_queries,
                    int(absolute_valid.sum().item()),
                )
            if int(selected.sum().item()) < desired_count:
                eligible_indices = torch.nonzero(
                    absolute_valid, as_tuple=False
                ).squeeze(-1)
                top_local = torch.topk(
                    scores[batch_index, eligible_indices],
                    k=desired_count,
                    sorted=True,
                ).indices
                selected = torch.zeros(source_count, device=scores.device, dtype=torch.bool)
                selected[eligible_indices[top_local]] = True
                threshold = scores[batch_index, eligible_indices[top_local]].amin()
            output[batch_index] = selected
            thresholds[batch_index] = threshold
        return output, thresholds

    def _compute_query_budgets(
        self,
        scores: Tensor,
        high_quality_mask: Tensor,
        region_valid_count: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute a per-sample budget from support count, quality and coverage.

        The thresholded high-quality count is always a hard upper bound.  When
        dynamic budgeting is enabled, mean reliability scales that upper bound;
        minimum query support and one base slot per valid region form conservative
        lower bounds.  Thus weak images may underfill without losing all spatial
        support.
        """

        high_quality_count = high_quality_mask.sum(dim=1)
        score_sum = (scores * high_quality_mask.float()).sum(dim=1)
        mean_valid_score = score_sum / high_quality_count.clamp_min(1).float()
        maximum_budget = torch.minimum(
            high_quality_count,
            torch.full_like(high_quality_count, self.max_queries),
        )
        if not self.config.use_dynamic_query_budget:
            return maximum_budget, high_quality_count, mean_valid_score

        quality_fraction = (
            mean_valid_score / self.config.dynamic_budget_target_mean_score
        ).clamp(0.0, 1.0)
        quality_budget = torch.ceil(
            maximum_budget.float() * quality_fraction
        ).long()
        if self.config.sampling_mode == "overlap_aware":
            # minimum_query_count and regional coverage must never inflate the
            # budget in a low-support pair. Padding is explicitly invalid.
            return quality_budget, high_quality_count, mean_valid_score
        valid_region_count = (region_valid_count > 0).sum(dim=1)
        coverage_floor = valid_region_count * self.config.min_quota_per_valid_region
        minimum_floor = torch.full_like(
            high_quality_count, self.config.minimum_query_count
        )
        supported_floor = torch.maximum(coverage_floor, minimum_floor)
        actual_budget = torch.minimum(
            maximum_budget,
            torch.maximum(quality_budget, supported_floor),
        )
        actual_budget = torch.where(
            high_quality_count > 0,
            actual_budget.clamp_min(1),
            torch.zeros_like(actual_budget),
        )
        return actual_budget, high_quality_count, mean_valid_score

    def _make_region_layout(
        self, height: int, width: int, device: torch.device
    ) -> tuple[Tensor, int, int]:
        y = torch.arange(height, device=device)[:, None].expand(-1, width)
        x = torch.arange(width, device=device)[None, :].expand(height, -1)
        if self.config.use_adaptive_density_sampling:
            grid_rows = math.ceil(height / self.config.region_height_cells)
            grid_cols = math.ceil(width / self.config.region_width_cells)
            region_y = torch.div(
                y, self.config.region_height_cells, rounding_mode="floor"
            )
            region_x = torch.div(
                x, self.config.region_width_cells, rounding_mode="floor"
            )
        else:
            grid_rows = min(self.legacy_grid_rows, height)
            grid_cols = min(self.legacy_grid_cols, width)
            region_y = torch.div(y * grid_rows, height, rounding_mode="floor")
            region_x = torch.div(x * grid_cols, width, rounding_mode="floor")
        return (region_y * grid_cols + region_x).reshape(-1), grid_rows, grid_cols

    @staticmethod
    def _region_statistics(
        scores: Tensor,
        valid_mask: Tensor,
        region_ids: Tensor,
        region_count: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = scores.shape[0]
        density = scores.new_zeros(batch, region_count)
        count = torch.zeros(
            batch, region_count, device=scores.device, dtype=torch.long
        )
        maximum = scores.new_zeros(batch, region_count)
        for batch_index in range(batch):
            valid = valid_mask[batch_index]
            if not valid.any():
                continue
            valid_regions = region_ids[valid]
            valid_scores = scores[batch_index, valid]
            density[batch_index].scatter_add_(0, valid_regions, valid_scores)
            count[batch_index].scatter_add_(
                0, valid_regions, torch.ones_like(valid_regions, dtype=torch.long)
            )
            maximum[batch_index].scatter_reduce_(
                0, valid_regions, valid_scores, reduce="amax", include_self=True
            )
        mean = density / count.clamp_min(1).to(dtype=density.dtype)
        return density, count, mean, maximum

    def _adaptive_select_single(
        self,
        scores: Tensor,
        valid_mask: Tensor,
        region_ids: Tensor,
        region_density: Tensor,
        region_valid_count: Tensor,
        region_mean_score: Tensor,
        region_max_score: Tensor,
        height: int,
        width: int,
        grid_rows: int,
        grid_cols: int,
        density_power: float,
        budget: int,
    ) -> tuple[Tensor, Tensor, Tensor, bool]:
        region_count = region_density.numel()
        budget = min(budget, self.max_queries, int(valid_mask.sum().item()))
        empty = torch.empty(0, device=scores.device, dtype=torch.long)
        if budget == 0:
            zeros = torch.zeros(
                region_count, device=scores.device, dtype=torch.long
            )
            return empty, zeros, zeros.clone(), False

        valid_regions = region_valid_count > 0
        valid_region_count = int(valid_regions.sum().item())
        base_target = torch.minimum(
            region_valid_count,
            torch.full_like(
                region_valid_count, self.config.min_quota_per_valid_region
            ),
        )
        quota = torch.zeros_like(region_valid_count)
        coverage_budget_clipped = int(base_target.sum().item()) > budget
        priority = self._region_priority(
            region_max_score, region_mean_score, region_density, valid_regions
        )
        if not coverage_budget_clipped:
            quota = base_target.clone()
        else:
            coverage_slots = min(budget, valid_region_count)
            covered_regions = self._choose_coverage_regions(
                priority,
                valid_regions,
                coverage_slots,
                grid_rows,
                grid_cols,
            )
            quota[covered_regions] = 1
            remaining_base = budget - coverage_slots
            while remaining_base > 0:
                active = quota < base_target
                if not active.any():
                    break
                choice_score = priority.masked_fill(~active, -torch.inf)
                choice = int(choice_score.argmax().item())
                quota[choice] += 1
                remaining_base -= 1

        if self.config.dynamic_max_quota:
            maximum_per_region = math.ceil(
                self.config.max_quota_multiplier
                * budget
                / max(valid_region_count, 1)
            )
            maximum_per_region = max(
                self.config.max_quota_lower_bound,
                min(maximum_per_region, self.config.max_quota_upper_bound),
            )
        else:
            maximum_per_region = self.config.max_quota_per_region
        region_max_quota = torch.minimum(
            region_valid_count,
            torch.full_like(region_valid_count, maximum_per_region),
        )
        region_max_quota = torch.maximum(region_max_quota, quota)
        quota = self._allocate_density_quota(
            quota,
            region_max_quota,
            region_density,
            budget,
            density_power,
        )

        selected_parts: list[Tensor] = []
        flat = torch.arange(scores.numel(), device=scores.device)
        coordinates = torch.stack(
            (
                torch.remainder(flat, width),
                torch.div(flat, width, rounding_mode="floor"),
            ),
            dim=-1,
        ).float()
        for region in torch.nonzero(quota > 0, as_tuple=False).squeeze(-1).tolist():
            region_indices = torch.nonzero(
                valid_mask & (region_ids == region), as_tuple=False
            ).squeeze(-1)
            requested = int(quota[region].item())
            region_area = int((region_ids == region).sum().item())
            selected_parts.append(
                self._select_inside_region(
                    region_indices,
                    scores,
                    coordinates,
                    requested,
                    region_area,
                )
            )
        selected = torch.cat(selected_parts) if selected_parts else empty
        selected = self._global_fill(
            selected,
            scores,
            valid_mask,
            coordinates,
            region_ids,
            region_max_quota,
            budget,
            height,
            width,
        )
        if selected.numel() > 0:
            order = torch.argsort(scores[selected], descending=True, stable=True)
            selected = selected[order]
        actual_quota = torch.bincount(
            region_ids[selected], minlength=region_count
        ) if selected.numel() else torch.zeros_like(region_valid_count)
        return selected, actual_quota, region_max_quota, coverage_budget_clipped

    def _legacy_select_single(
        self,
        scores: Tensor,
        valid_mask: Tensor,
        region_ids: Tensor,
        region_count: int,
        budget: int,
    ) -> tuple[Tensor, Tensor]:
        """Compatibility path for the previous equal-quota fixed-grid sampler."""

        budget = min(budget, self.max_queries, int(valid_mask.sum().item()))
        empty = torch.empty(0, device=scores.device, dtype=torch.long)
        if budget == 0:
            return empty, torch.zeros(
                region_count, device=scores.device, dtype=torch.long
            )
        quota_per_region = max(1, math.ceil(budget / region_count))
        selected_parts: list[Tensor] = []
        for region in range(region_count):
            indices = torch.nonzero(
                valid_mask & (region_ids == region), as_tuple=False
            ).squeeze(-1)
            if indices.numel() == 0:
                continue
            count = min(quota_per_region, int(indices.numel()))
            local = torch.topk(scores[indices], count, sorted=True).indices
            selected_parts.append(indices[local])
        selected = torch.cat(selected_parts) if selected_parts else empty
        if selected.numel() > budget:
            selected = selected[torch.topk(scores[selected], budget, sorted=True).indices]
        if selected.numel() < budget:
            chosen = torch.zeros(scores.numel(), device=scores.device, dtype=torch.bool)
            chosen[selected] = True
            remaining = torch.nonzero(
                valid_mask & ~chosen, as_tuple=False
            ).squeeze(-1)
            fill_count = min(budget - int(selected.numel()), int(remaining.numel()))
            if fill_count > 0:
                fill = remaining[torch.topk(scores[remaining], fill_count).indices]
                selected = torch.cat((selected, fill))
        if selected.numel() > 0:
            selected = selected[
                torch.argsort(scores[selected], descending=True, stable=True)
            ]
        actual_quota = torch.bincount(
            region_ids[selected], minlength=region_count
        ) if selected.numel() else torch.zeros(
            region_count, device=scores.device, dtype=torch.long
        )
        return selected, actual_quota

    def _allocate_density_quota(
        self,
        quota: Tensor,
        maximum_quota: Tensor,
        density: Tensor,
        budget: int,
        density_power: float,
    ) -> Tensor:
        quota = quota.clone()
        remaining = budget - int(quota.sum().item())
        while remaining > 0:
            capacity = maximum_quota - quota
            active = capacity > 0
            if not active.any():
                break
            weights = torch.where(
                active,
                (density + self.config.density_eps).pow(density_power),
                torch.zeros_like(density),
            )
            if float(weights.sum().item()) <= 0.0:
                weights = active.float()
            ideal = remaining * weights / weights.sum().clamp_min(
                self.config.density_eps
            )
            floor_allocation = torch.minimum(
                torch.floor(ideal).long(), capacity
            )
            allocated = int(floor_allocation.sum().item())
            if allocated > 0:
                quota += floor_allocation
                remaining -= allocated
                if remaining <= 0:
                    break
            capacity = maximum_quota - quota
            active = capacity > 0
            if not active.any():
                break
            fractional = ideal - torch.floor(ideal)
            fractional = fractional.masked_fill(~active, -torch.inf)
            order = torch.argsort(fractional, descending=True, stable=True)
            active_order = order[torch.isfinite(fractional[order])]
            give = min(remaining, int(active_order.numel()))
            if give <= 0:
                break
            quota[active_order[:give]] += 1
            remaining -= give
        return quota

    def _select_inside_region(
        self,
        region_indices: Tensor,
        scores: Tensor,
        coordinates: Tensor,
        quota: int,
        region_area: int,
    ) -> Tensor:
        order = torch.argsort(
            scores[region_indices], descending=True, stable=True
        )
        ordered_indices = region_indices[order]
        if quota >= ordered_indices.numel() or not self.config.use_local_nms:
            return ordered_indices[:quota]
        if self.config.use_dynamic_nms_radius:
            radius = 0.5 * math.sqrt(region_area / max(quota, 1))
            radius = max(
                self.config.min_nms_radius,
                min(radius, self.config.max_nms_radius),
            )
        else:
            radius = self.config.local_nms_radius
        pairwise_distance = torch.cdist(
            coordinates[ordered_indices], coordinates[ordered_indices]
        )
        available = torch.ones(
            ordered_indices.numel(), device=scores.device, dtype=torch.bool
        )
        selected_positions: list[int] = []
        while len(selected_positions) < quota and available.any():
            position = int(torch.nonzero(available, as_tuple=False)[0].item())
            selected_positions.append(position)
            available &= pairwise_distance[position] >= radius
            available[position] = False
        if len(selected_positions) < quota:
            selected_mask = torch.zeros_like(available)
            if selected_positions:
                selected_mask[torch.tensor(selected_positions, device=scores.device)] = True
            suppressed = torch.nonzero(~selected_mask, as_tuple=False).squeeze(-1)
            fill_count = min(quota - len(selected_positions), int(suppressed.numel()))
            selected_positions.extend(suppressed[:fill_count].tolist())
        return ordered_indices[
            torch.tensor(selected_positions[:quota], device=scores.device)
        ]

    def _global_fill(
        self,
        selected: Tensor,
        scores: Tensor,
        valid_mask: Tensor,
        coordinates: Tensor,
        region_ids: Tensor,
        region_max_quota: Tensor,
        budget: int,
        height: int,
        width: int,
    ) -> Tensor:
        chosen = torch.zeros(scores.numel(), device=scores.device, dtype=torch.bool)
        chosen[selected] = True
        region_selected_count = torch.bincount(
            region_ids[selected], minlength=region_max_quota.numel()
        ) if selected.numel() else torch.zeros_like(region_max_quota)
        image_diagonal = max(math.hypot(height - 1, width - 1), 1.0)
        while selected.numel() < budget:
            has_region_capacity = (
                region_selected_count[region_ids] < region_max_quota[region_ids]
            )
            eligible = valid_mask & ~chosen & has_region_capacity
            if not eligible.any():
                break
            if selected.numel() > 0:
                minimum_distance = torch.cdist(
                    coordinates, coordinates[selected]
                ).amin(dim=-1)
                normalized_distance = (minimum_distance / image_diagonal).clamp(0.0, 1.0)
            else:
                normalized_distance = torch.zeros_like(scores)
            fill_score = scores * (
                1.0 + self.config.coverage_weight * normalized_distance
            )
            fill_score = fill_score.masked_fill(~eligible, -torch.inf)
            choice = int(fill_score.argmax().item())
            if not torch.isfinite(fill_score[choice]):
                break
            selected = torch.cat(
                (selected, torch.tensor([choice], device=scores.device))
            )
            chosen[choice] = True
            region_selected_count[region_ids[choice]] += 1
        return selected

    def _choose_coverage_regions(
        self,
        priority: Tensor,
        valid_regions: Tensor,
        count: int,
        grid_rows: int,
        grid_cols: int,
    ) -> Tensor:
        valid_indices = torch.nonzero(valid_regions, as_tuple=False).squeeze(-1)
        if count >= valid_indices.numel():
            return valid_indices
        row = torch.div(valid_indices, grid_cols, rounding_mode="floor").float()
        col = torch.remainder(valid_indices, grid_cols).float()
        centers = torch.stack((col, row), dim=-1)
        selected_local: list[int] = []
        available = torch.ones(
            valid_indices.numel(), device=priority.device, dtype=torch.bool
        )
        diagonal = max(math.hypot(grid_rows - 1, grid_cols - 1), 1.0)
        for _ in range(count):
            if not selected_local:
                selection_score = priority[valid_indices]
            else:
                selected_tensor = torch.tensor(
                    selected_local, device=priority.device, dtype=torch.long
                )
                distance = torch.cdist(
                    centers, centers[selected_tensor]
                ).amin(dim=-1) / diagonal
                selection_score = priority[valid_indices] * (
                    1.0 + self.config.coverage_weight * distance
                )
            selection_score = selection_score.masked_fill(~available, -torch.inf)
            local_index = int(selection_score.argmax().item())
            selected_local.append(local_index)
            available[local_index] = False
        return valid_indices[
            torch.tensor(selected_local, device=priority.device, dtype=torch.long)
        ]

    @staticmethod
    def _region_priority(
        maximum: Tensor,
        mean: Tensor,
        density: Tensor,
        valid_regions: Tensor,
    ) -> Tensor:
        def normalize(value: Tensor) -> Tensor:
            scale = value[valid_regions].amax().clamp_min(1.0e-12)
            return value / scale

        priority = (
            normalize(maximum)
            + 1.0e-3 * normalize(mean)
            + 1.0e-6 * normalize(density)
        )
        return priority.masked_fill(~valid_regions, -torch.inf)

    def _effective_density_power(self, global_step: Optional[int]) -> float:
        target = min(self.config.density_power, self.config.density_power_max)
        if not self.config.use_density_schedule:
            return target
        warmup = min(self.config.density_power_warmup, self.config.density_power_max)
        if global_step is None:
            return warmup
        if global_step < 0:
            raise ValueError("global_step must be nonnegative when provided.")
        progress = min(
            float(global_step) / self.config.density_schedule_warmup_steps, 1.0
        )
        return warmup + progress * (target - warmup)

    @staticmethod
    def _build_diagnostics(
        source_xy_coarse: Tensor,
        selected_scores: Tensor,
        selected_entropy: Tensor,
        selected_valid: Tensor,
        selected_region_ids: Tensor,
        region_density: Tensor,
        region_valid_count: Tensor,
        region_mean_score: Tensor,
        region_max_score: Tensor,
        region_quota: Tensor,
        region_max_quota: Tensor,
        effective_density_power: Tensor,
        effective_threshold: Tensor,
        coverage_budget_clipped: Tensor,
        actual_budget: Tensor,
        high_quality_count: Tensor,
        mean_valid_score: Tensor,
        source_height: int,
        source_width: int,
        min_geometry_queries: int,
        grid_rows: int,
        grid_cols: int,
    ) -> Dict[str, Tensor]:
        batch = selected_valid.shape[0]
        device = selected_valid.device
        selected_count = selected_valid.sum(dim=1)
        valid_region_count = (region_valid_count > 0).sum(dim=1)
        covered_region_count = (region_quota > 0).sum(dim=1)
        coverage_ratio = (
            covered_region_count.float()
            / valid_region_count.clamp_min(1).float()
        )
        max_coverable_region_count = torch.minimum(
            valid_region_count, actual_budget
        )
        achievable_coverage_ratio = torch.where(
            max_coverable_region_count > 0,
            covered_region_count.float()
            / max_coverable_region_count.clamp_min(1).float(),
            torch.zeros_like(covered_region_count, dtype=torch.float32),
        )
        low_quality_sample = selected_count < min_geometry_queries
        budget_utilization = torch.where(
            actual_budget > 0,
            selected_count.float() / actual_budget.clamp_min(1).float(),
            torch.zeros_like(selected_count, dtype=torch.float32),
        )
        gini = torch.zeros(batch, device=device)
        mean_distance_cells = torch.zeros(batch, device=device)
        median_distance_cells = torch.zeros(batch, device=device)
        max_region_ratio = torch.zeros(batch, device=device)
        mean_score = torch.zeros(batch, device=device)
        mean_entropy = torch.zeros(batch, device=device)
        for batch_index in range(batch):
            valid = selected_valid[batch_index]
            count = int(valid.sum().item())
            if count == 0:
                continue
            selected_counts = region_quota[
                batch_index, region_valid_count[batch_index] > 0
            ].float()
            if selected_counts.numel() > 0 and selected_counts.sum() > 0:
                pairwise_count_difference = (
                    selected_counts[:, None] - selected_counts[None, :]
                ).abs().sum()
                gini[batch_index] = pairwise_count_difference / (
                    2.0 * selected_counts.numel() * selected_counts.sum()
                )
            points = source_xy_coarse[batch_index, valid]
            if points.shape[0] > 1:
                distances = torch.pdist(points)
                mean_distance_cells[batch_index] = distances.mean()
                median_distance_cells[batch_index] = distances.median()
            max_region_ratio[batch_index] = (
                region_quota[batch_index].max().float() / max(count, 1)
            )
            mean_score[batch_index] = selected_scores[batch_index, valid].mean()
            mean_entropy[batch_index] = selected_entropy[batch_index, valid].mean()
        source_diagonal = max(
            math.hypot(source_height - 1, source_width - 1), 1.0
        )
        mean_distance = mean_distance_cells / source_diagonal
        median_distance = median_distance_cells / source_diagonal
        return {
            "selected_count": selected_count,
            "actual_budget": actual_budget,
            "high_quality_count": high_quality_count,
            "mean_valid_score": mean_valid_score,
            "budget_utilization": budget_utilization,
            "low_quality_sample": low_quality_sample,
            "geometry_ready": ~low_quality_sample,
            "valid_region_count": valid_region_count,
            "covered_region_count": covered_region_count,
            "region_coverage_ratio": coverage_ratio,
            "max_coverable_region_count": max_coverable_region_count,
            "achievable_region_coverage_ratio": achievable_coverage_ratio,
            "selected_region_gini": gini,
            "mean_selected_pairwise_distance": mean_distance,
            "median_selected_pairwise_distance": median_distance,
            "mean_selected_pairwise_distance_cells": mean_distance_cells,
            "median_selected_pairwise_distance_cells": median_distance_cells,
            "max_region_selection_ratio": max_region_ratio,
            "mean_selected_score": mean_score,
            "mean_selected_entropy": mean_entropy,
            "region_density": region_density,
            "region_valid_count": region_valid_count,
            "region_mean_score": region_mean_score,
            "region_max_score": region_max_score,
            "region_quota": region_quota,
            "region_max_quota": region_max_quota,
            "effective_density_power": effective_density_power,
            "effective_score_threshold": effective_threshold,
            "coverage_budget_clipped": coverage_budget_clipped,
            "grid_rows": torch.full(
                (batch,), grid_rows, device=device, dtype=torch.long
            ),
            "grid_cols": torch.full(
                (batch,), grid_cols, device=device, dtype=torch.long
            ),
            "selected_region_ids": selected_region_ids,
        }

    @staticmethod
    def _gather(values: Tensor, query_indices: Tensor) -> Tensor:
        if values.ndim == 2:
            return torch.gather(values, dim=1, index=query_indices)
        if values.ndim == 3:
            expanded = query_indices[..., None].expand(-1, -1, values.shape[-1])
            return torch.gather(values, dim=1, index=expanded)
        raise ValueError("Only [B, S] and [B, S, K] tensors can be gathered.")

    @staticmethod
    def _prepare_mask(
        mask: Optional[Tensor], batch: int, count: int, device: torch.device
    ) -> Tensor:
        if mask is None:
            return torch.ones(batch, count, device=device, dtype=torch.bool)
        if mask.shape != (batch, count) or mask.dtype != torch.bool:
            raise ValueError(f"Mask must be bool {(batch, count)}, got {tuple(mask.shape)}.")
        return mask.to(device=device)


# New descriptive name while preserving the original class used by matcher.py.
GRMCandidateSampler = GRMTopKCandidateSampler
