"""End-to-end standalone Geometry-Role Multi-Hypothesis Matcher."""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone import build_grm_backbone
from .candidate_reranker import (
    GRMCandidateRerankContext,
    GRMCandidateRerankPlugin,
    GRMCandidateRerankRouter,
)
from .candidate_sampler import GRMSampledCandidates, GRMTopKCandidateSampler
from .coarse_transformer import GRMCoarseContextTransformer
from .config import GRMMatcherConfig
from .coordinates import flat_indices_to_image_xy, validate_image_size
from .correlation import GRMExplicitCorrelation
from .cycle_consistency import compute_cycle_consistency
from .cycle_consistency import GRMCycleConsistencyOutput
from .context_bridge import GRMContextBridge
from .feature_pyramid import GRMResidualFPN
from .fine_refiner import GRMFineRefiner
from .geometry_graph import (
    GRMCandidateTokenEncoder,
    GRMGeometryGraph,
    GRMGeometryTokenInputs,
)
from .mid_refiner import GRMMidRefiner
from .outputs import GRMMatcherAuxiliary, GRMMatcherOutput
from .role_head import GRMCandidateRerankHead, GRMGeometryRoleHead
from .inlier_head import GRMInlierHead
from .seed_geometry import GRMSeedHomographyEstimator


class GRMMatcher(nn.Module):
    """Independent matcher: two images in, correspondences and reliability out."""

    def __init__(self, config: GRMMatcherConfig) -> None:
        super().__init__()
        self.config = config
        if config.architecture_version == "v3":
            from .architecture_v3 import initialize_v3
            initialize_v3(self, config)
            return
        self.backbone = build_grm_backbone(config)
        self.fpn = GRMResidualFPN(config)
        self.coarse_transformer = GRMCoarseContextTransformer(config)
        self.correlation = GRMExplicitCorrelation(config)
        self.candidate_sampler = GRMTopKCandidateSampler(config)
        self.mid_refiner = GRMMidRefiner(config)
        self.fine_refiner = GRMFineRefiner(config)
        self.seed_geometry = GRMSeedHomographyEstimator(config)
        self.token_encoder = GRMCandidateTokenEncoder(config)
        self.geometry_graph = GRMGeometryGraph(config)
        self.rerank_head = GRMCandidateRerankHead(config)
        if config.classification_mode == "inlier":
            self.inlier_head = GRMInlierHead(config)
        elif config.classification_mode == "role":
            self.role_head = GRMGeometryRoleHead(config)
        self.context_bridge = GRMContextBridge(config) if config.architecture_version == "v2" else None
        if config.architecture_version == "v2":
            self.geometry_gain = nn.Parameter(torch.tensor(-2.0))
        # Parameter-free strategy router.  Legacy trainable components remain
        # registered under their original names for strict checkpoint loading.
        self.candidate_reranker = GRMCandidateRerankRouter(config.probability_eps)

    def forward(
        self,
        image_a: Tensor,
        image_b: Tensor,
        return_auxiliary: bool = False,
        *,
        detach_geometry_inputs: bool = False,
        candidate_rerank_mode: str | None = None,
        valid_a: Tensor | None = None,
        valid_b: Tensor | None = None,
    ) -> Union[GRMMatcherOutput, tuple[GRMMatcherOutput, GRMMatcherAuxiliary]]:
        self._validate_images(image_a, image_b)
        if self.config.architecture_version == "v3":
            if candidate_rerank_mode is not None:
                raise ValueError("V3 has no candidate reranker")
            from .architecture_v3 import forward_v3
            return forward_v3(self, image_a, image_b, return_auxiliary, valid_a, valid_b)
        batch, _, height, width = image_a.shape

        features_a, features_b, tokens_a, tokens_b = self.encode_pair(image_a, image_b, valid_a, valid_b)
        mask_a, mask_b = features_a["valid_coarse"].flatten(1), features_b["valid_coarse"].flatten(1)
        correlation = self.correlation(tokens_a, tokens_b, mask_a, mask_b)
        coarse_height, coarse_width = features_a["F_coarse"].shape[-2:]
        sampled_forward = self.candidate_sampler(
            correlation.probability,
            correlation.matchability_a,
            coarse_height,
            coarse_width,
            source_valid_mask=mask_a, target_valid_mask=mask_b,
        )

        candidate_state = self._refine_all_candidates(
            sampled_forward,
            features_a,
            features_b,
            coarse_width,
        )
        (
            flat_src_xy,
            flat_initial_dst_xy,
            flat_valid,
            mid_forward,
            fine_forward,
        ) = candidate_state

        reverse_probability = correlation.probability.transpose(1, 2)
        if self.config.architecture_version == "v2":
            cycle = self.direct_cycle(features_a, features_b, correlation.probability,
                                      flat_src_xy, fine_forward.dst_xy, flat_valid)
        else:
            cycle = self._legacy_cycle(reverse_probability, correlation.matchability_b,
                features_a, features_b, coarse_height, coarse_width,
                flat_src_xy, fine_forward.dst_xy, flat_valid)

        batch_size, query_count, topk = sampled_forward.candidate_indices.shape
        flat_coarse_probability = sampled_forward.candidate_global_probs.reshape(
            batch_size, query_count * topk
        )
        local_strength = torch.sqrt(
            (mid_forward.peak_probability * fine_forward.peak_probability).clamp_min(0.0)
        )
        flat_p_match = (
            flat_coarse_probability
            * local_strength
            * (0.5 + 0.5 * cycle.cycle_score)
        ).clamp(0.0, 1.0)

        reverse_all_top1 = self.candidate_sampler.topk_candidates(
            reverse_probability
        ).indices[..., 0]
        flat_target_indices = sampled_forward.candidate_indices.reshape(
            batch_size, query_count * topk
        )
        reverse_source_for_target = torch.gather(
            reverse_all_top1, 1, flat_target_indices
        )
        flat_query_indices = sampled_forward.query_indices[..., None].expand(
            -1, -1, topk
        ).reshape(batch_size, query_count * topk)
        mutual_top1 = reverse_source_for_target == flat_query_indices
        if self.config.architecture_version == "v2":
            forward_top1 = correlation.probability.argmax(dim=-1)
            mutual_top1 = mutual_top1 & (flat_target_indices == torch.gather(forward_top1, 1, flat_query_indices))

        flat_entropy = sampled_forward.candidate_entropy[..., None].expand(
            -1, -1, topk
        ).reshape(batch_size, query_count * topk)
        seed = self.seed_geometry(
            flat_src_xy,
            fine_forward.dst_xy,
            flat_p_match,
            flat_entropy,
            cycle.cycle_error,
            flat_valid,
            mutual_top1,
        )

        matchability_a = torch.gather(
            correlation.matchability_a, 1, sampled_forward.query_indices
        )[..., None].expand(-1, -1, topk).reshape(batch_size, query_count * topk)
        matchability_b = torch.gather(
            correlation.matchability_b, 1, flat_target_indices
        )
        flat_margin = sampled_forward.top1_top2_margin[..., None].expand(
            -1, -1, topk
        ).reshape(batch_size, query_count * topk)
        seed_valid_per_candidate = seed.homography_valid[:, None].expand_as(flat_valid)
        def geometry_input(value: Tensor) -> Tensor:
            return value.detach() if detach_geometry_inputs else value

        token = self.token_encoder(
            GRMGeometryTokenInputs(
                feature_a=geometry_input(fine_forward.source_descriptor),
                feature_b=geometry_input(fine_forward.target_descriptor),
                coarse_probability=geometry_input(flat_coarse_probability),
                mid_probability=geometry_input(mid_forward.peak_probability),
                fine_probability=geometry_input(fine_forward.peak_probability),
                candidate_entropy=geometry_input(flat_entropy),
                top1_top2_margin=geometry_input(flat_margin),
                matchability_a=geometry_input(matchability_a),
                matchability_b=geometry_input(matchability_b),
                src_xy=geometry_input(flat_src_xy),
                dst_xy=geometry_input(fine_forward.dst_xy),
                seed_residual_xy=geometry_input(seed.residual_xy),
                symmetric_transfer_error=geometry_input(
                    seed.symmetric_transfer_error
                ),
                cycle_error=geometry_input(cycle.cycle_error),
                seed_valid=seed_valid_per_candidate,
                fine_covariance=geometry_input(fine_forward.covariance_image),
                valid_mask=flat_valid,
                image_height=height,
                image_width=width,
            )
        )
        token_by_query = token.reshape(
            batch_size, query_count, topk, self.config.graph_hidden_dim
        )
        candidate_valid = flat_valid.reshape(batch_size, query_count, topk)
        p_match_by_query = flat_p_match.reshape(batch_size, query_count, topk)
        rerank_mode = (
            self.config.candidate_rerank_mode
            if candidate_rerank_mode is None
            else candidate_rerank_mode
        )
        rerank = self.candidate_reranker(
            rerank_mode,
            GRMCandidateRerankContext(
                candidate_tokens=token_by_query,
                candidate_prior=sampled_forward.candidate_probs,
                candidate_reliability=p_match_by_query,
                candidate_valid_mask=candidate_valid,
                flat_src_xy=geometry_input(flat_src_xy),
                flat_dst_xy=geometry_input(fine_forward.dst_xy),
                flat_seed_error=geometry_input(seed.symmetric_transfer_error),
                image_height=height,
                image_width=width,
            ),
            geometry_graph=self.geometry_graph,
            rerank_head=self.rerank_head,
        )
        if self.config.architecture_version == "v2" and rerank.used_geometry_graph and rerank.used_learned_rerank:
            # A bad/missing H cannot overwrite visual ranking. Bound the
            # log-score residual even when a geometrically plausible H exists.
            quality = self.geometry_quality(seed, flat_src_xy).detach()
            gain = quality[:, None, None] * self.geometry_gain.sigmoid()
            visual = torch.log(p_match_by_query.clamp_min(self.config.probability_eps))
            delta = self.rerank_head.score(rerank.candidate_tokens).squeeze(-1)
            logits = visual + 2.0 * gain * torch.tanh(delta)
            probabilities = logits.masked_fill(~candidate_valid, torch.finfo(logits.dtype).min).softmax(-1)
            probabilities = probabilities * candidate_valid.float()
            probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(self.config.probability_eps)
            rerank.logits, rerank.probabilities = logits, probabilities
            rerank.selected_indices = probabilities.argmax(-1)
            rerank.selected_probability_multiplier = torch.gather(probabilities, 2, rerank.selected_indices[..., None]).squeeze(-1)
        selected = rerank.selected_indices
        selected_valid = torch.gather(candidate_valid, 2, selected[..., None]).squeeze(-1)
        selected_token = self._gather_k(rerank.candidate_tokens, selected)
        inlier_logits = None
        visual_score = (self._gather_k(p_match_by_query, selected)
                        * rerank.selected_probability_multiplier).clamp(0.0, 1.0)
        if self.config.classification_mode == "inlier":
            inlier_logits = self.inlier_head(selected_token)
            match_score = inlier_logits.float().sigmoid() * selected_valid.float()
            # Legacy output slots are compatibility aliases, NOT geometry roles.
            p_global, p_local, p_false = match_score, torch.zeros_like(match_score), 1.0-match_score
            role_logits = selected_token.new_zeros((*selected_valid.shape, 3))
        elif self.config.classification_mode == "role":
            roles = self.role_head(selected_token, selected_valid)
            p_global, p_local, p_false = roles.probabilities.unbind(-1)
            role_logits = roles.logits
            match_score = visual_score
        else:
            match_score = visual_score * selected_valid.float()
            # Compatibility placeholders only. No class prediction is made.
            p_global = selected_valid.float()
            p_local = torch.zeros_like(match_score)
            p_false = 1.0-p_global
            role_logits = selected_token.new_zeros((*selected_valid.shape, 3))

        src_by_query = flat_src_xy.reshape(batch_size, query_count, topk, 2)
        dst_by_query = fine_forward.dst_xy.reshape(batch_size, query_count, topk, 2)
        cycle_by_query = cycle.cycle_error.reshape(batch_size, query_count, topk)
        covariance_by_query = fine_forward.covariance_image.reshape(
            batch_size, query_count, topk, 2, 2
        )
        output = GRMMatcherOutput(
            src_xy=self._gather_k(src_by_query, selected),
            dst_xy=self._gather_k(dst_by_query, selected),
            p_match=match_score,
            p_global=p_global,
            p_local=p_local,
            p_false=p_false,
            candidate_entropy=sampled_forward.candidate_entropy,
            cycle_error=self._gather_k(cycle_by_query, selected),
            fine_uncertainty=self._gather_k(covariance_by_query, selected),
            overlap_src=correlation.matchability_a.reshape(
                batch_size, 1, coarse_height, coarse_width
            ),
            overlap_dst=correlation.matchability_b.reshape(
                batch_size, 1, coarse_height, coarse_width
            ),
            candidate_indices=sampled_forward.candidate_indices,
            candidate_probs=rerank.probabilities,
            valid_mask=selected_valid,
        )
        if not return_auxiliary:
            return output
        auxiliary = GRMMatcherAuxiliary(
            coarse_similarity=correlation.similarity,
            coarse_probability=correlation.probability,
            matchability_a=correlation.matchability_a,
            matchability_b=correlation.matchability_b,
            coarse_height=coarse_height,
            coarse_width=coarse_width,
            query_indices=sampled_forward.query_indices,
            candidate_indices=sampled_forward.candidate_indices,
            candidate_prior=sampled_forward.candidate_probs,
            candidate_valid_mask=candidate_valid,
            candidate_src_xy=src_by_query,
            candidate_initial_dst_xy=flat_initial_dst_xy.reshape(
                batch_size, query_count, topk, 2
            ),
            mid_dst_xy=mid_forward.dst_xy.reshape(
                batch_size, query_count, topk, 2
            ),
            mid_probability=mid_forward.local_probability.reshape(
                batch_size,
                query_count,
                topk,
                self.config.mid_window_size,
                self.config.mid_window_size,
            ),
            fine_initial_dst_xy=mid_forward.dst_xy.reshape(
                batch_size, query_count, topk, 2
            ),
            fine_dst_xy=dst_by_query,
            fine_probability=fine_forward.local_probability.reshape(
                batch_size,
                query_count,
                topk,
                self.config.fine_window_size,
                self.config.fine_window_size,
            ),
            rerank_logits=rerank.logits,
            rerank_probabilities=rerank.probabilities,
            selected_indices=selected,
            role_logits=role_logits,
            inlier_logits=inlier_logits,
            selected_graph_tokens=selected_token,
            feature_a_fine_map=features_a["F_fine"],
            feature_b_fine_map=features_b["F_fine"],
            candidate_rerank_mode=rerank.mode,
            candidate_rerank_used_geometry=rerank.used_geometry_graph,
            candidate_rerank_used_learned_head=rerank.used_learned_rerank,
        )
        return output, auxiliary

    def encode_pair(self, image_a, image_b, valid_a=None, valid_b=None):
        """One feature contract shared by inference and supervised refinement."""
        self._validate_images(image_a, image_b)
        batch = image_a.shape[0]
        pyramid = self.fpn(self.backbone(torch.cat((image_a, image_b), dim=0)))
        a = {k:v[:batch] for k,v in pyramid.items()}
        b = {k:v[batch:] for k,v in pyramid.items()}
        for features, mask in ((a,valid_a),(b,valid_b)):
            if mask is None:
                mask = torch.ones_like(image_a[:, :1], dtype=torch.bool)
            if mask.ndim == 3:
                mask = mask[:, None]
            if mask.shape != image_a[:, :1].shape or mask.dtype != torch.bool:
                raise ValueError("valid image masks must be bool [B,1,H,W] or [B,H,W]")
            mask = mask.to(image_a.device)
            for level in ("coarse", "mid", "fine"):
                # V3 aligns masks with half-pixel feature centers. Retain the
                # historical mask convention for unchanged V1/V2 checkpoints.
                mode = "nearest-exact" if self.config.architecture_version == "v3" else "nearest"
                features[f"valid_{level}"] = F.interpolate(mask.float(), size=features[f"F_{level}"].shape[-2:], mode=mode)[:,0].bool()
            features["valid_image"] = mask
        ta, tb = self.coarse_transformer(a["F_coarse"],b["F_coarse"],a["valid_coarse"],b["valid_coarse"])
        if self.context_bridge is not None:
            a, b = self.context_bridge(a, ta), self.context_bridge(b, tb)
        return a, b, ta, tb

    @staticmethod
    def points_valid(points, features):
        mask = features.get("valid_image")
        if mask is None:
            return torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        h, w = mask.shape[-2:]
        grid = 2.0 * (points + .5) / points.new_tensor([w,h]) - 1.0
        return F.grid_sample(mask.float(), grid[:, :, None], mode="nearest", align_corners=False)[:,0,:,0] > .5

    def _legacy_cycle(self, probability, matchability, features_a, features_b,
                      height, width, source, destination, valid):
        sampled = self.candidate_sampler(probability, matchability, height, width,
            source_valid_mask=features_b["valid_coarse"].flatten(1),
            target_valid_mask=features_a["valid_coarse"].flatten(1))
        src = flat_indices_to_image_xy(sampled.query_indices, width, self.config.coarse_stride)
        dst = flat_indices_to_image_xy(sampled.candidate_indices[..., 0], width, self.config.coarse_stride)
        backward_valid = sampled.valid_mask & (sampled.candidate_global_probs[..., 0] > self.config.probability_eps)
        _, fine = self.refine_points(features_b, features_a, src, dst, backward_valid)
        return compute_cycle_consistency(source, destination, src, fine.dst_xy,
            sigma=self.config.cycle_sigma, forward_valid_mask=valid,
            backward_valid_mask=backward_valid & fine.valid_mask)

    def direct_cycle(self, features_a, features_b, probability, source, destination, valid):
        """Reverse-query each actual forward endpoint; no sparse-anchor hop."""
        h, w = features_b["F_coarse"].shape[-2:]
        cells = torch.round((destination.detach()+.5)/self.config.coarse_stride-.5).long()
        inside = (cells[...,0]>=0)&(cells[...,0]<w)&(cells[...,1]>=0)&(cells[...,1]<h)
        indices = cells[...,1].clamp(0,h-1)*w+cells[...,0].clamp(0,w-1)
        reverse_top1 = probability.argmax(dim=1)
        initial = flat_indices_to_image_xy(torch.gather(reverse_top1,1,indices),w,self.config.coarse_stride)
        valid = valid & inside & self.points_valid(destination,features_b)
        mid, fine = self.refine_points(features_b,features_a,destination,initial,valid)
        valid = valid & fine.valid_mask & self.points_valid(fine.dst_xy,features_a)
        error = torch.linalg.vector_norm(source-fine.dst_xy,dim=-1)
        error = torch.where(valid,error,torch.full_like(error,1e6))
        return GRMCycleConsistencyOutput(fine.dst_xy,indices,error,torch.exp(-error/self.config.cycle_sigma)*valid,valid)

    def refine_points(self, features_a, features_b, source, initial, valid):
        valid = valid & self.points_valid(source,features_a)
        mid = self.mid_refiner(features_a["F_mid"],features_b["F_mid"],source,initial,valid,
            source_valid_map=features_a.get("valid_mid"), target_valid_map=features_b.get("valid_mid"))
        fine = self.fine_refiner(features_a["F_fine"],features_b["F_fine"],source,mid.dst_xy,mid.valid_mask,
            source_valid_map=features_a.get("valid_fine"), target_valid_map=features_b.get("valid_fine"))
        fine.valid_mask = fine.valid_mask & self.points_valid(fine.dst_xy,features_b)
        return mid, fine

    @staticmethod
    def geometry_quality(seed, source):
        """Inlier support and 2-D spread within the observed overlap, not frame area."""
        qualities = []
        for i in range(source.shape[0]):
            selected = seed.selected_mask[i]
            inliers = seed.inlier_mask[i]
            # Duplicate hypotheses at one source do not count as independent support.
            all_points = torch.unique(source[i,selected].detach(), dim=0)
            points = torch.unique(source[i,inliers].detach(), dim=0)
            if not bool(seed.homography_valid[i]) or len(points)<4:
                qualities.append(source.new_tensor(0.)); continue
            centered = points-points.mean(0)
            eigen = torch.linalg.eigvalsh(centered.T@centered/max(len(points)-1,1)).clamp_min(0.)
            spread = (eigen[0]/eigen[-1].clamp_min(1e-6)/.05).clamp(0.,1.)
            support = len(points)/max(len(all_points),1)
            qualities.append(spread*support)
        return torch.stack(qualities)

    @property
    def available_candidate_rerank_modes(self) -> tuple[str, ...]:
        """Return registered runtime candidate-selection backends."""

        return () if self.config.architecture_version == "v3" else self.candidate_reranker.available_modes

    def register_candidate_reranker(
        self,
        plugin: GRMCandidateRerankPlugin,
        *,
        replace: bool = False,
    ) -> None:
        """Register a parameter-free experimental reranker.

        Learned parameters must remain explicit ``GRMMatcher`` submodules if
        they need optimizer/state-dict support.  This restriction is what keeps
        historical checkpoints strictly compatible.
        """

        self.candidate_reranker.register(plugin, replace=replace)

    def _refine_all_candidates(
        self,
        sampled: GRMSampledCandidates,
        features_a: dict[str, Tensor],
        features_b: dict[str, Tensor],
        coarse_width: int,
    ) -> tuple[Tensor, Tensor, Tensor, object, object]:
        batch, queries, topk = sampled.candidate_indices.shape
        source_by_query = flat_indices_to_image_xy(
            sampled.query_indices,
            coarse_width,
            self.config.coarse_stride,
        )
        source = source_by_query[..., None, :].expand(-1, -1, topk, -1).reshape(
            batch, queries * topk, 2
        )
        initial_target = flat_indices_to_image_xy(
            sampled.candidate_indices,
            coarse_width,
            self.config.coarse_stride,
        ).reshape(batch, queries * topk, 2)
        valid = (
            sampled.valid_mask[..., None]
            & (sampled.candidate_global_probs > self.config.probability_eps)
        ).reshape(batch, queries * topk)
        mid, fine = self.refine_points(features_a,features_b,source,initial_target,valid)
        valid = valid & fine.valid_mask
        return source, initial_target, valid, mid, fine

    def _validate_images(self, image_a: Tensor, image_b: Tensor) -> None:
        if image_a.shape != image_b.shape:
            raise ValueError("image_a and image_b must have identical [B, 3, H, W] shapes.")
        if image_a.ndim != 4 or image_a.shape[1] != 3:
            raise ValueError(f"Expected [B, 3, H, W], got {tuple(image_a.shape)}.")
        if not image_a.is_floating_point() or not image_b.is_floating_point():
            raise TypeError("Matcher images must use a floating-point dtype.")
        validate_image_size(
            int(image_a.shape[-2]),
            int(image_a.shape[-1]),
            self.config.coarse_stride,
        )
        if self.config.input_range == "zero_one":
            if torch.any(image_a < 0.0) or torch.any(image_a > 1.0):
                raise ValueError("image_a must lie in [0, 1] for input_range='zero_one'.")
            if torch.any(image_b < 0.0) or torch.any(image_b > 1.0):
                raise ValueError("image_b must lie in [0, 1] for input_range='zero_one'.")

    @staticmethod
    def _gather_k(values: Tensor, selected: Tensor) -> Tensor:
        index = selected
        while index.ndim < values.ndim:
            index = index.unsqueeze(-1)
        expansion = list(values.shape)
        expansion[2] = 1
        index = index.expand(*expansion)
        return torch.gather(values, dim=2, index=index).squeeze(2)
