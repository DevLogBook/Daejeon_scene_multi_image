"""Configuration for Geometry-Role Multi-Hypothesis Matcher."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Literal, Mapping, Optional, Tuple


GRM_BUILTIN_CANDIDATE_RERANK_MODES = (
    "geometry_graph",
    "graph_reliability",
    "token_mlp",
    "reliability",
    "coarse_prior",
)

GRMCandidateRerankMode = Literal[
    "geometry_graph",
    "graph_reliability",
    "token_mlp",
    "reliability",
    "coarse_prior",
]


@dataclass
class GRMCandidateSamplerConfig:
    """Configuration for adaptive reliability-density candidate sampling.

    Region dimensions are expressed in coarse feature cells.  ``topk_candidates``
    and ``probability_eps`` are included so this config can also instantiate the
    sampler independently of :class:`GRMMatcherConfig`.
    """

    max_source_queries: int = 256
    region_height_cells: int = 4
    region_width_cells: int = 4

    min_quota_per_valid_region: int = 1

    dynamic_max_quota: bool = True
    max_quota_per_region: int = 8
    max_quota_multiplier: float = 2.0
    max_quota_lower_bound: int = 4
    max_quota_upper_bound: int = 16

    density_power: float = 0.5
    density_power_warmup: float = 0.3
    density_power_max: float = 0.75

    use_density_schedule: bool = False
    density_schedule_warmup_steps: int = 5_000

    use_quantile_threshold: bool = True
    score_quantile: float = 0.50
    fixed_score_threshold: float = 0.01
    allow_underfilled_budget: bool = True
    minimum_query_count: int = 16

    use_dynamic_query_budget: bool = True
    dynamic_budget_target_mean_score: float = 0.05
    min_geometry_queries: int = 16

    entropy_weight: float = 0.5
    margin_weight: float = 0.5

    use_local_nms: bool = True
    local_nms_radius: float = 1.5
    use_dynamic_nms_radius: bool = True
    min_nms_radius: float = 1.0
    max_nms_radius: float = 3.0

    coverage_weight: float = 0.25
    density_eps: float = 1.0e-6

    use_adaptive_density_sampling: bool = True

    # Opt-in inference experiment; legacy checkpoints retain their sampler.
    sampling_mode: Literal["adaptive_density", "overlap_aware"] = "adaptive_density"
    support_min_matchability: float = 0.2
    support_relative_score: float = 0.1

    topk_candidates: int = 4
    candidate_peak_radius: int = 0
    probability_eps: float = 1.0e-8

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.candidate_peak_radius < 0:
            raise ValueError("candidate_peak_radius must be nonnegative")
        if self.sampling_mode not in ("adaptive_density", "overlap_aware"):
            raise ValueError("Unknown sampling_mode")
        for name in ("support_min_matchability", "support_relative_score"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        positive_ints = {
            "max_source_queries": self.max_source_queries,
            "region_height_cells": self.region_height_cells,
            "region_width_cells": self.region_width_cells,
            "max_quota_per_region": self.max_quota_per_region,
            "max_quota_lower_bound": self.max_quota_lower_bound,
            "max_quota_upper_bound": self.max_quota_upper_bound,
            "density_schedule_warmup_steps": self.density_schedule_warmup_steps,
            "topk_candidates": self.topk_candidates,
            "minimum_query_count": self.minimum_query_count,
            "min_geometry_queries": self.min_geometry_queries,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.min_quota_per_valid_region < 0:
            raise ValueError("min_quota_per_valid_region must be nonnegative.")
        if self.max_quota_lower_bound > self.max_quota_upper_bound:
            raise ValueError("max_quota_lower_bound must not exceed max_quota_upper_bound.")
        if self.max_quota_upper_bound < self.min_quota_per_valid_region:
            raise ValueError(
                "max_quota_upper_bound must be >= min_quota_per_valid_region."
            )
        if self.max_quota_per_region < self.min_quota_per_valid_region:
            raise ValueError(
                "max_quota_per_region must be >= min_quota_per_valid_region."
            )
        if self.max_quota_multiplier <= 0.0:
            raise ValueError("max_quota_multiplier must be positive.")
        if not 0.0 <= self.score_quantile <= 1.0:
            raise ValueError("score_quantile must lie in [0, 1].")
        if self.fixed_score_threshold < 0.0:
            raise ValueError("fixed_score_threshold must be nonnegative.")
        if self.dynamic_budget_target_mean_score <= 0.0:
            raise ValueError("dynamic_budget_target_mean_score must be positive.")
        if self.entropy_weight < 0.0 or self.margin_weight < 0.0:
            raise ValueError("entropy_weight and margin_weight must be nonnegative.")
        if self.entropy_weight + self.margin_weight <= 0.0:
            raise ValueError("entropy_weight + margin_weight must be positive.")
        if self.density_power < 0.0 or self.density_power_warmup < 0.0:
            raise ValueError("Density powers must be nonnegative.")
        if self.density_power_max < 0.0:
            raise ValueError("density_power_max must be nonnegative.")
        if self.density_power_warmup > self.density_power_max:
            raise ValueError("density_power_warmup must not exceed density_power_max.")
        if self.use_density_schedule and self.density_power_warmup > min(
            self.density_power, self.density_power_max
        ):
            raise ValueError(
                "density_power_warmup must not exceed the effective target density power."
            )
        if self.local_nms_radius <= 0.0:
            raise ValueError("local_nms_radius must be positive.")
        if self.min_nms_radius <= 0.0 or self.max_nms_radius <= 0.0:
            raise ValueError("Dynamic NMS radii must be positive.")
        if self.min_nms_radius > self.max_nms_radius:
            raise ValueError("min_nms_radius must not exceed max_nms_radius.")
        if self.coverage_weight < 0.0:
            raise ValueError("coverage_weight must be nonnegative.")
        if self.density_eps <= 0.0 or self.probability_eps <= 0.0:
            raise ValueError("density_eps and probability_eps must be positive.")


@dataclass
class GRMMatcherConfig:
    """All architecture and matcher-loss hyperparameters.

    Public coordinates are always ``(x, y)`` in original-image pixels.  With
    the default ``input_range='zero_one'``, callers must provide float images
    in ``[0, 1]``.  ``imagenet_normalized`` means normalization with
    ``normalization_mean`` and ``normalization_std`` has already been applied.
    Input height and width may differ from these defaults at runtime, but every
    spatial dimension must be divisible by ``coarse_stride``.
    """

    input_height: int = 512
    input_width: int = 512
    input_range: Literal["zero_one", "imagenet_normalized"] = "zero_one"
    normalization_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalization_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    coarse_stride: int = 16
    mid_stride: int = 8
    fine_stride: int = 4

    backbone_name: Literal["conv", "hybrid_mobilevit"] = "conv"
    backbone_stem_dim: int = 32
    backbone_stage_blocks: Tuple[int, int, int] = (2, 2, 2)
    mobilevit_depth: int = 2
    mobilevit_ffn_multiplier: float = 2.0
    coarse_dim: int = 128
    mid_dim: int = 128
    fine_dim: int = 96
    fpn_fine_alpha_init: float = 0.25

    transformer_layers: int = 4
    transformer_heads: int = 4
    transformer_ffn_dim: int = 256
    transformer_dropout: float = 0.0
    positional_temperature: float = 10_000.0

    topk_candidates: int = 4
    candidate_peak_radius: int = 0
    # Opt-in v2 modules: absent fields in historical checkpoints mean v1.
    architecture_version: Literal["v1", "v2", "v3"] = "v1"
    v3_confidence_head: bool = True
    v3_spatial_bonus: float = 0.2
    v3_spatial_sigma_cells: float = 2.0
    v3_mid_hidden: int = 64
    v3_fine_hidden: int = 32
    v3_confidence_weight: float = 0.1
    # Revision 0 preserves the first V3 checkpoint schema and selection policy.
    v3_revision: int = 0
    v3_min_matchability: float = 0.2
    v3_relative_reliability: float = 0.05
    classification_mode: Literal["role", "inlier", "none"] = "role"
    inlier_loss_weight: float = 0.5
    max_source_queries: int = 256
    # Legacy fixed-grid controls retained for checkpoint/API compatibility.
    sampling_grid_rows: int = 8
    sampling_grid_cols: int = 8

    region_height_cells: int = 4
    region_width_cells: int = 4
    min_quota_per_valid_region: int = 1
    dynamic_max_quota: bool = True
    max_quota_per_region: int = 8
    max_quota_multiplier: float = 2.0
    max_quota_lower_bound: int = 4
    max_quota_upper_bound: int = 16
    density_power: float = 0.5
    density_power_warmup: float = 0.3
    density_power_max: float = 0.75
    use_density_schedule: bool = False
    density_schedule_warmup_steps: int = 5_000
    use_quantile_threshold: bool = True
    score_quantile: float = 0.50
    fixed_score_threshold: float = 0.01
    allow_underfilled_budget: bool = True
    minimum_query_count: int = 16
    use_dynamic_query_budget: bool = True
    dynamic_budget_target_mean_score: float = 0.05
    min_geometry_queries: int = 16
    entropy_weight: float = 0.5
    margin_weight: float = 0.5
    use_local_nms: bool = True
    local_nms_radius: float = 1.5
    use_dynamic_nms_radius: bool = True
    min_nms_radius: float = 1.0
    max_nms_radius: float = 3.0
    coverage_weight: float = 0.25
    density_eps: float = 1.0e-6
    use_adaptive_density_sampling: bool = True

    sampling_mode: Literal["adaptive_density", "overlap_aware"] = "adaptive_density"
    support_min_matchability: float = 0.2
    support_relative_score: float = 0.1

    mid_source_patch_size: int = 3
    mid_window_size: int = 7
    fine_source_patch_size: int = 3
    fine_window_size: int = 5
    local_temperature: float = 0.1

    graph_hidden_dim: int = 128
    graph_heads: int = 4
    graph_layers: int = 2
    graph_neighbors: int = 12
    graph_dropout: float = 0.0
    graph_use_target_neighbors: bool = True
    role_appearance_dim: int = 32
    rerank_prior_weight: float = 1.0
    # ``geometry_graph`` preserves the original checkpoint behavior.  The
    # remaining modes are inference/ablation backends and add no parameters.
    candidate_rerank_mode: GRMCandidateRerankMode = "geometry_graph"

    temperature: float = 0.1
    probability_eps: float = 1.0e-8
    cycle_sigma: float = 4.0

    global_residual_threshold: float = 3.0
    local_residual_threshold: float = 24.0
    # Error to the exact dense correspondence used to decide whether a
    # predicted match is correct.  This is intentionally independent from the
    # residual-to-H thresholds above, which split correct matches into
    # Global/Local geometry roles.
    correspondence_correct_threshold: float = 5.0

    use_seed_homography: bool = True
    detach_seed_homography: bool = True
    seed_min_matches: int = 4
    seed_max_matches: int = 128
    seed_ransac_threshold: float = 3.0
    seed_confidence: float = 0.999
    seed_max_iterations: int = 10_000
    seed_max_condition_number: float = 1.0e8

    inference_min_match_probability: float = 0.0
    inference_max_cycle_error: Optional[float] = None

    coarse_loss_weight: float = 1.0
    mid_loss_weight: float = 1.0
    fine_loss_weight: float = 1.0
    rerank_loss_weight: float = 0.5
    role_loss_weight: float = 0.5
    matchability_loss_weight: float = 0.2
    calibration_loss_weight: float = 0.1
    coordinate_loss_beta: float = 1.0

    # Joint fine-tuning uses source-specific weights so the exact Homography
    # pairs preserve Stage-1/2 matching quality while controlled geometry
    # samples continue to provide Local/False Role supervision.
    joint_homography_coarse_loss_weight: float = 0.5
    joint_homography_mid_loss_weight: float = 0.5
    joint_homography_fine_loss_weight: float = 1.0
    joint_homography_rerank_loss_weight: float = 0.25
    joint_homography_role_loss_weight: float = 0.05
    joint_homography_matchability_loss_weight: float = 0.1
    joint_homography_calibration_loss_weight: float = 0.05

    joint_role_coarse_loss_weight: float = 0.25
    joint_role_mid_loss_weight: float = 0.5
    joint_role_fine_loss_weight: float = 1.0
    joint_role_rerank_loss_weight: float = 0.5
    joint_role_role_loss_weight: float = 0.5
    joint_role_matchability_loss_weight: float = 0.1
    joint_role_calibration_loss_weight: float = 0.1

    use_amp: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` when a configuration is internally invalid."""

        if self.architecture_version not in ("v1", "v2", "v3"):
            raise ValueError("architecture_version must be v1, v2 or v3")
        if self.architecture_version == "v3":
            if self.v3_revision not in (0, 1):
                raise ValueError("v3_revision must be 0 or 1")
            if not 0 <= self.v3_min_matchability < 1 or not 0 <= self.v3_relative_reliability < 1:
                raise ValueError("V3 support thresholds must be in [0,1)")
            if self.topk_candidates != 1 or self.classification_mode != "none":
                raise ValueError("V3 requires topk_candidates=1 and classification_mode=none")
            if not 0 <= self.v3_spatial_bonus <= .3 or self.v3_spatial_sigma_cells <= 0:
                raise ValueError("Invalid V3 spatial sampling parameters")
            if min(self.v3_mid_hidden, self.v3_fine_hidden) < 1 or self.v3_confidence_weight < 0:
                raise ValueError("Invalid V3 refinement/confidence parameters")
        if self.classification_mode not in ("role", "inlier", "none"):
            raise ValueError("classification_mode must be role, inlier or none")
        if self.inlier_loss_weight < 0:
            raise ValueError("inlier_loss_weight must be nonnegative")
        if (self.fine_stride, self.mid_stride, self.coarse_stride) != (4, 8, 16):
            raise ValueError("The current backbone requires fine/mid/coarse strides 4/8/16")

        positive_ints = {
            "input_height": self.input_height,
            "input_width": self.input_width,
            "coarse_stride": self.coarse_stride,
            "mid_stride": self.mid_stride,
            "fine_stride": self.fine_stride,
            "coarse_dim": self.coarse_dim,
            "mid_dim": self.mid_dim,
            "fine_dim": self.fine_dim,
            "backbone_stem_dim": self.backbone_stem_dim,
            "mobilevit_depth": self.mobilevit_depth,
            "transformer_layers": self.transformer_layers,
            "transformer_heads": self.transformer_heads,
            "transformer_ffn_dim": self.transformer_ffn_dim,
            "topk_candidates": self.topk_candidates,
            "max_source_queries": self.max_source_queries,
            "sampling_grid_rows": self.sampling_grid_rows,
            "sampling_grid_cols": self.sampling_grid_cols,
            "graph_hidden_dim": self.graph_hidden_dim,
            "graph_heads": self.graph_heads,
            "graph_layers": self.graph_layers,
            "graph_neighbors": self.graph_neighbors,
            "role_appearance_dim": self.role_appearance_dim,
            "seed_min_matches": self.seed_min_matches,
            "seed_max_matches": self.seed_max_matches,
            "seed_max_iterations": self.seed_max_iterations,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        if self.input_height % self.coarse_stride != 0:
            raise ValueError("input_height must be divisible by coarse_stride.")
        if self.input_width % self.coarse_stride != 0:
            raise ValueError("input_width must be divisible by coarse_stride.")
        if self.coarse_stride != 2 * self.mid_stride:
            raise ValueError("coarse_stride must equal 2 * mid_stride.")
        if self.mid_stride != 2 * self.fine_stride:
            raise ValueError("mid_stride must equal 2 * fine_stride.")
        if self.coarse_dim % self.transformer_heads != 0:
            raise ValueError("coarse_dim must be divisible by transformer_heads.")
        if self.coarse_dim % 4 != 0:
            raise ValueError("coarse_dim must be divisible by four for 2D positional encoding.")
        if self.graph_hidden_dim % self.graph_heads != 0:
            raise ValueError("graph_hidden_dim must be divisible by graph_heads.")
        if self.transformer_layers != 4:
            raise ValueError(
                "GRM v1 uses exactly four alternating self/cross transformer layers."
            )

        odd_sizes = {
            "mid_source_patch_size": self.mid_source_patch_size,
            "mid_window_size": self.mid_window_size,
            "fine_source_patch_size": self.fine_source_patch_size,
            "fine_window_size": self.fine_window_size,
        }
        for name, value in odd_sizes.items():
            if value <= 0 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer, got {value}.")

        if self.input_range not in {"zero_one", "imagenet_normalized"}:
            raise ValueError(f"Unsupported input_range: {self.input_range!r}.")
        if self.backbone_name not in {"conv", "hybrid_mobilevit"}:
            raise ValueError(f"Unsupported backbone_name: {self.backbone_name!r}.")
        if self.candidate_rerank_mode not in GRM_BUILTIN_CANDIDATE_RERANK_MODES:
            raise ValueError(
                "candidate_rerank_mode must be one of "
                f"{GRM_BUILTIN_CANDIDATE_RERANK_MODES}, got "
                f"{self.candidate_rerank_mode!r}."
            )
        if len(self.normalization_mean) != 3 or len(self.normalization_std) != 3:
            raise ValueError("normalization_mean/std must each contain three values.")
        if any(std <= 0.0 for std in self.normalization_std):
            raise ValueError("normalization_std values must be positive.")
        if len(self.backbone_stage_blocks) != 3:
            raise ValueError("backbone_stage_blocks must contain one value per c2/c3/c4 stage.")
        if any(blocks <= 0 for blocks in self.backbone_stage_blocks):
            raise ValueError("backbone_stage_blocks values must be positive.")
        if self.mobilevit_ffn_multiplier <= 0.0:
            raise ValueError("mobilevit_ffn_multiplier must be positive.")

        unit_interval_values = {
            "fpn_fine_alpha_init": self.fpn_fine_alpha_init,
            "seed_confidence": self.seed_confidence,
            "inference_min_match_probability": self.inference_min_match_probability,
        }
        for name, value in unit_interval_values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}.")

        nonnegative_values = {
            "transformer_dropout": self.transformer_dropout,
            "graph_dropout": self.graph_dropout,
            "inference_min_match_probability": self.inference_min_match_probability,
            "coarse_loss_weight": self.coarse_loss_weight,
            "mid_loss_weight": self.mid_loss_weight,
            "fine_loss_weight": self.fine_loss_weight,
            "rerank_loss_weight": self.rerank_loss_weight,
            "role_loss_weight": self.role_loss_weight,
            "matchability_loss_weight": self.matchability_loss_weight,
            "calibration_loss_weight": self.calibration_loss_weight,
            "joint_homography_coarse_loss_weight": self.joint_homography_coarse_loss_weight,
            "joint_homography_mid_loss_weight": self.joint_homography_mid_loss_weight,
            "joint_homography_fine_loss_weight": self.joint_homography_fine_loss_weight,
            "joint_homography_rerank_loss_weight": self.joint_homography_rerank_loss_weight,
            "joint_homography_role_loss_weight": self.joint_homography_role_loss_weight,
            "joint_homography_matchability_loss_weight": self.joint_homography_matchability_loss_weight,
            "joint_homography_calibration_loss_weight": self.joint_homography_calibration_loss_weight,
            "joint_role_coarse_loss_weight": self.joint_role_coarse_loss_weight,
            "joint_role_mid_loss_weight": self.joint_role_mid_loss_weight,
            "joint_role_fine_loss_weight": self.joint_role_fine_loss_weight,
            "joint_role_rerank_loss_weight": self.joint_role_rerank_loss_weight,
            "joint_role_role_loss_weight": self.joint_role_role_loss_weight,
            "joint_role_matchability_loss_weight": self.joint_role_matchability_loss_weight,
            "joint_role_calibration_loss_weight": self.joint_role_calibration_loss_weight,
        }
        for name, value in nonnegative_values.items():
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative, got {value}.")
        if self.transformer_dropout >= 1.0 or self.graph_dropout >= 1.0:
            raise ValueError("Dropout probabilities must be smaller than one.")

        strictly_positive = {
            "temperature": self.temperature,
            "local_temperature": self.local_temperature,
            "positional_temperature": self.positional_temperature,
            "probability_eps": self.probability_eps,
            "cycle_sigma": self.cycle_sigma,
            "global_residual_threshold": self.global_residual_threshold,
            "local_residual_threshold": self.local_residual_threshold,
            "correspondence_correct_threshold": self.correspondence_correct_threshold,
            "seed_ransac_threshold": self.seed_ransac_threshold,
            "seed_max_condition_number": self.seed_max_condition_number,
            "coordinate_loss_beta": self.coordinate_loss_beta,
            "rerank_prior_weight": self.rerank_prior_weight,
        }
        for name, value in strictly_positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.local_residual_threshold <= self.global_residual_threshold:
            raise ValueError(
                "local_residual_threshold must exceed global_residual_threshold."
            )
        if self.seed_max_matches < self.seed_min_matches:
            raise ValueError("seed_max_matches must be >= seed_min_matches.")
        if self.inference_max_cycle_error is not None:
            if self.inference_max_cycle_error <= 0.0:
                raise ValueError("inference_max_cycle_error must be positive or None.")
        # Reuse the standalone sampler config validator so every adaptive
        # hyperparameter has one authoritative validation path.
        self.candidate_sampler_config()

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialization-friendly configuration dictionary."""

        return asdict(self)

    def candidate_sampler_config(self) -> GRMCandidateSamplerConfig:
        """Extract the independently testable adaptive sampler configuration."""

        sampler_names = {field.name for field in fields(GRMCandidateSamplerConfig)}
        values = {name: getattr(self, name) for name in sampler_names}
        return GRMCandidateSamplerConfig(**values)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "GRMMatcherConfig":
        """Construct a config while rejecting misspelled or unknown keys."""

        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown GRMMatcherConfig fields: {unknown}.")
        return cls(**dict(values))
