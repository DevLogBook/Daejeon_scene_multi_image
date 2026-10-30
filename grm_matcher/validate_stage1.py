"""Evaluate only the components optimized during GRM-Matcher Stage 1.

This entry point deliberately stops after explicit coarse correlation,
matchability prediction, and candidate sampling.  Mid/fine refinement,
cycle consistency, seed geometry, reranking, and geometry-role predictions are
neither executed nor reported.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Dict, Iterable, Mapping, Optional, Sequence

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .candidate_sampler import GRMSampledCandidates
from .config import GRMMatcherConfig
from .coordinates import (
    feature_xy_to_image_xy,
    flat_indices_to_feature_xy,
    flat_indices_to_image_xy,
    image_xy_to_feature_xy,
)
from .matcher import GRMMatcher
from .metrics import candidate_recall_at_k, sample_dense_field
from .homography_pair_dataset import (
    GRMHomographyPairDataset,
    GRMResolutionBucketBatchSampler,
    parse_bucket_batch_sizes,
)
from .synthetic_dataset import SyntheticGRMDataConfig, SyntheticGRMDataset
from .visualization import (
    visualize_candidate_entropy,
    visualize_spatial_coverage,
    visualize_topk_candidates,
)


@dataclass
class GRMStage1BatchOutput:
    """Tensors produced by the trainable Stage 1 path."""

    probability: Tensor  # [B, Hc*Wc, Hc*Wc], global correlation probability
    matchability_a: Tensor  # [B, Hc*Wc]
    matchability_b: Tensor  # [B, Hc*Wc]
    sampled: GRMSampledCandidates
    coarse_height: int
    coarse_width: int


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the coarse matching and matchability learned in Stage 1."
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/grm_stage1_eval"))
    parser.add_argument(
        "--dataset_type",
        choices=("homography_pairs", "synthetic"),
        default="homography_pairs",
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--buckets", nargs="*", default=None)
    parser.add_argument(
        "--bucket_batch_size",
        action="append",
        default=None,
        metavar="BUCKET:BATCH",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--samples_per_source", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--visualize_samples", type=int, default=8)
    parser.add_argument("--max_visualized_queries", type=int, default=32)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow_non_stage1_checkpoint",
        action="store_true",
        help="Allow evaluation of a checkpoint whose metadata stage is not stage1.",
    )
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument(
        "--baseline_checkpoint",
        type=Path,
        default=None,
        help="Evaluate a second checkpoint on the exact same deterministic samples.",
    )
    baseline.add_argument(
        "--compare_untrained",
        action="store_true",
        help="Compare against a deterministically initialized untrained model.",
    )
    parser.add_argument("--min_recall_at_4", type=float, default=0.50)
    parser.add_argument("--min_selected_top1_precision", type=float, default=0.30)
    parser.add_argument("--min_matchability_auroc", type=float, default=0.60)
    parser.add_argument(
        "--min_recall_at_8",
        type=float,
        default=None,
        help="Optional full-correlation Recall@8 gate.",
    )
    parser.add_argument(
        "--min_oracle_top4_pck_16px",
        type=float,
        default=None,
        help="Optional gate for the fraction of valid GT points within 16 px of a top-4 candidate.",
    )
    parser.add_argument(
        "--max_oracle_top4_median_pixel_error",
        type=float,
        default=None,
        help="Optional upper bound for top-4 oracle median pixel error.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _seed_everything(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = _load_checkpoint(args.checkpoint)
    _validate_checkpoint_stage(
        checkpoint, args.checkpoint, allow_non_stage1=args.allow_non_stage1_checkpoint
    )
    config = _checkpoint_config(checkpoint)
    primary_model = _build_model(config, checkpoint, device)
    amp_requested = config.use_amp if args.amp is None else args.amp
    amp_enabled = bool(amp_requested and device.type == "cuda")

    bucket_batch_sizes = parse_bucket_batch_sizes(args.bucket_batch_size)
    source_image_count: Optional[int] = None
    if args.dataset_type == "homography_pairs":
        dataset: Dataset[Mapping[str, Tensor]] = GRMHomographyPairDataset(
            args.data_dir, args.split, config, args.buckets
        )
    else:
        image_paths = _find_images(args.data_dir)
        source_image_count = len(image_paths)
        dataset = _make_synthetic_dataset(
            image_paths, config, args.samples_per_source, args.seed
        )
    primary = _evaluate_model(
        primary_model,
        dataset,
        config,
        device,
        amp_enabled=amp_enabled,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        progress=args.progress,
        visualization_dir=args.output_dir / "visualizations",
        visualize_samples=args.visualize_samples,
        max_visualized_queries=args.max_visualized_queries,
        label="checkpoint",
        bucket_batch_sizes=bucket_batch_sizes,
    )

    baseline_summary: Optional[Dict[str, object]] = None
    if args.baseline_checkpoint is not None:
        baseline_checkpoint = _load_checkpoint(args.baseline_checkpoint)
        baseline_config = _checkpoint_config(baseline_checkpoint)
        if baseline_config.to_dict() != config.to_dict():
            raise ValueError("Baseline and evaluated checkpoint configs must match exactly.")
        baseline_model = _build_model(config, baseline_checkpoint, device)
        baseline_summary = _evaluate_model(
            baseline_model,
            dataset,
            config,
            device,
            amp_enabled=amp_enabled,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_batches=args.max_batches,
            progress=args.progress,
            visualization_dir=None,
            visualize_samples=0,
            max_visualized_queries=args.max_visualized_queries,
            label="baseline checkpoint",
            bucket_batch_sizes=bucket_batch_sizes,
        )
    elif args.compare_untrained:
        _seed_everything(args.seed)
        baseline_model = GRMMatcher(config).to(device).eval()
        baseline_summary = _evaluate_model(
            baseline_model,
            dataset,
            config,
            device,
            amp_enabled=amp_enabled,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_batches=args.max_batches,
            progress=args.progress,
            visualization_dir=None,
            visualize_samples=0,
            max_visualized_queries=args.max_visualized_queries,
            label="untrained baseline",
            bucket_batch_sizes=bucket_batch_sizes,
        )

    report: Dict[str, object] = {
        "schema_version": 1,
        "scope": "stage1_coarse_matching_matchability_and_sampling_only",
        "excluded_from_evaluation": [
            "mid_refiner",
            "fine_refiner",
            "cycle_consistency",
            "seed_homography",
            "geometry_graph",
            "rerank_head",
            "role_head",
        ],
        "checkpoint": str(args.checkpoint),
        "checkpoint_stage": checkpoint.get("stage"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "data_dir": str(args.data_dir),
        "dataset_type": args.dataset_type,
        "split": args.split,
        "selected_buckets": args.buckets,
        "dataset_sample_count": len(dataset),
        "source_image_count": source_image_count,
        "amp_enabled": amp_enabled,
        "evaluation": primary,
        "screening": _screen_stage1(primary["metrics"], args),
    }
    if baseline_summary is not None:
        report["baseline"] = baseline_summary
        report["comparison"] = _compare_metric_means(
            primary["metrics"], baseline_summary["metrics"]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "stage1_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Stage 1 report written to {report_path}", flush=True)


def forward_stage1(
    model: GRMMatcher,
    image_a: Tensor,
    image_b: Tensor,
) -> GRMStage1BatchOutput:
    """Run only the modules trained in Stage 1 plus deterministic sampling."""
    model._validate_images(image_a, image_b)
    batch = image_a.shape[0]
    combined = torch.cat((image_a, image_b), dim=0)
    pyramid = model.fpn(model.backbone(combined))
    coarse_a = pyramid["F_coarse"][:batch]
    coarse_b = pyramid["F_coarse"][batch:]
    tokens_a, tokens_b = model.coarse_transformer(coarse_a, coarse_b)
    correlation = model.correlation(tokens_a, tokens_b)
    coarse_height, coarse_width = coarse_a.shape[-2:]
    sampled = model.candidate_sampler(
        correlation.probability,
        correlation.matchability_a,
        coarse_height,
        coarse_width,
    )
    return GRMStage1BatchOutput(
        probability=correlation.probability,
        matchability_a=correlation.matchability_a,
        matchability_b=correlation.matchability_b,
        sampled=sampled,
        coarse_height=coarse_height,
        coarse_width=coarse_width,
    )


def _evaluate_model(
    model: GRMMatcher,
    dataset: Dataset[Mapping[str, Tensor]],
    config: GRMMatcherConfig,
    device: torch.device,
    *,
    amp_enabled: bool,
    batch_size: int,
    num_workers: int,
    max_batches: Optional[int],
    progress: bool,
    visualization_dir: Optional[Path],
    visualize_samples: int,
    max_visualized_queries: int,
    label: str,
    bucket_batch_sizes: Mapping[str, int],
) -> Dict[str, object]:
    if isinstance(dataset, GRMHomographyPairDataset):
        batch_sampler = GRMResolutionBucketBatchSampler(
            dataset.bucket_to_indices,
            batch_size,
            bucket_batch_sizes,
            shuffle=False,
            drop_last=False,
            seed=0,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
    iterator: Iterable[Mapping[str, Tensor]] = loader
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(loader, desc=f"Evaluating {label}", unit="batch")
        except ImportError:
            pass

    rows: list[Dict[str, float]] = []
    visualized = 0
    processed_batches = 0
    with torch.inference_mode():
        for batch_index, cpu_batch in enumerate(iterator):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = {key: value.to(device, non_blocking=True) for key, value in cpu_batch.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=amp_enabled,
            ):
                stage1 = forward_stage1(model, batch["image_a"], batch["image_b"])
            ground_truth = _coarse_ground_truth(batch, stage1, config.coarse_stride)
            rows.extend(
                _stage1_batch_metrics(
                    stage1,
                    ground_truth,
                    coarse_stride=config.coarse_stride,
                )
            )
            if visualization_dir is not None and visualized < visualize_samples:
                count = min(batch["image_a"].shape[0], visualize_samples - visualized)
                _save_visualizations(
                    batch,
                    stage1,
                    config,
                    visualization_dir,
                    first_index=visualized,
                    count=count,
                    max_queries=max_visualized_queries,
                )
                visualized += count
            processed_batches += 1
    if not rows:
        raise RuntimeError("Stage 1 validation processed no samples.")
    return {
        "metrics": _mean_metrics(rows),
        "metric_statistics": _metric_statistics(rows),
        "processed_samples": len(rows),
        "processed_batches": processed_batches,
    }


def _coarse_ground_truth(
    batch: Mapping[str, Tensor],
    output: GRMStage1BatchOutput,
    stride: int,
) -> Dict[str, Tensor]:
    device = output.probability.device
    count = output.coarse_height * output.coarse_width
    flat = torch.arange(count, device=device)
    source_feature_xy = flat_indices_to_feature_xy(flat, output.coarse_width)
    source_xy = feature_xy_to_image_xy(source_feature_xy, stride)
    source_xy = source_xy.unsqueeze(0).expand(output.probability.shape[0], -1, -1)
    gt_dst_xy = sample_dense_field(batch["gt_dense_mapping"], source_xy)
    overlap_a = sample_dense_field(batch["gt_overlap_a"].float(), source_xy)[..., 0] > 0.5
    finite_target = torch.isfinite(gt_dst_xy).all(dim=-1)
    safe_gt_dst_xy = torch.where(
        finite_target[..., None],
        gt_dst_xy,
        torch.zeros_like(gt_dst_xy),
    )
    target_feature_xy = image_xy_to_feature_xy(safe_gt_dst_xy, stride)
    rounded_target = target_feature_xy.round().long()
    inside = (
        finite_target
        & (rounded_target[..., 0] >= 0)
        & (rounded_target[..., 0] < output.coarse_width)
        & (rounded_target[..., 1] >= 0)
        & (rounded_target[..., 1] < output.coarse_height)
    )
    target_index = (
        rounded_target[..., 1].clamp(0, output.coarse_height - 1) * output.coarse_width
        + rounded_target[..., 0].clamp(0, output.coarse_width - 1)
    )
    # ``overlap_b`` supervises matchability over the regular B coarse grid.
    # ``target_overlap_b`` instead validates each A->B GT correspondence at its
    # mapped destination.  These tensors share [B, Hc*Wc] shape but have
    # different semantics and must not be interchanged.
    target_grid_xy = source_xy
    overlap_b = (
        sample_dense_field(batch["gt_overlap_b"].float(), target_grid_xy)[..., 0]
        > 0.5
    )
    target_overlap_b = (
        sample_dense_field(batch["gt_overlap_b"].float(), safe_gt_dst_xy)[..., 0]
        > 0.5
    )
    return {
        "source_xy": source_xy,
        "gt_dst_xy": gt_dst_xy,
        "target_feature_xy": target_feature_xy,
        "target_index": target_index,
        "valid_source": overlap_a & target_overlap_b & inside,
        "overlap_a": overlap_a,
        "overlap_b": overlap_b,
        "target_overlap_b": target_overlap_b,
    }


def _stage1_batch_metrics(
    output: GRMStage1BatchOutput,
    ground_truth: Mapping[str, Tensor],
    *,
    coarse_stride: int,
) -> list[Dict[str, float]]:
    if coarse_stride <= 0:
        raise ValueError("coarse_stride must be positive.")
    rows: list[Dict[str, float]] = []
    top1_indices = output.probability.argmax(dim=-1)
    predicted_feature_xy = flat_indices_to_feature_xy(top1_indices, output.coarse_width)
    for batch_index in range(output.probability.shape[0]):
        valid_source = ground_truth["valid_source"][batch_index]
        probability = output.probability[batch_index : batch_index + 1]
        target_index = ground_truth["target_index"][batch_index : batch_index + 1]
        recall = candidate_recall_at_k(
            probability,
            target_index,
            valid_source.unsqueeze(0),
            ks=(1, 4, 8),
        )
        row = dict(recall)
        row.update(
            _oracle_topk_metrics(
                probability,
                ground_truth["target_feature_xy"][batch_index : batch_index + 1],
                valid_source.unsqueeze(0),
                coarse_width=output.coarse_width,
                coarse_stride=coarse_stride,
                ks=(1, 4, 8),
                pixel_thresholds=(8.0, 16.0, 24.0),
            )
        )
        error_cells = torch.linalg.vector_norm(
            predicted_feature_xy[batch_index]
            - ground_truth["target_feature_xy"][batch_index],
            dim=-1,
        )[valid_source]
        row.update(_error_statistics(error_cells, prefix="top1_error_cells"))
        row.update(
            _binary_metrics(
                output.matchability_a[batch_index],
                ground_truth["overlap_a"][batch_index],
                prefix="matchability_a",
            )
        )
        row.update(
            _binary_metrics(
                output.matchability_b[batch_index],
                ground_truth["overlap_b"][batch_index],
                prefix="matchability_b",
            )
        )
        row.update(_selected_candidate_metrics(output, ground_truth, batch_index))
        for name, value in output.sampled.diagnostics.items():
            if value.ndim == 1 and value.shape[0] == output.probability.shape[0]:
                row[f"sampler_{name}"] = float(value[batch_index].float().cpu())
        row["mean_top1_global_probability"] = float(
            output.probability[batch_index].amax(dim=-1).mean().float().cpu()
        )
        rows.append(row)
    return rows


def _oracle_topk_metrics(
    probability: Tensor,
    gt_target_feature_xy: Tensor,
    valid_source_mask: Tensor,
    *,
    coarse_width: int,
    coarse_stride: int,
    ks: Sequence[int] = (1, 4, 8),
    pixel_thresholds: Sequence[float] = (8.0, 16.0, 24.0),
) -> Dict[str, float]:
    """Measure the closest top-k candidate to continuous GT in pixel units.

    Exact-cell Recall@K can be pessimistic close to a coarse-cell boundary.
    Oracle error complements it by retaining the continuous GT coordinate and
    measuring the nearest retrieved candidate center.  It does not use
    mid/fine refinement and therefore remains a coarse-only metric.
    """

    if probability.ndim != 3:
        raise ValueError("probability must be [B, S, T].")
    if gt_target_feature_xy.shape != (*probability.shape[:2], 2):
        raise ValueError("gt_target_feature_xy must be [B, S, 2].")
    if valid_source_mask.shape != probability.shape[:2]:
        raise ValueError("valid_source_mask must be [B, S].")
    if coarse_width <= 0 or coarse_stride <= 0:
        raise ValueError("coarse_width and coarse_stride must be positive.")
    if probability.shape[-1] % coarse_width != 0:
        raise ValueError("Target cell count must be divisible by coarse_width.")

    metrics: Dict[str, float] = {}
    for requested_k in ks:
        if requested_k <= 0:
            raise ValueError("Oracle k values must be positive.")
        effective_k = min(requested_k, probability.shape[-1])
        candidate_indices = torch.topk(
            probability,
            effective_k,
            dim=-1,
        ).indices
        candidate_feature_xy = flat_indices_to_feature_xy(
            candidate_indices,
            coarse_width,
        )
        error_pixels = (
            torch.linalg.vector_norm(
                candidate_feature_xy.float()
                - gt_target_feature_xy.float().unsqueeze(-2),
                dim=-1,
            )
            * float(coarse_stride)
        )
        oracle_error = error_pixels.amin(dim=-1)
        selected_error = oracle_error[valid_source_mask]
        prefix = f"oracle_top{requested_k}_pixel_error"
        metrics.update(_error_statistics(selected_error, prefix=prefix))
        for threshold in pixel_thresholds:
            if threshold <= 0.0:
                raise ValueError("Oracle PCK thresholds must be positive.")
            suffix = _format_pixel_threshold(threshold)
            correct = valid_source_mask & (oracle_error <= threshold)
            metrics[f"oracle_top{requested_k}_pck_{suffix}px"] = _safe_ratio(
                correct.sum(),
                valid_source_mask.sum(),
            )
    return metrics


def _safe_ratio(numerator: Tensor, denominator: Tensor) -> float:
    if int(denominator.detach().item()) == 0:
        return 0.0
    return float(
        (numerator.float() / denominator.float().clamp_min(1.0)).detach().cpu()
    )


def _format_pixel_threshold(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def _selected_candidate_metrics(
    output: GRMStage1BatchOutput,
    ground_truth: Mapping[str, Tensor],
    batch_index: int,
) -> Dict[str, float]:
    sampled = output.sampled
    selected = sampled.valid_mask[batch_index]
    selected_count = int(selected.sum())
    if selected_count == 0:
        return {
            "selected_top1_precision": 0.0,
            "selected_top4_precision": 0.0,
            "selected_top1_recall_all_gt": 0.0,
            "selected_top4_recall_all_gt": 0.0,
            "selected_overlap_precision": 0.0,
        }
    query_indices = sampled.query_indices[batch_index, selected]
    gt_valid = ground_truth["valid_source"][batch_index, query_indices]
    gt_index = ground_truth["target_index"][batch_index, query_indices]
    candidates = sampled.candidate_indices[batch_index, selected]
    correct = candidates == gt_index[:, None]
    correct_top1 = correct[:, 0] & gt_valid
    correct_top4 = correct.any(dim=-1) & gt_valid
    all_gt_count = int(ground_truth["valid_source"][batch_index].sum())
    return {
        "selected_top1_precision": float(correct_top1.float().mean().cpu()),
        "selected_top4_precision": float(correct_top4.float().mean().cpu()),
        "selected_top1_recall_all_gt": float(correct_top1.sum().float().cpu())
        / max(all_gt_count, 1),
        "selected_top4_recall_all_gt": float(correct_top4.sum().float().cpu())
        / max(all_gt_count, 1),
        "selected_overlap_precision": float(gt_valid.float().mean().cpu()),
    }


def _binary_metrics(scores: Tensor, target: Tensor, *, prefix: str) -> Dict[str, float]:
    scores = scores.detach().float().clamp(0.0, 1.0)
    target = target.detach().bool()
    finite = torch.isfinite(scores)
    scores = scores[finite]
    target = target[finite]
    if scores.numel() == 0:
        return {f"{prefix}_{name}": 0.0 for name in ("precision", "recall", "f1", "brier", "bce", "auroc")}
    predicted = scores >= 0.5
    true_positive = (predicted & target).sum().float()
    precision = true_positive / predicted.sum().clamp_min(1).float()
    recall = true_positive / target.sum().clamp_min(1).float()
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-12)
    brier = (scores - target.float()).square().mean()
    bce = F.binary_cross_entropy(scores.clamp(1.0e-6, 1.0 - 1.0e-6), target.float())
    auroc = _binary_auroc(scores, target)
    return {
        f"{prefix}_precision": float(precision.cpu()),
        f"{prefix}_recall": float(recall.cpu()),
        f"{prefix}_f1": float(f1.cpu()),
        f"{prefix}_brier": float(brier.cpu()),
        f"{prefix}_bce": float(bce.cpu()),
        f"{prefix}_auroc": auroc,
    }


def _binary_auroc(scores: Tensor, target: Tensor) -> float:
    positives = int(target.sum())
    negatives = int((~target).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True)
    sorted_target = target[order]
    true_positive_rate = torch.cat(
        (scores.new_zeros(1), sorted_target.float().cumsum(0) / positives)
    )
    false_positive_rate = torch.cat(
        (scores.new_zeros(1), (~sorted_target).float().cumsum(0) / negatives)
    )
    return float(torch.trapz(true_positive_rate, false_positive_rate).cpu())


def _error_statistics(values: Tensor, *, prefix: str) -> Dict[str, float]:
    if values.numel() == 0:
        return {f"{prefix}_{name}": float("nan") for name in ("mean", "median", "p90")}
    values = values.float()
    return {
        f"{prefix}_mean": float(values.mean().cpu()),
        f"{prefix}_median": float(values.median().cpu()),
        f"{prefix}_p90": float(torch.quantile(values, 0.9).cpu()),
    }


def _mean_metrics(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    names = sorted({name for row in rows for name in row})
    result: Dict[str, float] = {}
    for name in names:
        finite = [float(row[name]) for row in rows if name in row and math.isfinite(row[name])]
        result[name] = sum(finite) / len(finite) if finite else 0.0
    return result


def _metric_statistics(rows: Sequence[Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
    statistics: Dict[str, Dict[str, float]] = {}
    for name in sorted({name for row in rows for name in row}):
        values = np.asarray(
            [row[name] for row in rows if name in row and math.isfinite(row[name])],
            dtype=np.float64,
        )
        if values.size:
            statistics[name] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "count": int(values.size),
            }
    return statistics


def _compare_metric_means(
    evaluated: object,
    baseline: object,
) -> Dict[str, Dict[str, object]]:
    if not isinstance(evaluated, Mapping) or not isinstance(baseline, Mapping):
        raise TypeError("Metric summaries must be mappings.")
    result: Dict[str, Dict[str, object]] = {}
    for name in sorted(set(evaluated) & set(baseline)):
        current = float(evaluated[name])
        reference = float(baseline[name])
        direction = _metric_preference(name)
        values: Dict[str, object] = {
            "checkpoint": current,
            "baseline": reference,
            "delta": current - reference,
            "preferred_direction": direction,
        }
        if direction == "higher":
            values["improved"] = current > reference
        elif direction == "lower":
            values["improved"] = current < reference
        result[name] = values
    return result


def _metric_preference(name: str) -> str:
    lower_markers = (
        "error",
        "_brier",
        "_bce",
        "_gini",
        "max_region_selection_ratio",
    )
    higher_markers = (
        "recall",
        "precision",
        "pck",
        "_f1",
        "_auroc",
        "coverage_ratio",
        "pairwise_distance",
        "budget_utilization",
    )
    if any(marker in name for marker in lower_markers):
        return "lower"
    if any(marker in name for marker in higher_markers):
        return "higher"
    return "diagnostic_only"


def _screen_stage1(metrics: object, args: argparse.Namespace) -> Dict[str, object]:
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping.")
    checks = {
        "recall_at_4": float(metrics["recall_at_4"]) >= args.min_recall_at_4,
        "selected_top1_precision": float(metrics["selected_top1_precision"])
        >= args.min_selected_top1_precision,
        "matchability_a_auroc": float(metrics["matchability_a_auroc"])
        >= args.min_matchability_auroc,
    }
    thresholds: Dict[str, float] = {
        "min_recall_at_4": args.min_recall_at_4,
        "min_selected_top1_precision": args.min_selected_top1_precision,
        "min_matchability_auroc": args.min_matchability_auroc,
    }
    if args.min_recall_at_8 is not None:
        checks["recall_at_8"] = (
            float(metrics["recall_at_8"]) >= args.min_recall_at_8
        )
        thresholds["min_recall_at_8"] = args.min_recall_at_8
    if args.min_oracle_top4_pck_16px is not None:
        checks["oracle_top4_pck_16px"] = (
            float(metrics["oracle_top4_pck_16px"])
            >= args.min_oracle_top4_pck_16px
        )
        thresholds["min_oracle_top4_pck_16px"] = (
            args.min_oracle_top4_pck_16px
        )
    if args.max_oracle_top4_median_pixel_error is not None:
        checks["oracle_top4_pixel_error_median"] = (
            float(metrics["oracle_top4_pixel_error_median"])
            <= args.max_oracle_top4_median_pixel_error
        )
        thresholds["max_oracle_top4_median_pixel_error"] = (
            args.max_oracle_top4_median_pixel_error
        )
    return {
        "status": "pass" if all(checks.values()) else "needs_more_training_or_diagnosis",
        "checks": checks,
        "thresholds": thresholds,
        "note": "These are configurable screening thresholds, not production guarantees.",
    }


def _save_visualizations(
    batch: Mapping[str, Tensor],
    output: GRMStage1BatchOutput,
    config: GRMMatcherConfig,
    output_dir: Path,
    *,
    first_index: int,
    count: int,
    max_queries: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled = output.sampled
    source_xy = flat_indices_to_image_xy(
        sampled.query_indices, output.coarse_width, config.coarse_stride
    )
    candidate_xy = flat_indices_to_image_xy(
        sampled.candidate_indices, output.coarse_width, config.coarse_stride
    )
    for local_index in range(count):
        prefix = output_dir / f"sample_{first_index + local_index:04d}"
        figures = [
            visualize_topk_candidates(
                batch["image_a"][local_index],
                batch["image_b"][local_index],
                source_xy[local_index : local_index + 1],
                candidate_xy[local_index : local_index + 1],
                sampled.candidate_probs[local_index : local_index + 1],
                sampled.valid_mask[local_index : local_index + 1],
                max_queries=max_queries,
                save_path=f"{prefix}_topk.png",
            ),
            visualize_candidate_entropy(
                batch["image_a"][local_index],
                source_xy[local_index : local_index + 1],
                sampled.candidate_entropy[local_index : local_index + 1],
                sampled.valid_mask[local_index : local_index + 1],
                save_path=f"{prefix}_entropy.png",
            ),
            visualize_spatial_coverage(
                batch["image_a"][local_index],
                source_xy[local_index : local_index + 1],
                sampled.valid_mask[local_index : local_index + 1],
                save_path=f"{prefix}_coverage.png",
            ),
        ]
        for figure in figures:
            plt.close(figure)
        _save_matchability_heatmap(output, local_index, f"{prefix}_matchability.png")


def _save_matchability_heatmap(
    output: GRMStage1BatchOutput,
    batch_index: int,
    path: str | Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, values, name in (
        (axes[0], output.matchability_a[batch_index], "A matchability"),
        (axes[1], output.matchability_b[batch_index], "B matchability"),
    ):
        image = axis.imshow(
            values.float().reshape(output.coarse_height, output.coarse_width).cpu(),
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        axis.set_title(name)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _make_synthetic_dataset(
    image_paths: Sequence[Path],
    config: GRMMatcherConfig,
    samples_per_source: int,
    seed: int,
) -> SyntheticGRMDataset:
    return SyntheticGRMDataset(
        image_paths,
        SyntheticGRMDataConfig(
            output_height=config.input_height,
            output_width=config.input_width,
            samples_per_source=samples_per_source,
            seed=seed,
            global_residual_threshold=config.global_residual_threshold,
            local_residual_threshold=config.local_residual_threshold,
        ),
    )


def _load_checkpoint(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint {path} must contain a mapping.")
    return checkpoint


def _checkpoint_config(checkpoint: Mapping[str, object]) -> GRMMatcherConfig:
    values = checkpoint.get("config")
    if not isinstance(values, Mapping):
        raise KeyError("Checkpoint is missing the serialized GRM 'config'.")
    return GRMMatcherConfig.from_dict(dict(values))


def _build_model(
    config: GRMMatcherConfig,
    checkpoint: Mapping[str, object],
    device: torch.device,
) -> GRMMatcher:
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise KeyError("Checkpoint is missing the GRM 'model' state.")
    model = GRMMatcher(config).to(device)
    model.load_state_dict(state, strict=True)
    return model.eval()


def _validate_checkpoint_stage(
    checkpoint: Mapping[str, object],
    path: Path,
    *,
    allow_non_stage1: bool,
) -> None:
    stage = checkpoint.get("stage")
    if stage != "stage1" and not allow_non_stage1:
        raise ValueError(
            f"Checkpoint {path} reports stage={stage!r}; expected 'stage1'. "
            "Pass --allow_non_stage1_checkpoint only if this is intentional."
        )


def _find_images(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        raise NotADirectoryError(data_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not paths:
        raise FileNotFoundError(f"No supported images found under {data_dir}")
    return paths


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.dataset_type == "synthetic" and args.samples_per_source <= 0:
        raise ValueError("samples_per_source must be positive for synthetic data.")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")
    if args.num_workers < 0 or args.visualize_samples < 0:
        raise ValueError("num_workers and visualize_samples must be nonnegative.")
    if args.max_visualized_queries <= 0:
        raise ValueError("max_visualized_queries must be positive.")
    for name in (
        "min_recall_at_4",
        "min_selected_top1_precision",
        "min_matchability_auroc",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1].")
    for name in ("min_recall_at_8", "min_oracle_top4_pck_16px"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1] when provided.")
    if (
        args.max_oracle_top4_median_pixel_error is not None
        and args.max_oracle_top4_median_pixel_error < 0.0
    ):
        raise ValueError(
            "max_oracle_top4_median_pixel_error must be nonnegative when provided."
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
