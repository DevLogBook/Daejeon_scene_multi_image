"""Independent Stage-2 validator that stops before geometry and reranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .config import GRMMatcherConfig
from .homography_pair_dataset import (
    GRMHomographyPairDataset,
    GRMResolutionBucketBatchSampler,
    parse_bucket_batch_sizes,
)
from .matcher import GRMMatcher
from .stage2_metrics import (
    GRMStage2MetricAccumulator,
    GRMStage2Predictions,
    compute_stage2_metric_components,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate coarse-to-mid-to-fine refinement without Stage-3 modules."
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/grm_stage2_eval"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--buckets", nargs="*", default=None)
    parser.add_argument("--bucket_batch_size", action="append", default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_non_stage2_checkpoint", action="store_true")
    parser.add_argument("--max_fine_top1_error", type=float, default=5.0)
    parser.add_argument("--min_fine_top1_pck3", type=float, default=0.75)
    parser.add_argument("--min_fine_oracle_pck3", type=float, default=0.85)
    parser.add_argument("--min_mid_reachable_ratio", type=float, default=0.90)
    parser.add_argument("--min_fine_reachable_ratio", type=float, default=0.80)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping containing model and config.")
    stage = checkpoint.get("stage")
    if stage != "stage2" and not args.allow_non_stage2_checkpoint:
        raise ValueError(
            f"Expected a stage2 checkpoint, got stage={stage!r}. "
            "Use --allow_non_stage2_checkpoint only for an intentional baseline."
        )
    config_values = checkpoint.get("config")
    if not isinstance(config_values, Mapping):
        raise KeyError("Checkpoint does not contain a valid config mapping.")
    config = GRMMatcherConfig.from_dict(config_values)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    amp_requested = config.use_amp if args.amp is None else args.amp
    amp_enabled = bool(amp_requested and device.type == "cuda")

    model = GRMMatcher(config).to(device).eval()
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint model state must be a mapping.")
    model.load_state_dict(state, strict=True)

    dataset = GRMHomographyPairDataset(
        args.data_dir, args.split, config, args.buckets
    )
    bucket_batch_sizes = parse_bucket_batch_sizes(args.bucket_batch_size)
    batch_sampler = GRMResolutionBucketBatchSampler(
        dataset.bucket_to_indices,
        args.batch_size,
        bucket_batch_sizes,
        shuffle=False,
        drop_last=False,
        seed=0,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    iterator: Iterable[Mapping[str, Tensor]] = loader
    if args.progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(loader, desc="Evaluating Stage 2", unit="batch")
        except ImportError:
            pass

    overall = GRMStage2MetricAccumulator()
    per_bucket: Dict[str, GRMStage2MetricAccumulator] = {}
    processed_batches = 0
    processed_samples = 0
    with torch.inference_mode():
        for batch_index, cpu_batch in enumerate(iterator):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in cpu_batch.items()
                if isinstance(value, Tensor)
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=amp_enabled,
            ):
                predictions = forward_stage2(model, batch["image_a"], batch["image_b"])
            components = compute_stage2_metric_components(predictions, batch, config)
            overall.update(components)
            height, width = batch["image_a"].shape[-2:]
            bucket = f"{width}x{height}"
            per_bucket.setdefault(bucket, GRMStage2MetricAccumulator()).update(components)
            processed_batches += 1
            processed_samples += int(batch["image_a"].shape[0])

    if processed_batches == 0:
        raise RuntimeError("Stage-2 validation processed no batches.")
    metrics = overall.metrics()
    report = {
        "schema_version": 1,
        "scope": "stage2_coarse_mid_fine_without_geometry_rerank_or_roles",
        "excluded_from_evaluation": [
            "cycle_consistency",
            "seed_homography",
            "geometry_graph",
            "rerank_head",
            "role_head",
        ],
        "checkpoint": str(args.checkpoint),
        "checkpoint_stage": stage,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "data_dir": str(args.data_dir),
        "split": args.split,
        "selected_buckets": args.buckets,
        "dataset_sample_count": len(dataset),
        "processed_samples": processed_samples,
        "processed_batches": processed_batches,
        "max_batches": args.max_batches,
        "amp_enabled": amp_enabled,
        "metrics": metrics,
        "improvements": {
            "top1_coarse_to_mid_pixel": metrics["coarse_top1_pixel_error"]
            - metrics["mid_top1_pixel_error"],
            "top1_mid_to_fine_pixel": metrics["mid_top1_pixel_error"]
            - metrics["fine_top1_pixel_error"],
            "oracle_coarse_to_mid_pixel": metrics["coarse_oracle_pixel_error"]
            - metrics["mid_oracle_pixel_error"],
            "oracle_mid_to_fine_pixel": metrics["mid_oracle_pixel_error"]
            - metrics["fine_oracle_pixel_error"],
        },
        "per_bucket": {
            name: accumulator.metrics() for name, accumulator in sorted(per_bucket.items())
        },
        "screening": _screen(metrics, args),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "stage2_report.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Stage 2 report written to {output_path}", flush=True)


def forward_stage2(
    model: GRMMatcher, image_a: Tensor, image_b: Tensor
) -> GRMStage2Predictions:
    """Run through fine refinement and deliberately stop before Stage 3."""

    model._validate_images(image_a, image_b)
    batch = image_a.shape[0]
    combined = torch.cat((image_a, image_b), dim=0)
    pyramid = model.fpn(model.backbone(combined))
    features_a = {name: value[:batch] for name, value in pyramid.items()}
    features_b = {name: value[batch:] for name, value in pyramid.items()}
    tokens_a, tokens_b = model.coarse_transformer(
        features_a["F_coarse"], features_b["F_coarse"]
    )
    correlation = model.correlation(tokens_a, tokens_b)
    coarse_height, coarse_width = features_a["F_coarse"].shape[-2:]
    sampled = model.candidate_sampler(
        correlation.probability,
        correlation.matchability_a,
        coarse_height,
        coarse_width,
    )
    source, initial, valid, mid, fine = model._refine_all_candidates(
        sampled, features_a, features_b, coarse_width
    )
    query_count, topk = sampled.candidate_indices.shape[1:]
    coordinate_shape = (batch, query_count, topk, 2)
    return GRMStage2Predictions(
        src_xy=source.reshape(coordinate_shape),
        coarse_dst_xy=initial.reshape(coordinate_shape),
        mid_dst_xy=mid.dst_xy.reshape(coordinate_shape),
        fine_dst_xy=fine.dst_xy.reshape(coordinate_shape),
        candidate_valid_mask=valid.reshape(batch, query_count, topk),
    )


def _screen(metrics: Mapping[str, float], args: argparse.Namespace) -> Dict[str, object]:
    checks = {
        "fine_top1_pixel_error": metrics["fine_top1_pixel_error"]
        <= args.max_fine_top1_error,
        "fine_top1_pck_3": metrics["fine_top1_pck_3"]
        >= args.min_fine_top1_pck3,
        "fine_oracle_pck_3": metrics["fine_oracle_pck_3"]
        >= args.min_fine_oracle_pck3,
        "mid_top1_reachable_ratio": metrics["mid_top1_reachable_ratio"]
        >= args.min_mid_reachable_ratio,
        "fine_top1_reachable_ratio": metrics["fine_top1_reachable_ratio"]
        >= args.min_fine_reachable_ratio,
    }
    return {
        "status": "pass" if all(checks.values()) else "needs_improvement",
        "checks": checks,
        "thresholds": {
            "max_fine_top1_error": args.max_fine_top1_error,
            "min_fine_top1_pck3": args.min_fine_top1_pck3,
            "min_fine_oracle_pck3": args.min_fine_oracle_pck3,
            "min_mid_reachable_ratio": args.min_mid_reachable_ratio,
            "min_fine_reachable_ratio": args.min_fine_reachable_ratio,
        },
        "note": "Screening thresholds are configurable baselines, not production guarantees.",
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative.")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")
    if args.max_fine_top1_error <= 0.0:
        raise ValueError("max_fine_top1_error must be positive.")
    for name in (
        "min_fine_top1_pck3",
        "min_fine_oracle_pck3",
        "min_mid_reachable_ratio",
        "min_fine_reachable_ratio",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1].")


if __name__ == "__main__":
    main()
