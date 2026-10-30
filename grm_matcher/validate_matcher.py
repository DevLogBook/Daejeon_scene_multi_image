"""Independent synthetic validation for GRM-Matcher checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .config import GRM_BUILTIN_CANDIDATE_RERANK_MODES, GRMMatcherConfig
from .coordinates import feature_xy_to_image_xy, image_xy_to_feature_xy
from .homography_pair_dataset import (
    GRMHomographyPairDataset,
    GRMResolutionBucketBatchSampler,
    parse_bucket_batch_sizes,
)
from .matcher import GRMMatcher
from .metrics import (
    candidate_recall_at_k,
    evaluate_output_against_dense_targets,
    homography_quality_metrics,
    sample_dense_field,
)
from .outputs import GRMMatcherAuxiliary, GRMMatcherOutput
from .synthetic_dataset import SyntheticGRMDataConfig, SyntheticGRMDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a standalone GRM-Matcher checkpoint.")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/grm_validation/metrics.json"))
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
    parser.add_argument(
        "--candidate_rerank_mode",
        choices=GRM_BUILTIN_CANDIDATE_RERANK_MODES,
        default=None,
        help="Override candidate selection without changing checkpoint weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.dataset_type == "synthetic" and args.samples_per_source <= 0:
        raise ValueError("samples_per_source must be positive for synthetic data.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = GRMMatcherConfig.from_dict(checkpoint["config"])
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = GRMMatcher(config).to(device)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.eval()
    bucket_batch_sizes = parse_bucket_batch_sizes(args.bucket_batch_size)
    if args.dataset_type == "homography_pairs":
        dataset = GRMHomographyPairDataset(
            args.data_dir, args.split, config, args.buckets
        )
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
        )
    else:
        image_paths = _find_images(args.data_dir)
        dataset = SyntheticGRMDataset(
            image_paths,
            SyntheticGRMDataConfig(
                output_height=config.input_height,
                output_width=config.input_width,
                samples_per_source=args.samples_per_source,
                global_residual_threshold=config.global_residual_threshold,
                local_residual_threshold=config.local_residual_threshold,
            ),
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    amp_requested = config.use_amp if args.amp is None else args.amp
    amp_enabled = bool(amp_requested and device.type == "cuda")
    values: Dict[str, list[float]] = {}
    processed_batches = 0
    processed_samples = 0
    with torch.no_grad():
        for batch_index, cpu_batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = {
                key: value.to(device) if isinstance(value, Tensor) else value
                for key, value in cpu_batch.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=amp_enabled,
            ):
                output, auxiliary = model(
                    batch["image_a"],
                    batch["image_b"],
                    return_auxiliary=True,
                    candidate_rerank_mode=args.candidate_rerank_mode,
                )
            assert isinstance(output, GRMMatcherOutput)
            image_height, image_width = batch["image_a"].shape[-2:]
            metrics = evaluate_output_against_dense_targets(
                output,
                batch,
                image_height,
                image_width,
                correspondence_correct_threshold=(
                    config.correspondence_correct_threshold
                ),
            )
            metrics.update(_candidate_metrics(auxiliary, batch, config))
            legacy_role_homography = homography_quality_metrics(
                output.src_xy,
                output.dst_xy,
                output.p_global,
                output.valid_mask,
                batch["H_ab"],
                image_height,
                image_width,
            )
            metrics.update(legacy_role_homography)
            # Candidate-reranker ablations must not be confounded by Role-head
            # calibration.  This fixed downstream protocol gives every valid
            # selected match to the same MAGSAC estimator.
            all_valid_homography = homography_quality_metrics(
                output.src_xy,
                output.dst_xy,
                torch.ones_like(output.p_match),
                output.valid_mask,
                batch["H_ab"],
                image_height,
                image_width,
            )
            metrics.update(
                {
                    f"all_valid_{name}": value
                    for name, value in all_valid_homography.items()
                }
            )
            for name, value in metrics.items():
                values.setdefault(name, []).append(value)
            processed_batches += 1
            processed_samples += int(batch["image_a"].shape[0])
    if processed_batches == 0:
        raise RuntimeError("Validation processed no batches.")
    summary = {
        name: _mean_preserving_nan(metric_values) for name, metric_values in values.items()
    }
    summary["processed_batches"] = processed_batches
    summary["processed_samples"] = processed_samples
    summary["dataset_type"] = args.dataset_type
    summary["split"] = args.split
    summary["selected_buckets"] = args.buckets
    summary["candidate_rerank_mode"] = (
        args.candidate_rerank_mode or config.candidate_rerank_mode
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True))


def _candidate_metrics(
    auxiliary: GRMMatcherAuxiliary,
    batch: Dict[str, Tensor],
    config: GRMMatcherConfig,
) -> Dict[str, float]:
    batch_size = batch["image_a"].shape[0]
    height = auxiliary.coarse_height
    width = auxiliary.coarse_width
    flat = torch.arange(height * width, device=batch["image_a"].device)[None].expand(
        batch_size, -1
    )
    feature_xy = torch.stack(
        (
            torch.remainder(flat, width),
            torch.div(flat, width, rounding_mode="floor"),
        ),
        dim=-1,
    ).float()
    source_xy = feature_xy_to_image_xy(feature_xy, config.coarse_stride)
    gt_dst = sample_dense_field(batch["gt_dense_mapping"], source_xy)
    overlap = sample_dense_field(batch["gt_overlap_a"].float(), source_xy)[..., 0] > 0.5
    target_feature = image_xy_to_feature_xy(gt_dst, config.coarse_stride).round().long()
    inside = (
        (target_feature[..., 0] >= 0)
        & (target_feature[..., 0] < width)
        & (target_feature[..., 1] >= 0)
        & (target_feature[..., 1] < height)
    )
    target_index = (
        target_feature[..., 1].clamp(0, height - 1) * width
        + target_feature[..., 0].clamp(0, width - 1)
    )
    return candidate_recall_at_k(
        auxiliary.coarse_probability, target_index, overlap & inside, ks=(1, 4, 8)
    )


def _find_images(data_dir: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if not data_dir.is_dir():
        raise NotADirectoryError(data_dir)
    paths = sorted(
        path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() in extensions
    )
    if not paths:
        raise FileNotFoundError(f"No images found under {data_dir}")
    return paths


def _mean_preserving_nan(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


if __name__ == "__main__":
    main()
