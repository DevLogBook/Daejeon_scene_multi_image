"""Standalone two-image inference and visualization entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import torch
from torch import Tensor

from .config import GRM_BUILTIN_CANDIDATE_RERANK_MODES, GRMMatcherConfig
from .coordinates import flat_indices_to_image_xy
from .matcher import GRMMatcher
from .outputs import GRMMatcherOutput
from .visualization import (
    visualize_candidate_entropy,
    visualize_geometry_roles,
    visualize_matches,
    visualize_spatial_coverage,
    visualize_topk_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone GRM-Matcher inference.")
    parser.add_argument("--image_a", type=Path, required=True)
    parser.add_argument("--image_b", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/grm_demo"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--min_probability", type=float, default=0.0)
    parser.add_argument("--max_matches", type=int, default=200)
    parser.add_argument(
        "--candidate_rerank_mode",
        choices=GRM_BUILTIN_CANDIDATE_RERANK_MODES,
        default=None,
        help=(
            "Override the checkpoint's candidate reranker for an ablation. "
            "The default preserves its geometry-graph behavior."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_probability <= 1.0:
        raise ValueError("min_probability must lie in [0, 1].")
    if args.max_matches <= 0:
        raise ValueError("max_matches must be positive.")
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "config" not in checkpoint:
        raise KeyError("Checkpoint must contain a serialized GRMMatcherConfig under 'config'.")
    config = GRMMatcherConfig.from_dict(checkpoint["config"])
    model = GRMMatcher(config)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    original_a, input_a, scale_a = _load_image(
        args.image_a, config.input_height, config.input_width, config
    )
    original_b, input_b, scale_b = _load_image(
        args.image_b, config.input_height, config.input_width, config
    )
    amp_requested = config.use_amp if args.amp is None else args.amp
    amp_enabled = bool(amp_requested and device.type == "cuda")
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=amp_enabled,
    ):
        output = model(
            input_a.to(device),
            input_b.to(device),
            candidate_rerank_mode=args.candidate_rerank_mode,
        )
    assert isinstance(output, GRMMatcherOutput)
    output = output.to("cpu")

    src_xy = output.src_xy.clone()
    dst_xy = output.dst_xy.clone()
    src_xy[..., 0] /= scale_a[0]
    src_xy[..., 1] /= scale_a[1]
    dst_xy[..., 0] /= scale_b[0]
    dst_xy[..., 1] /= scale_b[1]
    valid = output.valid_mask & (output.p_match >= args.min_probability)
    if config.inference_max_cycle_error is not None:
        valid = valid & (output.cycle_error <= config.inference_max_cycle_error)

    covariance_scale = output.fine_uncertainty.new_tensor(
        [[1.0 / scale_b[0], 0.0], [0.0, 1.0 / scale_b[1]]]
    )
    covariance = torch.einsum(
        "ij,bnjk,lk->bnil",
        covariance_scale,
        output.fine_uncertainty,
        covariance_scale,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        args.output_dir / "matches.json",
        output,
        src_xy,
        dst_xy,
        covariance,
        valid,
        scale_a,
        scale_b,
        args.candidate_rerank_mode or config.candidate_rerank_mode,
    )

    figure = visualize_matches(
        original_a,
        original_b,
        src_xy,
        dst_xy,
        output.p_match,
        output.p_global,
        output.p_local,
        output.p_false,
        output.candidate_entropy,
        valid,
        max_matches=args.max_matches,
        save_path=args.output_dir / "matches.png",
    )
    _close(figure)
    figure = visualize_geometry_roles(
        original_a,
        original_b,
        src_xy,
        dst_xy,
        output.p_global,
        output.p_local,
        output.p_false,
        valid,
        save_path=args.output_dir / "roles.png",
    )
    _close(figure)
    figure = visualize_candidate_entropy(
        original_a,
        src_xy,
        output.candidate_entropy,
        valid,
        save_path=args.output_dir / "entropy.png",
    )
    _close(figure)
    figure = visualize_spatial_coverage(
        original_a,
        src_xy,
        valid & (output.p_match >= 0.5),
        grid_rows=config.sampling_grid_rows,
        grid_cols=config.sampling_grid_cols,
        save_path=args.output_dir / "coverage.png",
    )
    _close(figure)

    coarse_width = config.input_width // config.coarse_stride
    candidate_dst = flat_indices_to_image_xy(
        output.candidate_indices,
        coarse_width,
        config.coarse_stride,
    )
    candidate_dst[..., 0] /= scale_b[0]
    candidate_dst[..., 1] /= scale_b[1]
    figure = visualize_topk_candidates(
        original_a,
        original_b,
        src_xy,
        candidate_dst,
        output.candidate_probs,
        valid,
        save_path=args.output_dir / "topk.png",
    )
    _close(figure)


def _load_image(
    path: Path,
    height: int,
    width: int,
    config: GRMMatcherConfig,
) -> tuple[np.ndarray, Tensor, tuple[float, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        original = np.asarray(rgb, dtype=np.uint8)
        resized = rgb.resize((width, height), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    if config.input_range == "imagenet_normalized":
        mean = tensor.new_tensor(config.normalization_mean)[None, :, None, None]
        std = tensor.new_tensor(config.normalization_std)[None, :, None, None]
        tensor = (tensor - mean) / std
    scale = (width / original.shape[1], height / original.shape[0])
    return original, tensor, scale


def _write_json(
    path: Path,
    output: GRMMatcherOutput,
    src_xy: Tensor,
    dst_xy: Tensor,
    covariance: Tensor,
    valid_mask: Tensor,
    scale_a: tuple[float, float],
    scale_b: tuple[float, float],
    candidate_rerank_mode: str,
) -> None:
    valid = valid_mask[0]
    indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    records = []
    for index_tensor in indices:
        index = int(index_tensor)
        records.append(
            {
                "src_xy": src_xy[0, index].tolist(),
                "dst_xy": dst_xy[0, index].tolist(),
                "p_match": float(output.p_match[0, index]),
                "p_global": float(output.p_global[0, index]),
                "p_local": float(output.p_local[0, index]),
                "p_false": float(output.p_false[0, index]),
                "candidate_entropy": float(output.candidate_entropy[0, index]),
                "cycle_error": float(output.cycle_error[0, index]),
                "fine_uncertainty": covariance[0, index].tolist(),
                "candidate_indices": output.candidate_indices[0, index].tolist(),
                "candidate_probs": output.candidate_probs[0, index].tolist(),
            }
        )
    payload = {
        "coordinate_order": "xy",
        "coordinate_space": "original_input_images",
        "resize_scale_a_xy": list(scale_a),
        "resize_scale_b_xy": list(scale_b),
        "candidate_rerank_mode": candidate_rerank_mode,
        "match_count": len(records),
        "matches": records,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _close(figure: object) -> None:
    from matplotlib import pyplot as plt

    plt.close(figure)


if __name__ == "__main__":
    main()

