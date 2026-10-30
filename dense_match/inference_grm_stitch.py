"""Two-image GRM + residual mesh + independent seam stitching entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dense_match.geometry_pipeline import GeometryPipelineConfig
from dense_match.global_homography import GlobalHomographyConfig
from dense_match.grm_adapter import GRMAdapterConfig, GRMMatcherAdapter
from dense_match.grm_debug import write_grm_stitch_debug
from dense_match.grm_stitch_pipeline import (
    GRMStitchPipelineConfig,
    stitch_grm_pair,
)
from dense_match.instability_graphcut import InstabilitySeamConfig
from dense_match.narrow_band_blend import NarrowBandBlendConfig
from dense_match.residual_mesh import ResidualMeshConfig
from dense_match.role_guided_mesh_decoder import (
    RoleGuidedMeshDecoderConfig,
    RoleGuidedResidualMeshDecoder,
)
from grm_matcher import GRMMatcher, GRMMatcherConfig
from grm_matcher.routing_head import GRMGeometryRoutingHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_a", type=Path, required=True)
    parser.add_argument("--image_b", type=Path, required=True)
    parser.add_argument("--grm_checkpoint", type=Path, required=True)
    parser.add_argument("--routing_checkpoint", type=Path)
    parser.add_argument("--decoder_checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/grm_stitch.png"))
    parser.add_argument("--geometry_mode", choices=("h_only", "residual_mesh", "learned_residual_mesh"), default="residual_mesh")
    parser.add_argument("--routing_mode", choices=("none", "legacy_role", "soft_routing"), default="none")
    parser.add_argument("--h_match_mode", choices=("all_reliable", "legacy_global_only"), default="all_reliable")
    parser.add_argument("--mesh_match_mode", choices=("all_reliable", "legacy_local_only"), default="all_reliable")
    parser.add_argument("--seam_mode", choices=("none", "graphcut", "graphcut_multiband"), default="graphcut_multiband")
    parser.add_argument("--minimum_match_probability", type=float, default=0.05)
    parser.add_argument("--maximum_cycle_error", type=float)
    parser.add_argument("--mesh_rows", type=int, default=12)
    parser.add_argument("--mesh_cols", type=int, default=12)
    parser.add_argument("--max_residual_px", type=float, default=24.0)
    parser.add_argument("--max_decoder_correction_px", type=float, default=4.0)
    parser.add_argument("--seam_band_width", type=int, default=12)
    parser.add_argument("--seam_backend", choices=("auto", "opencv_graphcut", "dynamic_programming"), default="auto")
    parser.add_argument("--save_debug", action="store_true")
    parser.add_argument("--debug_dir", type=Path)
    parser.add_argument("--device")
    return parser.parse_args()


def _read(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _network_tensor(
    image_bgr: np.ndarray,
    config: GRMMatcherConfig,
    device: torch.device,
) -> torch.Tensor:
    resized = cv2.resize(
        image_bgr,
        (config.input_width, config.input_height),
        interpolation=cv2.INTER_LINEAR,
    )
    rgb = resized[..., ::-1].copy()
    tensor = torch.from_numpy(rgb).to(device=device, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
    if config.input_range == "imagenet_normalized":
        mean = tensor.new_tensor(config.normalization_mean)[None, :, None, None]
        std = tensor.new_tensor(config.normalization_std)[None, :, None, None]
        tensor = (tensor - mean) / std
    return tensor


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(args.grm_checkpoint, map_location="cpu", weights_only=False)
    if "config" not in checkpoint:
        raise KeyError("GRM checkpoint must contain 'config'.")
    grm_config = GRMMatcherConfig.from_dict(checkpoint["config"])
    matcher = GRMMatcher(grm_config)
    matcher.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    matcher.to(device).eval()
    adapter = GRMMatcherAdapter(
        matcher,
        GRMAdapterConfig(
            minimum_match_probability=args.minimum_match_probability,
            maximum_cycle_error=args.maximum_cycle_error,
        ),
    )

    routing_head = None
    if args.routing_checkpoint is not None:
        routing_checkpoint = torch.load(
            args.routing_checkpoint, map_location="cpu", weights_only=False
        )
        hidden = int(routing_checkpoint.get("hidden_dim", grm_config.graph_hidden_dim))
        routing_head = GRMGeometryRoutingHead(hidden)
        routing_head.load_state_dict(
            routing_checkpoint.get("routing_head", routing_checkpoint), strict=True
        )
        routing_head.to(device).eval()

    decoder = None
    if args.decoder_checkpoint is not None:
        decoder = RoleGuidedResidualMeshDecoder(
            RoleGuidedMeshDecoderConfig(
                feature_channels=grm_config.fine_dim,
                max_correction_px=args.max_decoder_correction_px,
                max_total_residual_px=args.max_residual_px,
            )
        )
        decoder_checkpoint = torch.load(
            args.decoder_checkpoint, map_location="cpu", weights_only=False
        )
        decoder.load_state_dict(
            decoder_checkpoint.get("decoder", decoder_checkpoint.get("state_dict", decoder_checkpoint)),
            strict=True,
        )
        decoder.to(device).eval()

    geometry_config = GeometryPipelineConfig(
        geometry_mode=args.geometry_mode,
        routing_mode=args.routing_mode,
        h_match_mode=args.h_match_mode,
        mesh_match_mode=args.mesh_match_mode,
        homography=GlobalHomographyConfig(),
        residual_mesh=ResidualMeshConfig(
            mesh_rows=args.mesh_rows,
            mesh_cols=args.mesh_cols,
            max_residual_px=args.max_residual_px,
        ),
    )
    pipeline_config = GRMStitchPipelineConfig(
        geometry=geometry_config,
        seam_mode=args.seam_mode,
        seam=InstabilitySeamConfig(backend=args.seam_backend),
        blend=NarrowBandBlendConfig(band_width_px=args.seam_band_width),
    )
    image_a = _read(args.image_a)
    image_b = _read(args.image_b)
    tensor_a = _network_tensor(image_a, grm_config, device)
    tensor_b = _network_tensor(image_b, grm_config, device)
    with torch.no_grad():
        result = stitch_grm_pair(
            image_a,
            image_b,
            tensor_a,
            tensor_b,
            adapter,
            config=pipeline_config,
            routing_head=routing_head,
            learned_decoder=decoder,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), result.final_stitched_image)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(
        json.dumps(result.diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.save_debug:
        debug_dir = args.debug_dir or args.output.with_name(args.output.stem + "_debug")
        write_grm_stitch_debug(debug_dir, image_a, image_b, result)
    print(f"[save] stitched={args.output.resolve()}")
    print(f"[save] report={report_path.resolve()}")
    print(f"[fallback] {result.fallback_reason or 'none'}")


if __name__ == "__main__":
    main()
