"""Build offline soft geometry-routing labels from cached GRM correspondences."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .routing_head import (
    GRMGeometryRoutingConfig,
    build_geometry_routing_targets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sigma-h-px", type=float, default=3.0)
    parser.add_argument("--global-threshold-px", type=float, default=3.0)
    parser.add_argument("--max-deform-threshold-px", type=float, default=24.0)
    parser.add_argument("--local-consistency-threshold-px", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(records, list):
        raise ValueError("Input cache must be a list of dictionaries.")
    config = GRMGeometryRoutingConfig(
        sigma_h_px=args.sigma_h_px,
        global_residual_threshold_px=args.global_threshold_px,
        max_deform_threshold_px=args.max_deform_threshold_px,
        local_consistency_threshold_px=args.local_consistency_threshold_px,
    )
    output = []
    for index, record in enumerate(records):
        required = {
            "src_xy", "dst_xy", "H_ab", "p_match", "cycle_error",
            "uncertainty_score", "valid_mask", "selected_graph_tokens",
        }
        missing = sorted(required - set(record))
        if missing:
            raise KeyError(f"Record {index} is missing fields: {missing}")
        targets = build_geometry_routing_targets(
            record["src_xy"],
            record["dst_xy"],
            record["H_ab"],
            record["p_match"],
            record["cycle_error"],
            record["uncertainty_score"],
            record["valid_mask"],
            config=config,
        )
        labeled = dict(record)
        labeled.update(
            {
                "target_h": targets.target_h,
                "target_deform": targets.target_deform,
                "h_mask": targets.h_mask,
                "deform_mask": targets.deform_mask,
            }
        )
        output.append(labeled)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"saved={args.output} records={len(output)}")


if __name__ == "__main__":
    main()
