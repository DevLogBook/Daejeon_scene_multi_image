"""Joint Stage-1/2/3 fine-tuning contracts."""

from __future__ import annotations

from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader

from grm_matcher import GRMMatcher, GRMMatcherConfig
from grm_matcher.losses import GRMMatcherLoss
from grm_matcher.joint_training import (
    GRMJointMixedBatchLoader,
    apply_joint_checkpoint_gate,
    joint_source_name,
    merge_joint_validation_metrics,
)
from grm_matcher.train_matcher import (
    _build_stage_optimizer,
    _configure_stage,
    _initial_stage_best,
    _run_epoch,
    _update_stage_best_metrics,
    parse_args,
)


class _TinyLoader:
    def __init__(self, count: int, offset: int) -> None:
        self.count = count
        self.offset = offset

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        for index in range(self.count):
            yield {"sample_index": torch.tensor([self.offset + index])}


def _small_config() -> GRMMatcherConfig:
    return GRMMatcherConfig(
        input_height=32,
        input_width=32,
        max_source_queries=4,
        fixed_score_threshold=0.0,
        backbone_stage_blocks=(1, 1, 1),
        graph_layers=1,
        graph_neighbors=3,
    )


def test_joint_mixed_loader_preserves_homography_epoch_and_ratio() -> None:
    loader = GRMJointMixedBatchLoader(
        _TinyLoader(5, 0),
        _TinyLoader(1, 100),
        homography_batches=2,
        role_batches=1,
    )

    batches = list(loader)
    sources = [joint_source_name(batch) for batch in batches]

    assert len(loader) == 8
    assert sources.count("homography") == 5
    assert sources.count("role_augmented") == 3
    homography_indices = [
        int(batch["sample_index"].item())
        for batch in batches
        if joint_source_name(batch) == "homography"
    ]
    assert homography_indices == [0, 1, 2, 3, 4]


def test_joint_validation_uses_homography_matching_and_role_domain_f1() -> None:
    homography = {
        "candidate_recall": 0.99,
        "pixel_error": 0.9,
        "fine_top1_pck_3": 0.97,
        "role_f1": 0.2,
    }
    role = {
        "candidate_recall": 0.95,
        "pixel_error": 1.2,
        "fine_top1_pck_3": 0.90,
        "role_f1": 0.61,
    }

    merged = merge_joint_validation_metrics(homography, role)

    assert merged["candidate_recall"] == 0.99
    assert merged["pixel_error"] == 0.9
    assert merged["fine_top1_pck_3"] == 0.97
    assert merged["role_f1"] == 0.61
    assert merged["homography_role_f1"] == 0.2
    assert merged["role_augmented_candidate_recall"] == 0.95


def test_best_joint_requires_matching_quality_gate() -> None:
    baseline = {
        "candidate_recall": 0.99,
        "pixel_error": 1.0,
        "fine_top1_pck_3": 0.95,
        "role_f1": 0.50,
    }
    degraded = apply_joint_checkpoint_gate(
        {
            "candidate_recall": 0.97,
            "pixel_error": 1.0,
            "fine_top1_pck_3": 0.95,
            "role_f1": 0.80,
        },
        baseline,
        maximum_recall_drop=0.002,
        maximum_pixel_error_ratio=1.05,
        maximum_pck3_drop=0.005,
    )
    best = _initial_stage_best("joint")
    improved = _update_stage_best_metrics("joint", degraded, best)

    assert degraded["joint_eligible"] == 0.0
    assert "joint_score" not in improved
    assert "role_f1" in improved

    eligible = apply_joint_checkpoint_gate(
        {
            "candidate_recall": 0.989,
            "pixel_error": 1.02,
            "fine_top1_pck_3": 0.949,
            "role_f1": 0.62,
        },
        baseline,
        maximum_recall_drop=0.002,
        maximum_pixel_error_ratio=1.05,
        maximum_pck3_drop=0.005,
    )
    improved = _update_stage_best_metrics("joint", eligible, best)
    assert eligible["joint_eligible"] == 1.0
    assert improved["joint_score"] == "best_joint.pt"


def test_joint_optimizer_uses_discriminative_learning_rates() -> None:
    model = GRMMatcher(_small_config())
    args = parse_args(
        [
            "--data_dir",
            "unused",
            "--stage",
            "joint",
            "--joint_backbone_lr",
            "0",
            "--joint_coarse_lr",
            "1e-6",
            "--joint_refinement_lr",
            "3e-6",
            "--joint_geometry_lr",
            "1e-5",
        ]
    )
    _configure_stage(model, "joint", joint_backbone_trainable=False)
    optimizer = _build_stage_optimizer(model, args)
    learning_rates = {
        str(group["name"]): float(group["lr"])
        for group in optimizer.param_groups
    }

    assert learning_rates == {
        "coarse": 1.0e-6,
        "refinement": 3.0e-6,
        "geometry": 1.0e-5,
    }
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())


def test_detached_geometry_inputs_protect_matching_features_from_role_loss() -> None:
    torch.manual_seed(7)
    model = GRMMatcher(_small_config())
    _configure_stage(model, "joint", joint_backbone_trainable=False)
    image = torch.rand(1, 3, 32, 32)

    _, auxiliary = model(
        image,
        image,
        return_auxiliary=True,
        detach_geometry_inputs=True,
    )
    auxiliary.role_logits.sum().backward()

    assert model.role_head.classifier[-1].weight.grad is not None
    assert model.geometry_graph.layers[0].query.weight.grad is not None
    assert model.fpn.fine_lateral[0].weight.grad is None
    assert model.fine_refiner.correlation.descriptor_projection.weight.grad is None


def test_joint_mixed_epoch_runs_with_source_specific_losses() -> None:
    config = _small_config()
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    dense_mapping = torch.stack((xx, yy), dim=0).float()
    overlap = torch.ones(1, 32, 32)
    sample = {
        "image_a": torch.rand(3, 32, 32),
        "image_b": torch.rand(3, 32, 32),
        "H_ab": torch.eye(3),
        "gt_dense_mapping": dense_mapping,
        "gt_overlap_a": overlap,
        "gt_overlap_b": overlap,
        "gt_global_mask": overlap,
        "gt_local_mask": torch.zeros_like(overlap),
        "local_flow": torch.zeros(2, 32, 32),
    }
    mixed = GRMJointMixedBatchLoader(
        DataLoader([sample], batch_size=1),
        DataLoader([sample], batch_size=1),
        homography_batches=1,
        role_batches=1,
    )
    model = GRMMatcher(config)
    args = parse_args(["--data_dir", "unused", "--stage", "joint"])
    _configure_stage(model, "joint", joint_backbone_trainable=False)
    optimizer = _build_stage_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    with tempfile.TemporaryDirectory() as directory:
        metrics, global_step = _run_epoch(
            model,
            GRMMatcherLoss(config),
            mixed,
            optimizer,
            scaler,
            torch.device("cpu"),
            "joint",
            False,
            1.0,
            Path(directory),
            train=True,
            writer=None,
            global_step=0,
            epoch=0,
            progress_enabled=False,
            log_every=0,
            visualize_enabled=False,
            visualize_every=0,
            histogram_every=0,
            max_visualized_matches=10,
            max_consecutive_amp_overflows=0,
            min_amp_scale=1.0,
            detach_geometry_inputs=True,
        )

    assert global_step == 2
    assert torch.isfinite(torch.tensor(metrics["loss_total"]))
    assert metrics["joint_source_role_augmented"] == 0.5
    assert metrics["joint_geometry_inputs_detached"] == 1.0
    assert "fine_top1_pck_3" in metrics
    assert "rerank_accuracy" in metrics
