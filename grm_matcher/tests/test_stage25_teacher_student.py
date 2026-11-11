"""Stage-2.5 real-pair Teacher/Student contract tests."""

from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np
import torch

from grm_matcher import GRMMatcher, GRMMatcherConfig
from grm_matcher.real_pair_dataset import GRMRealPairDataset
from grm_matcher.local_correlation import GRMRefinementOutput
from grm_matcher.stage25 import (
    GRMStage25Config,
    GRMStage25ConsistencyLoss,
    GRMStage25PseudoLabels,
    GRMStage25StudentOutput,
    build_dense_mapping_supervision_labels,
    build_homography_supervision_labels,
    forward_student_refinement,
    generate_teacher_pseudo_labels,
    make_ema_teacher,
    make_student_augmentation,
    transform_points,
    transform_pseudo_labels,
    update_ema_teacher,
)
from grm_matcher.train_matcher import _configure_stage
from grm_matcher.train_stage25 import (
    _checkpoint_state,
    _configure_real_phase,
    _make_optimizer,
    _run_controlled_geometry_epoch,
    _run_epoch,
)
from grm_matcher.visualization import visualize_stage25_refinement


def _small_matcher_config() -> GRMMatcherConfig:
    return GRMMatcherConfig(
        input_height=32,
        input_width=32,
        max_source_queries=4,
        fixed_score_threshold=0.0,
        backbone_stage_blocks=(1, 1, 1),
        graph_layers=1,
        graph_neighbors=3,
    )


def _permissive_stage25_config(**changes: object) -> GRMStage25Config:
    values: dict[str, object] = {
        "min_pseudo_matches": 1,
        "max_candidate_normalized_entropy": 1.0,
        "min_mid_peak_probability": 0.0,
        "min_fine_peak_probability": 0.0,
        "max_cycle_error": 1.0e6,
        "max_fine_std": 1.0e6,
        "crop_scale_min": 1.0,
        "horizontal_flip_probability": 0.0,
        "brightness_jitter": 0.0,
        "contrast_jitter": 0.0,
        "saturation_jitter": 0.0,
        "gamma_jitter": 0.0,
        "noise_std": 0.0,
        "blur_probability": 0.0,
    }
    values.update(changes)
    return GRMStage25Config(**values)


def test_transform_points_and_identity_student_augmentation() -> None:
    points = torch.tensor([[[2.0, 3.0], [10.0, 7.0]]])
    horizontal_flip = torch.tensor(
        [[[-1.0, 0.0, 31.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    transformed = transform_points(points, horizontal_flip)
    assert torch.allclose(transformed, torch.tensor([[[29.0, 3.0], [21.0, 7.0]]]))

    config = _small_matcher_config()
    stage25 = _permissive_stage25_config()
    image_a = torch.rand(2, 3, 32, 32)
    image_b = torch.rand(2, 3, 32, 32)
    augmented = make_student_augmentation(image_a, image_b, config, stage25)
    identity = torch.eye(3)[None].expand(2, -1, -1)
    assert torch.allclose(augmented.transform_a, identity)
    assert torch.allclose(augmented.transform_b, identity)
    assert torch.allclose(augmented.image_a, image_a, atol=1.0e-5)
    assert torch.allclose(augmented.image_b, image_b, atol=1.0e-5)


def test_ema_teacher_update_is_named_and_deterministic() -> None:
    student = torch.nn.Linear(2, 1, bias=False)
    teacher = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        student.weight.fill_(3.0)
        teacher.weight.fill_(1.0)
    update_ema_teacher(teacher, student, decay=0.75)
    assert torch.allclose(teacher.weight, torch.full_like(teacher.weight, 1.5))


def test_teacher_generation_bypasses_stage3_and_student_gradients_are_scoped() -> None:
    torch.manual_seed(19)
    matcher_config = _small_matcher_config()
    stage25_config = _permissive_stage25_config()
    student = GRMMatcher(matcher_config)
    _configure_stage(student, "stage2")
    teacher = make_ema_teacher(student)

    def forbidden(*_: object, **__: object) -> object:
        raise AssertionError("Stage-3 module must not run during Stage 2.5.")

    teacher.seed_geometry.forward = forbidden  # type: ignore[method-assign]
    teacher.geometry_graph.forward = forbidden  # type: ignore[method-assign]
    teacher.rerank_head.forward = forbidden  # type: ignore[method-assign]
    teacher.role_head.forward = forbidden  # type: ignore[method-assign]
    image = torch.rand(1, 3, 32, 32)
    valid = torch.ones(1, 1, 32, 32, dtype=torch.bool)
    labels = generate_teacher_pseudo_labels(
        teacher, image, image, stage25_config, valid, valid
    )
    augmented = make_student_augmentation(
        image, image, matcher_config, stage25_config
    )
    labels = transform_pseudo_labels(
        labels,
        augmented.transform_a,
        augmented.transform_b,
        32,
        32,
        matcher_config,
        stage25_config,
    )
    prediction = forward_student_refinement(
        student, augmented.image_a, augmented.image_b, labels
    )
    losses = GRMStage25ConsistencyLoss(matcher_config, stage25_config)(
        prediction, labels
    )
    assert torch.isfinite(losses["total"])
    assert int(losses["valid_count"].item()) > 0
    losses["total"].backward()
    assert student.fpn.mid_lateral[0].weight.grad is not None
    assert student.fpn.fine_lateral[0].weight.grad is not None
    assert student.mid_refiner.correlation.descriptor_projection.weight.grad is not None
    assert student.fine_refiner.correlation.descriptor_projection.weight.grad is not None
    assert student.backbone.stem[0].weight.grad is None
    assert student.fpn.coarse_lateral[0].weight.grad is None
    assert student.coarse_transformer.layers[0].attention.in_proj_weight.grad is None


def test_transform_pseudo_labels_does_not_pad_by_duplication() -> None:
    matcher_config = _small_matcher_config()
    stage25_config = _permissive_stage25_config(min_pseudo_matches=2)
    labels = GRMStage25PseudoLabels(
        src_xy=torch.tensor([[[4.0, 4.0], [28.0, 28.0], [31.0, 31.0]]]),
        coarse_dst_xy=torch.tensor([[[4.0, 4.0], [28.0, 28.0], [31.0, 31.0]]]),
        mid_dst_xy=torch.tensor([[[4.0, 4.0], [28.0, 28.0], [31.0, 31.0]]]),
        fine_dst_xy=torch.tensor([[[4.0, 4.0], [28.0, 28.0], [31.0, 31.0]]]),
        confidence=torch.ones(1, 3),
        cycle_error=torch.zeros(1, 3),
        fine_std=torch.ones(1, 3),
        valid_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    translation = torch.tensor(
        [[[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]]
    )
    transformed = transform_pseudo_labels(
        labels,
        translation,
        translation,
        32,
        32,
        matcher_config,
        stage25_config,
    )
    assert transformed.valid_mask.tolist() == [[True, True, False]]
    assert transformed.confidence.tolist() == [[1.0, 1.0, 0.0]]


def _dummy_refinement(dst_xy: torch.Tensor, window: int) -> GRMRefinementOutput:
    batch, queries, _ = dst_xy.shape
    probability = torch.full(
        (batch, queries, window, window),
        1.0 / float(window * window),
        dtype=dst_xy.dtype,
        device=dst_xy.device,
    )
    covariance = torch.eye(2, dtype=dst_xy.dtype)[None, None].expand(
        batch, queries, -1, -1
    )
    descriptor = torch.zeros(batch, queries, 4, dtype=dst_xy.dtype)
    return GRMRefinementOutput(
        src_xy=torch.zeros_like(dst_xy),
        dst_xy=dst_xy,
        delta_feature_xy=torch.zeros_like(dst_xy),
        local_probability=probability,
        entropy=torch.zeros(batch, queries),
        covariance_feature=covariance,
        covariance_image=covariance,
        peak_probability=probability.flatten(-2).amax(dim=-1),
        source_descriptor=descriptor,
        target_descriptor=descriptor,
        valid_mask=torch.ones(batch, queries, dtype=torch.bool),
    )


def test_absolute_batch_reliability_scales_low_confidence_loss() -> None:
    matcher_config = _small_matcher_config()
    stage25_config = _permissive_stage25_config(
        mid_coordinate_weight=0.0,
        fine_coordinate_weight=1.0,
        mid_nll_weight=0.0,
        fine_nll_weight=0.0,
        absolute_confidence_reference=0.01,
        reliable_match_count_reference=1,
        reliable_region_count_reference=1,
    )
    predicted = torch.tensor([[[2.0, 0.0]]], requires_grad=True)
    student = GRMStage25StudentOutput(
        mid=_dummy_refinement(predicted, matcher_config.mid_window_size),
        fine=_dummy_refinement(predicted, matcher_config.fine_window_size),
    )

    def labels(confidence: float) -> GRMStage25PseudoLabels:
        target = torch.zeros(1, 1, 2)
        return GRMStage25PseudoLabels(
            src_xy=target,
            coarse_dst_xy=target,
            mid_dst_xy=target,
            fine_dst_xy=target,
            confidence=torch.tensor([[confidence]]),
            cycle_error=torch.zeros(1, 1),
            fine_std=torch.zeros(1, 1),
            valid_mask=torch.ones(1, 1, dtype=torch.bool),
            region_ids=torch.zeros(1, 1, dtype=torch.long),
        )

    loss = GRMStage25ConsistencyLoss(matcher_config, stage25_config)
    high = loss(student, labels(0.01))
    low = loss(student, labels(0.001))
    assert torch.allclose(high["batch_reliability"], torch.tensor(1.0))
    assert torch.allclose(low["batch_reliability"], torch.tensor(0.1))
    assert torch.allclose(low["total"], high["total"] * 0.1, atol=1.0e-6)


def test_homography_supervision_uses_a_to_b_and_checks_reverse_cycle() -> None:
    source = torch.tensor([[[4.0, 5.0], [10.0, 12.0]]])
    labels = GRMStage25PseudoLabels(
        src_xy=source,
        coarse_dst_xy=source.clone(),
        mid_dst_xy=source.clone(),
        fine_dst_xy=source.clone(),
        confidence=torch.ones(1, 2),
        cycle_error=torch.zeros(1, 2),
        fine_std=torch.zeros(1, 2),
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        base_valid_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    h_ab = torch.tensor(
        [[[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]]]
    )
    h_ba = torch.linalg.inv(h_ab)
    supervised = build_homography_supervision_labels(
        labels, h_ab, h_ba, torch.tensor([True]), 0.1
    )
    assert torch.allclose(
        supervised.fine_dst_xy,
        source + torch.tensor([3.0, -2.0]),
    )
    assert supervised.valid_mask.all()
    assert supervised.cycle_error.max() < 1.0e-5


def test_dense_mapping_supervision_preserves_local_flow() -> None:
    source = torch.tensor([[[4.0, 5.0], [10.0, 12.0]]])
    labels = GRMStage25PseudoLabels(
        src_xy=source,
        coarse_dst_xy=source.clone(),
        mid_dst_xy=source.clone(),
        fine_dst_xy=source.clone(),
        confidence=torch.ones(1, 2),
        cycle_error=torch.ones(1, 2),
        fine_std=torch.ones(1, 2),
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        base_valid_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    dense_mapping = torch.stack((xx + 2, yy - 1), dim=0).float().unsqueeze(0)
    overlap = torch.ones(1, 1, 16, 16, dtype=torch.bool)
    supervised = build_dense_mapping_supervision_labels(
        labels, dense_mapping, overlap, overlap
    )
    torch.testing.assert_close(
        supervised.fine_dst_xy,
        source + torch.tensor([2.0, -1.0]),
        rtol=0.0,
        atol=1.0e-5,
    )
    assert supervised.valid_mask.all()
    assert torch.count_nonzero(supervised.cycle_error) == 0
    assert torch.count_nonzero(supervised.fine_std) == 0


def test_stage25a_fine_scope_only_unfreezes_fine_refiner() -> None:
    model = GRMMatcher(_small_matcher_config())
    _configure_real_phase(model, "stage2.5a", "fine")
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith("fine_refiner.") for name in trainable)


def test_fixed_teacher_checkpoint_exports_adapted_student() -> None:
    student = GRMMatcher(_small_matcher_config())
    teacher = make_ema_teacher(student)
    _configure_real_phase(student, "stage2.5a", "fine")
    with torch.no_grad():
        next(student.fine_refiner.parameters()).add_(1.0)
    optimizer = _make_optimizer(student, "stage2.5a", 1.0e-5, 1.0e-6, 0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    state = _checkpoint_state(
        student,
        teacher,
        optimizer,
        scaler,
        _small_matcher_config(),
        _permissive_stage25_config(),
        "stage2.5a",
        0,
        1,
        0.0,
        {},
        {},
        "fixed",
        "fine",
    )
    parameter_name = next(
        name for name in state["student"] if name.startswith("fine_refiner.")
    )
    assert torch.equal(state["model"][parameter_name], state["student"][parameter_name])
    assert not torch.equal(state["model"][parameter_name], state["teacher"][parameter_name])


def test_stage25a_v2_combines_homography_and_source_replay(tmp_path: Path) -> None:
    matcher_config = _small_matcher_config()
    stage25_config = _permissive_stage25_config(
        absolute_confidence_reference=1.0e-8,
        reliable_match_count_reference=1,
        reliable_region_count_reference=1,
    )
    student = GRMMatcher(matcher_config)
    _configure_real_phase(student, "stage2.5a", "fine")
    teacher = make_ema_teacher(student)
    loss_module = GRMStage25ConsistencyLoss(matcher_config, stage25_config)
    optimizer = _make_optimizer(student, "stage2.5a", 1.0e-5, 1.0e-6, 0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    image = torch.rand(1, 3, 32, 32)
    mask = torch.ones(1, 1, 32, 32, dtype=torch.bool)
    identity = torch.eye(3).unsqueeze(0)
    real_batch = {
        "image_a": image,
        "image_b": image.clone(),
        "valid_a": mask,
        "valid_b": mask,
        "overlap_a": mask,
        "overlap_b": mask,
        "overlap_usable_a": torch.tensor([True]),
        "overlap_usable_b": torch.tensor([True]),
        "H_ab": identity,
        "H_ba": identity,
        "homography_valid": torch.tensor([True]),
        "pair_bucket": ["global"],
        "pair_id": ["real"],
    }
    source_batch = {
        "image_a": image,
        "image_b": image.clone(),
        "valid_a": mask,
        "valid_b": mask,
        "gt_overlap_a": mask,
        "gt_overlap_b": mask,
        "H_ab": identity,
        "gt_dense_mapping": torch.stack(
            torch.meshgrid(
                torch.arange(32), torch.arange(32), indexing="xy"
            ),
            dim=0,
        ).float().unsqueeze(0),
    }
    metrics, global_step = _run_epoch(
        student,
        teacher,
        loss_module,
        [real_batch],  # type: ignore[arg-type]
        optimizer,
        scaler,
        torch.device("cpu"),
        matcher_config,
        stage25_config,
        "stage2.5a",
        False,
        1.0,
        20,
        tmp_path,
        train=True,
        epoch=0,
        global_step=0,
        writer=None,
        log_every=0,
        progress_enabled=False,
        source_loader=[source_batch],  # type: ignore[arg-type]
        source_replay_interval=1,
        teacher_mode="fixed",
    )
    assert global_step == 1
    assert np.isfinite(metrics["loss_total"])
    assert metrics["loss_homography_valid_count"] > 0
    assert metrics["loss_source_replay_valid_count"] > 0
    assert all(parameter.grad is None for parameter in student.mid_refiner.parameters())
    assert any(parameter.grad is not None for parameter in student.fine_refiner.parameters())


def test_controlled_only_epoch_uses_exact_dense_supervision(tmp_path: Path) -> None:
    matcher_config = _small_matcher_config()
    stage25_config = _permissive_stage25_config()
    student = GRMMatcher(matcher_config)
    _configure_real_phase(student, "stage2.5a", "fine")
    teacher = make_ema_teacher(student)
    loss_module = GRMStage25ConsistencyLoss(matcher_config, stage25_config)
    optimizer = _make_optimizer(student, "stage2.5a", 1.0e-5, 1.0e-6, 0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    image = torch.rand(1, 3, 32, 32)
    mask = torch.ones(1, 1, 32, 32, dtype=torch.bool)
    dense_mapping = torch.stack(
        torch.meshgrid(torch.arange(32), torch.arange(32), indexing="xy"),
        dim=0,
    ).float().unsqueeze(0)
    batch = {
        "image_a": image,
        "image_b": image.clone(),
        "valid_a": mask,
        "valid_b": mask,
        "gt_overlap_a": mask,
        "gt_overlap_b": mask,
        "gt_dense_mapping": dense_mapping,
        "sample_index": torch.tensor([0]),
    }
    metrics, global_step = _run_controlled_geometry_epoch(
        student,
        teacher,
        loss_module,
        [batch],  # type: ignore[arg-type]
        optimizer,
        scaler,
        torch.device("cpu"),
        matcher_config,
        stage25_config,
        False,
        1.0,
        20,
        tmp_path,
        train=True,
        epoch=0,
        global_step=0,
        writer=None,
        log_every=0,
        progress_enabled=False,
        teacher_mode="fixed",
        visualize_every=0,
        visualize_max_matches=16,
    )
    assert global_step == 1
    assert np.isfinite(metrics["loss_total"])
    assert metrics["exact_count"] > 0
    assert all(parameter.grad is None for parameter in student.mid_refiner.parameters())
    assert any(parameter.grad is not None for parameter in student.fine_refiner.parameters())


def test_stage25_refinement_visualization_detaches_tensors(tmp_path: Path) -> None:
    points = torch.tensor([[[4.0, 4.0], [12.0, 12.0]]], requires_grad=True)
    probability = torch.softmax(torch.randn(1, 2, 5, 5), dim=-1)
    path = tmp_path / "stage25.png"
    figure = visualize_stage25_refinement(
        torch.rand(1, 3, 32, 32),
        torch.rand(1, 3, 32, 32),
        points,
        points,
        points,
        points,
        points,
        probability,
        probability,
        torch.ones(1, 2, dtype=torch.bool),
        save_path=path,
    )
    assert path.is_file()
    assert len(figure.axes) >= 4


def test_real_pair_dataset_scans_matching_input_names(tmp_path: Path) -> None:
    split = tmp_path / "train"
    (split / "input1").mkdir(parents=True)
    (split / "input2").mkdir(parents=True)
    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(split / "input1" / "pair.png"), image)
    assert cv2.imwrite(str(split / "input2" / "pair.png"), image)
    dataset = GRMRealPairDataset(tmp_path, "train", _small_matcher_config())
    sample = dataset[0]
    assert len(dataset) == 1
    assert sample["image_a"].shape == (3, 32, 32)
    assert sample["valid_a"].all()
    assert not bool(sample["overlap_known"])
    assert dataset.bucket_to_indices == {"32x32": [0]}


def test_real_pair_dataset_uses_exported_bucket_not_mask_guess(tmp_path: Path) -> None:
    split = tmp_path / "train"
    for directory in ("input1", "input2", "labels", "masks/overlap1", "masks/overlap2"):
        (split / directory).mkdir(parents=True)
    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    overlap = np.zeros((32, 32), dtype=np.uint8)
    overlap[:, :16] = 255
    for name in ("simple.png", "hard.png", "negative.png"):
        assert cv2.imwrite(str(split / "input1" / name), image)
        assert cv2.imwrite(str(split / "input2" / name), image)
    assert cv2.imwrite(str(split / "masks/overlap1/simple.png"), overlap)
    assert cv2.imwrite(str(split / "masks/overlap1/hard.png"), overlap)
    assert cv2.imwrite(str(split / "masks/overlap2/simple.png"), overlap)
    assert cv2.imwrite(str(split / "masks/overlap2/hard.png"), overlap)
    assert cv2.imwrite(
        str(split / "masks/overlap1/negative.png"),
        np.zeros((32, 32), dtype=np.uint8),
    )
    assert cv2.imwrite(
        str(split / "masks/overlap2/negative.png"),
        np.zeros((32, 32), dtype=np.uint8),
    )
    (split / "labels/simple.json").write_text(
        '{"bucket":"global","label":1}', encoding="utf-8"
    )
    (split / "labels/hard.json").write_text(
        '{"bucket":"hard_positive","label":1}', encoding="utf-8"
    )
    (split / "labels/negative.json").write_text(
        '{"bucket":"negative","label":0}', encoding="utf-8"
    )
    config = _small_matcher_config()
    simple = GRMRealPairDataset(tmp_path, "train", config, difficulty="simple")
    hard = GRMRealPairDataset(tmp_path, "train", config, difficulty="hard")
    negative = GRMRealPairDataset(tmp_path, "train", config, difficulty="negative")
    assert len(simple) == len(hard) == len(negative) == 1
    assert bool(simple[0]["overlap_known"])
    assert bool(simple[0]["overlap_usable"])
    assert simple[0]["pair_bucket"] == "global"
    assert int(simple[0]["overlap_a"].sum()) == 32 * 16
    assert bool(hard[0]["overlap_known"])
    assert bool(hard[0]["overlap_usable"])
    assert hard[0]["pair_bucket"] == "hard_positive"
    assert negative[0]["pair_bucket"] == "negative"
    assert int(negative[0]["pair_label"]) == 0
    assert not negative[0]["overlap_a"].any()
    assert float(negative[0]["overlap_fraction"]) == 0.0


def test_real_pair_dataset_reads_stage06_manifest_and_both_overlap_masks(
    tmp_path: Path,
) -> None:
    split = tmp_path / "train"
    for directory in (
        "input1",
        "input2",
        "masks/valid1",
        "masks/valid2",
        "masks/overlap1",
        "masks/overlap2",
        "homography",
        "homography_ba",
    ):
        (split / directory).mkdir(parents=True)
    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    full = np.full((32, 32), 255, dtype=np.uint8)
    records = []
    for stem, bucket, label, overlap_value in (
        ("positive", "global", 1, full),
        ("negative", "negative", 0, np.zeros_like(full)),
    ):
        assert cv2.imwrite(str(split / "input1" / f"{stem}.jpg"), image)
        assert cv2.imwrite(str(split / "input2" / f"{stem}.jpg"), image)
        for directory in ("valid1", "valid2"):
            assert cv2.imwrite(str(split / "masks" / directory / f"{stem}.png"), full)
        for directory in ("overlap1", "overlap2"):
            assert cv2.imwrite(
                str(split / "masks" / directory / f"{stem}.png"), overlap_value
            )
        np.save(
            split / "homography" / f"{stem}.npy",
            np.eye(3, dtype=np.float64),
            allow_pickle=False,
        )
        np.save(
            split / "homography_ba" / f"{stem}.npy",
            np.eye(3, dtype=np.float64),
            allow_pickle=False,
        )
        records.append(
            {
                "schema": "real_stitch_export_v1",
                "pair_id": stem,
                "image_a": f"input1/{stem}.jpg",
                "image_b": f"input2/{stem}.jpg",
                "valid_a": f"masks/valid1/{stem}.png",
                "valid_b": f"masks/valid2/{stem}.png",
                "overlap_a": f"masks/overlap1/{stem}.png",
                "overlap_b": f"masks/overlap2/{stem}.png",
                "h_ab": f"homography/{stem}.npy",
                "h_ba": f"homography_ba/{stem}.npy",
                "bucket": bucket,
                "label": label,
                "target_bucket": "32x32",
            }
        )
    (split / "manifest.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    dataset = GRMRealPairDataset(
        tmp_path,
        "train",
        _small_matcher_config(),
        selected_pair_buckets=("global",),
    )
    sample = dataset[0]
    assert len(dataset) == 1
    assert dataset.pair_bucket_counts == {"global": 1}
    assert sample["pair_bucket"] == "global"
    assert int(sample["pair_label"]) == 1
    assert sample["overlap_a"].all()
    assert sample["overlap_b"].all()
    assert bool(sample["homography_valid"])
    assert torch.allclose(sample["H_ab"], torch.eye(3))
    assert torch.allclose(sample["H_ba"], torch.eye(3))
