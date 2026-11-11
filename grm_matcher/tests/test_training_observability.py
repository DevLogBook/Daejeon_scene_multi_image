"""Training progress, visualization, and TensorBoard logging tests."""

from __future__ import annotations

from pathlib import Path
import warnings

from matplotlib import pyplot as plt
import torch
from torch import nn

from grm_matcher.config import GRMMatcherConfig
from grm_matcher.matcher import GRMMatcher
from grm_matcher.train_matcher import (
    _amp_overflow_is_fatal,
    _is_log_step,
    _log_parameter_histograms,
    _log_run_metadata,
    _load_role_data_config,
    _write_training_visualizations,
    parse_args,
)


class _RecordingWriter:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.scalars: list[str] = []
        self.histograms: list[str] = []
        self.figures: list[str] = []
        self.hparams: dict[str, object] = {}

    def add_text(self, tag: str, *_: object) -> None:
        self.texts.append(tag)

    def add_scalar(self, tag: str, *_: object) -> None:
        self.scalars.append(tag)

    def add_histogram(self, tag: str, *_: object) -> None:
        self.histograms.append(tag)

    def add_hparams(self, values: dict[str, object], *_: object) -> None:
        self.hparams.update(values)

    def add_figure(
        self, tag: str, figure: object, *_: object, close: bool = False
    ) -> None:
        self.figures.append(tag)
        if close:
            plt.close(figure)


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


def test_training_cli_exposes_observability_controls() -> None:
    args = parse_args(
        [
            "--data_dir",
            "images",
            "--no-progress",
            "--no-visualize",
            "--log_every",
            "7",
            "--visualize_every",
            "13",
            "--histogram_every",
            "17",
            "--max_visualized_matches",
            "23",
        ]
    )

    assert not args.progress
    assert not args.visualize
    assert args.log_every == 7
    assert args.visualize_every == 13
    assert args.histogram_every == 17
    assert args.max_visualized_matches == 23
    assert args.amp_init_scale == 8192.0
    assert args.amp_growth_interval == 2000
    assert args.max_consecutive_amp_overflows == 16
    assert args.min_amp_scale == 1.0
    assert _is_log_step(14, 7)
    assert not _is_log_step(14, 0)


def test_stage3_role_data_config_uses_matcher_coordinate_contract() -> None:
    matcher_config = _small_config()
    role_config = _load_role_data_config(None, matcher_config, seed=71)
    assert role_config.output_height == matcher_config.input_height
    assert role_config.output_width == matcher_config.input_width
    assert role_config.global_residual_threshold == (
        matcher_config.global_residual_threshold
    )
    assert role_config.local_residual_threshold == (
        matcher_config.local_residual_threshold
    )
    assert role_config.seed == 71
    assert role_config.local_flow_max_pixels < (
        matcher_config.local_residual_threshold
    )


def test_tensorboard_metadata_and_parameter_histograms_are_recorded() -> None:
    writer = _RecordingWriter()
    model = GRMMatcher(_small_config())
    args = parse_args(["--data_dir", "images", "--stage", "stage1"])
    _log_run_metadata(
        writer,
        model,
        model.config,
        args,
        amp_enabled=False,
        device=torch.device("cpu"),
    )

    layer = nn.Linear(3, 2)
    layer(torch.ones(1, 3)).sum().backward()
    _log_parameter_histograms(writer, layer, global_step=5)  # type: ignore[arg-type]

    assert "run/configuration" in writer.texts
    assert "run/parameters_total" in writer.scalars
    assert "run/parameters_trainable" in writer.scalars
    assert writer.hparams["stage"] == "stage1"
    assert any(tag.startswith("parameters/") for tag in writer.histograms)
    assert any(tag.startswith("gradients/") for tag in writer.histograms)
    assert any(tag.startswith("parameter_norm/") for tag in writer.scalars)
    assert any(tag.startswith("gradient_norm/") for tag in writer.scalars)


def test_training_visualizations_are_saved_and_sent_to_tensorboard(
    tmp_path: Path,
) -> None:
    torch.manual_seed(5)
    model = GRMMatcher(_small_config()).eval()
    image_a = torch.rand(1, 3, 32, 32)
    image_b = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        output, auxiliary = model(image_a, image_b, return_auxiliary=True)
    writer = _RecordingWriter()
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    dense_mapping = torch.stack((xx, yy), dim=0).float().unsqueeze(0)
    overlap = torch.ones(1, 1, 32, 32, dtype=torch.bool)

    _write_training_visualizations(
        output,
        auxiliary,
        {
            "image_a": image_a,
            "image_b": image_b,
            "gt_dense_mapping": dense_mapping,
            "gt_overlap_a": overlap,
            "gt_global_mask": overlap,
            "gt_local_mask": torch.zeros_like(overlap),
        },
        tmp_path,
        split="train",
        step=3,
        writer=writer,
        max_matches=4,
    )

    image_paths = list((tmp_path / "visualizations" / "train").glob("*.png"))
    assert len(image_paths) == 5
    assert len(writer.figures) == 5
    assert any(path.name.endswith("_role_gt.png") for path in image_paths)
    assert all(path.stat().st_size > 0 for path in image_paths)


def test_amp_overflow_policy_allows_normal_scale_backoff() -> None:
    assert not _amp_overflow_is_fatal(3, 1024.0, 16, 1.0)
    assert not _amp_overflow_is_fatal(15, 1.0, 16, 1.0)
    assert _amp_overflow_is_fatal(16, 1.0, 16, 1.0)
    assert not _amp_overflow_is_fatal(100, 0.25, 0, 1.0)


def test_visualization_detaches_training_tensors(tmp_path: Path) -> None:
    from grm_matcher.visualization import visualize_matches

    image_a = torch.rand(1, 3, 16, 16)
    image_b = torch.rand(1, 3, 16, 16)
    src_xy = torch.tensor([[[2.0, 3.0], [8.0, 9.0]]], requires_grad=True)
    dst_xy = torch.tensor([[[3.0, 3.0], [9.0, 9.0]]], requires_grad=True)
    probabilities = torch.tensor([[0.8, 0.6]], requires_grad=True)
    roles = torch.softmax(torch.randn(1, 2, 3, requires_grad=True), dim=-1)
    entropy = torch.tensor([[0.2, 0.7]], requires_grad=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figure = visualize_matches(
            image_a,
            image_b,
            src_xy,
            dst_xy,
            probabilities,
            roles[..., 0],
            roles[..., 1],
            roles[..., 2],
            entropy,
            torch.ones(1, 2, dtype=torch.bool),
            save_path=tmp_path / "requires_grad.png",
        )
        plt.close(figure)

    assert not any("requires_grad=True" in str(item.message) for item in caught)
