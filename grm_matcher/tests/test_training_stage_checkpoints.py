"""Stage-aware checkpoint restore and best-metric selection tests."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from grm_matcher.train_matcher import (
    _apply_checkpoint,
    _atomic_torch_save,
    _epoch_checkpoint_name,
    _initial_stage_best,
    _save_stage_best_snapshots,
    _stage_best_snapshot_name,
    _update_stage_best_metrics,
    parse_args,
)


def _checkpoint(stage: str = "stage1", epoch: int = 19) -> dict[str, object]:
    source_model = nn.Linear(3, 2)
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=1.0e-3)
    source_model(torch.ones(1, 3)).sum().backward()
    source_optimizer.step()
    return {
        "model": source_model.state_dict(),
        "optimizer": source_optimizer.state_dict(),
        "scaler": {},
        "stage": stage,
        "epoch": epoch,
        "best": {
            "candidate_recall": 0.75,
            "pixel_error": 8.0,
            "role_f1": 0.25,
        },
    }


def _target_state() -> tuple[nn.Module, torch.optim.Optimizer, torch.amp.GradScaler]:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, optimizer, scaler


def test_same_stage_resume_restores_epoch_optimizer_and_stage_best() -> None:
    model, optimizer, scaler = _target_state()
    start_epoch, best = _apply_checkpoint(
        _checkpoint(),
        model,
        optimizer,
        scaler,
        requested_stage="stage1",
        resume_training=True,
    )

    assert start_epoch == 20
    assert best == {"candidate_recall": 0.75}
    assert optimizer.state


def test_cross_stage_init_resets_epoch_optimizer_and_best() -> None:
    model, optimizer, scaler = _target_state()
    start_epoch, best = _apply_checkpoint(
        _checkpoint(stage="stage1", epoch=19),
        model,
        optimizer,
        scaler,
        requested_stage="stage2",
        resume_training=False,
    )

    assert start_epoch == 0
    assert best.keys() == {"pixel_error"}
    assert math.isinf(best["pixel_error"]) and best["pixel_error"] > 0
    assert not optimizer.state


def test_cross_stage_resume_is_rejected_with_actionable_message() -> None:
    model, optimizer, scaler = _target_state()
    with pytest.raises(ValueError, match="--init_checkpoint"):
        _apply_checkpoint(
            _checkpoint(stage="stage1"),
            model,
            optimizer,
            scaler,
            requested_stage="stage2",
            resume_training=True,
        )


@pytest.mark.parametrize(
    ("stage", "expected_metric", "expected_file"),
    [
        ("stage1", "candidate_recall", "best_candidate_recall.pt"),
        ("stage2", "pixel_error", "best_pixel_error.pt"),
        ("stage3", "role_f1", "best_role_f1.pt"),
    ],
)
def test_best_checkpoint_selection_is_stage_specific(
    stage: str, expected_metric: str, expected_file: str
) -> None:
    best = _initial_stage_best(stage)
    improved = _update_stage_best_metrics(
        stage,
        {"candidate_recall": 0.8, "pixel_error": 3.0, "role_f1": 0.6},
        best,
    )

    assert improved == {expected_metric: expected_file}
    assert best == {
        expected_metric: {
            "candidate_recall": 0.8,
            "pixel_error": 3.0,
            "role_f1": 0.6,
        }[expected_metric]
    }


def test_resume_and_init_checkpoint_cli_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--data_dir",
                "images",
                "--resume",
                "same_stage.pt",
                "--init_checkpoint",
                "previous_stage.pt",
            ]
        )


def test_epoch_checkpoint_cli_defaults() -> None:
    args = parse_args(["--data_dir", "images"])

    assert args.save_every_epoch
    assert args.best_snapshot_interval == 10


def test_epoch_and_best_snapshot_names_are_one_based() -> None:
    assert _epoch_checkpoint_name(1) == "epoch_0001.pt"
    assert _epoch_checkpoint_name(10) == "epoch_0010.pt"
    assert (
        _stage_best_snapshot_name("best_candidate_recall.pt", 20)
        == "best_candidate_recall_epoch_0020.pt"
    )


def test_stage_best_milestone_snapshot_is_an_independent_checkpoint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "best_candidate_recall.pt"
    _atomic_torch_save(
        {"stage": "stage1", "epoch": 7, "value": torch.tensor([3.0])},
        source,
    )

    saved = _save_stage_best_snapshots(tmp_path, "stage1", epoch_number=10)

    assert saved == [tmp_path / "best_candidate_recall_epoch_0010.pt"]
    assert saved[0].is_file()
    loaded = torch.load(saved[0], map_location="cpu", weights_only=False)
    assert loaded["epoch"] == 7
    assert torch.equal(loaded["value"], torch.tensor([3.0]))
    assert not list(tmp_path.glob("*.tmp"))
