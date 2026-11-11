"""Tests for true scratch initialization, exact-H resizing and LR scheduling."""
import json

import cv2
import numpy as np
import pytest
import torch

from grm_matcher.config import GRMMatcherConfig
from grm_matcher.matcher import GRMMatcher
from grm_matcher.scratch_transfer import (
    GEOMETRY_PREFIXES, ResizedHomographyDataset, ScratchTransferObjective,
    check_homography_splits, configure_optimizer, lr_multiplier,
)
from grm_matcher.train_scratch_transfer import parse_args, run_epoch


def small_config():
    return GRMMatcherConfig(input_height=64, input_width=64, max_source_queries=16,
                            backbone_stage_blocks=(1, 1, 1), graph_layers=1)


def test_warmup_cosine_endpoints():
    values = [lr_multiplier(i, 100, 10) for i in range(100)]
    assert values[0] == pytest.approx(.1)
    assert values[9] == pytest.approx(1.) and values[10] == pytest.approx(1.)
    assert values[-1] == pytest.approx(.05)
    assert all(a <= b for a, b in zip(values[:9], values[1:10]))
    assert all(a >= b for a, b in zip(values[10:-1], values[11:]))
    assert lr_multiplier(150, 100, 10) == pytest.approx(.05)


def test_scratch_disallows_warm_start_and_checks_agri_source():
    with pytest.raises(SystemExit):
        parse_args(["--stage", "imc", "--data-dir", "x", "--output-dir", "y", "--init-checkpoint", "old.pt"])
    with pytest.raises(SystemExit):
        parse_args(["--stage", "agri", "--data-dir", "x", "--output-dir", "y"])
    args = parse_args(["--stage", "imc", "--data-dir", "x", "--output-dir", "y"])
    assert args.epochs == 30 and args.warmup_epochs == 2 and args.accumulation_steps == 8


def test_scratch_all_matching_parameters_unfrozen():
    model = GRMMatcher(small_config())
    opt = configure_optimizer(model, "imc")
    assert all(p.requires_grad for n, p in model.named_parameters() if not n.startswith(GEOMETRY_PREFIXES))
    assert all(not p.requires_grad for n, p in model.named_parameters() if n.startswith(GEOMETRY_PREFIXES))
    assert opt.param_groups[0]["peak_lr"] == 2e-4
    opt = configure_optimizer(model, "agri")
    assert all(p.requires_grad for p in model.parameters())
    assert {g["name"]: g["peak_lr"] for g in opt.param_groups} == {
        "backbone": 1e-5, "coarse": 2e-5, "refinement": 5e-5, "geometry": 1e-4}


def test_exact_h_resizing_and_source_isolation(tmp_path):
    config = small_config()
    for split, source in (("train", "apple.jpg"), ("val", "pear.jpg"), ("test", "plum.jpg")):
        folder = tmp_path / split
        row = {"source_image": source, "direction": "A_to_B", "target_size": [64, 64], "target_bucket": "64x64"}
        for key, directory in (("image_a", "input1"), ("image_b", "input2"),
                               ("valid_a", "masks/valid1"), ("valid_b", "masks/valid2"), ("overlap_a", "masks/overlap1")):
            path = folder / directory / "pair.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), np.full((64, 64, 3) if key.startswith("image") else (64, 64), 255, dtype=np.uint8))
            row[key] = str(path)
        h_path = folder / "homography/pair.npy"
        h_path.parent.mkdir()
        np.save(h_path, np.array([[1., 0., 8.], [0., 1., 0.], [0., 0., 1.]]))
        row["h_gt_a_to_b"] = str(h_path)
        (folder / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    counts = check_homography_splits(tmp_path)
    assert counts["train"]["pairs"] == 1
    sample = ResizedHomographyDataset(tmp_path, "train", config, size=32)[0]
    assert sample["image_a"].shape == (3, 32, 32)
    assert sample["H_ab"][0, 2] == 4
    assert torch.allclose(sample["gt_dense_mapping"][:, 10, 10], torch.tensor([14., 10.]))
    assert sample["gt_overlap_a"][0, 10, 10]
    assert not sample["gt_overlap_a"][0, 10, 31]
    assert not sample["gt_local_mask"].any()
    path = tmp_path / "val/manifest.jsonl"
    text = path.read_text(encoding="utf-8").replace("pear.jpg", "apple.jpg")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        check_homography_splits(tmp_path)


def test_imc_from_random_has_backbone_gradients_and_direct_f_gradients():
    torch.manual_seed(5)
    config = small_config()
    model = GRMMatcher(config)
    configure_optimizer(model, "imc")
    y, x = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
    points = (torch.stack((x, y), -1).float().reshape(1, -1, 2) + .5) * 16 - .5
    batch = {"image_a": torch.rand(1, 3, 64, 64), "image_b": torch.rand(1, 3, 64, 64),
             "sparse_a": points, "sparse_b": points + torch.tensor([1., 0.]),
             "sparse_valid": torch.ones(1, 16, dtype=torch.bool), "sparse_confidence": torch.ones(1, 16),
             "F_ab": torch.tensor([[[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]]])}
    objective = ScratchTransferObjective(config)
    loss, metrics = objective.imc(model, batch, False)
    assert torch.isfinite(loss) and metrics["epipolar"] > 0
    loss.backward()
    for prefix in ("backbone.", "coarse_transformer.", "mid_refiner.", "fine_refiner."):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for n, p in model.named_parameters() if n.startswith(prefix))
    assert all(p.grad is None for n, p in model.named_parameters() if n.startswith(GEOMETRY_PREFIXES))
    # Subtracting the zero-F-weight objective isolates the direct differentiable F term.
    model.zero_grad(set_to_none=True)
    objective_without_f = ScratchTransferObjective(config, epipolar_weight=0.)
    base, _ = objective_without_f.imc(model, batch, False)
    with_f, _ = objective.imc(model, batch, False)
    (with_f - base).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.fine_refiner.parameters())


def test_accumulation_averages_partial_windows_and_skips_empty():
    from types import SimpleNamespace
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    optimizer = torch.optim.SGD([{"params": model.parameters(), "lr": .1, "peak_lr": .1, "name": "test"}])
    class Objective:
        def imc(self, model, batch, train):
            if batch["empty"]:
                return None, {"valid_count": 0}
            loss = (model(torch.ones(1, 1)) - 1).square().mean()
            return loss, {"loss": loss.detach(), "valid_count": 1.}
    args = SimpleNamespace(stage="imc", accumulation_steps=3, max_empty_batches=10,
                           gradient_clip=100., min_lr_ratio=1., log_every=0)
    metrics, steps = run_epoch(model, Objective(), [{"empty": False}, {"empty": True}, {"empty": False}, {"empty": False}],
        optimizer, args, torch.device("cpu"), 0, 0, 10, 0, train=True)
    assert steps == 2 and metrics["optimizer_updates"] == 2
    # First update: average of two gradients=-2; partial window gradient=-1.6.
    assert model.weight.item() == pytest.approx(.36)
