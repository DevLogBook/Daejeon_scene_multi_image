# GRM: Geometry-Aware Reliable Matching for UAV Image Stitching

GRM is an image matcher for agricultural UAV imagery. It extracts shared multi-scale features and uses hierarchical reliability aggregation (HRA) to build a compact grid for global image-to-image interaction. Reliability and spatial coverage guide query selection. Mid and fine refinement then localize the correspondences, and the model assigns a confidence score to each valid match.

## Setup

Use a Python environment with PyTorch installed for your hardware. Run the following from the repository root.

```bash
python -m pip install -r requirements.txt
```

Datasets, trained checkpoints, and generated results are stored separately. Use a GRM checkpoint containing both the model state and its configuration.

## Match an image pair

```bash
python -m grm_matcher.inference_matcher --image_a path/to/image_a.jpg --image_b path/to/image_b.jpg --checkpoint path/to/grm.pt --output_dir outputs/matches --device cuda
```

This writes `matches.json` and a `matches.png` visualization. Use `--device cpu` for a CPU run.

## Train and evaluate

| Task | Entry point | Input |
| --- | --- | --- |
| Synthetic pretraining | `python -m grm_matcher.train_scratch_transfer --help` | Paired images with known homographies |
| Real-data training | `python -m grm_matcher.train_real_curriculum --help` | MegaDepth or UAV3DCrop geometry caches and, for R2, a precomputed agricultural teacher cache |
| Multi-domain replay | `python -m grm_matcher.train_real_replay --help` | Prepared geometry and teacher caches, synthetic pairs, and an initialization checkpoint |
| Agricultural pair evaluation | [Evaluation guide](GRM_AGRI_EVALUATION.md) | A synthetic homography split or real pairs with reference point annotations |
| Four-dataset benchmark | `python -m tools.evaluate_grm_four_hpc --help` | MegaDepth1500, prepared UAV3DCrop, real agricultural pairs, and synthetic pairs |

The F8/HRA configuration for synthetic pretraining is in [grm_f8_hra_foundation.json](grm_matcher/configs/grm_f8_hra_foundation.json). [prepare_grm_depth_data.py](tools/prepare_grm_depth_data.py) builds depth and pose supervision caches. Real-data training can start from a compatible checkpoint with `--init-checkpoint`. Use `--resume` to continue the same run with its matching training code and configuration.
