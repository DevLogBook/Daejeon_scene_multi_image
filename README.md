# GRM: Geometry-Aware Reliable Matching

PyTorch implementation of GRM image matching, supervised training, geometry data
preparation, and image stitching. The model configuration is stored in each
checkpoint and restored at inference time.

## Installation

Use Python 3.10 or newer. Install a PyTorch build compatible with your GPU driver,
then install the remaining dependencies in the same environment:

```bash
python -m pip install -r requirements.txt
```

Datasets, pretrained weights, teacher caches, experiment logs, evaluation outputs,
and paper materials are not distributed in this source snapshot. Supply your own
paths. RoMaV2 teacher-cache generation additionally requires the upstream RoMaV2
package, its weights, and its backbone dependencies; matcher inference does not
require the teacher.

## Match two images

```bash
python -m grm_matcher.inference_matcher --image_a path/to/a.jpg --image_b path/to/b.jpg --checkpoint path/to/best_agri.pt --output_dir outputs/matches --device cuda
```

The entry point exports match coordinates in original-image pixels and matching
visualizations. Checkpoint files must contain the model state and its serialized
GRM configuration. Use `--device cpu` for CPU inference.

## Stitch two images or a sequence

```bash
python -m dense_match.inference_grm_stitch --image_a path/to/a.jpg --image_b path/to/b.jpg --grm_checkpoint path/to/best_agri.pt --output outputs/stitch.png --device cuda
python tools/generate_grm_panoramas.py --help
```

The pair stitcher combines GRM correspondences with robust homography estimation,
optional residual-mesh alignment, and seam blending.

## Prepare data and train

Run the following entry points from the repository root to inspect their required
input paths, sampling settings, and optimizer options:

```bash
python dataset/generate_single_image_homography_pairs.py --help
python tools/prepare_grm_depth_data.py --help
python tools/prepare_udis_pairs.py --help
python -m dense_match.precompute_teacher --help
python -m grm_matcher.train_synthetic_robustness --help
python -m grm_matcher.train_udis_finetune --help
python -m grm_matcher.train_depth_curriculum --help
python -m grm_matcher.train_coarse_recovery --help
```

Depth supervision uses paired images, camera calibration, camera poses, and depth
maps. The preparation tool validates visibility, depth agreement, and cycle
consistency, then writes compact geometry caches. Preserve scene/field split
isolation when preparing training and validation data.

For curriculum training, run each phase separately. Use `--init-checkpoint` when
starting a new phase; `--resume` restores an interrupted run of the same phase,
including optimizer and learning-rate state. Use a separate output directory for
each new run. Generated outputs are ignored by Git.

## Evaluation

```bash
python evaluate_agri_synthetic.py --help
python evaluate_agri_real.py --help
```

Evaluation accepts external datasets and checkpoints. Ground truth is used for
scoring rather than matcher inference. Evaluation results are written locally.

## Core regression tests

```bash
python -m pytest -q grm_matcher/tests tests/test_depth_curriculum.py
```

The included tests exercise coordinates, correlation, local refinement, V3 query
selection, architecture gradients, and curriculum-training contracts using small
synthetic inputs. They do not require private datasets or trained weights.
