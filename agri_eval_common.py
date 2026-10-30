"""Identical evaluation protocol copied into each baseline repository.

OpenCV coordinates: pixel centers at integer indices. GT is used for scoring only.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

THRESHOLDS = (1, 3, 5, 10)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_json(value):
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(safe_json(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_image(path, mask=False):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE if mask else cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return image > 0 if mask else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def project(points, h):
    p = np.c_[np.asarray(points).reshape(-1, 2), np.ones(len(points))] @ np.asarray(h).T
    good = np.isfinite(p).all(1) & (np.abs(p[:, 2]) > 1e-10)
    q = np.full((len(p), 2), np.nan)
    q[good] = p[good, :2] / p[good, 2:]
    return q


def resize_transform(shape, size):
    height, width = shape[:2]
    sx, sy = size / width, size / height
    return np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]], dtype=float)


def in_bounds(points, shape):
    h, w = shape[:2]
    return np.isfinite(points).all(1) & (points[:, 0] >= 0) & (points[:, 0] <= w-1) & (points[:, 1] >= 0) & (points[:, 1] <= h-1)


def sample_mask(mask, points):
    good = in_bounds(points, mask.shape)
    out = np.zeros(len(points), dtype=bool)
    xy = np.rint(points[good]).astype(int)
    out[good] = mask[xy[:, 1], xy[:, 0]]
    return out


def resolve_synthetic(root, split, value, folder):
    normalized = str(value).replace("\\", "/")
    # Prefer the supplied dataset root over stale manifest paths.
    path = root / split / folder / normalized.rsplit("/", 1)[-1]
    if path.is_file():
        return path
    for path in (root / normalized, Path(normalized)):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {folder}: {value} under {root}")


def unique_map(rows, label):
    result = {}
    for row in rows:
        key = row["pair_id"]
        if key in result:
            raise ValueError(f"Duplicate pair_id in {label}: {key}")
        result[key] = row
    return result


def load_pairs(args, mode):
    if mode == "synthetic":
        manifest = args.data_root / args.split / "manifest.jsonl"
        raw = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        unique_map(raw, str(manifest))
        pairs = []
        for row in raw:
            if row.get("direction", "A_to_B") != "A_to_B":
                raise ValueError("Only A_to_B homography manifests are supported")
            item = {"pair_id": row["pair_id"], "group": row.get("target_bucket", "unknown")}
            for key, folder in (("image_a", "input1"), ("image_b", "input2"), ("h_gt_a_to_b", "homography"),
                                ("valid_a", "masks/valid1"), ("valid_b", "masks/valid2"), ("overlap_a", "masks/overlap1")):
                item[key] = resolve_synthetic(args.data_root, args.split, row[key], folder)
            pairs.append(item)
        meta = {"manifest_sha256": digest(manifest), "total_records": len(raw), "excluded": []}
    else:
        annotation_path = args.annotations or args.data_root / "annotations.json"
        list_path = args.pair_list or args.data_root / "pair_list.json"
        ann = unique_map(read_json(annotation_path), "annotations") if annotation_path.is_file() else {}
        listing = unique_map(read_json(list_path), "pair_list")
        root = args.image_root or args.data_root
        pairs, excluded = [], []
        for key, row in listing.items():
            label = ann.get(key)
            labeled = label is not None and label.get("status") in args.statuses.split(",")
            pa, pb = ([np.asarray(label[k], dtype=float).reshape(-1, 2) for k in ("points_a", "points_b")]
                      if labeled else (np.empty((0, 2)), np.empty((0, 2))))
            if labeled and (len(pa) != len(pb) or len(pa) == 0 or not np.isfinite(pa).all() or not np.isfinite(pb).all()):
                raise ValueError(f"Malformed landmarks: {key}")
            item = {"pair_id": key, "group": row.get("video_id", "unknown"), "points_a": pa, "points_b": pb,
                    "has_ground_truth": labeled, "annotation_status": label.get("status", "missing") if label else "missing"}
            for field in ("image_a", "image_b"):
                if label is not None and field in label and str(row[field]).replace("\\", "/") != str(label[field]).replace("\\", "/"):
                    raise ValueError(f"Image path disagreement between lists: {key}, {field}")
                item[field] = root / str(row[field]).replace("\\", "/")
                if not item[field].is_file():
                    raise FileNotFoundError(item[field])
            pairs.append(item)
        meta = {"annotations_sha256": digest(annotation_path) if annotation_path.is_file() else None, "pair_list_sha256": digest(list_path),
                "total_records": len(listing), "excluded": excluded, "unused_annotation_ids": sorted(set(ann)-set(listing))}
        meta["labeled_pairs"] = sum(pair["has_ground_truth"] for pair in pairs)
        meta["unlabeled_pairs"] = len(pairs) - meta["labeled_pairs"]
    meta["eligible_pairs_before_limit"] = len(pairs)
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        raise ValueError("No eligible pairs")
    return pairs, meta


def load_ground_truth(pair, mode, a, b):
    if mode == "real":
        for key, image in (("points_a", a), ("points_b", b)):
            # Continuous subpixel annotations may lie within the edge pixel's
            # footprint, e.g. y=1079.13 in a 1080-pixel-high image.
            points = pair[key]
            height, width = image.shape[:2]
            inside = np.isfinite(points).all(1) & (points[:, 0] >= -.5) & (points[:, 0] <= width-.5) & (points[:, 1] >= -.5) & (points[:, 1] <= height-.5)
            if not inside.all():
                raise ValueError(f"Landmarks outside original image: {pair['pair_id']} {key}")
        return pair
    h = np.load(pair["h_gt_a_to_b"], allow_pickle=False)
    if h.shape != (3, 3) or not np.isfinite(h).all() or np.linalg.matrix_rank(h) != 3:
        raise ValueError(f"Invalid GT H: {pair['pair_id']}")
    masks = {key: read_image(pair[key], True) for key in ("valid_a", "valid_b", "overlap_a")}
    for key, shape in (("valid_a", a.shape[:2]), ("valid_b", b.shape[:2]), ("overlap_a", a.shape[:2])):
        if masks[key].shape != shape:
            raise ValueError(f"GT mask/image shape mismatch: {pair['pair_id']} {key}")
    if not (masks["valid_a"] & masks["overlap_a"]).any():
        raise ValueError(f"Empty GT overlap: {pair['pair_id']}")
    return {"h": h, **masks}


def fit_homography(pa, pb, seed, threshold):
    if len(pa) < 4:
        return None, 0, "fewer_than_four_matches"
    cv2.setRNGSeed(seed)
    try:
        h, mask = cv2.findHomography(pa, pb, cv2.USAC_MAGSAC, threshold, maxIters=10000, confidence=.999)
    except cv2.error as exc:
        return None, 0, f"homography_error: {exc}"
    if h is None or not np.isfinite(h).all() or np.linalg.matrix_rank(h) != 3:
        return None, 0, "invalid_homography"
    n = int(mask.sum()) if mask is not None else 0
    if n < 4:
        return None, n, "fewer_than_four_inliers"
    return h, n, "ok"


def distances(a, b):
    error = np.linalg.norm(a-b, axis=1)
    return np.where(np.isfinite(error), error, np.inf)


def coverage(points, overlap, grid=8):
    occupied = cv2.resize(overlap.astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA) > 0
    found = np.zeros((grid, grid), bool)
    if len(points):
        xy = np.floor(points * np.array([grid/overlap.shape[1], grid/overlap.shape[0]])).astype(int)
        xy = np.clip(xy, 0, grid-1)
        found[xy[:, 1], xy[:, 0]] = True
    return float((found & occupied).sum() / max(1, occupied.sum()))


def score_synthetic(pa, pb, h, gt, a_shape, b_shape):
    truth = project(pa, gt["h"])
    eligible = sample_mask(gt["valid_a"] & gt["overlap_a"], pa) & sample_mask(gt["valid_b"], truth)
    error = distances(pb, truth)
    error[~eligible | ~sample_mask(gt["valid_b"], pb)] = np.inf
    result = {"gt_eligible_matches": int(eligible.sum())}
    for t in THRESHOLDS:
        correct = error <= t
        result[f"correct_matches_{t}px"] = int(correct.sum())
        result[f"match_precision_{t}px"] = float(correct.sum()/len(pa)) if len(pa) else 0.
    result["coverage_3px"] = coverage(pa[error <= 3], gt["overlap_a"] & gt["valid_a"])
    height, width = a_shape[:2]
    corners = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], float)
    corner_error = distances(project(corners, h), project(corners, gt["h"])) if h is not None else np.full(4, np.inf)
    result["corner_mean_px"] = float(corner_error.mean())
    # Visible-area grid supplements corners that may lie outside overlap.
    y, x = np.meshgrid(np.linspace(0, height-1, 25), np.linspace(0, width-1, 25), indexing="ij")
    grid = np.stack((x.ravel(), y.ravel()), 1)
    grid_truth = project(grid, gt["h"])
    valid = sample_mask(gt["valid_a"] & gt["overlap_a"], grid) & sample_mask(gt["valid_b"], grid_truth)
    grid = grid[valid]; grid_truth = grid_truth[valid]
    result["overlap_grid_mean_px"] = float(distances(project(grid, h), grid_truth).mean()) if h is not None and len(grid) else math.inf
    return result, error


def score_real(h, pair):
    n = len(pair["points_a"])
    if not n:
        result = {"landmarks": 0, "landmark_mean_px": None, "landmark_median_px": None, "landmark_p90_px": None}
        for t in THRESHOLDS:
            result[f"landmark_correct_{t}px"] = None
            result[f"landmark_pck_{t}px"] = None
        return result, np.empty(0)
    error = distances(project(pair["points_a"], h), pair["points_b"]) if h is not None else np.full(n, np.inf)
    result = {"landmarks": n, "landmark_mean_px": float(error.mean()),
              "landmark_median_px": float(np.median(error)), "landmark_p90_px": float(np.quantile(error, .9)) if np.isfinite(error).all() else math.inf}
    for t in THRESHOLDS:
        result[f"landmark_correct_{t}px"] = int((error <= t).sum())
        result[f"landmark_pck_{t}px"] = float((error <= t).mean())
    return result, error


def save_stitch(path, a, b, h, max_side=2048):
    # A diagnostic average blend; identical backend for every matcher.
    ha, wa = a.shape[:2]; hb, wb = b.shape[:2]
    corners = np.array([[0, 0], [wa-1, 0], [wa-1, ha-1], [0, ha-1]], float)
    denominator = np.c_[corners, np.ones(4)] @ h[2]
    if np.min(denominator) <= 0 <= np.max(denominator):
        raise ValueError("Homography pole crosses image")
    warped = project(corners, h)
    if not np.isfinite(warped).all():
        raise ValueError("Nonfinite warped corners")
    all_points = np.r_[warped, [[0, 0], [wb-1, hb-1]]]
    low = np.floor(all_points.min(0)); high = np.ceil(all_points.max(0))
    extent = high-low+1
    if extent.max() > 20*max(ha, wa, hb, wb):
        raise ValueError("Excessive canvas extent")
    scale = min(1., max_side/extent.max())
    canvas = tuple(np.maximum(1, np.ceil(extent*scale)).astype(int))
    transform = np.array([[scale, 0, -scale*low[0]], [0, scale, -scale*low[1]], [0, 0, 1]])
    aw = cv2.warpPerspective(a, transform@h, canvas).astype(float)
    bw = cv2.warpPerspective(b, transform, canvas).astype(float)
    ma = cv2.warpPerspective(np.ones((ha, wa), np.uint8), transform@h, canvas, flags=cv2.INTER_NEAREST)
    mb = cv2.warpPerspective(np.ones((hb, wb), np.uint8), transform, canvas, flags=cv2.INTER_NEAREST)
    image = ((aw+ bw)/np.maximum(1, ma+mb)[..., None]).clip(0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    encoded.tofile(str(path))


def error_auc(errors, threshold):
    ordered = np.sort(np.asarray(errors, float))
    finite = ordered[ordered < threshold]
    x = np.r_[0., finite, threshold]
    y = np.r_[0., np.arange(1, len(finite)+1)/len(ordered), len(finite)/len(ordered)]
    return float(np.sum(np.diff(x)*(y[:-1]+y[1:])/2)/threshold)


def aggregate(rows, mode):
    n = len(rows)
    success = [r for r in rows if r["status"] == "ok"]
    key = "corner_mean_px" if mode == "synthetic" else "landmark_mean_px"
    evaluation_rows = rows if mode == "synthetic" else [r for r in rows if r.get("landmarks", 0) > 0]
    errors = [r.get(key, math.inf) for r in evaluation_rows]
    finite = [e for e in errors if math.isfinite(e)]
    result = {"pairs": n, "estimated_h_pairs": len(success), "homography_failure_rate": 1-len(success)/n,
        "geometry_evaluation_pairs": len(evaluation_rows), "unlabeled_pairs": n-len(evaluation_rows),
        "scorable_pairs": len(finite), "pair_mean_error_px_success_only": float(np.mean(finite)) if finite else None,
        "pair_median_error_px_success_only": float(np.median(finite)) if finite else None,
        "pair_p90_error_px_success_only": float(np.quantile(finite, .9)) if finite else None,
        "inference_error_pairs": sum(r["status"] == "inference_error" for r in rows),
        "mean_selected_matches": float(np.mean([r["matches"] for r in rows])),
        "mean_matching_seconds": float(np.mean([r["matching_seconds"] for r in rows])),
        "mean_pipeline_seconds": float(np.mean([r["pipeline_seconds"] for r in rows]))}
    for t in THRESHOLDS:
        result[f"pair_success_{t}px"] = sum(e <= t for e in errors)/len(errors) if errors else None
        result[f"pair_error_auc_{t}px"] = error_auc(errors, t) if errors else None
        if mode == "synthetic":
            result[f"macro_match_precision_{t}px"] = float(np.mean([r[f"match_precision_{t}px"] for r in rows]))
            denominator = sum(r["matches"] for r in rows)
            result[f"micro_match_precision_{t}px"] = sum(r[f"correct_matches_{t}px"] for r in rows)/denominator if denominator else 0.
        else:
            result[f"macro_landmark_pck_{t}px"] = float(np.mean([r[f"landmark_pck_{t}px"] for r in evaluation_rows])) if evaluation_rows else None
            result[f"micro_landmark_pck_{t}px"] = sum(r[f"landmark_correct_{t}px"] for r in evaluation_rows)/sum(r["landmarks"] for r in evaluation_rows) if evaluation_rows else None
    if mode == "synthetic":
        result["macro_coverage_3px"] = float(np.mean([r["coverage_3px"] for r in rows]))
    for key in ("pair_total_seconds", "input_and_gt_seconds", "matching_seconds", "geometry_seconds",
                "scoring_seconds", "matches_save_seconds", "stitch_save_seconds"):
        values = [r[key] for r in rows if key in r]
        if values:
            result[f"total_{key}"] = float(np.sum(values))
            result[f"mean_{key}"] = float(np.mean(values))
            result[f"median_{key}"] = float(np.median(values))
            result[f"p90_{key}"] = float(np.quantile(values, .9))
    result["stitches_saved"] = sum(r.get("stitch_file") is not None for r in rows)
    return result


def parser(model, mode):
    p = argparse.ArgumentParser(description=f"{model}: {mode} agriculture evaluation; GT is never used to fit H")
    p.add_argument("--data-root", type=Path, default=Path("D:/Admin/Desktop/homography_dataset" if mode == "synthetic" else "D:/Admin/Desktop/test"))
    p.add_argument("--output", type=Path, default=Path("evaluation_outputs") / f"{mode}_512")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--size", type=int, default=512, help="Square input canvas shared by all models; divisible by 32")
    p.add_argument("--max-matches", type=int, default=2048, help="Top confidence budget; 0 means uncapped")
    p.add_argument("--min-confidence", type=float, default=None, help="Optional postfilter; upstream thresholds remain enabled")
    p.add_argument("--ransac-threshold", type=float, default=3., help="MAGSAC threshold in resized input pixels")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--save-visuals", type=int, default=-1 if mode == "real" else 10,
                   help="-1 saves every eligible pair, 0 disables, N saves first N (real default: all)")
    p.add_argument("--skip-warmup", action="store_true")
    p.add_argument("--check-data", action="store_true", help="Validate all selected files/labels without loading any model")
    p.add_argument("--continue-on-error", action="store_true", help="Record inference errors as failed pairs; default aborts")
    p.add_argument("--roma-samples", type=int, default=2048)
    p.add_argument("--dinov3-dir", type=Path, help="RoMaV2: local DINOv3 source directory containing hubconf.py")
    p.add_argument("--allow-download", action="store_true", help="Allow RoMaV2 constructor to download official weights")
    if model == "GRM":
        p.add_argument("--grm-rerank-mode", choices=("checkpoint", "geometry_graph", "token_mlp", "coarse_prior"), default="checkpoint",
                       help="token_mlp bypasses graph messages but retains the learned rerank head; coarse_prior bypasses both")
        p.add_argument("--grm-sampling-mode", choices=("checkpoint", "adaptive_density", "overlap_aware"), default="checkpoint")
        p.add_argument("--grm-sampling-preset", choices=("checkpoint", "uniform"), default="checkpoint")
        p.add_argument("--grm-topk-candidates", type=int, choices=(1, 2, 4), default=None)
        p.add_argument("--grm-support-min-matchability", type=float, default=None)
        p.add_argument("--grm-support-relative-score", type=float, default=None)
    if mode == "synthetic":
        p.add_argument("--split", choices=("train", "val", "test"), default="test")
    else:
        p.add_argument("--annotations", type=Path)
        p.add_argument("--pair-list", type=Path)
        p.add_argument("--image-root", type=Path)
        p.add_argument("--statuses", default="labeled", help="Annotation statuses trusted for scoring only; all listed pairs are processed")
    return p


def main(model, mode, project_root, argv=None):
    run_start = time.perf_counter()
    p = parser(model, mode)
    args = p.parse_args(argv)
    if args.size < 64 or args.size % 32 or min(args.max_matches, args.limit) < 0 or args.save_visuals < -1 or args.ransac_threshold <= 0 or args.roma_samples < 1:
        p.error("Invalid size, budget, limit, visual count or threshold")
    if model == "RoMaV2" and args.roma_samples*4 > (args.size//4)**2:
        p.error("RoMa sampling expansion exceeds its unidirectional final grid; lower --roma-samples")
    pairs, meta = load_pairs(args, mode)
    print(f"[{model}/{mode}] selected={len(pairs)} excluded={len(meta['excluded'])} size={args.size} pixels=original", flush=True)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Use a new output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {"model": model, "mode": mode, "arguments": vars(args), "data": meta,
        "evaluator_sha256": digest(__file__),
        "coordinate_convention": "OpenCV integer pixel centers; half-pixel resize transform",
        "scoring_units": "original image pixels", "ransac_units": "square network input pixels",
        "preprocessing": "RGB decode; resize each image directly to size x size (no aspect-ratio preservation)",
        "pair_ids": [pair["pair_id"] for pair in pairs], "ground_truth_used_for_estimation": False}
    write_json(args.output / "settings.json", settings)
    if args.check_data:
        for i, pair in enumerate(pairs):
            a, b = read_image(pair["image_a"]), read_image(pair["image_b"])
            load_ground_truth(pair, mode, a, b)
            if (i+1) % 50 == 0 or i+1 == len(pairs):
                print(f"[DATA OK] {i+1}/{len(pairs)}", flush=True)
        write_json(args.output / "data_check.json", {"checked_pairs": len(pairs), "ok": True, **meta})
        return
    import torch
    from agri_eval_adapter import Matcher
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    cv2.setNumThreads(1)
    init_start = time.perf_counter()
    matcher = Matcher(model, Path(project_root), args)
    matcher.sync()
    model_init_seconds = time.perf_counter() - init_start
    settings["runtime"] = matcher.metadata
    settings["checkpoint_sha256"] = digest(args.checkpoint) if args.checkpoint else "official auto-download; inspect torch hub cache"
    write_json(args.output / "settings.json", settings)
    (args.output / "matches").mkdir()
    (args.output / "stitches").mkdir()
    rows = []
    warmup_seconds = 0.
    for index, pair in enumerate(pairs):
        pair_start = time.perf_counter()
        a, b = read_image(pair["image_a"]), read_image(pair["image_b"])
        gt = load_ground_truth(pair, mode, a, b)
        ta, tb = resize_transform(a.shape, args.size), resize_transform(b.shape, args.size)
        ia, ib = [cv2.resize(x, (args.size, args.size), interpolation=cv2.INTER_LINEAR) for x in (a, b)]
        input_and_gt_seconds = time.perf_counter() - pair_start
        pair_warmup_seconds = 0.
        if index == 0 and not args.skip_warmup:
            print("[WARMUP] one inference excluded from timing", flush=True)
            matcher.sync(); warmup_start = time.perf_counter()
            matcher(ia, ib); matcher.sync()
            pair_warmup_seconds = time.perf_counter() - warmup_start
            warmup_seconds += pair_warmup_seconds
        # Pair-specific RNG makes limiting/pair order reproducible.
        seed = (args.seed + int(hashlib.sha256(pair["pair_id"].encode()).hexdigest()[:8], 16)) % (2**31-1)
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        if matcher.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(matcher.device)
        matcher.sync(); start = time.perf_counter()
        inference_error = None
        try:
            pa, pb, conf = matcher(ia, ib)
            pa, pb, conf = np.asarray(pa).reshape(-1, 2), np.asarray(pb).reshape(-1, 2), np.asarray(conf).reshape(-1)
            if len(pa) != len(pb) or len(pa) != len(conf):
                raise ValueError("Inconsistent matcher output lengths")
        except Exception as exc:
            if not args.continue_on_error:
                raise
            inference_error = f"{type(exc).__name__}: {exc}"
            pa = pb = np.empty((0, 2)); conf = np.empty(0)
        matcher.sync(); matching_seconds = time.perf_counter()-start
        geometry_start = time.perf_counter()
        raw_count = len(pa)
        valid = in_bounds(pa, ia.shape) & in_bounds(pb, ib.shape) & np.isfinite(conf)
        if args.min_confidence is not None:
            valid &= conf >= args.min_confidence
        pa, pb, conf = pa[valid], pb[valid], conf[valid]
        order = np.argsort(-conf, kind="stable")
        if args.max_matches:
            order = order[:args.max_matches]
        pa, pb, conf = pa[order], pb[order], conf[order]
        h_net, inliers, status = fit_homography(pa, pb, seed, args.ransac_threshold)
        h = np.linalg.inv(tb) @ h_net @ ta if h_net is not None else None
        pa, pb = project(pa, np.linalg.inv(ta)), project(pb, np.linalg.inv(tb))
        pipeline_seconds = time.perf_counter()-start
        geometry_seconds = time.perf_counter() - geometry_start
        if inference_error:
            status = "inference_error"
        scoring_start = time.perf_counter()
        metrics, errors = (score_synthetic(pa, pb, h, gt, a.shape, b.shape) if mode == "synthetic" else score_real(h, pair))
        scoring_seconds = time.perf_counter() - scoring_start
        row = {"pair_id": pair["pair_id"], "group": pair["group"], "status": status,
            "has_ground_truth": pair.get("has_ground_truth", True),
            "annotation_status": pair.get("annotation_status", "synthetic_ground_truth"),
            "error": inference_error, "raw_matches": raw_count, "matches": len(pa), "ransac_inliers": inliers,
            "matching_seconds": matching_seconds, "pipeline_seconds": pipeline_seconds,
            "input_and_gt_seconds": input_and_gt_seconds, "geometry_seconds": geometry_seconds,
            "scoring_seconds": scoring_seconds,
            "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(matcher.device)/2**30 if matcher.device.type == "cuda" else None,
            **metrics}
        name = f"{index:05d}_{hashlib.sha256(pair['pair_id'].encode()).hexdigest()[:10]}"
        row["matches_file"] = f"matches/{name}.npz"
        save_start = time.perf_counter()
        np.savez_compressed(args.output / row["matches_file"], points_a=pa, points_b=pb, confidence=conf,
            homography_a_to_b=h if h is not None else np.full((3, 3), np.nan), evaluation_errors_px=errors)
        row["matches_save_seconds"] = time.perf_counter() - save_start
        row["visual_error"] = None
        row["stitch_file"] = None
        row["stitch_save_seconds"] = 0.
        requested = args.save_visuals == -1 or index < args.save_visuals
        row["visual_status"] = "skipped_no_homography" if requested and h is None else "not_requested"
        if requested and h is not None:
            save_start = time.perf_counter()
            try:
                save_stitch(args.output / "stitches" / f"{name}.jpg", a, b, h)
                row["stitch_file"] = f"stitches/{name}.jpg"
                row["visual_status"] = "saved"
            except (ValueError, RuntimeError, OSError, cv2.error) as exc:
                row["visual_error"] = str(exc)
                row["visual_status"] = "failed"
            row["stitch_save_seconds"] = time.perf_counter() - save_start
        # Includes data IO, scoring and artifacts, but excludes first-pair warmup
        # and this row's own log/CSV serialization.
        row["pair_total_seconds"] = time.perf_counter() - pair_start - pair_warmup_seconds
        rows.append(row)
        with (args.output / "pairs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(safe_json(row), ensure_ascii=False, allow_nan=False)+"\n")
        print(f"[{model}/{mode}] {index+1}/{len(pairs)} {pair['pair_id']} status={status} matches={len(pa)} "
              f"match={matching_seconds:.3f}s H={geometry_seconds:.3f}s stitch={row['stitch_save_seconds']:.3f}s "
              f"pair_total={row['pair_total_seconds']:.3f}s visual={row['visual_status']}", flush=True)
    with (args.output / "pairs.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(safe_json(row) for row in rows)
    summary = aggregate(rows, mode)
    summary.update(model=model, mode=mode, complete=True,
        groups={str(g): aggregate([r for r in rows if r["group"] == g], mode) for g in sorted({r["group"] for r in rows})},
        model_init_seconds=model_init_seconds, warmup_seconds=warmup_seconds,
        run_wall_seconds_before_summary_write=time.perf_counter()-run_start,
        note="All listed pairs contribute runtime and H failure rate. Only GT-labeled pairs contribute geometry accuracy/PCK/AUC, including their failures. Unlabeled errors are null, not zero. matching_seconds includes adapter preprocessing/inference/sampling/CPU transfer; geometry_seconds includes filtering/H estimation/coordinate restoration. pair_total_seconds includes input IO/GT/resize/scoring/NPZ/stitch output, excludes warmup and row logging. Run wall includes initialization/warmup and output until summary write; component timers are nested, not additive with pipeline/total.")
    write_json(args.output / "summary.json", summary)
    print(json.dumps(safe_json(summary), ensure_ascii=False), flush=True)
