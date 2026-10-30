"""Versioned, canvas-coordinate offline teacher correspondences for IMC2022."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .imc_dataset import sampson_squared


def pair_key(row, height, width):
    payload = json.dumps(dict(version=1, row=row, height=height, width=width), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def filter_matches(a, b, confidence, masks, fundamental, threshold=2., minimum_confidence=.5):
    a, b, confidence = np.asarray(a, np.float32), np.asarray(b, np.float32), np.asarray(confidence, np.float32)
    if a.ndim != 2 or a.shape[1:] != (2,) or b.shape != a.shape or confidence.shape != (len(a),):
        raise ValueError("Teacher arrays must be a/b [N,2], confidence [N]")
    keep = np.isfinite(a).all(-1) & np.isfinite(b).all(-1) & np.isfinite(confidence)
    keep &= (confidence >= minimum_confidence) & (confidence <= 1.)
    for points, mask in zip((a,b), masks):
        h,w = mask.shape
        inside = ((points >= 0) & (points <= [w-1,h-1])).all(-1)
        safe = np.rint(np.nan_to_num(points, nan=0., posinf=0., neginf=0.)).astype(np.int64)
        keep &= inside & mask[safe[:,1].clip(0,h-1), safe[:,0].clip(0,w-1)]
    a,b,confidence = a[keep],b[keep],confidence[keep]
    if len(a):
        distance = sampson_squared(torch.from_numpy(a)[None], torch.from_numpy(b)[None],
            torch.as_tensor(fundamental)[None])[0].numpy()
        keep = np.isfinite(distance) & (distance <= threshold**2)
        a,b,confidence = a[keep],b[keep],confidence[keep]
    order = np.argsort(-confidence, kind="stable")
    return a[order], b[order], confidence[order]


def unique_coarse(a, b, confidence, stride):
    """Avoid contradictory CE targets per source/target cell; retain raw cache separately."""
    cells = [np.rint((p+.5)/stride-.5).astype(np.int64) for p in (a,b)]
    used_a,used_b,chosen = set(),set(),[]
    for i in np.argsort(-confidence, kind="stable"):
        x,y=tuple(cells[0][i]),tuple(cells[1][i])
        if x not in used_a and y not in used_b:
            used_a.add(x); used_b.add(y); chosen.append(i)
    return a[chosen],b[chosen],confidence[chosen]


def load_pair(cache_dir, row, height, width):
    key = pair_key(row,height,width)
    path = Path(cache_dir)/f"{key}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing RoMaV2 teacher cache: {path}; generate labels before training")
    with np.load(path, allow_pickle=False) as data:
        if data["key"].item() != key:
            raise ValueError(f"Teacher cache signature mismatch: {path}")
        return data["a"].copy(),data["b"].copy(),data["confidence"].copy()
