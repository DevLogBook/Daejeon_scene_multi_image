"""Camera/depth adapters for DISK MegaDepth and UAV3DCrop (no learned Teacher)."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np


@dataclass
class Frame:
    key: str
    image: Path
    depth: Path
    k: np.ndarray
    distortion: np.ndarray
    c2w: np.ndarray
    width: int
    height: int


@dataclass
class Scene:
    domain: str
    name: str
    group: str
    frames: list[Frame]
    pairs: list[tuple[int, int]] | None = None


def read_rgb(path):
    raw = Path(path).read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f'Cannot decode image: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), hashlib.sha256(raw).hexdigest()


@lru_cache(maxsize=6)
def read_depth(path: str):
    path = Path(path)
    if path.suffix.lower() in ('.h5', '.hdf5'):
        import h5py
        with h5py.File(path, 'r') as f:
            if 'depth' not in f:
                raise ValueError(f'{path}: expected HDF5 dataset depth; found {list(f)}')
            value = np.asarray(f['depth'], np.float32)
    else:
        import tifffile
        value = tifffile.imread(path).astype(np.float32)
    if value.ndim != 2:
        raise ValueError(f'{path}: expected 2D Z-depth, found {value.shape}')
    return value


def letterbox_geometry(width, height, size):
    ratio = size / max(width, height)
    nw, nh = max(1, round(width * ratio)), max(1, round(height * ratio))
    left, top = (size - nw) // 2, (size - nh) // 2
    sx, sy = nw / width, nh / height
    transform = np.array([[sx, 0, left + (sx - 1) / 2],
                          [0, sy, top + (sy - 1) / 2], [0, 0, 1]], np.float64)
    return transform, (left, top, nw, nh)


def coarse_grid(size, stride=16):
    y, x = np.meshgrid(np.arange(size // stride), np.arange(size // stride), indexing='ij')
    return (np.c_[x.ravel(), y.ravel()] + .5) * stride - .5


def map_points(points, matrix):
    value = np.c_[points, np.ones(len(points))] @ matrix.T
    return value[:, :2] / value[:, 2:]


def unproject(points, depth, k, distortion):
    xy = cv2.undistortPointsIter(points[:, None].astype(np.float64), k, distortion,
        None, None, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10))[:, 0]
    return np.c_[xy, np.ones(len(xy))] * depth[:, None]


def project(xyz, k, distortion):
    return cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), k, distortion)[0][:, 0]


def xyz_transform(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def sample_depth(frame, points):
    value = read_depth(str(frame.depth))
    h, w = value.shape
    # Calibration K is expressed at RGB resolution. COLMAP depth can be smaller.
    xy = (points + .5) * [w / frame.width, h / frame.height] - .5
    finite = np.isfinite(xy).all(1)
    safe = np.where(finite[:, None], xy, 0)
    ij = np.rint(safe).astype(np.int64)
    inside = finite & (points >= 0).all(1) & (points <= [frame.width-1, frame.height-1]).all(1)
    z = value[ij[:, 1].clip(0, h-1), ij[:, 0].clip(0, w-1)]
    return z, inside & np.isfinite(z) & (z > 0)


def direction(a, b, size=512, relative_depth=.01, cycle_px=1.0):
    """Trusted positive labels on the *training* coarse grid; all others unknown.

    cycle_px is measured on the resized/padded training image, never raw pixels.
    Depth/cycle agreement is a quality check, not independent GT accuracy.
    """
    ta, _ = letterbox_geometry(a.width, a.height, size)
    tb, _ = letterbox_geometry(b.width, b.height, size)
    source = coarse_grid(size)
    original = map_points(source, np.linalg.inv(ta))
    za, good_a = sample_depth(a, original)
    ab = np.linalg.inv(b.c2w) @ a.c2w
    xb = xyz_transform(unproject(original, np.where(good_a, za, 1), a.k, a.distortion), ab)
    raw_target = project(xb, b.k, b.distortion)
    zb, good_b = sample_depth(b, raw_target)
    # Prevent distortion polynomial extrapolation folding out-of-FOV rays into RGB.
    inverse_ray = unproject(np.nan_to_num(raw_target), np.ones(len(source)), b.k, b.distortion)
    normalized = xb[:, :2] / np.where(np.abs(xb[:, 2:]) > 1e-8, xb[:, 2:], 1e-8)
    ray_px = np.linalg.norm(inverse_ray[:, :2] - normalized, axis=1) * np.mean([b.k[0,0], b.k[1,1]])
    back = xyz_transform(unproject(np.nan_to_num(raw_target), np.where(good_b, zb, 1),
                                  b.k, b.distortion), np.linalg.inv(ab))
    returned = map_points(project(back, a.k, a.distortion), ta)
    target = map_points(raw_target, tb)
    cycle = np.linalg.norm(returned-source, axis=1)
    residual = np.abs(xb[:,2]-zb) / np.maximum(np.maximum(np.abs(xb[:,2]), zb), 1e-6)
    valid = good_a & good_b & (xb[:,2] > 0) & (back[:,2] > 0) & (ray_px <= .5)
    valid &= np.isfinite(target).all(1) & (target >= 0).all(1) & (target <= size-1).all(1)
    valid &= np.isfinite(cycle) & (cycle <= cycle_px) & (residual <= relative_depth)
    # Conservative supervision mask: target cell center must remain in real RGB,
    # not in padding, even if its subpixel target itself is valid.
    cells = np.rint((target+.5)/16-.5)
    centers = (cells+.5)*16-.5
    original_centers = map_points(centers, np.linalg.inv(tb))
    valid &= (original_centers >= 0).all(1) & (original_centers <= [b.width-1,b.height-1]).all(1)
    weight = np.exp(-.5*(cycle/max(cycle_px,1e-8))**2 - .5*(residual/max(relative_depth,1e-8))**2)
    return {'target': np.where(valid[:,None], target, 0).astype(np.float32),
            'valid': valid, 'weight': np.where(valid, weight, 0).astype(np.float32),
            'cycle': np.where(valid, cycle, 0).astype(np.float32)}


def discover_uav(rgb_root, depth_root, year='2024'):
    scenes, rejected = [], []
    if rgb_root is None:
        return scenes, rejected
    for meta_path in sorted((Path(rgb_root)/year).glob('*/transforms.json')):
        meta = json.loads(meta_path.read_text(encoding='utf-8-sig'))
        frames = []
        for f in meta['frames']:
            def get(key, fallback=None): return f.get(key, meta.get(key, fallback))
            if get('camera_model') != 'OPENCV':
                raise ValueError(f'{meta_path}: only OPENCV cameras supported')
            image = meta_path.parent/f['file_path']
            depth = Path(depth_root)/year/(meta_path.parent.name+'_depth')/(image.stem+'_depth.tif')
            if not image.is_file() or not depth.is_file():
                rejected.append(str(image)); continue
            k = np.array([[get('fl_x'),0,get('cx')],[0,get('fl_y'),get('cy')],[0,0,1]],np.float64)
            d = np.array([get(key,0) for key in ('k1','k2','p1','p2','k3')],np.float64)
            pose = np.asarray(f['transform_matrix'],np.float64) @ np.diag([1.,-1.,-1.,1.])
            frames.append(Frame(image.name,image,depth,k,d,pose,int(get('w')),int(get('h'))))
        group = re.sub(r'^Day\d+_', '', meta_path.parent.name).casefold()
        if len(frames)>1:
            scenes.append(Scene('uav',meta_path.parent.name,'uav/'+group,frames))
    return scenes, rejected


def discover_disk(root, dataset_json=None):
    import h5py
    root = Path(root)
    dataset_json = Path(dataset_json) if dataset_json else root.parent/'dataset.json'
    catalog = json.loads(dataset_json.read_text()) if dataset_json.is_file() else {}
    scenes, rejected = [], []
    for scene in sorted(path for path in root.iterdir() if path.is_dir()):
        if not (scene/'images').is_dir(): continue
        entry = catalog.get(scene.name, {})
        names = entry.get('images') or sorted(p.name for p in (scene/'images').iterdir()
                                            if p.suffix.lower() in ('.jpg','.jpeg','.png'))
        frames, original_indices = [], {}
        for original_index, name in enumerate(names):
            name = Path(name).name
            image = scene/'images'/name
            calibration = scene/'calibration'/('calibration_'+name+'.h5')
            candidates = [scene/folder/(stem+ext) for folder in ('depth_maps','depth')
                          for stem in (Path(name).stem,name) for ext in ('.h5','.hdf5')]
            depth = next((p for p in candidates if p.is_file()), None)
            if not image.is_file() or not calibration.is_file() or depth is None:
                rejected.append(str(image)); continue
            with h5py.File(calibration,'r') as f:
                if not all(key in f for key in ('K','R','T')):
                    raise ValueError(f'{calibration}: expected K,R,T; got {list(f)}')
                k,r,t = (np.asarray(f[key],np.float64) for key in ('K','R','T'))
            if k.shape != (3,3) or r.shape != (3,3) or t.size != 3:
                raise ValueError(f'{calibration}: unsupported K/R/T shapes')
            # DISK uses COLMAP world-to-camera R,T, with OpenCV camera axes.
            w2c = np.eye(4); w2c[:3,:3] = r; w2c[:3,3] = t.reshape(3)
            from PIL import Image
            with Image.open(image) as im:
                w,h=im.size
            original_indices[original_index] = len(frames)
            frames.append(Frame(name,image,depth,k,np.zeros(5),np.linalg.inv(w2c),w,h))
        pairs = []
        for item in entry.get('tuples', []):
            for i in range(len(item)):
                for j in range(i+1,len(item)):
                    if item[i] in original_indices and item[j] in original_indices:
                        pairs.append((original_indices[item[i]],original_indices[item[j]]))
        if len(frames)>1:
            scenes.append(Scene('megadepth',scene.name,'megadepth/'+scene.name,frames,
                                list(dict.fromkeys(pairs)) if pairs else None))
    return scenes,rejected


def group_splits(scenes, seed=2026, override=None):
    groups = sorted({s.group for s in scenes})
    if override is not None:
        if set(override) != set(groups) or any(v not in ('train','val') for v in override.values()):
            raise ValueError('Split JSON must map EVERY discovered group to train or val')
        result = dict(override)
    else:
        result = {}
        rng = np.random.default_rng(seed)
        for domain in ('megadepth','uav'):
            selected = [g for g in groups if g.startswith(domain+'/')]
            if len(selected)<2:
                raise ValueError(f'{domain}: need >=2 distinct groups for isolated train/val; '
                                 'run --inspect and supply more groups or an explicit split JSON')
            if domain=='uav' and 'uav/corn2' in selected:
                validation = {'uav/corn2'}
            else:
                validation = set(rng.permutation(selected)[:max(1,round(.2*len(selected)))])
            result.update({g:'val' if g in validation else 'train' for g in selected})
    for domain in ('megadepth','uav'):
        if {result[g] for g in groups if g.startswith(domain+'/')} != {'train','val'}:
            raise ValueError(f'{domain}: each domain needs train AND val groups')
    return result


def candidate_pairs(scene, seed=2026, neighbors=16):
    rng = np.random.default_rng(seed)
    if scene.pairs:
        order = rng.permutation(len(scene.pairs))
        for index in order: yield scene.pairs[index]
        return
    positions = np.array([f.c2w[:3,3] for f in scene.frames])
    seen = set()
    for i in rng.permutation(len(scene.frames)):
        nearest = np.argsort(np.linalg.norm(positions-positions[i],axis=1))[1:neighbors+1]
        additional = rng.choice(len(scene.frames),min(neighbors,len(scene.frames)),replace=False)
        for j in np.r_[nearest, additional]:
            key = tuple(sorted((int(i),int(j))))
            if i != j and key not in seen:
                seen.add(key); yield key
