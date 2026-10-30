"""Read-only RGB/depth inventory and sample camera/depth reprojection audit.

Writes reports and diagnostic overlays only to --output. Never changes source data.
Requires numpy, opencv-python, tifffile (and imagecodecs for compressed TIFFs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile


def camera(metadata, frame):
    def value(key, default=None):
        return frame.get(key, metadata.get(key, default))
    if value('camera_model') != 'OPENCV':
        raise ValueError('Only the released OPENCV camera model is supported')
    k = np.array([[value('fl_x'), 0, value('cx')],
                  [0, value('fl_y'), value('cy')], [0, 0, 1]], np.float64)
    distortion = np.array([value(key, 0) for key in ('k1', 'k2', 'p1', 'p2', 'k3')], np.float64)
    # Nerfstudio camera-to-world uses OpenGL axes. Convert camera axes to OpenCV.
    c2w = np.asarray(frame['transform_matrix'], np.float64) @ np.diag([1., -1., -1., 1.])
    return k, distortion, c2w


def unproject(points, z, k, distortion):
    normalized = cv2.undistortPointsIter(
        points[:, None, :].astype(np.float64), k, distortion, None, None,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10))[:, 0]
    return np.c_[normalized, np.ones(len(points))] * z[:, None]


def transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def project(points, k, distortion):
    return cv2.projectPoints(points, np.zeros(3), np.zeros(3), k, distortion)[0][:, 0]


def sample_nearest(depth, points):
    finite = np.isfinite(points).all(1)
    safe = np.where(finite[:, None], points, 0)
    pixels = np.rint(safe).astype(np.int64)
    h, w = depth.shape
    inside = finite & (points[:, 0] >= 0) & (points[:, 0] <= w - 1)
    inside &= (points[:, 1] >= 0) & (points[:, 1] <= h - 1)
    values = depth[pixels[:, 1].clip(0, h - 1), pixels[:, 0].clip(0, w - 1)]
    return values, inside & np.isfinite(values) & (values > 0)


def depth_path(depth_scene, frame):
    return depth_scene / (Path(frame['file_path']).stem + '_depth.tif')


def overlay(a_path, b_path, source, target, accepted, output):
    images = [cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR) for p in (a_path, b_path)]
    if any(image is None for image in images):
        raise OSError('Could not decode RGB image')
    width, height = 640, 480
    canvas = np.concatenate([cv2.resize(im, (width, height)) for im in images], axis=1)
    ids = np.flatnonzero(accepted)
    if len(ids) > 45:
        ids = ids[np.linspace(0, len(ids) - 1, 45).astype(int)]
    for index in ids:
        pa = source[index] * [width / images[0].shape[1], height / images[0].shape[0]]
        pb = target[index] * [width / images[1].shape[1], height / images[1].shape[0]] + [width, 0]
        pa, pb = tuple(np.rint(pa).astype(int)), tuple(np.rint(pb).astype(int))
        cv2.line(canvas, pa, pb, (60, 220, 100), 1, cv2.LINE_AA)
        cv2.circle(canvas, pa, 3, (0, 230, 255), -1)
        cv2.circle(canvas, pb, 3, (0, 230, 255), -1)
    cv2.rectangle(canvas, (0, 0), (1280, 27), (20, 20, 20), -1)
    cv2.putText(canvas, f'Geometry reference; {len(ids)} displayed / {accepted.sum()} consistent samples',
                (12, 19), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode('.jpg', canvas)
    if not ok:
        raise OSError('Could not encode diagnostic overlay')
    encoded.tofile(output)


def probe(scene, depth_scene, metadata, first, second, output, rel_tol, cycle_tol):
    da = tifffile.imread(depth_path(depth_scene, first)).astype(np.float32)
    db = tifffile.imread(depth_path(depth_scene, second)).astype(np.float32)
    if da.ndim != 2 or db.ndim != 2:
        raise ValueError('Expected single-channel depth maps')
    h, w = da.shape
    if da.shape != (int(metadata['h']), int(metadata['w'])) or db.shape != da.shape:
        raise ValueError('Depth size differs from RGB camera metadata')
    y, x = np.meshgrid(np.linspace(0, h - 1, 40), np.linspace(0, w - 1, 48), indexing='ij')
    source = np.c_[x.ravel(), y.ravel()]
    z_a, good_a = sample_nearest(da, source)
    ka, dist_a, ca = camera(metadata, first)
    kb, dist_b, cb = camera(metadata, second)
    ab = np.linalg.inv(cb) @ ca
    xyz_b = transform(unproject(source, np.where(good_a, z_a, 1), ka, dist_a), ab)
    target = project(xyz_b, kb, dist_b)
    z_b, good_b = sample_nearest(db, target)
    # A distortion polynomial extrapolated outside the camera's field of view can
    # fold an invalid ray back into the image. Check the inverse ray as well.
    inverse_rays = unproject(np.nan_to_num(target), np.ones(len(target)), kb, dist_b)
    expected_rays = xyz_b[:, :2] / np.where(np.abs(xyz_b[:, 2:]) > 1e-8, xyz_b[:, 2:], 1e-8)
    inverse_ray_px = np.linalg.norm(inverse_rays[:, :2] - expected_rays, axis=1) * (kb[0, 0] + kb[1, 1]) / 2
    supported = good_a & good_b & (xyz_b[:, 2] > 0) & (inverse_ray_px <= .5)
    rel_error = np.abs(xyz_b[:, 2] - z_b) / np.maximum(np.maximum(np.abs(xyz_b[:, 2]), z_b), 1e-6)
    returned_xyz = transform(unproject(np.nan_to_num(target),
        np.where(good_b, z_b, 1), kb, dist_b), np.linalg.inv(ab))
    returned = project(returned_xyz, ka, dist_a)
    cycle = np.linalg.norm(returned - source, axis=1)
    accepted = supported & (returned_xyz[:, 2] > 0) & (rel_error <= rel_tol) & np.isfinite(cycle) & (cycle <= cycle_tol)
    pair = Path(first['file_path']).stem + '_' + Path(second['file_path']).stem
    overlay(scene / first['file_path'], scene / second['file_path'], source, target, accepted,
            output / (scene.name + '_' + pair + '.jpg'))
    valid_values = da[np.isfinite(da) & (da > 0)]
    median = lambda v: float(np.median(v)) if len(v) else None
    return {'scene': scene.name, 'pair': pair, 'depth_shape': list(da.shape),
        'sample_count': len(source), 'source_valid_fraction': float(good_a.mean()),
        'mutually_supported_count': int(supported.sum()), 'consistent_count': int(accepted.sum()),
        'consistent_fraction_of_source_samples': float(accepted.mean()),
        'relative_depth_error_median_supported': median(rel_error[supported]),
        'cycle_px_median_supported': median(cycle[supported]),
        'cycle_px_median_accepted': median(cycle[accepted]),
        'source_depth_median_m': median(valid_values),
        'baseline_m': float(np.linalg.norm(ca[:3, 3] - cb[:3, 3]))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rgb-root', type=Path, default=Path(r'E:\UAV3DCrop'))
    p.add_argument('--depth-root', type=Path, default=Path(r'E:\UAV3DCrop_depth'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--probe-scenes', type=int, default=3)
    p.add_argument('--relative-depth-tolerance', type=float, default=.01)
    p.add_argument('--cycle-px', type=float, default=2., help='Original-resolution pixels; diagnostic only')
    args = p.parse_args()
    if args.probe_scenes < 0 or min(args.relative_depth_tolerance, args.cycle_px) <= 0:
        p.error('Invalid probe count or geometry thresholds')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'rgb_root': str(args.rgb_root), 'depth_root': str(args.depth_root),
        'scope': '2024', 'scenes': [], 'probes': [], 'probe_errors': [],
        'diagnostic_thresholds': {'relative_depth': args.relative_depth_tolerance,
                                  'cycle_original_px': args.cycle_px},
        'caveat': 'Reference geometry from MVS, not independent exact GT. '
                  'Depth/cycle agreement does not prove visual correspondence accuracy.'}
    for root, manifest_name in ((args.rgb_root, 'download_2024_manifest.json'),
                                (args.depth_root, 'download_depth_2024_manifest.json')):
        manifest = json.loads((root / manifest_name).read_text(encoding='utf-8'))
        missing, wrong_size = [], []
        for entry in manifest['files']:
            path = root / entry['path']
            if not path.is_file():
                missing.append(entry['path'])
            elif path.stat().st_size != entry['size']:
                wrong_size.append(entry['path'])
        key = 'rgb_inventory' if root == args.rgb_root else 'depth_inventory'
        report[key] = {'files_expected': len(manifest['files']), 'missing': missing,
                       'wrong_size': wrong_size, 'revision': manifest['revision']}
    scenes = sorted(path.parent for path in (args.rgb_root / '2024').glob('*/transforms.json'))
    if not scenes:
        raise FileNotFoundError('No 2024 scene transforms.json found')
    paired_scenes = []
    for scene in scenes:
        metadata = json.loads((scene / 'transforms.json').read_text(encoding='utf-8-sig'))
        depth_scene = args.depth_root / '2024' / (scene.name + '_depth')
        frames = metadata['frames']
        both = [f for f in frames if (scene / f['file_path']).is_file() and depth_path(depth_scene, f).is_file()]
        row = {'scene': scene.name, 'camera_frames': len(frames), 'paired_files': len(both),
            'rgb_missing': [f['file_path'] for f in frames if not (scene / f['file_path']).is_file()],
            'depth_missing': [f['file_path'] for f in frames if not depth_path(depth_scene, f).is_file()],
            'camera_model': metadata.get('camera_model'),
            'distortion': {key: metadata.get(key, 0) for key in ('k1', 'k2', 'k3', 'p1', 'p2')}}
        report['scenes'].append(row)
        if len(both) > 1:
            paired_scenes.append((scene, depth_scene, metadata, both))
        print(f"{scene.name}: {len(both)}/{len(frames)} RGB/depth pairs", flush=True)
    if args.probe_scenes and paired_scenes:
        ids = np.linspace(0, len(paired_scenes) - 1, min(args.probe_scenes, len(paired_scenes))).astype(int)
        for index in ids:
            scene, ds, meta, frames = paired_scenes[index]
            for offset in sorted({min(step, len(frames) - 1) for step in (1, 10, 50)}):
                try:
                    result = probe(scene, ds, meta, frames[0], frames[offset], args.output,
                                   args.relative_depth_tolerance, args.cycle_px)
                    report['probes'].append(result)
                    print(json.dumps(result), flush=True)
                except Exception as exc:
                    report['probe_errors'].append({'scene': scene.name, 'offset': offset, 'error': str(exc)})
    (args.output / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Report:', args.output / 'audit.json', flush=True)
    if report['probe_errors']:
        raise RuntimeError('Some reprojection probes failed; see audit.json')


if __name__ == '__main__':
    main()
