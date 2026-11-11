"""Evaluate GRM on calibrated MegaDepth1500 pairs; GT is never a model input."""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import html
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.cache/megadepth_eval_dependencies'))
import cv2
import h5py
import numpy as np
from agri_eval_common import digest, letterbox_geometry, project, read_image, resize_transform
from agri_eval_adapter import Matcher


def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, Path): return str(x)
    return x


def write_json(path, obj):
    path.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def skew(t):
    x, y, z = t
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0.]])


def normalized(points, k):
    p = np.c_[points, np.ones(len(points))] @ np.linalg.inv(k).T
    return p[:, :2] / p[:, 2:]


def epi_error(pa, pb, ka, kb, rotation, translation):
    a = np.c_[normalized(pa, ka), np.ones(len(pa))]
    b = np.c_[normalized(pb, kb), np.ones(len(pb))]
    essential = skew(translation) @ rotation
    ea, eb = a @ essential.T, b @ essential
    residual = np.sum(b * ea, axis=1) ** 2
    da, db = np.sum(ea[:, :2] ** 2, axis=1), np.sum(eb[:, :2] ** 2, axis=1)
    return residual * (1 / np.maximum(da, 1e-30) + 1 / np.maximum(db, 1e-30))


def estimate_pose(pa, pb, ka, kb, pixel_threshold, seed):
    if len(pa) < 5: return None
    a, b = normalized(pa, ka), normalized(pb, kb)
    # Explicit four-axis mean: unlike a historical duplicated-entry focal mean,
    # this remains symmetric under the benchmark's square, anisotropic resize.
    focal = np.mean([ka[0, 0], ka[1, 1], kb[0, 0], kb[1, 1]])
    cv2.setRNGSeed(seed)
    try:
        e, mask = cv2.findEssentialMat(a, b, np.eye(3), method=cv2.RANSAC,
                                     prob=.99999, threshold=pixel_threshold/focal, maxIters=1000)
        if e is None or mask is None: return None
        best, best_n = None, 0
        for candidate in np.asarray(e).reshape(-1, 3, 3):
            n, r, t, recovered, _ = cv2.recoverPose(candidate, a, b, np.eye(3),
                                                   distanceThresh=1e9, mask=mask.copy())
            if n > best_n:
                best, best_n = (r, t.reshape(3), recovered.reshape(-1) > 0), n
        return best
    except cv2.error:
        return None


def angular_errors(r, t, gt_r, gt_t):
    rot = np.degrees(np.arccos(np.clip((np.trace(r.T @ gt_r)-1)/2, -1., 1.)))
    den = np.linalg.norm(t) * np.linalg.norm(gt_t)
    trans = np.degrees(np.arccos(np.clip(np.dot(t, gt_t)/den, -1., 1.))) if den > 1e-15 else np.inf
    return float(rot), float(min(trans, 180-trans))


def pose_auc(errors, threshold):
    values = np.r_[0., np.sort(errors)]
    recall = np.linspace(0., 1., len(values))
    end = np.searchsorted(values, threshold)
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return float(integrate(np.r_[recall[:end], recall[end-1]],
                              np.r_[values[:end], threshold]) / threshold)


@functools.lru_cache(maxsize=8)
def read_depth(path):
    with h5py.File(path, 'r') as f: depth = np.asarray(f['depth'], dtype=np.float32)
    if depth.ndim != 2: raise ValueError(f'Non-2D depth: {path}')
    return depth


def sample_depth(depth, points):
    inside = np.isfinite(points).all(1) & (points[:, 0] >= 0) & (points[:, 1] >= 0)
    inside &= (points[:, 0] <= depth.shape[1]-1) & (points[:, 1] <= depth.shape[0]-1)
    out = np.zeros(len(points), dtype=float)
    xy = np.rint(points[inside]).astype(int)
    out[inside] = depth[xy[:, 1], xy[:, 0]]
    return out


def preprocess_image(rgb,size,mode):
    if mode=='stretch':
        return cv2.resize(rgb,(size,size),interpolation=cv2.INTER_LINEAR),resize_transform(rgb.shape,size),None
    height,width=rgb.shape[:2]
    matrix,(left,top,nw,nh)=letterbox_geometry(width,height,size)
    canvas=np.zeros((size,size,3),np.uint8)
    valid=np.zeros((size,size),bool)
    canvas[top:top+nh,left:left+nw]=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_AREA)
    valid[top:top+nh,left:left+nw]=True
    return canvas,matrix,valid


def depth_errors(pa, pb, pair, shape_a, shape_b, relative_tolerance):
    da, db = (read_depth(str(pair[k])) for k in ['depth_a', 'depth_b'])
    if da.shape != shape_a[:2] or db.shape != shape_b[:2]:
        raise ValueError(f'Depth/RGB resolution mismatch: {pair["pair_id"]}')
    depth = sample_depth(da, pa)
    rays = np.c_[normalized(pa, pair['ka']), np.ones(len(pa))]
    xyz = (rays * depth[:, None]) @ pair['rotation'].T + pair['translation']
    projected = xyz @ pair['kb'].T
    gt = np.full_like(pb, np.nan, dtype=float)
    good_z = np.isfinite(xyz).all(1) & (xyz[:, 2] > 1e-8)
    gt[good_z] = projected[good_z, :2] / projected[good_z, 2:]
    target_depth = sample_depth(db, gt)
    discrepancy = np.abs(target_depth-xyz[:, 2])/np.maximum(np.maximum(target_depth, xyz[:, 2]), 1e-8)
    eligible = (depth > 0) & np.isfinite(depth) & good_z & (target_depth > 0)
    eligible &= np.isfinite(target_depth) & (discrepancy <= relative_tolerance)
    error = np.linalg.norm(pb-gt, axis=1)
    error[~eligible] = np.nan
    return gt, error, eligible, discrepancy


def inventory(root):
    lines = (root / 'pairs_calibrated.txt').read_text().splitlines()
    plain = [line.split() for line in (root / 'pairs.txt').read_text().splitlines() if line.strip()]
    views = {}
    for line in (root / 'views.txt').read_text().splitlines():
        t = line.split()
        v = np.array(t[1:13], float)
        if t[13] != 'PINHOLE': raise ValueError('Only PINHOLE views supported')
        width, height = map(int, t[14:16])
        fx, fy, cx, cy = map(float, t[16:20])
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
        views[t[0]] = (v[:9].reshape(3, 3), v[9:], k, (height, width))
    pairs, missing, seen = [], [], set()
    max_rotation_delta, max_translation_delta = 0., 0.
    for i, line in enumerate(lines):
        t = line.split()
        if len(t) != 32: raise ValueError(f'Expected 32 columns, line {i+1}')
        if t[:2] != plain[i]: raise ValueError('Plain/calibrated pair order mismatch')
        if tuple(t[:2]) in seen: raise ValueError('Duplicate directed pair')
        seen.add(tuple(t[:2]))
        v = np.array(t[2:], float)
        ka, kb, rotation, translation = v[:9].reshape(3,3), v[9:18].reshape(3,3), v[18:27].reshape(3,3), v[27:]
        ra, ta, vka, shape_a = views[t[0]]
        rb, tb, vkb, shape_b = views[t[1]]
        max_rotation_delta = max(max_rotation_delta, float(abs(rb@ra.T-rotation).max()))
        max_translation_delta = max(max_translation_delta, float(abs(tb-rb@ra.T@ta-translation).max()))
        if not np.allclose(ka, vka, atol=.002) or not np.allclose(kb, vkb, atol=.002):
            raise ValueError('Intrinsic mismatch between views and pairs')
        pair = dict(index=i, pair_id=f'{i:04d}_'+hashlib.sha256(line.encode()).hexdigest()[:12],
                    scene=t[0].split('/')[0], name_a=t[0], name_b=t[1], ka=ka, kb=kb,
                    rotation=rotation, translation=translation, shape_a=shape_a, shape_b=shape_b)
        for key, folder, name in [('image_a','images',t[0]), ('image_b','images',t[1]),
                                  ('depth_a','depths',str(Path(t[0]).with_suffix('.h5'))),
                                  ('depth_b','depths',str(Path(t[1]).with_suffix('.h5')))]:
            pair[key] = root / folder / name
            if not pair[key].is_file(): missing.append(str(pair[key]))
        pairs.append(pair)
    if len(pairs) != 1500 or len(plain) != 1500: raise ValueError('Expected exactly 1500 benchmark pairs')
    if missing: raise FileNotFoundError(f'Missing files ({len(missing)}): {missing[:5]}')
    if max_rotation_delta > 1e-4 or max_translation_delta > 1e-4: raise ValueError('Pose convention mismatch')
    return pairs, dict(pairs=len(pairs), unique_images=len({p[k] for p in pairs for k in ['name_a','name_b']}),
                       scenes={s:sum(p['scene']==s for p in pairs) for s in sorted({p['scene'] for p in pairs})},
                       missing_files=0, max_rotation_component_delta=max_rotation_delta,
                       max_translation_component_delta=max_translation_delta,
                       pair_sha256=digest(root/'pairs_calibrated.txt'), views_sha256=digest(root/'views.txt'))


def aggregate(rows):
    errors = [r['pose_error_deg'] if r['pose_error_deg'] is not None else np.inf for r in rows]
    rotation_errors=np.array([r['rotation_error_deg'] if r['rotation_error_deg'] is not None else np.inf for r in rows])
    translation_errors=np.array([r['translation_error_deg'] if r['translation_error_deg'] is not None else np.inf for r in rows])
    total_matches = sum(r['matches'] for r in rows)
    total_eligible = sum(r['depth_eligible'] for r in rows)
    result = dict(pairs=len(rows), pose_estimated=sum(r['status']=='ok' for r in rows),
                  pose_failures=sum(r['status']!='ok' for r in rows),
                  pose_error_median_deg=float(np.median(errors)),
                  pose_error_p90_deg=float(np.quantile(errors,.9)),
                  rotation_error_median_deg=float(np.median(rotation_errors)),
                  translation_error_median_deg=float(np.median(translation_errors)),
                  rotation_success_5deg=float(np.mean(rotation_errors<=5)),
                  translation_success_5deg=float(np.mean(translation_errors<=5)),
                  rotation_le5_translation_gt20_pairs=int(((rotation_errors<=5)&(translation_errors>20)).sum()),
                  micro_epipolar_precision_1e_4=sum(r['epi_correct_1e_4'] for r in rows)/max(total_matches,1),
                  macro_epipolar_precision_1e_4=float(np.mean([r['epi_precision_1e_4'] for r in rows])),
                  mean_matches=total_matches/len(rows), depth_eligible_matches=total_eligible,
                  depth_eligible_fraction=total_eligible/max(total_matches,1),
                  mean_matching_ms=1000*float(np.mean([r['matching_seconds'] for r in rows])),
                  mean_pipeline_ms=1000*float(np.mean([r['pipeline_seconds'] for r in rows])))
    for t in [5,10,20]:
        result[f'pose_auc_{t}deg'] = pose_auc(errors,t)
        result[f'pose_success_{t}deg'] = float(np.mean(np.asarray(errors)<=t))
    for t in [1,3,5,10]:
        result[f'depth_pck_{t}px_eligible'] = sum(r[f'depth_correct_{t}px'] for r in rows)/max(total_eligible,1)
    return result


def visualize(output, pairs, rows, top):
    # Diagnostic native image rendering, not manuscript artwork. GT affects colors only.
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 19)
    selected = {r['pair_id'] for r in sorted(rows,key=lambda r:r['pose_error_deg'] if r['pose_error_deg'] is not None else np.inf,reverse=True)[:top]}
    selected |= {r['pair_id'] for r in sorted(rows,key=lambda r:r['epi_precision_1e_4'])[:top]}
    selected |= {r['pair_id'] for r in rows[:10]}
    folder = output/'visualizations'; folder.mkdir(exist_ok=True)
    byid = {p['pair_id']:p for p in pairs}
    gallery = []
    for row in rows:
        if row['pair_id'] not in selected: continue
        pair = byid[row['pair_id']]
        a,b = (read_image(pair[k]) for k in ['image_a','image_b'])
        with np.load(output/row['matches_file']) as z:
            pa,pb,c,e,derr,eligible = (z[k] for k in ['points_a','points_b','confidence','epipolar_error','depth_error_px','depth_eligible'])
        canvas = Image.new('RGB',(1320,650),'white'); draw = ImageDraw.Draw(canvas)
        transforms=[]
        for image,x in [(a,10),(b,670)]:
            h,w=image.shape[:2];scale=min(640/w,490/h)
            thumb=Image.fromarray(image).resize((round(w*scale),round(h*scale)),Image.Resampling.LANCZOS)
            y=105+(490-thumb.height)//2
            canvas.paste(thumb,(x,y));transforms.append((scale,x,y))
        pose_label=f"{row['pose_error_deg']:.2f}" if row['pose_error_deg'] is not None else 'FAILED'
        draw.text((10,10),f"{row['pair_id']}  scene={row['scene']}  pose={pose_label} deg",font=font,fill='black')
        draw.text((10,37),f"Matches={len(pa)}  epipolar precision={100*row['epi_precision_1e_4']:.1f}%  depth-eligible={row['depth_eligible']}",font=font,fill='black')
        draw.text((10,64),'GT-color-stratified sparse lines (<=24); numeric scores use ALL matches.',font=font,fill='black')
        # Up to 12 high-confidence correct and 12 incorrect epipolar matches, spatially spread.
        picked=[]
        for mask in [e<1e-4,e>=1e-4]:
            candidates=np.flatnonzero(mask)
            if not len(candidates): continue
            current=int(candidates[np.argmax(c[candidates])]); group=[current]
            distance=np.sum((pa[candidates]-pa[current])**2,axis=1)
            for _ in range(min(12,len(candidates))-1):
                distance[np.isin(candidates,group)]=-1
                current=int(candidates[np.argmax(distance)]);group.append(current)
                distance=np.minimum(distance,np.sum((pa[candidates]-pa[current])**2,axis=1))
            picked.extend(group)
        for i in picked:
            s,x,y=transforms[0]; q1=(float(pa[i,0]*s+x),float(pa[i,1]*s+y))
            s,x,y=transforms[1]; q2=(float(pb[i,0]*s+x),float(pb[i,1]*s+y))
            color='#0072B2' if e[i]<1e-4 else '#D33F49'
            draw.line([q1,q2],fill=color,width=1)
            for q in [q1,q2]: draw.ellipse((q[0]-3,q[1]-3,q[0]+3,q[1]+3),outline=color,width=2)
        draw.text((10,615),'Blue: epi error <1e-4; red: >=1e-4. Depth references are MVS estimates; no H stitching assumed.',font=font,fill='black')
        filename=row['pair_id']+'.jpg';canvas.save(folder/filename,quality=94)
        gallery.append(f'<article><h3>{html.escape(pair["name_a"])} / {html.escape(pair["name_b"])}</h3>'
                       f'<a href="visualizations/{filename}"><img loading="lazy" width="100%" src="visualizations/{filename}"></a></article>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>MegaDepth1500 GRM诊断</title>'
       '<style>body{max-width:1350px;margin:30px auto;font:16px Arial}article{border-bottom:1px solid #ccc;margin:30px 0}</style>'
       '<h1>MegaDepth1500：位姿误差尾部、最低极线精度与前10对</h1><p>稀疏线按GT误差分层展示，不代表全部点的正确率。</p>'+''.join(gallery),encoding='utf-8')
    return len(selected)


def write_report(output, result, rows, args):
    table=[]
    for name,r in [('全部',result),*result['scenes'].items()]:
        table.append(f"| {name} | {r['pairs']} | {100*r['pose_auc_5deg']:.2f} | {100*r['pose_auc_10deg']:.2f} | "
                     f"{100*r['pose_auc_20deg']:.2f} | {100*r['micro_epipolar_precision_1e_4']:.2f} | "
                     f"{100*r['macro_epipolar_precision_1e_4']:.2f} | {r['pose_failures']} |")
    diagnostics=[]
    for row in rows:
        flags=[]
        if row['status']!='ok': flags.append('pose_estimation_failed')
        if row['pose_error_deg'] is None or row['pose_error_deg']>20: flags.append('pose_over_20deg')
        if row['epi_precision_1e_4']<.5: flags.append('epipolar_precision_below_50pct')
        if flags: diagnostics.append({**row,'diagnostic_flags':';'.join(flags)})
    fields=[*rows[0],'diagnostic_flags']
    with (output/'diagnostic_cases.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(diagnostics)
    counts={k:sum(k in r['diagnostic_flags'] for r in diagnostics)
            for k in ['pose_estimation_failed','pose_over_20deg','epipolar_precision_below_50pct']}
    write_json(output/'diagnostic_summary.json',{'counts':counts,'union_pairs':len(diagnostics),
               'note':'These overlapping thresholds are diagnostic criteria, not calibrated pass/fail standards.'})
    overlap_table=''
    if result.get('overlap_buckets'):
        overlap_table='\n## 按官方重叠率分组\n\n| 重叠率组 | 图对 | AUC@5° % | @10° % | @20° % | 极线精度micro % |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
        for bucket,r in result['overlap_buckets'].items():
            overlap_table+=f"| {bucket} | {r['pairs']} | {100*r['pose_auc_5deg']:.2f} | {100*r['pose_auc_10deg']:.2f} | {100*r['pose_auc_20deg']:.2f} | {100*r['micro_epipolar_precision_1e_4']:.2f} |\n"
        overlap_table+='\n组别来自本地LoFTR官方MegaDepth1500配对资产；以有方向的图像名逐对精确关联，见benchmark_pair_metadata.csv及其来源哈希。\n'
        overlap_table+='\n不同重叠率组的场景组成并不相同：0.5–0.7只有0022，不能把组间差异完全归因于重叠率。\n'
        overlap_table+='\n| 场景/重叠率 | 图对 | AUC@5° % | @10° % | @20° % | 极线精度macro % |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
        for key,r in result.get('scene_overlap_buckets',{}).items():
            overlap_table+=f"| {key} | {r['pairs']} | {100*r['pose_auc_5deg']:.2f} | {100*r['pose_auc_10deg']:.2f} | {100*r['pose_auc_20deg']:.2f} | {100*r['macro_epipolar_precision_1e_4']:.2f} |\n"
    (output/'REPORT.md').write_text(f'''# GRM V3：MegaDepth1500验证

权重：`{args.checkpoint}`。本次验证{len(rows)}对，输入{args.size}×{args.size}、FP32，预处理模式`{args.preprocess}`，保存配置中的默认采样器，不额外按置信度硬筛点。
可视化入口：[index.html](index.html)；全部逐对数据：[pairs.csv](pairs.csv)；规则命中清单：[diagnostic_cases.csv](diagnostic_cases.csv)。

| 场景 | 图对 | Pose AUC@5° % ↑ | @10° % ↑ | @20° % ↑ | 极线精度micro % ↑ | 极线精度macro % ↑ | 位姿求解失败 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table)}
{overlap_table}

## 指标含义与协议

位姿误差取旋转角误差和符号无关平移方向角误差的较大者；求解失败按无穷误差进入全量分母，未剔除失败样本。
极线精度采用归一化相机坐标下的平方对称极线距离<1e-4；micro按点汇总，macro先算每对再平均，零匹配图对精度记0。
满足极线约束并不保证二维点对应完全正确，所以另列深度重投影定位误差；两类指标不能互相替代。
指标族参考 [LoFTR官方评估实现](https://github.com/zju3dv/LoFTR/blob/master/src/utils/metrics.py)。

本次具体协议：RGB预处理`{args.preprocess}`，使用半像素坐标变换；E-RANSAC阈值{args.ransac_threshold}输入像素，置信度0.99999，最多1000次迭代。
stretch表示直接拉伸；letterbox表示与深度训练缓存相同的保持比例缩放、居中补零，传入有效RGB掩码，并在几何估计前剔除落在任一图像填充区的点。
归一化阈值采用两相机fx/fy四轴焦距均值；这与某些历史实现重复使用两项焦距的方式不同。
使用已知内参估计E和相对位姿，GT位姿只用于评分；不把GT深度、GT位姿或极线误差送入匹配模型。
这属于本项目输入协议，不能直接与论文中不同分辨率或不同RANSAC设置的数字作公平排名。

位姿误差中位数：{result['pose_error_median_deg']:.3f}°。平均输出{result['mean_matches']:.1f}个点。
旋转误差中位数{result['rotation_error_median_deg']:.3f}°，平移方向误差中位数{result['translation_error_median_deg']:.3f}°。
旋转≤5°占{100*result['rotation_success_5deg']:.2f}%，平移方向≤5°占{100*result['translation_success_5deg']:.2f}%。
其中旋转≤5°但平移方向>20°的有{result['rotation_le5_translation_gt20_pairs']}对；这些统计用于区分旋转/平移误差，不能单独证明几何退化或架构问题。
平均匹配耗时：{result['mean_matching_ms']:.1f}ms；平均完整单对处理耗时：{result['mean_pipeline_ms']:.1f}ms。
前者含图像张量转换、网络及结果取回，不含解码、深度读取与几何求解；后者还含这些步骤和匹配缓存写入，均不含3次预热。

## 深度重投影补充检查

从A预测源点读取MVS深度，结合内参和GT相对位姿投影到B。要求源深度、目标深度有效且为正，投影位于B内且z>0，
两者相对深度差≤{100*args.depth_relative_tolerance:g}%后才纳入深度定位统计。深度最近邻采样，误差单位为B原图像素。

- 有效深度支持点：{result['depth_eligible_matches']}，占全部预测点{100*result['depth_eligible_fraction']:.2f}%。
- 有效支持点PCK@1/3/5/10px：{100*result['depth_pck_1px_eligible']:.2f}% / {100*result['depth_pck_3px_eligible']:.2f}% / {100*result['depth_pck_5px_eligible']:.2f}% / {100*result['depth_pck_10px_eligible']:.2f}%。

这组PCK是条件统计，不能忽略支持率，也不能与合成数据的全量点PCK直接比较。MVS深度存在缺失、边缘误差和重建不确定性，不是精确人工/合成真值。

## 失败与薄弱案例

位姿求解失败{counts['pose_estimation_failed']}对；位姿误差>20°或求解失败{counts['pose_over_20deg']}对；
单对极线精度<50%的{counts['epipolar_precision_below_50pct']}对。去重后{len(diagnostics)}对，类别交叠。
50%为诊断阈值，不是经消融确定的通用合格标准。“求解成功”只表示E/位姿已返回，不代表估计正确。

全部1500对匹配保存在matches/；诊断页保存位姿误差最大的20对、极线精度最低的20对及前10对的并集，共{result['visualized_pairs']}张连线图。
连线图按GT极线误差分层抽取至多24条，故意展示部分错点；颜色比例不是全量正确率。全部点的坐标、置信度、GT误差与支持标记保存在NPZ中。
蓝色仅表示满足本次极线误差阈值，不等同于精确GT对应点。
没有将本数据当作单平面估计拼接H。

## 可比性边界

训练使用过部分MegaDepth，当前本地材料不能核实训练场景/图像是否与本测试集0015、0022重合。
在补齐HPC训练缓存清单之前，不能将本结果称为已排除泄漏的独立泛化成绩。
当前只测试此权重；未据此宣称优于V1、其他V3阶段或经典模型。

## 复现

```powershell
& 'D:/Software/Miniconda/envs/Competitions/python.exe' tools/evaluate_grm_megadepth1500.py --data-root "{args.data_root}" --checkpoint "{args.checkpoint}" --output "{args.output}" --size {args.size} --preprocess {args.preprocess} --limit {args.limit} --ransac-threshold {args.ransac_threshold} --depth-relative-tolerance {args.depth_relative_tolerance} --seed {args.seed} --resume
```

数据与权重校验见settings.json、source_fingerprints.json，逐场景结果见summary.json。
''',encoding='utf-8')


def attach_official_overlap(output, pairs, rows, directory):
    """Optional post-hoc metadata join; cannot alter inference or geometric scores."""
    if not directory.is_dir(): return {}
    lookup, sources = {}, {}
    for path in sorted(directory.glob('*.npz')):
        # Trusted local LoFTR assets store a pickled dict despite the .npz suffix.
        z=np.load(path,allow_pickle=True)
        sources[path.name]=digest(path)
        for (i,j),score,_ in z['pair_infos']:
            def name(k):
                p=Path(str(z['image_paths'][k]));return p.parts[-3]+'/'+p.name
            key=(name(i),name(j))
            if key in lookup: raise ValueError('Duplicate official directed pair')
            lookup[key]={'overlap_bucket':path.stem.split('_',1)[1],'official_overlap_score':float(score),'asset':path.name}
    if not lookup: return {}
    byid={p['pair_id']:p for p in pairs}
    metadata=[]
    for row in rows:
        pair=byid[row['pair_id']];key=(pair['name_a'],pair['name_b'])
        if key not in lookup: raise ValueError(f'No exact directed official pair metadata: {key}')
        info=lookup[key]
        row.update(overlap_bucket=info['overlap_bucket'],official_overlap_score=info['official_overlap_score'])
        metadata.append(dict(pair_id=pair['pair_id'],image_a=key[0],image_b=key[1],**info))
    with (output/'benchmark_pair_metadata.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(metadata[0]));w.writeheader();w.writerows(metadata)
    write_json(output/'benchmark_pair_metadata_sources.json',{'directory':str(directory),'asset_sha256':sources,
               'exact_directed_matches':len(rows),'official_directed_pairs':len(lookup),'reverse_only_matches':0})
    return {bucket:aggregate([r for r in rows if r['overlap_bucket']==bucket])
            for bucket in sorted({r['overlap_bucket'] for r in rows})}


def verify_outputs(output,pairs,rows,result,contract,args):
    from PIL import Image
    byid={p['pair_id']:p for p in pairs}
    if {r['pair_id'] for r in rows}!=set(byid) or len(rows)!=len(byid):
        raise ValueError('Incomplete/duplicate evaluation rows')
    total=0;max_epi_delta=0.;max_angle_delta=0.
    for row in rows:
        p=byid[row['pair_id']]
        with np.load(output/row['matches_file'],allow_pickle=False) as z:
            a,b,c,e,d,valid=(z[k] for k in ['points_a','points_b','confidence','epipolar_error','depth_error_px','depth_eligible'])
            if len(a)!=row['matches'] or len(b)!=len(a) or len(c)!=len(a): raise ValueError('Cache point count mismatch')
            if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(c).all(): raise ValueError('Nonfinite output')
            if (c<0).any() or (c>1).any(): raise ValueError('Invalid confidence range')
            truth=epi_error(a,b,p['ka'],p['kb'],p['rotation'],p['translation'])
            if not np.allclose(truth,e,rtol=1e-10,atol=1e-12): raise ValueError('Cached epipolar errors differ')
            max_epi_delta=max(max_epi_delta,float(np.max(abs(truth-e),initial=0.)))
            if int((e<1e-4).sum())!=row['epi_correct_1e_4']: raise ValueError('Epipolar precision count mismatch')
            if int(valid.sum())!=row['depth_eligible']: raise ValueError('Depth support count mismatch')
            for threshold in [1,3,5,10]:
                if int((d<=threshold).sum())!=row[f'depth_correct_{threshold}px']: raise ValueError('Depth score mismatch')
            if not np.isfinite(d[valid]).all() or np.isfinite(d[~valid]).any(): raise ValueError('Depth support/error mask mismatch')
            if row['status']=='ok':
                re,te=angular_errors(z['estimated_rotation'],z['estimated_translation'],p['rotation'],p['translation'])
                delta=max(abs(re-row['rotation_error_deg']),abs(te-row['translation_error_deg']))
                max_angle_delta=max(max_angle_delta,delta)
                if delta>1e-7: raise ValueError('Pose angle mismatch')
            total+=len(a)
    visuals=list((output/'visualizations').glob('*.jpg'))
    for path in visuals:
        with Image.open(path) as im: im.verify()
    if len(visuals)!=result['visualized_pairs']: raise ValueError('Diagnostic image count mismatch')
    if digest(args.checkpoint)!=contract['checkpoint_sha256']: raise ValueError('Checkpoint changed')
    if digest(args.data_root/'pairs_calibrated.txt')!=contract['inventory']['pair_sha256']: raise ValueError('Pair list changed')
    if digest(args.data_root/'views.txt')!=contract['inventory']['views_sha256']: raise ValueError('Calibration changed')
    report=dict(complete=True,pairs=len(rows),all_1500_pairs=len(rows)==1500,cache_files=len(rows),
                checked_point_rows=total,diagnostic_images=len(visuals),all_images_readable=True,
                max_epipolar_error_recompute_delta=max_epi_delta,max_angular_error_recompute_delta_deg=max_angle_delta,
                checkpoint_and_calibrations_unchanged=True,training_test_overlap_verified=False)
    write_json(output/'verification.json',report)
    return report


def self_check():
    rng=np.random.default_rng(53)
    ka=np.array([[800.,0,320],[0,810,240],[0,0,1]])
    kb=np.array([[780.,0,300],[0,790,235],[0,0,1]])
    r=cv2.Rodrigues(np.array([.03,.12,-.02]))[0];t=np.array([.6,.1,.05])
    world=np.c_[rng.uniform(-1,1,(120,2)),rng.uniform(4,9,120)]
    a=world@ka.T;a=a[:,:2]/a[:,2:]
    other=world@r.T+t;b=other@kb.T;b=b[:,:2]/b[:,2:]
    assert epi_error(a,b,ka,kb,r,t).max()<1e-20
    pose=estimate_pose(a,b,ka,kb,.5,2026);assert pose is not None
    re,te=angular_errors(*pose[:2],r,t);assert max(re,te)<.01,(re,te)
    assert pose_auc([np.inf,np.inf],5)==0
    assert abs(pose_auc([0.,0.],5)-1)<1e-12
    print('Geometry self-check PASS',re,te,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('D:/Program/Stitching/Dataset/megadepth1500/megadepth1500'))
    p.add_argument('--checkpoint',type=Path,default=Path('checkpoints/grm_v3_depth_mix_r1/best_agri.pt'))
    p.add_argument('--output',type=Path,default=ROOT/'evaluation_outputs/grm_megadepth1500_20261021/best_depth_mix_ep02')
    p.add_argument('--size',type=int,default=512)
    p.add_argument('--preprocess',choices=['stretch','letterbox'],default='stretch',help='Use letterbox to match the depth-cache training preprocessing; stretch preserves the historical diagnostic run')
    p.add_argument('--device',default='cuda')
    p.add_argument('--ransac-threshold',type=float,default=.5)
    p.add_argument('--depth-relative-tolerance',type=float,default=.05)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--self-check',action='store_true')
    p.add_argument('--official-pair-info-dir',type=Path,default=ROOT.parent/'LoFTR/assets/megadepth_test_1500_scene_info')
    args=p.parse_args()
    if args.self_check: self_check();return
    if args.size%32: raise ValueError('Input size must be divisible by 32')
    pairs,inv=inventory(args.data_root)
    if args.limit: pairs=pairs[:args.limit]
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'matches').mkdir(exist_ok=True)
    contract=dict(arguments=vars(args),inventory=inv,checkpoint_sha256=digest(args.checkpoint),
                  coordinate_convention='OpenCV integer pixel centers; half-pixel direct square resize',
                  preprocessing=f'RGB, saved normalization, FP32, {args.preprocess}; half-pixel transform; letterbox passes valid RGB masks and excludes padding matches',
                  protocol='E RANSAC, 0.5 input px by default, confidence .99999, maxIters 1000, 4-axis focal mean; no confidence hard filter',
                  primary='max(rotation error, sign-ambiguous translation angular error); AUC5/10/20 degrees; normalized squared symmetric epipolar precision 1e-4',
                  supplemental='A->B MVS depth reprojection, nearest depth sample, 5% relative depth agreement, conditional PCK in ORIGINAL B pixels',
                  reference='https://github.com/zju3dv/LoFTR/blob/master/src/utils/metrics.py',
                  comparison_caveat='Same metric family, custom GRM 512-square protocol; do not equate scores with literature settings. Training/test scene overlap not yet verified.')
    settings=args.output/'settings.json'
    if settings.exists():
        if not args.resume: raise FileExistsError('Output exists; use --resume for this exact evaluation')
        old=json.loads(settings.read_text(encoding='utf-8'))
        for k in ['checkpoint_sha256','inventory','protocol']:
            if old[k]!=clean(contract[k]): raise ValueError(f'Resume mismatch: {k}')
        for k in ['size','ransac_threshold','depth_relative_tolerance','limit','seed','data_root']:
            if old['arguments'][k]!=clean(vars(args)[k]): raise ValueError(f'Resume argument mismatch: {k}')
        if old['arguments'].get('preprocess','stretch')!=args.preprocess: raise ValueError('Resume preprocessing mismatch')
        if args.preprocess=='letterbox' and old.get('preprocessing')!=contract['preprocessing']:
            raise ValueError('Letterbox mask protocol changed; use a new output directory')
    else: write_json(settings,contract)
    journal=args.output/'pairs.jsonl'
    rows=[json.loads(line) for line in journal.read_text().splitlines() if line.strip()] if args.resume and journal.exists() else []
    done={r['pair_id'] for r in rows}
    if len(done)!=len(rows): raise ValueError('Duplicate progress records')
    matcher=Matcher('GRM',ROOT,SimpleNamespace(checkpoint=args.checkpoint,device=args.device,size=args.size))
    matcher.torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    matcher.torch.set_num_threads(1)
    cv2.setNumThreads(1)
    write_json(args.output/'model_metadata.json',matcher.metadata)
    # Exclude checkpoint loading and three warm-up calls from timing.
    first=pairs[0];warm=[preprocess_image(read_image(first[k]),args.size,args.preprocess) for k in ['image_a','image_b']]
    for _ in range(3): matcher(warm[0][0],warm[1][0],valid_a=warm[0][2],valid_b=warm[1][2])
    matcher.sync();started=time.perf_counter()
    with journal.open('a',encoding='utf-8') as f:
        for pair in pairs:
            if pair['pair_id'] in done: continue
            begin=time.perf_counter()
            a,b=(read_image(pair[k]) for k in ['image_a','image_b'])
            if a.shape[:2]!=pair['shape_a'] or b.shape[:2]!=pair['shape_b']: raise ValueError('RGB/view shape mismatch')
            ia,sa,va=preprocess_image(a,args.size,args.preprocess)
            ib,sb,vb=preprocess_image(b,args.size,args.preprocess)
            matcher.sync();tick=time.perf_counter();pa0,pb0,confidence=matcher(ia,ib,valid_a=va,valid_b=vb);matcher.sync();match_seconds=time.perf_counter()-tick
            if not np.isfinite(pa0).all() or not np.isfinite(pb0).all(): raise ValueError('Nonfinite predicted coordinates')
            pa,pb=project(pa0,np.linalg.inv(sa)),project(pb0,np.linalg.inv(sb))
            raw_count=len(pa)
            if args.preprocess=='letterbox':
                keep=np.ones(len(pa),bool)
                for points,shape in [(pa,a.shape),(pb,b.shape)]:
                    keep&=(points[:,0]>=0)&(points[:,1]>=0)&(points[:,0]<=shape[1]-1)&(points[:,1]<=shape[0]-1)
                pa0,pb0,pa,pb,confidence=(v[keep] for v in [pa0,pb0,pa,pb,confidence])
            epi=epi_error(pa,pb,pair['ka'],pair['kb'],pair['rotation'],pair['translation'])
            pose=estimate_pose(pa0,pb0,sa@pair['ka'],sb@pair['kb'],args.ransac_threshold,args.seed+pair['index'])
            if pose is None:
                re,te,inliers,pr,pt=math.inf,math.inf,np.zeros(len(pa),bool),np.full((3,3),np.nan),np.full(3,np.nan)
            else:
                pr,pt,inliers=pose;re,te=angular_errors(pr,pt,pair['rotation'],pair['translation'])
            target,depth_error,eligible,depth_discrepancy=depth_errors(pa,pb,pair,a.shape,b.shape,args.depth_relative_tolerance)
            rel=f'matches/{pair["pair_id"]}.npz'
            np.savez_compressed(args.output/rel,points_a=pa,points_b=pb,confidence=confidence,
                                epipolar_error=epi,depth_error_px=depth_error,depth_eligible=eligible,
                                depth_target=target,relative_depth_discrepancy=depth_discrepancy,
                                estimated_rotation=pr,estimated_translation=pt,pose_inlier_mask=inliers)
            row=dict(pair_id=pair['pair_id'],index=pair['index'],scene=pair['scene'],image_a=pair['name_a'],image_b=pair['name_b'],
                     raw_matches=raw_count,padding_matches_removed=raw_count-len(pa),
                     status='ok' if pose is not None else 'pose_failed',matches=len(pa),pose_inliers=int(inliers.sum()),
                     rotation_error_deg=re,translation_error_deg=te,pose_error_deg=max(re,te),
                     epi_correct_1e_4=int((epi<1e-4).sum()),epi_precision_1e_4=float((epi<1e-4).mean()) if len(pa) else 0.,
                     depth_eligible=int(eligible.sum()),depth_eligible_fraction=float(eligible.mean()) if len(pa) else 0.,
                     **{f'depth_correct_{t}px':int((depth_error<=t).sum()) for t in [1,3,5,10]},
                     matching_seconds=match_seconds,pipeline_seconds=time.perf_counter()-begin,matches_file=rel)
            row=clean(row);rows.append(row);f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
            if len(rows)%25==0 or len(rows)==1 or len(rows)==len(pairs):
                result=aggregate(rows);print(json.dumps({'completed':len(rows),'total':len(pairs),'elapsed_seconds':round(time.perf_counter()-started,1),
                    'pose_auc_5deg':result['pose_auc_5deg'],'epi_precision':result['micro_epipolar_precision_1e_4'],
                    'last_pose_error_deg':row['pose_error_deg']},allow_nan=False),flush=True)
                write_json(args.output/'summary_partial.json',result)
    rows.sort(key=lambda r:r['index'])
    overlap_result=attach_official_overlap(args.output,pairs,rows,args.official_pair_info_dir)
    with (args.output/'pairs.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=aggregate(rows);result['scenes']={scene:aggregate([r for r in rows if r['scene']==scene]) for scene in inv['scenes'] if any(r['scene']==scene for r in rows)}
    result['overlap_buckets']=overlap_result
    result['scene_overlap_buckets']={key:aggregate([r for r in rows if r.get('scene')+'/'+r.get('overlap_bucket','unknown')==key])
         for key in sorted({r['scene']+'/'+r['overlap_bucket'] for r in rows if 'overlap_bucket' in r})}
    result['visualized_pairs']=visualize(args.output,pairs,rows,20)
    result['complete']=len(rows)==len(pairs);result['all_1500_pairs']=len(rows)==1500
    if digest(args.checkpoint)!=contract['checkpoint_sha256']:
        raise ValueError('Checkpoint changed during evaluation')
    write_json(args.output/'summary.json',result)
    write_report(args.output,result,rows,args)
    verify_outputs(args.output,pairs,rows,result,contract,args)
    write_json(args.output/'source_fingerprints.json',{'checkpoint_sha256':digest(args.checkpoint),
               'pairs_sha256':digest(args.data_root/'pairs_calibrated.txt'),'script_sha256':digest(Path(__file__)),
               'adapter_sha256':digest(ROOT/'agri_eval_adapter.py'),'data_root':str(args.data_root),
               'model_python_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'grm_matcher').rglob('*.py')}})
    print('COMPLETE',json.dumps(clean(result),ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__': main()
