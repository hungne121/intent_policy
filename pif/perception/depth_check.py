"""Metric accuracy of the estimated 3D on HABIT S8789, against a robot-kinematics reference (PIF_P1 §3.5).

Within one arm's base frame, the distance between the release clusters of two adjacent containers is their true
distance: containers 1-2 = 0.248 m (robot0), 3-4 = 0.260 m (robot1); cluster spread std 1.5-2.5 cm
(outputs/habit/label_check/s8789_targets.json, pif/data/habit_label_check.py).

For each episode: Grounding DINO on the first frames (the person waits, the arms are away), frames with exactly 4
containers kept; the table plane from the depth of those frames (RANSAC); each container's table-frame centre by
  (a) depth: median depth in the central part of its box, back-projected;
  (b) plane: the camera ray through the middle of the box's bottom edge intersected with the table plane (needs no
      per-frame depth, only the plane and the focal length);
containers sorted left to right (table +y = robot's left); error = |d12 - 0.248|, |d34 - 0.260| (median over frames).
Writes <out>/depth_check_<model>.json.
"""
import argparse
import json
import time

import numpy as np
import yaml

from intent_policy.utils import resolve
from pif.data.habit import episode_indices, paths
from pif.perception.habit_perception import Detector, backproject, decode, fit_plane, make_depth, table_frame

REF = {'d12': 0.248, 'd34': 0.260}


def ray_plane(u, v, f, cx, cy, n, d):
    ray = np.array([(u - cx) / f, (v - cy) / f, 1.0])
    s = -d / (n @ ray)
    return s * ray


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/pif/habit_perception.yaml')
    p.add_argument('--model', help='override depth.model (e.g. Ruicheng/moge-2-vits-normal, apple/DepthPro-hf)')
    p.add_argument('--episodes', type=int, default=10, help='number of S8789 episodes (spread over the task)')
    p.add_argument('--frames', type=int, default=5, help='frames per episode, from the first 3 s')
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', default='docs/reports/PIF_P1')
    args = p.parse_args()
    cfg = yaml.safe_load(resolve(args.config).read_text())
    if args.model:
        cfg['depth']['model'] = args.model
    depth_model = make_depth(cfg['depth'], args.device)
    det = Detector(cfg['detect'], {'airtight container': 'container'}, args.device)
    rng = np.random.default_rng(0)
    eps = episode_indices('S8789')
    eps = [eps[i] for i in np.linspace(0, len(eps) - 1, args.episodes).round().astype(int)]
    rows, t_depth = [], []
    for e in eps:
        frames = decode(paths(e)[1])[:30:max(1, 30 // args.frames)][:args.frames]
        h, w = frames[0].shape[:2]
        cx, cy = w / 2, h / 2
        per = []
        for rgb in frames:
            boxes = sorted([d['box'] for d in det(rgb)], key=lambda b: b[0])
            t0 = time.time()
            depth, f = depth_model(rgb)
            t_depth.append(time.time() - t0)
            per.append((rgb, boxes, depth, f))
        f = float(np.median([x[3] for x in per]))
        pts = []
        for _, boxes, depth, _ in per:
            vv, uu = np.mgrid[int(h * cfg['table']['min_row_frac']):h:4, 0:w:4]
            mask = np.ones_like(vv, bool)
            for b in boxes:                                      # table points only: drop the containers' pixels
                mask &= ~((uu >= b[0]) & (uu <= b[2]) & (vv >= b[1]) & (vv <= b[3]))
            pts.append(backproject(uu[mask], vv[mask], depth[vv[mask], uu[mask]], f, cx, cy))
        pts = np.concatenate(pts)
        pts = pts[np.isfinite(pts).all(1)]
        n, d, frac = fit_plane(pts, cfg['table']['ransac_iters'], cfg['table']['inlier_m'], rng)
        R, o = table_frame(n, d)
        res = dict(a=[], b=[])
        for _, boxes, depth, _ in per:
            if len(boxes) != 4:
                continue
            ca, cb = [], []
            for x0, y0, x1, y1 in boxes:
                m = cfg['detect']['core_frac'] / 2
                core = depth[int(y0 + (0.5 - m) * (y1 - y0)):int(y0 + (0.5 + m) * (y1 - y0)) + 1,
                             int(x0 + (0.5 - m) * (x1 - x0)):int(x0 + (0.5 + m) * (x1 - x0)) + 1]
                z = np.nanmedian(core)
                ca.append((backproject((x0 + x1) / 2, (y0 + y1) / 2, z, f, cx, cy) - o) @ R.T)
                cb.append((ray_plane((x0 + x1) / 2, y1, f, cx, cy, n, d) - o) @ R.T)
            for key, c in (('a', ca), ('b', cb)):
                c = sorted(c, key=lambda q: -q[1])                   # left to right = decreasing table y
                res[key].append(dict(d12=float(np.linalg.norm((c[0] - c[1])[:2])), d34=float(np.linalg.norm((c[2] - c[3])[:2])),
                                     d23=float(np.linalg.norm((c[1] - c[2])[:2]))))
        row = dict(episode=e, focal_px=round(f, 1), plane_inliers=round(frac, 3), camera_height_m=round(d, 3),
                   frames_with_4=len(res['a']))
        for key in ('a', 'b'):
            if res[key]:
                for k in ('d12', 'd34', 'd23'):
                    row[f'{key}_{k}'] = round(float(np.median([r[k] for r in res[key]])), 3)
        rows.append(row)
        print(row, flush=True)
    summary = dict(model=cfg['depth']['model'], reference_m=REF, depth_s_per_frame=round(float(np.median(t_depth)), 3),
                   episodes=rows)
    for key in ('a', 'b'):
        err = [abs(r[f'{key}_{k}'] - REF[k]) for r in rows for k in REF if f'{key}_{k}' in r]
        if err:
            summary[f'{key}_abs_error_m'] = dict(median=round(float(np.median(err)), 3), p90=round(float(np.percentile(err, 90)), 3),
                                                n=len(err))
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = cfg['depth']['model'].split('/')[-1]
    (out / f'depth_check_{name}.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'episodes'}, indent=1))


if __name__ == '__main__':
    main()
