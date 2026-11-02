"""Stage 2 (cheap): stage-1 perception -> PIF episode in the table frame (pif/data/tokens.py format).

  python -m pif.perception.lift --tasks S8789      (all extracted episodes; seconds per episode)

Table frame (same convention as the sim): z = table normal towards the camera side; origin = the camera centre
projected on the table (the camera sits between the arms on the robot side; the sim origin is under the robot base);
x = the optical axis projected on the table (robot side -> human); y = z x x (the robot's left).
Table plane: RANSAC + least squares on the depth samples of stage 1 (lower image half, outside detected boxes). One
focal length per episode (median, static camera); principal point = image centre (HABIT has no intrinsics).
Objects: per prompt, exactly `count` tracks (configs/pif/habit.yaml): detections are matched to tracks by 3D distance
(Hungarian), a new track starts only while fewer than `count` exist; a hidden object keeps its last position with
visible = 0 (causal hold). The robot gripper is detected only to decide `holding`: when a gripper of the robot is
closed (parquet + task threshold), the robot holds the `graspable` track nearest to a detected gripper in the image.
has_item / done are not built for HABIT yet (zeros).
"""
from __future__ import annotations
import argparse
import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from pif.utils import resolve
from pif.config import load_config
from pif.data.habit import gripper_closed, read_episode


def backproject(u, v, z, f, cx, cy) -> np.ndarray:
    return np.stack([(np.asarray(u) - cx) * z / f, (np.asarray(v) - cy) * z / f, np.asarray(z, float)], -1)


def fit_plane(points: np.ndarray, iters: int, inlier: float, rng) -> tuple[np.ndarray, float, float]:
    """RANSAC plane n.x + d = 0, refined by least squares on the inliers; d > 0 (camera at the origin on the positive
    side, so n points to the camera side). Returns n, d, inlier fraction."""
    best, best_n = None, -1
    for _ in range(iters):
        q = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(q[1] - q[0], q[2] - q[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs(points @ n - n @ q[0]) < inlier
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    q = points[best]
    c = q.mean(0)
    n = np.linalg.svd(q - c, full_matrices=False)[2][-1]       # thin SVD: the full one builds an N x N matrix
    d = -n @ c
    if d < 0:
        n, d = -n, -d
    return n, float(d), float(best_n / len(points))


def table_frame(n: np.ndarray, d: float) -> tuple[np.ndarray, np.ndarray]:
    """R (rows = table x, y, z in camera coordinates) and origin o: p_table = R (p_cam - o)."""
    x = np.array([0.0, 0.0, 1.0]) - n * n[2]
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(n, x), n]), -d * n


def track(frames: list[list[dict]], count: int) -> np.ndarray:
    """[T, count, 3] positions (NaN = not seen this frame) of one object kind, at most `count` tracks."""
    T = len(frames)
    pos = np.full((T, count, 3), np.nan)
    last = []
    for t, dets in enumerate(frames):
        xyz = [d['xyz'] for d in sorted(dets, key=lambda d: -d['score']) if np.isfinite(d['xyz']).all()]
        if last and xyz:
            cost = np.linalg.norm(np.array(last)[:, None] - np.array(xyz)[None], axis=-1)
            rows, cols = linear_sum_assignment(cost)
            used = set()
            for r, c in zip(rows, cols):
                pos[t, r], last[r] = xyz[c], xyz[c]
                used.add(c)
            xyz = [x for c, x in enumerate(xyz) if c not in used]
        for x in xyz:                                            # new tracks while fewer than `count` exist
            if len(last) < count:
                pos[t, len(last)] = x
                last.append(x)
    return pos


def lift_episode(ex: dict, cfg: dict, keypoints: list[str], object_types: list[str], rng) -> dict:
    lc, task = cfg['lift'], cfg['tasks'][str(ex['sid'])]
    T, (h, w) = int(ex['T']), ex['image_size']
    cx, cy, f = w / 2.0, h / 2.0, float(np.median(ex['focal']))
    puvz = ex['plane_uvz']
    pts = backproject(puvz[:, 0], puvz[:, 1], puvz[:, 2], f, cx, cy)
    pts = pts[np.isfinite(pts).all(1)]
    n, d, frac = fit_plane(pts, lc['table']['ransac_iters'], lc['table']['inlier_m'], rng)
    R, o = table_frame(n, d)
    to_table = lambda p: (p - o) @ R.T
    idx = [lc['keypoint_index'][k] for k in keypoints]
    uv, z = ex['kp2d'][:, idx], ex['kp_depth'][:, idx]
    human = to_table(backproject(uv[..., 0], uv[..., 1], z, f, cx, cy))
    b = lc['border_px']
    inside = (uv[..., 0] >= b) & (uv[..., 0] < w - b) & (uv[..., 1] >= b) & (uv[..., 1] < h - b)
    vis = (ex['kp_score'][:, idx] >= lc['kpt_thr']) & inside & np.isfinite(human).all(-1)
    human[~vis] = 0.0
    dets = json.loads(str(ex['dets_json']))
    for fd in dets:
        for dd in fd:
            x0, y0, x1, y1 = dd['box']
            dd['uv'] = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            dd['xyz'] = to_table(backproject(dd['uv'][0], dd['uv'][1], dd['z'], f, cx, cy))
    names, types, tracks = [], [], []
    for label, spec in task['objects'].items():
        p = track([[dd for dd in fd if dd['label'] == label] for fd in dets], spec['count'])
        for k in range(spec['count']):
            if np.isfinite(p[:, k]).all(-1).mean() >= lc['min_object_frac']:
                names.append(label)
                types.append(object_types.index(spec['type']))
                tracks.append(p[:, k])
    K = len(tracks)
    seen = np.stack([np.isfinite(p).all(-1) for p in tracks], 1) if K else np.zeros((T, 0), bool)
    held = np.zeros((T, K, 3))
    for k, p in enumerate(tracks):                               # causal hold of the last seen position
        last = np.zeros(3)
        for t in range(T):
            if seen[t, k]:
                last = p[t]
            held[t, k] = last
    # holding: graspable track nearest (image) to a detected gripper while a gripper is closed
    rob = read_episode(int(ex['index']), cfg['root'])
    closed = np.zeros(T, bool)
    for arm in ('robot0', 'robot1'):
        closed |= gripper_closed(rob, arm, task['gripper_threshold'], cfg['gripper_min_frames'])[:T]
    proj = lambda p: (lambda q: np.array([f * q[0] / q[2] + cx, f * q[1] / q[2] + cy]))(p @ R + o)
    grasp_k = [k for k, nm in enumerate(names) if nm == task['graspable']]
    holding, last = np.full(T, -1), (grasp_k[0] if grasp_k else -1)
    for t in np.flatnonzero(closed):
        grips = [dd['uv'] for dd in dets[t] if dd['label'] == cfg['perception']['gripper_prompt']]
        cand = [(min(np.linalg.norm(g - proj(held[t, k])) for g in grips), k) for k in grasp_k if seen[t, k]] if grips else []
        if cand:
            last = min(cand)[1]
        holding[t] = last
    return dict(fps=10.0, t=np.arange(T) / 10.0, human_pos=human, human_vis=vis, obj_pos=held, obj_vis=seen,
                obj_type=np.array(types, int), obj_has_item=np.zeros((T, K), bool), obj_done=np.zeros((T, K), bool),
                holding=holding, obj_names=names,
                meta=dict(domain='habit', sid=str(ex['sid']), task=str(ex['sid']), episode_index=int(ex['index']), stride=1,
                          focal=f, plane_inliers=frac, camera_height_m=d, R=R.tolist(), origin=o.tolist()))


def save_episode(ep: dict, path) -> None:
    arrays = {k: v for k, v in ep.items() if k not in ('meta', 'obj_names')}
    np.savez_compressed(path, **arrays, obj_names=np.array(ep['obj_names']), meta=json.dumps(ep['meta']))


def load_episode(path) -> dict:
    z = np.load(path)
    ep = {k: z[k] for k in z.files if k not in ('meta', 'obj_names')}
    ep.update(fps=float(ep['fps']), obj_names=[str(x) for x in z['obj_names']], meta=json.loads(str(z['meta'])))
    return ep


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/pif/pif.yaml')
    p.add_argument('--set', nargs='*', default=[])
    p.add_argument('--tasks', nargs='+', default=['S8789'])
    args = p.parse_args()
    full = load_config(args.config, args.set)
    cfg = full['habit']
    rng = np.random.default_rng(0)
    for sid in args.tasks:
        src, dst = resolve(cfg['extract_dir']) / sid, resolve(cfg['tokens_dir']) / sid
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.glob('episode_*.npz')):
            ep = lift_episode(dict(np.load(f)), cfg, full['tokens']['keypoints'], full['tokens']['object_types'], rng)
            save_episode(ep, dst / f.name)
            m = ep['meta']
            print(f"{sid} {f.stem}: objects {ep['obj_names']}, visible kp {ep['human_vis'].mean():.2f}, "
                  f"table inliers {m['plane_inliers']:.2f}, camera {m['camera_height_m']:.2f} m, "
                  f"holding {np.mean(ep['holding'] >= 0):.2f}", flush=True)


if __name__ == '__main__':
    main()
