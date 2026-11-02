"""Stage 1 (heavy, once per episode): per-frame raw perception of a HABIT camera video.

  python -m pif.perception.extract --tasks S8789 [--per-task N | --episodes ...] [--shard i/n]

Models come from configs/pif/habit.yaml `habit.perception` (pif/perception/components.py). Per frame it stores the
whole-body 2D keypoints + scores, the depth around every keypoint, the focal length, the detected objects (box, score,
label, depth at the box centre) and, on a few frames, table-candidate depth samples. Everything geometric (table
frame, 3D, tracks, visibility) is stage 2 (pif/perception/lift.py), cheap to redo with other settings.
The pose model (CPU) runs in a thread while the GPU models work on the same frame.
Output: <extract_dir>/<sid>/episode_NNNNNN.npz; log in outputs/pif_extract/<date>/<time>_<tasks>.
"""
from __future__ import annotations
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import av
import numpy as np

from pif.utils import log_run, resolve, run_dir
from pif.config import build, load_config
from pif.data.habit import episode_indices, paths
import pif.perception.components  # noqa: F401  (registers the models)


def decode(video) -> list[np.ndarray]:
    with av.open(str(video)) as c:
        return [f.to_ndarray(format='rgb24') for f in c.decode(video=0)]


def depth_at(depth: np.ndarray, u, v, win: int) -> np.ndarray:
    """Median depth in a win x win window around each (u, v); NaN outside the image or without values."""
    h, w = depth.shape
    r = win // 2
    out = np.full(len(u), np.nan)
    for i, (x, y) in enumerate(zip(np.round(u).astype(int), np.round(v).astype(int))):
        if 0 <= x < w and 0 <= y < h:
            patch = depth[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
            if np.isfinite(patch).any():
                out[i] = np.nanmedian(patch)
    return out


def box_depth(depth: np.ndarray, box, core: float) -> float:
    x0, y0, x1, y1 = box
    m = core / 2
    patch = depth[int(y0 + (0.5 - m) * (y1 - y0)):int(y0 + (0.5 + m) * (y1 - y0)) + 1,
                  int(x0 + (0.5 - m) * (x1 - x0)):int(x0 + (0.5 + m) * (x1 - x0)) + 1]
    return float(np.nanmedian(patch)) if patch.size and np.isfinite(patch).any() else float('nan')


def extract_episode(index: int, sid: str, models: dict, cfg: dict) -> dict:
    pc = cfg['perception']
    frames = decode(paths(index, cfg['root'], cfg['camera'])[1])
    T, (h, w) = len(frames), frames[0].shape[:2]
    kp2d, kpsc, kpz = np.zeros((T, 133, 2), np.float32), np.zeros((T, 133), np.float32), np.zeros((T, 133), np.float32)
    focal = np.zeros(T)
    dets, plane = [], []
    plane_frames = set(np.linspace(0, T - 1, pc['plane_frames']).round().astype(int).tolist())
    with ThreadPoolExecutor(1) as pool:
        fut = pool.submit(models['pose'], frames[0])
        for t, rgb in enumerate(frames):
            depth, focal[t] = models['depth'](rgb)
            fd = models['detect'](rgb)
            kp2d[t], kpsc[t] = fut.result()
            if t + 1 < T:
                fut = pool.submit(models['pose'], frames[t + 1])
            kpz[t] = depth_at(depth, kp2d[t, :, 0], kp2d[t, :, 1], pc['depth_window_px'])
            for d in fd:
                d['z'] = box_depth(depth, d['box'], pc['object_core_frac'])
            dets.append([dict(label=d['label'], score=round(d['score'], 3), box=d['box'].round(1).tolist(),
                              z=round(d['z'], 4)) for d in fd])
            if t in plane_frames:                       # lower half, outside every detected box, every 4th pixel
                vv, uu = np.mgrid[h // 2:h:4, 0:w:4]
                keep = np.ones(vv.shape, bool)
                for d in fd:
                    x0, y0, x1, y1 = d['box']
                    keep &= ~((uu >= x0) & (uu <= x1) & (vv >= y0) & (vv <= y1))
                z = depth[vv[keep], uu[keep]]
                plane.append(np.c_[uu[keep], vv[keep], z, np.full(z.shape, t)].astype(np.float32))
    return dict(index=index, sid=sid, T=T, image_size=np.array([h, w]), kp2d=kp2d, kp_score=kpsc, kp_depth=kpz,
                focal=focal, dets_json=json.dumps(dets), plane_uvz=np.concatenate(plane),
                models=json.dumps({k: pc[k] for k in ('pose', 'depth', 'detect')}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/pif/habit.yaml')
    p.add_argument('--set', nargs='*', default=[])
    p.add_argument('--tasks', nargs='+', default=['S8789'])
    p.add_argument('--per-task', type=int, help='N episodes spread evenly over each task')
    p.add_argument('--episodes', nargs='*', type=int)
    p.add_argument('--shard', default='0/1', help='i/n: every n-th episode from i (parallel runs)')
    p.add_argument('--log-dir')
    args = p.parse_args()
    cfg = load_config(args.config, args.set)['habit']
    log_run(run_dir('pif_extract', '_'.join(args.tasks), args.log_dir), 'pif.perception.extract')
    jobs = []
    for sid in args.tasks:
        eps = episode_indices(sid, cfg['root'])
        if args.episodes:
            eps = [e for e in eps if e in args.episodes]
        elif args.per_task:
            eps = [eps[i] for i in np.unique(np.linspace(0, len(eps) - 1, args.per_task).round().astype(int))]
        jobs += [(sid, e) for e in eps]
    si, sn = map(int, args.shard.split('/'))
    jobs = jobs[si::sn]
    pc = cfg['perception']
    models = dict(pose=build('pose', pc['pose']), depth=build('depth', pc['depth']))
    t0, done = time.time(), 0
    for n, (sid, e) in enumerate(jobs):
        dst = resolve(cfg['extract_dir']) / sid / f'episode_{e:06d}.npz'
        if dst.exists():
            continue
        if models.get('sid') != sid:
            prompts = list(cfg['tasks'][sid]['objects']) + [pc['gripper_prompt']]
            models.update(sid=sid, detect=build('detect', pc['detect'], prompts=prompts))
        te = time.time()
        r = extract_episode(e, sid, models, cfg)
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, **r)
        done += 1
        dt = time.time() - te
        print(f"[{n + 1}/{len(jobs)}] {sid} ep {e}: {r['T']} frames, {dt / r['T']:.2f} s/frame, focal {np.median(r['focal']):.0f}px "
              f"[total {(time.time() - t0) / 3600:.2f} h]", flush=True)


if __name__ == '__main__':
    main()
