"""Checks of the HABIT perception.

  python -m pif.perception.tools depth-check [--set habit.perception.depth.resolution_level=0] [--episodes 20]
      metric accuracy on S8789 against the robot-kinematics reference: within one arm's base frame the distance between
      the release clusters of two adjacent containers is their true distance (main session; pif/data/habit.py labels).
      First frames of each episode (the person waits, the arms are away), frames with exactly 4 containers; container
      centre = depth at the box centre, back-projected; error = |d12 - ref|, |d34 - ref| (median over frames).
  python -m pif.perception.tools view --tasks S8789 [--episodes ...]
      side-by-side video of an episode: camera + keypoints + boxes | top and side view of the 3D tokens.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np

from pif.utils import resolve
from pif.config import build, load_config
from pif.data.habit import episode_indices, load_labels, paths
from pif.perception.extract import box_depth, decode
from pif.perception.lift import backproject, fit_plane, load_episode, table_frame
import pif.perception.components  # noqa: F401


def depth_check(full: dict, n_episodes: int, n_frames: int, out: Path) -> dict:
    cfg = full['habit']
    pc = cfg['perception']
    lab = load_labels(cfg, 'S8789')
    ref = lab['reference_m']
    trusted = {r['episode'] for r in lab['episodes'] if r.get('trusted')}
    depth = build('depth', pc['depth'])
    det = build('detect', pc['detect'], prompts=['airtight container'])
    rng = np.random.default_rng(0)
    eps = [e for e in episode_indices('S8789', cfg['root']) if e in trusted]
    eps = [eps[i] for i in np.linspace(0, len(eps) - 1, n_episodes).round().astype(int)]
    rows, errs = [], []
    for e in eps:
        frames = decode(paths(e, cfg['root'], cfg['camera'])[1])[:30:max(1, 30 // n_frames)][:n_frames]
        h, w = frames[0].shape[:2]
        per = [(sorted([d['box'] for d in det(rgb)], key=lambda b: b[0]), *depth(rgb)) for rgb in frames]
        f = float(np.median([x[2] for x in per]))
        pts = []
        for boxes, dmap, _ in per:
            vv, uu = np.mgrid[h // 2:h:4, 0:w:4]
            keep = np.ones(vv.shape, bool)
            for b in boxes:
                keep &= ~((uu >= b[0]) & (uu <= b[2]) & (vv >= b[1]) & (vv <= b[3]))
            pts.append(backproject(uu[keep], vv[keep], dmap[vv[keep], uu[keep]], f, w / 2, h / 2))
        pts = np.concatenate(pts)
        n, d, _ = fit_plane(pts[np.isfinite(pts).all(1)], cfg['lift']['table']['ransac_iters'], cfg['lift']['table']['inlier_m'], rng)
        R, o = table_frame(n, d)
        dist = []
        for boxes, dmap, _ in per:
            if len(boxes) != 4:
                continue
            c = [(backproject((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, box_depth(dmap, b, pc['object_core_frac']), f, w / 2, h / 2) - o) @ R.T
                 for b in boxes]
            c = sorted(c, key=lambda q: -q[1])                       # left to right = decreasing table y
            dist.append([np.linalg.norm((c[0] - c[1])[:2]), np.linalg.norm((c[2] - c[3])[:2])])
        if dist:
            d12, d34 = np.median(dist, 0)
            errs += [abs(d12 - ref['d12']), abs(d34 - ref['d34'])]
            rows.append(dict(episode=e, d12=round(float(d12), 3), d34=round(float(d34), 3), frames=len(dist)))
            print(rows[-1], flush=True)
    res = dict(depth=pc['depth'], reference_m=ref, episodes=rows,
               abs_error_m=dict(median=round(float(np.median(errs)), 4), p90=round(float(np.percentile(errs, 90)), 4), n=len(errs)))
    out.mkdir(parents=True, exist_ok=True)
    name = f"depth_check_{pc['depth']['name']}_L{pc['depth'].get('resolution_level', '')}.json"
    (out / name).write_text(json.dumps(res, indent=1))
    print(res['abs_error_m'])
    return res


COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
          (240, 50, 230), (210, 245, 60), (250, 190, 212)]
BONES = [('r_shoulder', 'r_elbow'), ('r_elbow', 'r_wrist'), ('l_shoulder', 'l_elbow'), ('l_elbow', 'l_wrist'),
         ('r_wrist', 'r_index_base'), ('r_index_base', 'r_index_tip'), ('l_wrist', 'l_index_base'),
         ('l_index_base', 'l_index_tip'), ('r_shoulder', 'l_shoulder')]


def scene_panel(ep: dict, t: int, names: list[str], size) -> np.ndarray:
    """Top view (x up = away from the robot, y left = robot's left) above a side view (x right, height up)."""
    w, h = size
    img = np.full((h, w, 3), 255, np.uint8)
    p = np.concatenate([ep['human_pos'][ep['human_vis']], ep['obj_pos'][ep['obj_vis']]])
    lo, hi = np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)
    x0, x1 = min(lo[0], 0.0) - 0.05, hi[0] + 0.1
    ymax = max(abs(lo[1]), abs(hi[1])) + 0.1
    th = int(h * 0.62)
    s = min((th - 20) / (x1 - x0), (w - 20) / (2 * ymax))
    top = lambda q: (int(w / 2 - q[1] * s), int(th - 10 - (q[0] - x0) * s))
    ss = min((w - 20) / (x1 - x0), (h - th - 25) / (max(hi[2] + 0.1, 0.5) + 0.05))
    side = lambda q: (int(10 + (q[0] - x0) * ss), int(h - 15 - q[2] * ss))
    cv2.line(img, (0, th), (w, th), (180, 180, 180), 1)
    cv2.putText(img, 'top view (table frame)', (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
    cv2.putText(img, 'side view: height above table', (5, th + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
    cv2.circle(img, top(np.zeros(3)), 4, (0, 0, 0), -1)
    cv2.putText(img, 'camera / robot', (top(np.zeros(3))[0] + 5, top(np.zeros(3))[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.line(img, side((x0, 0, 0)), side((x1, 0, 0)), (120, 80, 40), 2)
    for k, nm in enumerate(ep['obj_names']):
        if not ep['obj_vis'][:t + 1, k].any():
            continue
        col = COLORS[k % len(COLORS)]
        for proj in (top, side):
            cv2.circle(img, proj(ep['obj_pos'][t, k]), 6, col, -1 if ep['obj_vis'][t, k] else 1)
        u, v = top(ep['obj_pos'][t, k])
        tag = f'{k}:{nm[:9]}' + (' (held)' if ep['holding'][t] == k else '')
        cv2.putText(img, tag, (u + 7, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)
    J = {n: i for i, n in enumerate(names)}
    kp, vis = ep['human_pos'][t], ep['human_vis'][t]
    for side_name, col in (('r', (200, 0, 0)), ('l', (0, 0, 200))):
        j = J[f'{side_name}_wrist']
        tr = [ep['human_pos'][s_, j] for s_ in range(max(0, t - 20), t + 1) if ep['human_vis'][s_, j]]
        for a, b in zip(tr[:-1], tr[1:]):
            cv2.line(img, top(a), top(b), tuple(int(c * 0.5 + 127) for c in col), 1)
    for a, b in BONES:
        if vis[J[a]] and vis[J[b]]:
            for proj in (top, side):
                cv2.line(img, proj(kp[J[a]]), proj(kp[J[b]]), (200, 0, 0) if a[0] == 'r' else (0, 0, 200), 2)
    for nm, i in J.items():
        if vis[i]:
            for proj in (top, side):
                cv2.circle(img, proj(kp[i]), 3, (0, 0, 0), -1)
    cv2.putText(img, 'red = right hand, blue = left; hollow = object not detected now', (5, h - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (90, 90, 90), 1)
    return img


def view(full: dict, sid: str, episode: int, sheet: int = 6) -> Path:
    cfg = full['habit']
    names = full['tokens']['keypoints']
    ep = load_episode(resolve(cfg['tokens_dir']) / sid / f'episode_{episode:06d}.npz')
    ex = np.load(resolve(cfg['extract_dir']) / sid / f'episode_{episode:06d}.npz')
    dets = json.loads(str(ex['dets_json']))
    idx = [cfg['lift']['keypoint_index'][k] for k in names]
    frames = decode(paths(episode, cfg['root'], cfg['camera'])[1])
    h, w = frames[0].shape[:2]
    out = []
    for t, rgb in enumerate(frames):
        cam = np.ascontiguousarray(rgb.copy())
        for d in dets[t]:
            x0, y0, x1, y1 = map(int, d['box'])
            cv2.rectangle(cam, (x0, y0), (x1, y1), (255, 200, 0), 1)
            cv2.putText(cam, f"{d['label'][:12]} {d['score']:.2f}", (x0, max(y0 - 3, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)
        for j, k in enumerate(idx):
            u, v = ex['kp2d'][t, k]
            cv2.circle(cam, (int(u), int(v)), 3, (0, 255, 0) if ep['human_vis'][t, j] else (255, 0, 0), -1)
        cv2.putText(cam, f't={t / 10:.1f}s', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        out.append(np.hstack([cam, scene_panel(ep, t, names, (w, h))]))
    dst = resolve(cfg['tokens_dir']) / sid / f'episode_{episode:06d}_view.mp4'
    with av.open(str(dst), 'w') as c:
        s = c.add_stream('h264', rate=10)
        s.width, s.height, s.pix_fmt = out[0].shape[1], out[0].shape[0], 'yuv420p'
        for img in out:
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format='rgb24')):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    if sheet:
        pick = np.linspace(0, len(out) - 1, sheet + 2).round().astype(int)[1:-1]
        img = np.vstack([cv2.resize(out[i], (out[i].shape[1] // 2, out[i].shape[0] // 2)) for i in pick])
        cv2.imwrite(str(dst.with_suffix('.jpg')), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    return dst


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['depth-check', 'view'])
    p.add_argument('--config', default='configs/pif/pif.yaml')
    p.add_argument('--set', nargs='*', default=[])
    p.add_argument('--episodes', nargs='*', type=int)
    p.add_argument('--n-episodes', type=int, default=20)
    p.add_argument('--frames', type=int, default=5)
    p.add_argument('--tasks', nargs='+', default=['S8789'])
    p.add_argument('--out', default='docs/reports/PIF_P1')
    args = p.parse_args()
    full = load_config(args.config, args.set)
    if args.command == 'depth-check':
        depth_check(full, args.n_episodes, args.frames, resolve(args.out))
    else:
        for sid in args.tasks:
            eps = args.episodes or sorted(int(f.stem.split('_')[1]) for f in (resolve(full['habit']['tokens_dir']) / sid).glob('episode_*.npz'))
            for e in eps:
                print('wrote', view(full, sid, e))


if __name__ == '__main__':
    main()
