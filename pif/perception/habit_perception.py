"""HABIT front_view video -> per-frame perception and estimated 3D tokens in the table frame.

Per frame (configs/pif/habit_perception.yaml): RTMW whole-body 2D keypoints of the human, Depth Pro metric depth and
focal length, Grounding DINO boxes of the task's objects. Per episode: the table plane (RANSAC on the depth of a few
frames, lower image half) gives the table frame; keypoints and objects are lifted with the depth around them.

Table frame (same convention as the sim, pif/data/episode_reader.py): z = table normal pointing to the camera side,
origin = the camera centre projected onto the table (the camera sits between the two arms on the robot side, the sim
origin is under the robot base), x = the optical axis projected onto the table (from the robot side towards the
human), y = z x x (the robot's left).

The principal point is the image centre (Depth Pro assumes it; HABIT has no intrinsics, PIF_P0 §2.1).
Output per episode: <out>/<sid>/episode_NNNNNN.npz (raw 2D results + 3D tokens) and, with --video, an overlay mp4.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import av
import numpy as np
import yaml

from intent_policy.utils import log_run, resolve, run_dir
from pif.data.habit import episode_indices, paths


def decode(video: Path) -> list[np.ndarray]:
    with av.open(str(video)) as c:
        return [f.to_ndarray(format='rgb24') for f in c.decode(video=0)]


# ------------------------------------------------------------------ models
class Pose2D:
    def __init__(self, cfg: dict, device: str):
        from rtmlib import Wholebody
        self.model = Wholebody(mode=cfg['mode'], backend='onnxruntime', device=device)

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """[133, 2] keypoints, [133] scores of the person with the highest summed body score (the robot arms or
        background are not people; the one human of HABIT is the most confident detection)."""
        kp, sc = self.model(rgb[..., ::-1])               # rtmlib expects BGR
        if len(kp) == 0:
            return np.zeros((133, 2), np.float32), np.zeros(133, np.float32)
        best = int(np.argmax(sc[:, :17].sum(1)))
        return kp[best].astype(np.float32), sc[best].astype(np.float32)


class Depth:
    def __init__(self, cfg: dict, device: str):
        import torch
        from transformers import DepthProForDepthEstimation, DepthProImageProcessor
        self.torch, self.device = torch, device
        self.proc = DepthProImageProcessor.from_pretrained(cfg['model'])
        self.model = DepthProForDepthEstimation.from_pretrained(cfg['model'], dtype=getattr(torch, cfg['dtype'])).to(device).eval()
        self.dtype = getattr(torch, cfg['dtype'])

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, float]:
        """Metric depth [H, W] (m) and focal length (px) at the image size."""
        h, w = rgb.shape[:2]
        with self.torch.inference_mode():
            x = self.proc(images=rgb, return_tensors='pt').to(self.device)
            out = self.model(pixel_values=x['pixel_values'].to(self.dtype))
            post = self.proc.post_process_depth_estimation(out, target_sizes=[(h, w)])[0]
        return post['predicted_depth'].float().cpu().numpy(), float(post['focal_length'])


class MoGeDepth:
    """MoGe-2 (Wang et al. 2025, NeurIPS; github.com/microsoft/MoGe, MIT): metric depth + intrinsics from one image.
    The ViT-S model (35M parameters) fits a 4 GB GPU, unlike Depth Pro."""

    def __init__(self, cfg: dict, device: str):
        import torch
        from moge.model.v2 import MoGeModel
        self.torch, self.device = torch, device
        self.model = MoGeModel.from_pretrained(cfg['model']).to(device).eval()
        self.fp16 = cfg.get('dtype') == 'float16' and device != 'cpu'

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, float]:
        """Metric depth [H, W] (m, NaN where MoGe masks the pixel out) and focal length (px)."""
        x = self.torch.tensor(rgb / 255.0, dtype=self.torch.float32, device=self.device).permute(2, 0, 1)
        with self.torch.inference_mode():
            out = self.model.infer(x, use_fp16=self.fp16)
        depth = out['depth'].float().cpu().numpy()
        depth[~np.isfinite(depth)] = np.nan
        fx = float(out['intrinsics'][0, 0]) * rgb.shape[1]          # MoGe intrinsics are normalised by the image size
        return depth, fx


def make_depth(cfg: dict, device: str):
    return MoGeDepth(cfg, device) if 'moge' in cfg['model'].lower() else Depth(cfg, device)


class Detector:
    def __init__(self, cfg: dict, prompts: dict, device: str):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.torch, self.device, self.cfg = torch, device, cfg
        self.labels = list(prompts)
        self.text = '. '.join(self.labels) + '.'
        self.proc = AutoProcessor.from_pretrained(cfg['model'])
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg['model']).to(device).eval()

    def __call__(self, rgb: np.ndarray) -> list[dict]:
        h, w = rgb.shape[:2]
        with self.torch.inference_mode():
            x = self.proc(images=rgb, text=self.text, return_tensors='pt').to(self.device)
            out = self.model(**x)
            r = self.proc.post_process_grounded_object_detection(out, x['input_ids'], threshold=self.cfg['box_threshold'],
                                                                 text_threshold=self.cfg['text_threshold'],
                                                                 target_sizes=[(h, w)])[0]
        names = r.get('text_labels', r.get('labels'))
        dets = []
        for box, score, name in zip(r['boxes'].cpu().numpy(), r['scores'].cpu().numpy(), names):
            label = next((l for l in self.labels if l in str(name) or str(name) in l), None) if name else None
            if label is not None:
                dets.append(dict(box=box.astype(np.float32), score=float(score), label=label))
        return dets


# ------------------------------------------------------------------ geometry
def backproject(u, v, z, f, cx, cy) -> np.ndarray:
    return np.stack([(np.asarray(u) - cx) * z / f, (np.asarray(v) - cy) * z / f, z], -1)


def depth_at(depth: np.ndarray, u: float, v: float, win: int) -> float:
    h, w = depth.shape
    r = win // 2
    iu, iv = int(round(u)), int(round(v))
    if not (0 <= iu < w and 0 <= iv < h):
        return float('nan')
    patch = depth[max(iv - r, 0):iv + r + 1, max(iu - r, 0):iu + r + 1]
    return float(np.nanmedian(patch)) if np.isfinite(patch).any() else float('nan')


def fit_plane(points: np.ndarray, iters: int, inlier: float, rng) -> tuple[np.ndarray, float, float]:
    """RANSAC plane n.x + d = 0 (|n| = 1, d > 0: the camera at the origin is on the positive side, so n points to
    the camera side of the table); refined by least squares on the inliers. Returns n, d, inlier fraction."""
    best, best_n = None, -1
    for _ in range(iters):
        p = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs(points @ n - n @ p[0]) < inlier
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
    """Rotation R (rows x, y, z of the table frame in camera coords) and origin o: p_table = R (p_cam - o)."""
    z = n
    o = -d * n
    x = np.array([0.0, 0.0, 1.0]) - z * z[2]
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z]), o


# ------------------------------------------------------------------ episode
def track(dets_per_frame: list[list[dict]], gate: float) -> list[dict]:
    """Greedy tracking by label and 3D distance: a detection joins the nearest live track of its label within `gate`
    (each track at most once per frame), otherwise starts a new track. Tracks keep their last position."""
    tracks = []
    for t, dets in enumerate(dets_per_frame):
        used = set()
        for d in sorted(dets, key=lambda d: -d['score']):
            if not np.all(np.isfinite(d['xyz'])):
                continue
            cand = [(np.linalg.norm(tr['last'] - d['xyz']), i) for i, tr in enumerate(tracks)
                    if tr['label'] == d['label'] and i not in used]
            dist, i = min(cand, default=(np.inf, -1))
            if dist > gate:
                tracks.append(dict(label=d['label'], frames={}, last=d['xyz']))
                i = len(tracks) - 1
            used.add(i)
            tracks[i]['frames'][t] = d
            tracks[i]['last'] = d['xyz']
    return tracks


def process_episode(index: int, models: dict, cfg: dict, base_cfg: dict, sid: str, rng) -> dict:
    frames = decode(paths(index, cam=cfg['camera'])[1])
    T, (h, w) = len(frames), frames[0].shape[:2]
    cx, cy = w / 2.0, h / 2.0
    names = base_cfg['tokens']['keypoints']
    idx = [cfg['pose']['index'][k] for k in names]
    kp2d, kpsc = np.zeros((T, 133, 2), np.float32), np.zeros((T, 133), np.float32)
    focal = np.zeros(T)
    kp_depth = np.full((T, len(names)), np.nan)
    dets = []
    plane_pts = []
    plane_frames = set(np.linspace(0, T - 1, cfg['table']['frames']).round().astype(int).tolist())
    win = cfg['depth']['window_px']
    with_depth = models.get('depth') is not None
    for t, rgb in enumerate(frames):
        kp2d[t], kpsc[t] = models['pose'](rgb)
        fd = models['detect'](rgb)
        for d in fd:
            x0, y0, x1, y1 = d['box']
            d['uv'] = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)
        dets.append(fd)
        if not with_depth:
            continue
        depth, focal[t] = models['depth'](rgb)
        for j, k in enumerate(idx):
            kp_depth[t, j] = depth_at(depth, *kp2d[t, k], win)
        for d in fd:
            x0, y0, x1, y1 = d['box']
            m = cfg['detect']['core_frac'] / 2
            core = depth[int(y0 + (0.5 - m) * (y1 - y0)):int(y0 + (0.5 + m) * (y1 - y0)) + 1,
                         int(x0 + (0.5 - m) * (x1 - x0)):int(x0 + (0.5 + m) * (x1 - x0)) + 1]
            d['z'] = float(np.nanmedian(core)) if core.size and np.isfinite(core).any() else float('nan')
        if t in plane_frames:
            vv, uu = np.mgrid[int(h * cfg['table']['min_row_frac']):h:4, 0:w:4]
            plane_pts.append((uu.ravel(), vv.ravel(), depth[vv.ravel(), uu.ravel()], t))
    if not with_depth:                                  # 2D only (a GPU too small for Depth Pro): no 3D tokens
        return dict(index=index, sid=sid, T=T, image_size=(h, w), kp2d=kp2d, kp_score=kpsc, keypoints=np.array(names),
                    kp_vis2d=kpsc[:, idx] >= cfg['pose']['kpt_thr'],
                    dets_json=json.dumps([[dict(label=d_['label'], score=round(d_['score'], 3), box=d_['box'].round(1).tolist())
                                           for d_ in fd] for fd in dets]))
    f = float(np.median(focal))                         # static camera: one focal length per episode
    pts = np.concatenate([backproject(u, v, z, f, cx, cy) for u, v, z, _ in plane_pts])
    pts = pts[np.isfinite(pts).all(1)]
    n, d, inlier_frac = fit_plane(pts, cfg['table']['ransac_iters'], cfg['table']['inlier_m'], rng)
    R, o = table_frame(n, d)
    to_table = lambda p: (p - o) @ R.T
    hum_cam = backproject(kp2d[:, idx, 0], kp2d[:, idx, 1], kp_depth, f, cx, cy)
    human_pos = to_table(hum_cam)
    human_vis = (kpsc[:, idx] >= cfg['pose']['kpt_thr']) & np.isfinite(human_pos).all(-1)
    human_pos[~human_vis] = 0.0
    for fd in dets:
        for d_ in fd:
            d_['xyz'] = to_table(backproject(d_['uv'][0], d_['uv'][1], d_['z'], f, cx, cy))
    types = cfg['detect']['prompts'][sid]
    tracks = track(dets, cfg['detect']['gate_m'])
    obj = [tr for tr in tracks if types[tr['label']] != 'gripper']
    grip = [tr for tr in tracks if types[tr['label']] == 'gripper']
    K = len(obj)
    obj_pos, obj_vis = np.zeros((T, K, 3)), np.zeros((T, K), bool)
    for k, tr in enumerate(obj):
        for t, d_ in tr['frames'].items():
            obj_pos[t, k], obj_vis[t, k] = d_['xyz'], True
    gripper_uv = np.full((T, 2, 2), np.nan)              # up to two grippers per frame, image centre of the box
    for t, fd in enumerate(dets):
        g = sorted([d_ for d_ in fd if types[d_['label']] == 'gripper'], key=lambda d_: -d_['score'])[:2]
        for i, d_ in enumerate(g):
            gripper_uv[t, i] = d_['uv']
    type_names = base_cfg['tokens']['object_types']
    return dict(index=index, sid=sid, T=T, image_size=(h, w), focal=f, focal_per_frame=focal, plane_n=n, plane_d=d,
                plane_inlier_frac=inlier_frac, R=R, origin=o, kp2d=kp2d, kp_score=kpsc, keypoints=np.array(names),
                kp_depth=kp_depth, human_pos=human_pos.astype(np.float32), human_vis=human_vis,
                obj_pos=obj_pos.astype(np.float32), obj_vis=obj_vis, obj_label=np.array([tr['label'] for tr in obj]),
                obj_type=np.array([type_names.index(types[tr['label']]) for tr in obj]),
                obj_first_frame=np.array([min(tr['frames']) for tr in obj]), obj_n_frames=np.array([len(tr['frames']) for tr in obj]),
                gripper_uv=gripper_uv, dets_json=json.dumps([[dict(label=d_['label'], score=round(d_['score'], 3),
                                                                     box=d_['box'].round(1).tolist()) for d_ in fd] for fd in dets]),
                n_gripper_tracks=len(grip))


def overlay(frames: list[np.ndarray], r: dict, out: Path, fps: float = 10.0) -> None:
    """Preview video: keypoints used as tokens (green = visible), boxes, table-frame axes."""
    import cv2
    dets = json.loads(str(r['dets_json']))
    with av.open(str(out), 'w') as c:
        s = c.add_stream('h264', rate=int(fps))
        s.width, s.height, s.pix_fmt = frames[0].shape[1], frames[0].shape[0], 'yuv420p'
        h, w = frames[0].shape[:2]
        vis = r['human_vis'] if 'human_vis' in r else r['kp_vis2d']
        f, cx, cy = r.get('focal', 1.0), w / 2, h / 2
        proj = lambda p_table: (lambda pc: (f * pc[0] / pc[2] + cx, f * pc[1] / pc[2] + cy))(p_table @ r['R'] + r['origin'])
        for t, rgb in enumerate(frames):
            img = np.ascontiguousarray(rgb.copy())
            for d_ in dets[t]:
                x0, y0, x1, y1 = map(int, d_['box'])
                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 200, 0), 1)
                cv2.putText(img, f"{d_['label'][:12]} {d_['score']:.2f}", (x0, max(y0 - 3, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)
            for j, kp in enumerate(r['kp_index']):
                u, v = r['kp2d'][t, kp]
                col = (0, 255, 0) if vis[t, j] else (255, 0, 0)
                cv2.circle(img, (int(u), int(v)), 3, col, -1)
            o = proj(np.zeros(3)) if 'R' in r else None
            for axis, col in ((np.array([0.2, 0, 0]), (255, 0, 0)), (np.array([0, 0.2, 0]), (0, 255, 0)), (np.array([0, 0, 0.2]), (0, 0, 255))):
                if o is None:
                    break
                a = proj(axis)
                if all(np.isfinite([*o, *a])):
                    cv2.line(img, (int(o[0]), int(o[1])), (int(a[0]), int(a[1])), col, 2)
            cv2.putText(img, f't={t / fps:.1f}s', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format='rgb24')):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/pif/habit_perception.yaml')
    p.add_argument('--base-config', default='configs/pif/base.yaml')
    p.add_argument('--sids', nargs='+', default=['S8789', 'S8791', 'S6760'])
    p.add_argument('--per-sid', type=int, help='first N episodes of each sid (preview)')
    p.add_argument('--episodes', nargs='*', type=int)
    p.add_argument('--out', default='outputs/habit/perception')
    p.add_argument('--device', default='cuda', help='Depth Pro + Grounding DINO')
    p.add_argument('--depth-device', help="default: --device; 'none' = 2D only (no Depth Pro, no 3D tokens)")
    p.add_argument('--detect-device', help='default: --device')
    p.add_argument('--pose-device', default='cpu', help='onnxruntime device of RTMW (cuda needs onnxruntime-gpu)')
    p.add_argument('--video', action='store_true', help='also write an overlay mp4 per episode')
    p.add_argument('--shard', default='0/1', help='i/n: this process takes every n-th episode from i (parallel runs)')
    p.add_argument('--log-dir')
    args = p.parse_args()
    cfg, base = yaml.safe_load(resolve(args.config).read_text()), yaml.safe_load(resolve(args.base_config).read_text())
    out = resolve(args.out)
    log_run(run_dir('pif_habit_perception', '_'.join(args.sids), args.log_dir), 'pif.perception.habit_perception')
    jobs = []
    for sid in args.sids:
        eps = episode_indices(sid)
        eps = [e for e in eps if args.episodes is None or e in args.episodes][:args.per_sid]
        jobs += [(sid, e) for e in eps]
    si, sn = map(int, args.shard.split('/'))
    jobs = jobs[si::sn]
    rng = np.random.default_rng(0)
    dd = args.depth_device or args.device
    models = dict(pose=Pose2D(cfg['pose'], args.pose_device), depth=None if dd == 'none' else make_depth(cfg['depth'], dd))
    detectors = {}
    t0 = time.time()
    for n, (sid, e) in enumerate(jobs):
        dst = out / sid / f'episode_{e:06d}.npz'
        if dst.exists():
            continue
        if sid not in detectors:
            detectors.clear()
            detectors[sid] = Detector(cfg['detect'], cfg['detect']['prompts'][sid], args.detect_device or args.device)
        models['detect'] = detectors[sid]
        te = time.time()
        r = process_episode(e, models, cfg, base, sid, rng)
        r['kp_index'] = np.array([cfg['pose']['index'][k] for k in base['tokens']['keypoints']])
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, **r)
        if args.video:
            overlay(decode(paths(e, cam=cfg['camera'])[1]), r, dst.with_suffix('.mp4'))
        if 'focal' in r:
            msg = (f"focal {r['focal']:.0f}px, table inliers {r['plane_inlier_frac']:.2f}, visible kp {r['human_vis'].mean():.2f}, "
                   f"objects {len(r['obj_label'])}")
        else:
            msg = f"2D only, visible kp {r['kp_vis2d'].mean():.2f}"
        print(f"[{n + 1}/{len(jobs)}] {sid} ep {e}: {r['T']} frames, {msg} [{time.time() - te:.0f}s, total {time.time() - t0:.0f}s]",
              flush=True)


if __name__ == '__main__':
    main()
