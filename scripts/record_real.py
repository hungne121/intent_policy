"""Record real hand keypoints (RGB-D + MediaPipe Hands) to measure the PredictedSource noise (INTENT_ACT_GUIDE_v2.md
§3.3, M0 answer): the 6 contract keypoints (wrist, index base = MCP, index tip, both hands), the pointing ray (index
base -> tip), detection flags and timestamps.

  record  ./run.sh -m scripts.record_real record --out outputs/real/session1 [--seconds 60] [--extrinsic T.json]
          [--point-at x,y,z]      (the person points at this known point, robot-base frame, for the angle error)
  stats   ./run.sh -m scripts.record_real stats outputs/real/session1/keypoints.npz [--point-at x,y,z]
          -> jitter_m (keypoint std while the hand is still), dropout_prob, latency_s (capture -> keypoints),
             angle_deg (ray error to --point-at while still): the `predicted.noise` block of configs/intent_schema.yaml

Camera: an Intel RealSense (pyrealsense2, depth aligned to colour) by default; `--camera webcam` records colour only
(keypoints in normalised image coordinates, z = NaN: no metric jitter). Keypoints are in the camera frame unless
`--extrinsic` gives the 4x4 camera->robot-base transform (JSON {"T": [[...]]}, TODO(user): hand-eye calibration).
Needs `pip install mediapipe pyrealsense2 opencv-python` (not part of the simulation environment).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

KEYPOINTS = ['r_wrist', 'r_index_base', 'r_index_tip', 'l_wrist', 'l_index_base', 'l_index_tip']
LANDMARK = {'wrist': 0, 'index_base': 5, 'index_tip': 8}          # MediaPipe Hands landmark ids


def deproject(u: float, v: float, depth_m: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Pinhole back-projection of pixel (u, v) at depth z (camera frame, metres)."""
    return np.array([(u - cx) / fx * depth_m, (v - cy) / fy * depth_m, depth_m], float)


def robust_depth(depth: np.ndarray, u: float, v: float, scale: float, win: int = 2) -> float:
    """Median of the valid depths in a (2 win + 1)^2 window (fingertips often fall on invalid pixels)."""
    h, w = depth.shape
    x, y = int(round(u)), int(round(v))
    patch = depth[max(y - win, 0):min(y + win + 1, h), max(x - win, 0):min(x + win + 1, w)].astype(float) * scale
    patch = patch[patch > 0]
    return float(np.median(patch)) if patch.size else float('nan')


def hands_to_keypoints(hands: list[dict], depth, intr, scale, mirrored: bool) -> tuple[np.ndarray, np.ndarray]:
    """hands: [{'label': 'Left'|'Right', 'landmarks': [(u_px, v_px), ...21]}] -> keypoints (6, 3), valid (6,)."""
    kp, valid = np.full((6, 3), np.nan), np.zeros(6, bool)
    for hd in hands:
        side = hd['label'].lower()[0]
        if mirrored:                                   # MediaPipe labels assume a mirrored (selfie) image
            side = 'l' if side == 'r' else 'r'
        for name, lid in LANDMARK.items():
            i = KEYPOINTS.index(f'{side}_{name}')
            u, v = hd['landmarks'][lid]
            if depth is None:
                kp[i] = [u, v, np.nan]
                valid[i] = True
                continue
            z = robust_depth(depth, u, v, scale)
            if np.isfinite(z):
                kp[i] = deproject(u, v, z, *intr)
                valid[i] = True
    return kp, valid


def transform(kp: np.ndarray, T: np.ndarray | None) -> np.ndarray:
    return kp if T is None else kp @ T[:3, :3].T + T[:3, 3]


def noise_stats(t: np.ndarray, kp: np.ndarray, valid: np.ndarray, latency: np.ndarray, point_at=None,
                still_speed: float = 0.03, fps: float = 30.0) -> dict:
    """Noise parameters from a recording: jitter = std of the high-pass keypoint (minus a 5-frame moving average)
    while the hand is still; dropout = fraction of invalid keypoints; latency = median capture -> output delay;
    angle = mean ray error (index base -> tip of the most valid hand) to `point_at` while still."""
    out = dict(frames=int(len(t)), dropout_prob=float(1.0 - valid.mean()), latency_s=float(np.nanmedian(latency)))
    v = np.where(valid[..., None], kp, np.nan)
    smooth = np.copy(v)
    for k in range(2, len(v) - 2):
        smooth[k] = np.nanmean(v[k - 2:k + 3], axis=0)
    speed = np.nanmax(np.linalg.norm(np.diff(smooth, axis=0), axis=-1), axis=-1) * fps
    still = np.r_[False, speed < still_speed]
    resid = (v - smooth)[still]
    out['jitter_m'] = float(np.nanstd(resid)) if np.isfinite(resid).any() else float('nan')
    out['still_frames'] = int(still.sum())
    if point_at is not None:
        errs = []
        for s in ('r', 'l'):
            b, tip = KEYPOINTS.index(f'{s}_index_base'), KEYPOINTS.index(f'{s}_index_tip')
            ok = still & valid[:, b] & valid[:, tip]
            for k in np.flatnonzero(ok):
                d, q = kp[k, tip] - kp[k, b], np.asarray(point_at) - kp[k, tip]
                errs.append(np.degrees(np.arccos(np.clip(d @ q / (np.linalg.norm(d) * np.linalg.norm(q) + 1e-12), -1, 1))))
        out['angle_deg'] = float(np.mean(errs)) if errs else float('nan')
        out['angle_p95_deg'] = float(np.percentile(errs, 95)) if errs else float('nan')
    return out


def record(args) -> None:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as e:
        raise SystemExit(f'{e}: install mediapipe and opencv-python to record real keypoints')
    T = np.asarray(json.loads(Path(args.extrinsic).read_text())['T'], float) if args.extrinsic else None
    hands_model = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=1, min_detection_confidence=0.5)
    if args.camera == 'realsense':
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise SystemExit(f'{e}: install pyrealsense2 (or use --camera webcam)')
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
        prof = pipe.start(cfg)
        align = rs.align(rs.stream.color)
        scale = prof.get_device().first_depth_sensor().get_depth_scale()
        ci = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        intr = (ci.fx, ci.fy, ci.ppx, ci.ppy)

        def grab():
            f = align.process(pipe.wait_for_frames())
            return np.asanyarray(f.get_color_frame().get_data()), np.asanyarray(f.get_depth_frame().get_data()), f.get_timestamp() / 1000.0
    else:
        cap = cv2.VideoCapture(args.device)
        scale, intr, pipe = 1.0, None, None

        def grab():
            ok, img = cap.read()
            if not ok:
                raise SystemExit('webcam read failed')
            return img, None, time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = dict(t=[], t_capture=[], keypoints=[], valid=[], latency=[])
    t0 = time.time()
    try:
        while time.time() - t0 < args.seconds:
            color, depth, t_cap = grab()
            t_in = time.time()
            res = hands_model.process(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
            hands = []
            if res.multi_hand_landmarks:
                h, w = color.shape[:2]
                for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    hands.append(dict(label=hd.classification[0].label,
                                      landmarks=[(p.x * w, p.y * h) if depth is not None else (p.x, p.y) for p in lm.landmark]))
            kp, valid = hands_to_keypoints(hands, depth, intr, scale, args.mirrored)
            rows['t'].append(t_in - t0)
            rows['t_capture'].append(t_cap)
            rows['keypoints'].append(transform(kp, T))
            rows['valid'].append(valid)
            rows['latency'].append(time.time() - t_in)          # processing delay (camera pipeline delay not included)
    finally:
        if pipe is not None:
            pipe.stop()
    np.savez_compressed(out / 'keypoints.npz', names=np.array(KEYPOINTS), frame='base' if T is not None else 'camera',
                        **{k: np.asarray(v) for k, v in rows.items()})
    print(f"wrote {out / 'keypoints.npz'} ({len(rows['t'])} frames)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('record')
    r.add_argument('--out', required=True)
    r.add_argument('--seconds', type=float, default=60.0)
    r.add_argument('--camera', choices=['realsense', 'webcam'], default='realsense')
    r.add_argument('--device', type=int, default=0, help='webcam index')
    r.add_argument('--width', type=int, default=640)
    r.add_argument('--height', type=int, default=480)
    r.add_argument('--fps', type=int, default=30)
    r.add_argument('--extrinsic', help='JSON {"T": 4x4 camera -> robot base}, TODO(user)')
    r.add_argument('--mirrored', action='store_true', help='the image is mirrored (selfie view)')
    s = sub.add_parser('stats')
    s.add_argument('npz')
    s.add_argument('--point-at', help='x,y,z of the point the person points at (same frame as the keypoints)')
    s.add_argument('--fps', type=float, default=30.0)
    args = p.parse_args()
    if args.cmd == 'record':
        record(args)
    else:
        d = np.load(args.npz)
        pa = np.array([float(x) for x in args.point_at.split(',')]) if args.point_at else None
        st = noise_stats(d['t'], d['keypoints'], d['valid'], d['latency'], pa, fps=args.fps)
        print(json.dumps(st, indent=2))
        print('-> configs/intent_schema.yaml predicted.noise: '
              + json.dumps({k: round(st[k], 4) for k in ('jitter_m', 'dropout_prob', 'latency_s', 'angle_deg') if k in st}))


if __name__ == '__main__':
    main()
