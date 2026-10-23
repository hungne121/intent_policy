"""Side-by-side preview of one HABIT episode's perception (pif/perception/habit_perception.py output).

Left: the front_view frame with the token keypoints (green = visible) and the detected boxes. Right: the estimated 3D
scene in the table frame, top view (x = away from the robot, up; y = robot's left, left, as in the camera image) and
side view (x right, height z up): hands (wrist -> index tip), arms, head, objects (filled = detected this frame, hollow
= last known position), 2 s trail of both wrists.
Writes <npz>_3d.mp4 and, with --sheet, a contact sheet <npz>_3d.jpg of a few frames.
"""
import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np

from pif.data.habit import paths
from pif.perception.habit_perception import decode

COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
          (240, 50, 230), (210, 245, 60), (250, 190, 212)]


def panel_maps(r: dict, size: tuple[int, int]):
    """Pixel maps of the top view (x, y) and the side view (x, z), sharing the x scale."""
    w, h = size
    pts = [r['human_pos'][r['human_vis']], r['obj_pos'][r['obj_vis']]]
    p = np.concatenate([q for q in pts if len(q)])
    lo, hi = np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)
    x0, x1 = min(lo[0], 0.0) - 0.05, hi[0] + 0.1
    ymax = max(abs(lo[1]), abs(hi[1])) + 0.1
    top_h = int(h * 0.62)
    s = min((top_h - 20) / (x1 - x0), (w - 20) / (2 * ymax))
    top = lambda q: (int(w / 2 - q[1] * s), int(top_h - 10 - (q[0] - x0) * s))
    z1 = max(hi[2] + 0.1, 0.5)
    side_h = h - top_h
    ss = min((w - 20) / (x1 - x0), (side_h - 25) / (z1 + 0.05))
    side = lambda q: (int(10 + (q[0] - x0) * ss), int(h - 15 - q[2] * ss))
    return top, side, top_h, (x0, x1, z1, ss)


def draw_scene(r: dict, t: int, size: tuple[int, int], last: dict, labels: list[str]) -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), 255, np.uint8)
    top, side, top_h, (x0, x1, z1, ss) = panel_maps(r, size)
    cv2.line(img, (0, top_h), (w, top_h), (180, 180, 180), 1)
    cv2.putText(img, 'top view (table frame)', (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
    cv2.putText(img, 'side view: height above table', (5, top_h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
    o = top(np.zeros(3))
    cv2.circle(img, o, 4, (0, 0, 0), -1)
    cv2.putText(img, 'camera / robot', (o[0] + 5, o[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.line(img, side((x0, 0, 0)), side((x1, 0, 0)), (120, 80, 40), 2)          # table surface
    for k in range(len(labels)):                                                    # objects
        if r['obj_vis'][t, k]:
            last[k] = r['obj_pos'][t, k]
        if k not in last:
            continue
        col = COLORS[k % len(COLORS)]
        fill = -1 if r['obj_vis'][t, k] else 1
        for proj in (top, side):
            cv2.circle(img, proj(last[k]), 6, col, fill)
        u, v = top(last[k])
        cv2.putText(img, str(r['obj_label'][k])[:10], (u + 7, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)
    names = [str(n) for n in r['keypoints']]
    kp, vis = r['human_pos'][t], r['human_vis'][t]
    J = {n: i for i, n in enumerate(names)}
    bones = [('r_shoulder', 'r_elbow'), ('r_elbow', 'r_wrist'), ('l_shoulder', 'l_elbow'), ('l_elbow', 'l_wrist'),
             ('r_wrist', 'r_index_base'), ('r_index_base', 'r_index_tip'), ('l_wrist', 'l_index_base'),
             ('l_index_base', 'l_index_tip'), ('r_shoulder', 'l_shoulder')]
    for side_name, col in (('r', (200, 0, 0)), ('l', (0, 0, 200))):                # 2 s wrist trails
        j = J[f'{side_name}_wrist']
        tr = [r['human_pos'][s, j] for s in range(max(0, t - 20), t + 1) if r['human_vis'][s, j]]
        for a, b in zip(tr[:-1], tr[1:]):
            cv2.line(img, top(a), top(b), tuple(int(c * 0.5 + 127) for c in col), 1)
    for a, b in bones:
        if vis[J[a]] and vis[J[b]]:
            col = (200, 0, 0) if a[0] == 'r' else (0, 0, 200)
            for proj in (top, side):
                cv2.line(img, proj(kp[J[a]]), proj(kp[J[b]]), col, 2)
    for n, i in J.items():
        if vis[i]:
            for proj in (top, side):
                cv2.circle(img, proj(kp[i]), 4 if 'tip' in n or 'wrist' in n else 3, (0, 0, 0), -1)
    for n in ('r_index_tip', 'l_index_tip'):
        if vis[J[n]]:
            u, v = side(kp[J[n]])
            cv2.putText(img, f'{kp[J[n]][2] * 100:.0f}cm', (u + 5, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)
    cv2.putText(img, 'red = right hand, blue = left hand; hollow object = not detected now', (5, h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (90, 90, 90), 1)
    return img


def camera_overlay(rgb: np.ndarray, r: dict, t: int, dets: list) -> np.ndarray:
    img = np.ascontiguousarray(rgb.copy())
    for d in dets[t]:
        x0, y0, x1, y1 = map(int, d['box'])
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 200, 0), 1)
        cv2.putText(img, f"{d['label'][:12]} {d['score']:.2f}", (x0, max(y0 - 3, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)
    for j, k in enumerate(r['kp_index']):
        u, v = r['kp2d'][t, k]
        cv2.circle(img, (int(u), int(v)), 3, (0, 255, 0) if r['human_vis'][t, j] else (255, 0, 0), -1)
    cv2.putText(img, f't={t / 10:.1f}s', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('npz', nargs='+')
    p.add_argument('--sheet', type=int, default=6, help='frames in the contact sheet (0: none)')
    args = p.parse_args()
    for f in args.npz:
        r = dict(np.load(f, allow_pickle=True))
        frames = decode(paths(int(r['index']))[1])
        dets = json.loads(str(r['dets_json']))
        h, w = frames[0].shape[:2]
        labels = sorted(set(str(x) for x in r['obj_label']))
        last, out = {}, []
        for t, rgb in enumerate(frames):
            out.append(np.hstack([camera_overlay(rgb, r, t, dets), draw_scene(r, t, (w, h), last, labels)]))
        dst = Path(f).with_name(Path(f).stem + '_3d.mp4')
        with av.open(str(dst), 'w') as c:
            s = c.add_stream('h264', rate=10)
            s.width, s.height, s.pix_fmt = out[0].shape[1], out[0].shape[0], 'yuv420p'
            for img in out:
                for pkt in s.encode(av.VideoFrame.from_ndarray(img, format='rgb24')):
                    c.mux(pkt)
            for pkt in s.encode():
                c.mux(pkt)
        if args.sheet:
            pick = np.linspace(0, len(out) - 1, args.sheet + 2).round().astype(int)[1:-1]
            sheet = np.vstack([cv2.resize(out[i], (out[i].shape[1] // 2, out[i].shape[0] // 2)) for i in pick])
            cv2.imwrite(str(Path(f).with_name(Path(f).stem + '_3d.jpg')), sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        print('wrote', dst)


if __name__ == '__main__':
    main()
