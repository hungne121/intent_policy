"""Clips of the measuring device for the intent report (2026-10-19): what the device outputs, frame by frame.

  sim   from a collected dataset (the device ran in the loop; no re-simulation): the robot's side camera large, top /
        wrist small, and a panel with the time, the human's and the robot's stage (simulator ground truth), the
        device's 6 gesture-class probabilities, its belief over the candidates (the true target of the current gesture
        marked), and a strip of the belief over the whole episode with the 0.85 / 0.9 / 0.95 levels and a cursor
  real  a real video and two DeePoint outputs (scripts/deepoint_infer.py): the released model's p(pointing) and the
        sim fine-tuned model's 6 class probabilities, as strips with a cursor

  ./run.sh -m scripts.device_clips sim --root <dataset> --episodes 3 5 --out outputs/device_report/clips
  ./run.sh -m scripts.device_clips real --video v.mp4 --released r.npz --finetuned f.npz --out outputs/device_report/clips
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CLASSES = ['none', 'point', 'reach', 'palm_up', 'wait', 'withdraw']
CLASS_VN = ['khong', 'chi tay', 'vuon lay', 'ngua tay', 'ra hieu cho', 'rut tay']
FPS = 20
W_PANEL = 560
GREEN, RED, GREY, WHITE, YELLOW, CYAN = (80, 200, 80), (60, 60, 230), (150, 150, 150), (240, 240, 240), (0, 220, 255), (230, 200, 60)


def put(img, text, xy, scale=0.5, color=WHITE, thick=1):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def bar(img, x, y, w, h, v, color, label, value_text=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(w * max(0.0, min(1.0, v))), y + h), color, -1)
    put(img, label, (x - 150, y + h - 3), 0.45)
    put(img, value_text if value_text is not None else f'{v:.2f}', (x + w + 6, y + h - 3), 0.45)


def strip(img, x, y, w, h, series, cursor, levels=(0.85, 0.9, 0.95), marks=()):
    """series: list of (values (T,), color, label); cursor: frame index."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
    T = max(len(s[0]) for s in series)
    for lv in levels:
        yy = y + int(h * (1 - lv))
        cv2.line(img, (x, yy), (x + w, yy), (90, 90, 90), 1)
        put(img, f'{lv:.2f}', (x + w + 4, yy + 4), 0.35, GREY)
    for t, color in marks:
        xx = x + int(w * t / max(T - 1, 1))
        cv2.line(img, (xx, y), (xx, y + h), color, 1)
    for vals, color, _ in series:
        pts = [(x + int(w * i / max(T - 1, 1)), y + int(h * (1 - float(np.nan_to_num(v))))) for i, v in enumerate(vals)]
        cv2.polylines(img, [np.asarray(pts, np.int32)], False, color, 2, cv2.LINE_AA)
    xx = x + int(w * cursor / max(T - 1, 1))
    cv2.line(img, (xx, y - 4), (xx, y + h + 4), YELLOW, 2)
    lx = x
    for _, color, label in series:
        put(img, label, (lx, y + h + 16), 0.42, color)
        lx += 12 + 9 * len(label)


def stage_at(stages, f):
    for s in stages:
        if s['start'] <= f <= s['end']:
            return s['name']
    return '-'


def sim_clips(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import torch
    root = Path(args.root)
    metas = {m['episode_index']: m for m in map(json.loads, (root / 'meta/hri_episodes.jsonl').read_text().splitlines())}
    schema = json.loads((root / 'meta/intent_schema.json').read_text())
    T_names = schema['targets']
    segs = json.loads((root / 'meta/intent_segments.json').read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ei in args.episodes:
        m = metas[ei]
        ds = LeRobotDataset(args.repo_id or f'local/{root.name}', root=root, episodes=[ei])
        cams = ds.meta.camera_keys
        main = next((k for k in cams if 'sensor_side' in k), cams[0])
        small = [k for k in cams if k != main]
        rows = []
        for batch in torch.utils.data.DataLoader(ds, batch_size=32, num_workers=0):
            for j in range(len(batch['index'])):
                rows.append({k: batch[k][j] for k in [main] + small + ['intent_pd.p_target', 'intent_pd.p_gesture']})
        P = np.stack([r['intent_pd.p_target'].numpy() for r in rows])
        G = np.stack([r['intent_pd.p_gesture'].numpy() for r in rows])
        layout = m['spec'].get('layout') or {}
        slot_obj = {v: k for k, v in layout.items()}
        ub = m['spec'].get('u_blocks') or []
        name_of = lambda t: slot_obj.get(t) or ({'U1': ub[0], 'U2': ub[1]}.get(t) if len(ub) == 2 else None) or t
        cand = [i for i in range(P.shape[1]) if P[:, i].max() > 0.001 and T_names[i] != 'none']
        # the true target of the gesture under way at each frame (simulator segments)
        true_at = [None] * len(rows)
        for s in segs.get(str(ei), []):
            if s['gesture'] in ('point', 'reach', 'palm_up') and s['target'] in T_names:
                for f in range(int(s['t_onset']), min(int(s['t_end']) + 1, len(rows))):
                    true_at[f] = s['target']
        # normalised belief per candidate kind (objects vs hand zones) for the strip
        kinds = {t: k for k, ts in schema['target_kinds'].items() for t in ts}
        def norm_kind(f, i):
            ks = [c for c in cand if kinds.get(T_names[c]) == kinds.get(T_names[i])]
            s = P[f, ks].sum()
            return P[f, i] / s if s > 1e-9 else np.nan
        Bn = np.array([[norm_kind(f, i) for i in cand] for f in range(len(rows))])
        onsets = [(int(s['t_onset']), CYAN) for s in segs.get(str(ei), []) if s['gesture'] in ('point', 'reach', 'palm_up', 'palm_forward')]
        commits = [(int(round(c['t'] * FPS)), RED) for c in m.get('expert_commits') or []]
        H, Wm = 576, 768
        vw = cv2.VideoWriter(str(out / f"sim_ep{ei:03d}_{m['list_task']}_seed{m['seed']}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'),
                             FPS, (Wm + W_PANEL, H + 160))
        palette = [GREEN, (230, 120, 60), (200, 80, 200), (60, 180, 230)]
        for f, r in enumerate(rows):
            img = np.zeros((H + 160, Wm + W_PANEL, 3), np.uint8)
            fr = (r[main].permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1]
            img[:H, :Wm] = cv2.resize(fr, (Wm, H))
            for k, key in enumerate(small[:2]):
                sm = (r[key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1]
                img[H:H + 150, k * 200:(k + 1) * 200] = cv2.resize(sm, (200, 150))
            x0 = Wm + 160
            put(img, f"ep {ei} {m['list_task']} seed {m['seed']}   t = {f / FPS:5.2f} s", (Wm + 10, 24), 0.55, YELLOW)
            put(img, f"nguoi: {stage_at(m['human_stages'], f)}", (Wm + 10, 48), 0.5)
            put(img, f"robot: {stage_at(m['expert_stages'], f)}", (Wm + 10, 70), 0.5)
            put(img, 'Cu chi (DeePoint):', (Wm + 10, 98), 0.5, CYAN)
            for c in range(6):
                bar(img, x0, 108 + 22 * c, 260, 14, float(G[f, c]), CYAN, CLASS_VN[c])
            put(img, 'Belief (dich that: vien do):', (Wm + 10, 254), 0.5, CYAN)
            for k, i in enumerate(cand):
                t = T_names[i]
                y = 264 + 22 * k
                v = float(Bn[f, k]) if np.isfinite(Bn[f, k]) else 0.0
                bar(img, x0, y, 260, 14, v, palette[k % len(palette)], f'{t} ({name_of(t)})')
                if true_at[f] == t:
                    cv2.rectangle(img, (x0 - 2, y - 2), (x0 + 262, y + 16), RED, 2)
            strip(img, Wm + 10, H - 10, W_PANEL - 60, 110, [(Bn[:, k], palette[k % len(palette)], T_names[i]) for k, i in enumerate(cand)],
                  f, marks=onsets + commits)
            put(img, 'xanh: nguoi bat dau cu chi | do: expert chot dich', (Wm + 10, H + 135), 0.42, GREY)
            vw.write(img)
        vw.release()
        print(f'{out}/sim_ep{ei:03d}: {len(rows)} frames, candidates {[T_names[i] for i in cand]}', flush=True)


def real_clip(args):
    r, ft = np.load(args.released), np.load(args.finetuned)
    pr, pf = r['probs'][:, 1], ft['probs']
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Wv, Hv = 960, 720
    vw = cv2.VideoWriter(str(out / 'real_deepoint_demo.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (Wv + W_PANEL, Hv))
    f = 0
    colors = [GREY, RED, (230, 120, 60), GREEN, CYAN, (200, 80, 200)]
    while True:
        ok, fr = cap.read()
        if not ok or f >= len(pr):
            break
        img = np.zeros((Hv, Wv + W_PANEL, 3), np.uint8)
        img[:, :Wv] = cv2.resize(fr, (Wv, Hv))
        x0 = Wv + 160
        put(img, f'Nguoi that (video demo DeePoint)  t = {f / fps:4.1f} s', (Wv + 10, 24), 0.55, YELLOW)
        put(img, 'Model goc DeePoint:', (Wv + 10, 56), 0.5, CYAN)
        bar(img, x0, 66, 260, 14, float(np.nan_to_num(pr[f])), RED, 'p(chi tay)')
        put(img, 'Model fine-tune tren sim (bo do):', (Wv + 10, 110), 0.5, CYAN)
        for c in range(6):
            bar(img, x0, 120 + 22 * c, 260, 14, float(np.nan_to_num(pf[f, c])), colors[c], CLASS_VN[c])
        strip(img, Wv + 10, 300, W_PANEL - 60, 120, [(pr, RED, 'goc: chi tay')], f, levels=(0.5,))
        strip(img, Wv + 10, 470, W_PANEL - 60, 120, [(pf[:, 1], RED, 'sim: chi tay'), (pf[:, 3], GREEN, 'sim: ngua tay'),
                                                    (pf[:, 4], CYAN, 'sim: cho')], f, levels=(0.5,))
        put(img, 'cua so 15 frame: 0.9 s dau chua co dau ra', (Wv + 10, 680), 0.42, GREY)
        vw.write(img)
        f += 1
    vw.release()
    print(f'{out}/real_deepoint_demo.mp4: {f} frames')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    s = sub.add_parser('sim')
    s.add_argument('--root', required=True)
    s.add_argument('--repo-id')
    s.add_argument('--episodes', type=int, nargs='+', required=True)
    s.add_argument('--out', required=True)
    r = sub.add_parser('real')
    r.add_argument('--video', required=True)
    r.add_argument('--released', required=True)
    r.add_argument('--finetuned', required=True)
    r.add_argument('--out', required=True)
    args = ap.parse_args()
    sim_clips(args) if args.mode == 'sim' else real_clip(args)


if __name__ == '__main__':
    main()
