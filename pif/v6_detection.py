"""M5: ManiCast's own decision protocol on the v6 gestures (configs/pif/pif.yaml v6_detection; their
third_party/manicast/eval/handover.py): a target is detected when the wrist enters its region; the forecast detects
with the wrist forecast 1 s ahead (the last forecast frame), the late reference with the current wrist. Detection
advantage = t(current) - t(forecast) for gestures where both pick the right target. No belief threshold: regions come
from the layout (half the nearest-neighbour spacing, as distance or, for targets beyond reach, as direction). As in
their handover (the goal = where the hand ends), a region counts only where the wrist stops: speed below
`still_fraction` of the gesture's peak speed so far (current wrist) / at the end of the forecast (forecast).
Episodes / candidates / gestures as pif/v6_timing.py (its sets), with its keypoint noise.
  ./run.sh -m pif.v6_detection [--set "v6_detection.observers={ft: outputs/forecast_ft/<run>/best.pt}"]
  -> outputs/v6_detection/<date>/<time>_m5/{results.json, report.md}
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from pif.config import load_config
from pif.forecast import ManiCastForecaster
from pif.utils import log_run, run_dir
from pif.v6_timing import record

NEED = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist']


def regions(pos: np.ndarray, kind: str, origin: np.ndarray) -> np.ndarray:
    """Region size per candidate: half the distance (point) or half the angle seen from `origin` (angle) to its
    nearest neighbour."""
    if kind == 'point':
        d = np.linalg.norm(pos[:, None, :2] - pos[None, :, :2], axis=2)
    else:
        u = pos[:, :2] - origin[:2]
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        d = np.arccos(np.clip(u @ u.T, -1, 1))
    np.fill_diagonal(d, np.inf)
    return d.min(1) / 2


def inside(w: np.ndarray, pos: np.ndarray, size: np.ndarray, kind: str, origin: np.ndarray, moving_m: float) -> int | None:
    """Candidate whose region the wrist position w is in (None: none)."""
    if kind == 'point':
        d = np.linalg.norm(pos[:, :2] - w[:2], axis=1)
        k = int(np.argmin(d))
        return k if d[k] < size[k] else None
    v = w[:2] - origin[:2]
    if np.linalg.norm(v) < moving_m:
        return None
    u = pos[:, :2] - origin[:2]
    ang = np.arccos(np.clip(u @ v / (np.linalg.norm(u, axis=1) * np.linalg.norm(v)), -1, 1))
    k = int(np.argmin(ang))
    return k if ang[k] < size[k] else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/pif/pif.yaml')
    ap.add_argument('--set', nargs='*', default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    vd, vt, kps = cfg['v6_detection'], cfg['v6_timing'], cfg['tokens']['keypoints']
    out = run_dir('v6_detection', 'm5')
    log_run(out, 'pif.v6_detection')
    torch.set_num_threads(4)
    sets = {n: dict(vt['sets'][n], **s) for n, s in vd['sets'].items()}
    recs = [record(sid, vd['seed0'] + i, sets, kps, vt.get('scenario_overrides'))
            for sid, n in vd['episodes'].items() for i in range(n)]
    rng = np.random.default_rng(0)
    for r in recs:
        r['ep']['human_pos'] = r['ep']['human_pos'] + rng.normal(0, vt['keypoint_noise_m'], r['ep']['human_pos'].shape)
    J = {k: i for i, k in enumerate(kps)}
    fps = float(cfg['fps'])
    t_hist, t_out = np.arange(-4, 1) / fps, np.array([1.0 - 1.0 / fps, 1.0])   # end position + end speed
    observers = {}
    for name, path in vd['observers'].items():
        observers[name] = ManiCastForecaster(dict(cfg, forecast=dict(cfg['forecast'], checkpoint_path=path)))
    observers['constant velocity'] = None
    rows = []
    for r in recs:
        e, tt = r['ep'], r['ep']['t']
        wrist = e['human_pos'][:, J['r_wrist']]
        fc0 = next(o for o in observers.values() if o is not None)
        mk, waist = fc0.to_comad({k: e['human_pos'][:, J[k]] for k in NEED})
        idx = np.arange(4, len(tt))
        H = np.stack([mk[j - 4:j + 1] for j in idx])
        W = np.stack([waist[j - 4:j + 1] for j in idx])
        fut, cur_speed = {}, np.r_[0.0, np.linalg.norm(np.diff(wrist, axis=0), axis=1) * fps]
        for name, o in observers.items():
            f, v = np.full((len(tt), 3), np.nan), np.full(len(tt), np.nan)
            if o is None:                                                  # their CVM baseline, 1 s ahead (constant
                vel = (wrist[idx] - wrist[idx - 4]) / (4 / fps)            #   velocity never stops: speed = current)
                f[idx], v[idx] = wrist[idx] + vel * 1.0, np.linalg.norm(vel, axis=1)
            else:
                p = np.concatenate([o.forecast(H[a:a + 256], W[a:a + 256], t_hist, t_out) for a in range(0, len(H), 256)])
                f[idx] = o.from_comad(p[:, 1, 6])
                v[idx] = np.linalg.norm(o.from_comad(p[:, 1, 6]) - o.from_comad(p[:, 0, 6]), axis=1) * fps
            fut[name] = (f, v)
        for g in r['gestures']:
            s = sets[g['set']]
            pos = r['pos'][g['set']]
            a = int(np.searchsorted(tt, g['t0'] + vd['window_s'][0]))
            z = min(int(np.searchsorted(tt, g['t1'] + vd['window_s'][1])), len(tt) - 1)
            origin = wrist[int(np.searchsorted(tt, g['t0']))]
            size = regions(pos, s['region'], origin)

            peak = np.maximum.accumulate(cur_speed[a:z + 1])                # observed peak speed of the gesture so far

            def detect(track, speed):
                for j in range(a, z + 1):
                    if np.isnan(track[j]).any() or speed[j] >= vd['still_fraction'] * max(peak[j - a], 1e-6):
                        continue                                           # not (predicted to be) stopped there
                    k = inside(track[j], pos, size, s['region'], origin, vd['moving_m'])
                    if k is not None:
                        return tt[j], k
                return None, None
            tc, kc = detect(wrist, cur_speed)
            row = dict(set=g['set'], label=g['label'], seed=r['seed'], K=len(pos), dur=round(g['t1'] - g['t0'], 2),
                       current_t=None if tc is None else round(tc - g['t0'], 2), current_ok=kc == g['target'])
            for name, (f, v) in fut.items():
                tf, kf = detect(f, v)
                row[f'{name}_t'] = None if tf is None else round(tf - g['t0'], 2)
                row[f'{name}_ok'] = kf == g['target']
                row[f'{name}_adv'] = round(tc - tf, 2) if (kf == g['target'] and kc == g['target']) else None
            rows.append(row)
    (out / 'results.json').write_text(json.dumps(rows, indent=1))
    names = list(observers)
    by = defaultdict(list)
    for r in rows:
        by[r['set']].append(r)
    lines = [f"# v6 detection, ManiCast protocol ({out.name}); keypoint noise {vt['keypoint_noise_m']} m", '',
             'Times in s after the motion start (median); correct = first region entered is the target; advantage = '
             't(current) - t(forecast) where both are correct (median, n).', '',
             '| set | n | K | gesture s | current: correct / t | ' + ' | '.join(f'{n}: correct / t / advantage (n)' for n in names) + ' |',
             '|' + '---|' * (5 + len(names))]
    med = lambda v: f"{np.median(v):.2f}" if len(v) else 'nan'
    for name, rs in by.items():
        cur = [r['current_t'] for r in rs if r['current_ok']]
        cells = []
        for n in names:
            t = [r[f'{n}_t'] for r in rs if r[f'{n}_ok']]
            adv = [r[f'{n}_adv'] for r in rs if r[f'{n}_adv'] is not None]
            cells.append(f"{np.mean([r[f'{n}_ok'] for r in rs]):.2f} / {med(t)} / {med(adv)} ({len(adv)})")
        lines.append(f"| {name} | {len(rs)} | {rs[0]['K']} | {med([r['dur'] for r in rs])} | "
                     f"{np.mean([r['current_ok'] for r in rs]):.2f} / {med(cur)} | " + ' | '.join(cells) + ' |')
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
