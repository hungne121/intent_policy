"""M5: how early does inverse planning on the observed + forecast hand path identify the human's target in v6?
(configs/pif/pif.yaml v6_goal). P(G) ∝ exp(beta [d(S, G) - d(Q, G)]) with S the wrist lookback_s ago and Q the wrist
forecast 1 s ahead (Dragan, Lee, Srinivasa 2013; Baker, Saxe, Tenenbaum 2009); beta = 1 / the forecaster's measured
error. No threshold: per gesture, the earliest time from which the true target stays the most probable until the
motion ends ('stable'), in s after the motion start and before its end. Episodes / candidates as pif/v6_timing.py.
  ./run.sh -m pif.v6_goal [--set "v6_goal.observers={released: null, fine-tuned: <best.pt>}"]
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


def posterior(S: np.ndarray, Q: np.ndarray, cand: np.ndarray, beta: float, ray=None, kappa: float = 0.0,
              stopped=None, reachable=None) -> np.ndarray:
    """[T, K] goal posterior from progress d(S, G) - d(Q, G) (uniform prior); with `ray` = (base [T, 3], tip [T, 3])
    the pointing evidence kappa cos(angle(base -> tip, tip -> G)) (a vMF on the pointing direction) replaces the path
    term on the frames where ray[2] (the observable pointing hand shape) is set."""
    prog = np.linalg.norm(S[:, None] - cand[None], axis=2) - np.linalg.norm(Q[:, None] - cand[None], axis=2)
    z = beta * prog
    if stopped is not None:                  # termination: a hand that stops (or is forecast to) stops at its goal,
        near = np.linalg.norm(Q[:, None, :2] - cand[None, :, :2], axis=2)      # p(stop at Q | G) ∝ exp(-beta d(Q, G))
        mask = stopped[:, None] & (np.ones(len(cand), bool) if reachable is None else reachable)[None]
        z = z - beta * near * mask           # goals beyond reach: the hand stops at its reach limit whatever the goal
    if ray is not None:
        d = ray[1] - ray[0]
        d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        u = cand[None] - ray[1][:, None]
        u = u / (np.linalg.norm(u, axis=2, keepdims=True) + 1e-9)
        pointing = ray[2][:, None] if len(ray) > 2 else 1.0           # only while the hand has the pointing shape
        z = np.where(pointing, kappa * np.einsum('tc,tkc->tk', d, u), z)  #   (the ray then replaces the path term)
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/pif/pif.yaml')
    ap.add_argument('--set', nargs='*', default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    vg, vt, kps = cfg['v6_goal'], cfg['v6_timing'], cfg['tokens']['keypoints']
    out = run_dir('v6_goal', 'm5')
    log_run(out, 'pif.v6_goal')
    torch.set_num_threads(4)
    sets = {n: vt['sets'][n] for n in vg['sets']}
    recs = [record(sid, vg['seed0'] + i, sets, kps, vt.get('scenario_overrides'))
            for sid, n in vg['episodes'].items() for i in range(n)]
    rng = np.random.default_rng(0)
    for r in recs:
        r['ep']['human_pos'] = r['ep']['human_pos'] + rng.normal(0, vt['keypoint_noise_m'], r['ep']['human_pos'].shape)
    J = {k: i for i, k in enumerate(kps)}
    fps, beta = float(cfg['fps']), 1000.0 / float(vg['beta_from_error_mm'])
    lb = int(round(vg['lookback_s'] * fps))
    t_hist = np.arange(-4, 1) / fps
    t_out = np.arange(1, int(vg['forecast_steps']) + 1) / int(vg['forecast_steps'])
    fcs = {n: ManiCastForecaster(dict(cfg, forecast=dict(cfg['forecast'], checkpoint_path=p))) for n, p in vg['observers'].items()}
    kappa = 1.0 / np.radians(float(vg['pointing_sigma_deg'])) ** 2
    names = ['observed only'] + list(fcs) + ['constant velocity']
    rows = []
    for r in recs:
        e, tt = r['ep'], r['ep']['t']
        wrist = e['human_pos'][:, J['r_wrist']]
        T = len(tt)
        S = wrist[np.maximum(np.arange(T) - lb, 0)]
        Qs = {'observed only': wrist.copy()}
        idx = np.arange(4, T)
        any_fc = next(iter(fcs.values()))
        mk, waist = any_fc.to_comad({k: e['human_pos'][:, J[k]] for k in NEED})
        H = np.stack([mk[j - 4:j + 1] for j in idx])
        W = np.stack([waist[j - 4:j + 1] for j in idx])
        for n, fc in fcs.items():
            p = np.concatenate([fc.forecast(H[a:a + 256], W[a:a + 256], t_hist, t_out) for a in range(0, len(H), 256)])
            Q = wrist.copy()
            Q[idx] = fc.from_comad(p[:, -1, 6])
            Qs[n] = Q
        Q = wrist.copy()
        Q[idx] = wrist[idx] + (wrist[idx] - wrist[idx - 4]) / (4 / fps)
        Qs['constant velocity'] = Q
        speed = {'observed only': np.r_[0.0, np.linalg.norm(np.diff(wrist, axis=0), axis=1) * fps]}
        for n, fc in fcs.items():                                          # forecast speed at its end (0.9 -> 1.0 s)
            p = np.concatenate([fc.forecast(H[a:a + 256], W[a:a + 256], t_hist, t_out[-2:]) for a in range(0, len(H), 256)])
            v = speed['observed only'].copy()
            v[idx] = np.linalg.norm(fc.from_comad(p[:, 1, 6]) - fc.from_comad(p[:, 0, 6]), axis=1) / (t_out[-1] - t_out[-2])
            speed[n] = v
        speed['constant velocity'] = speed['observed only']
        for g in r['gestures']:
            cand = r['pos'][g['set']]
            a = max(int(np.searchsorted(tt, g['t0'] + vg['window_s'][0])), 0)
            z = min(int(np.searchsorted(tt, g['t1'] + vg['window_s'][1])), T - 1)
            row = dict(set=g['set'], label=g['label'], seed=r['seed'], K=len(cand), dur=round(g['t1'] - g['t0'], 2))
            ray = (e['human_pos'][a:z + 1, J['r_index_base']], e['human_pos'][a:z + 1, J['r_index_tip']], e['pointing'][a:z + 1])
            peak = np.maximum.accumulate(speed['observed only'][a:z + 1])     # observed peak speed of the gesture so far
            for n in names + ['observed + pointing', 'fine-tuned + pointing']:
                base_n = {'observed + pointing': 'observed only', 'fine-tuned + pointing': 'fine-tuned'}.get(n, n)
                if base_n not in Qs:
                    continue
                stopped = speed[base_n][a:z + 1] < vg['still_fraction'] * np.maximum(peak, 1e-6)
                P = posterior(S[a:z + 1], Qs[base_n][a:z + 1], cand, beta, ray if n.endswith('+ pointing') else None, kappa,
                              stopped, r['reach'][g['set']])
                ok = np.argmax(P, 1) == g['target']
                bad = np.flatnonzero(~ok)
                k = 0 if not len(bad) else (bad[-1] + 1 if bad[-1] + 1 < len(ok) else None)
                row[f'{n}_stable_s'] = None if k is None else round(tt[a + k] - g['t0'], 2)
                row[f'{n}_p_end'] = round(float(P[-1, g['target']]), 3)
            rows.append(row)
    names = [n for n in names + ['observed + pointing', 'fine-tuned + pointing'] if f'{n}_p_end' in rows[0]]
    (out / 'results.json').write_text(json.dumps(rows, indent=1))
    by = defaultdict(list)
    for r in rows:
        by[r['set']].append(r)
    med = lambda v: f'{np.median(v):.2f}' if len(v) else 'nan'
    lines = [f"# v6 goal inference, inverse planning on observed + forecast path ({out.name}); beta 1/{vg['beta_from_error_mm']} mm, "
             f"keypoint noise {vt['keypoint_noise_m']} m", '',
             "Per observer: correct at the motion end (fraction) / stable-correct time, s after the motion start (median over "
             "the correct ones; negative = before the motion starts) / mean P(true) at the end.", '',
             '| set | n | K | motion s | ' + ' | '.join(names) + ' |', '|' + '---|' * (4 + len(names))]
    for s, rs in by.items():
        cells = []
        for n in names:
            v = [r[f'{n}_stable_s'] for r in rs if r[f'{n}_stable_s'] is not None]
            cells.append(f"{len(v) / len(rs):.2f} / {med(v)} / {np.mean([r[f'{n}_p_end'] for r in rs]):.2f}")
        lines.append(f"| {s} | {len(rs)} | {rs[0]['K']} | {med([r['dur'] for r in rs])} | " + ' | '.join(cells) + ' |')
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
