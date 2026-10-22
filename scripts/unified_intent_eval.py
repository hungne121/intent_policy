"""TRIAL (2026-10-18): offline check of the two new parts of the unified intent state (intent/unified.py) on the recorded
device outputs (intent_pd.*), episodes outside the device's training / validation pool (as scripts/device_early_curve).

1. hold = p_gesture(palm_forward), against the simulator's 'stop_signal' segments (T5 'wait!', t_onset .. t_end):
   detection delay (first frame >= 0.5 after t_onset, w.r.t. t_onset and to t_clear, the pose reached), misses,
   false alarms (frames outside every stop_signal with hold >= 0.5, per task), binary log score against the
   base-rate forecast (a proper-score reference, Gneiting & Raftery 2007).
2. time to ready (T2 / T5 palm-up 'reach_out' segments; truth: t_clear, the open hand in place = `hand_ready`), from the
   device's own onset (first frame with p_gesture(palm_up) >= 0.5):
     prior   ready_prior: the onset-to-ready durations of the OTHER episodes, conditioned on not being ready yet
             (the prior term of Maeda et al. 2017's phase estimate)
     tte     the v3.2 contract field intent_pd.tte (rule-based, from the simulator's keypoints: privileged)
   absolute / signed error of the predicted remaining time, by time since the onset, and at the frame the device's
   hand-zone belief first reaches 0.9 (when the robot would start its final approach).
Simulated humans move with minimum-jerk durations drawn from a range, so these numbers say how good the estimates
are in simulation only.

  ./run.sh -m scripts.unified_intent_eval --run outputs/prelim/2026-10-13_slow --out outputs/dp_prediction/2026-10-18
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from intent_policy.intent.unified import ready_prior, unified_belief

FPS = 20


def load(root: Path, seen: set):
    schema = json.loads((root / 'meta/intent_schema.json').read_text())
    classes = list(schema['gestures']) + ['withdraw']
    eps = {e['episode_index']: e for e in map(json.loads, (root / 'meta/hri_episodes.jsonl').read_text().splitlines())}
    segs = json.loads((root / 'meta/intent_segments.json').read_text())
    cols = ['episode_index', 'frame_index', 'intent_pd.p_target', 'intent_pd.p_gesture', 'intent_pd.tte']
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(root / 'data/**/*.parquet'), recursive=True))])
    out = []
    for ei, g in df.groupby('episode_index'):
        e = eps[int(ei)]
        if e['seed'] in seen:
            continue
        g = g.sort_values('frame_index')
        out.append(dict(ep=f'{root.name}:{ei}', task=e['list_task'], P=np.stack(g['intent_pd.p_target'].values),
                        G=np.stack(g['intent_pd.p_gesture'].values), tte=np.asarray([np.ravel(v)[0] for v in g['intent_pd.tte'].values], float),
                        segs=segs.get(str(ei), []), schema=schema, classes=classes))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--datasets', nargs='+', default=['intent_act_device_early_prelim', 'intent_act_device_late_prelim'])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    run = Path(args.run)
    seen = set(json.loads((run / 'device/device_folds.json').read_text()).get('seen_seeds', []))
    seen |= {json.loads(l)['seed'] for l in (run / 'lists/pilot.jsonl').read_text().splitlines() if l.strip()}
    E = [x for d in args.datasets for x in load(run / 'datasets' / d, seen)]
    classes = E[0]['classes']
    T = E[0]['schema']['targets']
    hand_idx = [T.index(t) for t in E[0]['schema']['target_kinds']['hand']]
    L = [f'# Unified intent trial: offline check (episodes outside the device pool: {len(E)})\n']

    # ---- 1. hold
    sums, delays, delays_clear, missed, n_stop = [], [], [], 0, 0
    fa = {}
    lab_all, pi_all = [], []
    for x in E:
        U = unified_belief(x['P'], x['G'], classes)
        sums.append(np.abs(U.sum(1) - 1).max())
        pi = U[:, -1]
        lab = np.zeros(len(pi), bool)
        for s in x['segs']:
            if s['gesture'] == 'palm_forward':
                n_stop += 1
                t0, tc, te = int(s['t_onset']), int(s['t_clear']), min(int(s['t_end']), len(pi) - 1)
                lab[t0:te + 1] = True
                hit = np.flatnonzero(pi[t0:te + 1] >= 0.5)
                if len(hit):
                    delays.append(hit[0] / FPS), delays_clear.append((t0 + hit[0] - tc) / FPS)
                else:
                    missed += 1
        f = fa.setdefault(x['task'], [0, 0])
        f[0] += int((pi[~lab] >= 0.5).sum())
        f[1] += int((~lab).sum())
        lab_all.append(lab), pi_all.append(pi)
    lab_all, pi_all = np.concatenate(lab_all), np.clip(np.concatenate(pi_all), 1e-6, 1 - 1e-6)
    base = lab_all.mean()
    ls = float(np.mean(np.where(lab_all, np.log(pi_all), np.log(1 - pi_all))))
    ls0 = float(np.mean(np.where(lab_all, np.log(base), np.log(1 - base))))
    L += ['## 1. hold = p_gesture(palm_forward)\n',
          f'u sums to 1 within {max(sums):.1e}. stop_signal segments: {n_stop}, missed (never >= 0.5): {missed}.',
          f'Detection delay after the onset: median {np.median(delays):.2f} s (IQR {np.percentile(delays, 25):.2f}-'
          f'{np.percentile(delays, 75):.2f}); w.r.t. the stop pose reached: median {np.median(delays_clear):+.2f} s.',
          f'Binary log score of hold: {ls:.3f} (base-rate forecast {ls0:.3f}, base rate {base:.3f}).', '',
          '| task | false-alarm share (hold >= 0.5 outside a stop_signal) | frames |', '|---|---|---|']
    L += [f'| {t} | {a / max(b, 1):.4f} | {b} |' for t, (a, b) in sorted(fa.items())]

    # ---- 2. time to ready
    seg_rows, durations = [], {}
    for x in E:
        pu = x['G'][:, classes.index('palm_up')]
        for s in x['segs']:
            if s['gesture'] != 'palm_up':
                continue
            t0, tc = int(s['t_onset']), int(s['t_clear'])
            win = np.flatnonzero(pu[max(0, t0 - 10):tc + 1] >= 0.5)
            if not len(win):
                seg_rows.append(dict(ep=x['ep'], detected=False))
                continue
            td = max(0, t0 - 10) + int(win[0])
            durations.setdefault(x['ep'], []).append((tc - td) / FPS)
            P = x['P'][:, hand_idx]
            share = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
            dec = next((t for t in range(td, tc + 1) if share[t].max() >= 0.9), None)
            seg_rows.append(dict(ep=x['ep'], detected=True, td=td, tc=tc, t0=t0, dec=dec, tte=x['tte']))
    rows = []
    for r in seg_rows:
        if not r['detected']:
            continue
        others = np.array([d for ep, ds in durations.items() if ep != r['ep'] for d in ds])
        for t in range(r['td'], r['tc'] + 1):
            el, truth = (t - r['td']) / FPS, (r['tc'] - t) / FPS
            m, _ = ready_prior(el, others)
            rows.append(dict(el=el, truth=truth, prior=m, tte=float(r['tte'][t]), at_dec=(t == r['dec'])))
    R = pd.DataFrame(rows)
    det = [r for r in seg_rows if r['detected']]
    allD = np.concatenate([np.array(v) for v in durations.values()])
    L += ['', '## 2. Time to ready (palm-up reach_out; truth = t_clear)\n',
          f"Segments: {len(seg_rows)}, detected by the device: {len(det)}. Device onset - simulator onset: median "
          f"{np.median([(r['td'] - r['t0']) / FPS for r in det]):+.2f} s. Onset-to-ready duration: median {np.median(allD):.2f} s "
          f"(IQR {np.percentile(allD, 25):.2f}-{np.percentile(allD, 75):.2f}).", '',
          '| time since the device onset [s] | frames | |error| prior [s] | |error| tte [s] | signed error prior | signed error tte |',
          '|---|---|---|---|---|---|']
    for lo in np.arange(0, 2.01, 0.25):
        x = R[(R.el >= lo) & (R.el < lo + 0.25)]
        if len(x) < 5:
            continue
        L.append(f"| {lo:.2f} | {len(x)} | {np.median(np.abs(x.prior - x.truth)):.2f} | {np.median(np.abs(x.tte - x.truth)):.2f} | "
                 f"{np.median(x.prior - x.truth):+.2f} | {np.median(x.tte - x.truth):+.2f} |")
    d = R[R.at_dec]
    if len(d):
        L += ['', f"At the hand-zone decision (belief >= 0.9; {len(d)} gestures): truth remaining median {np.median(d.truth):.2f} s; "
                  f"|error| prior {np.median(np.abs(d.prior - d.truth)):.2f} s, tte {np.median(np.abs(d.tte - d.truth)):.2f} s; "
                  f"time from the device onset to the decision median {np.median(d.el):.2f} s."]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'unified_intent_eval.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
