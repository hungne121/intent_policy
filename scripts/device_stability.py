"""From which belief level does the measuring device stop flip-flopping between candidates? (user request 2026-10-18:
the robot starts moving towards the target once the belief has passed a level after which it only grows, and grasps
at 0.9.) Same gestures and belief as scripts/device_early_curve.py (simulation, episodes outside the device's
training / validation pool, intent_pd.p_target recorded online, restricted to the true target's kind).

Per directed gesture and level θ, the first frame whose leading candidate's belief reaches θ (frames t_onset .. t_end):
  crossed    the level is reached before the end of the gesture
  wrong      the leading candidate at that frame is not the human's target
  flip       after that frame the leading candidate changes at least once (another candidate's belief exceeds the
             chosen one's): the "strong oscillation" of the request
  drop       the largest fall of the chosen candidate's belief after that frame (how far from "grows steadily")
  τ, ρ       time since the motion onset [s] and the share of the approach motion seen, (t - t_onset) / (t_clear - t_onset)
Shares with 95 % bootstrap intervals over episodes.

  ./run.sh -m scripts.device_stability --run outputs/prelim/2026-10-13_slow --out outputs/dp_prediction/2026-10-18
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.device_early_curve import boot, load

LEVELS = np.round(np.arange(0.55, 0.951, 0.05), 2)


def gesture_rows(D: pd.DataFrame) -> list[dict]:
    """One row per (gesture, level)."""
    rows = []
    for seg, g in D.groupby('seg', sort=False):
        g = g.sort_values('tau')
        bt, bw = g['b_true'].values, g['b_wrong'].values
        lead_true = bt >= bw                       # the leading candidate is the human's target
        lead = np.maximum(bt, bw)
        for th in LEVELS:
            hit = np.flatnonzero(lead >= th)
            r = dict(seg=seg, ep=g['ep'].iloc[0], task=g['task'].iloc[0], level=float(th), crossed=bool(len(hit)))
            if len(hit):
                i = hit[0]
                chosen_true = bool(lead_true[i])
                chosen = bt[i:] if chosen_true else bw[i:]
                other = bw[i:] if chosen_true else bt[i:]
                r.update(wrong=not chosen_true, flip=bool((other > chosen).any()),
                         drop=float(np.max(np.maximum.accumulate(chosen) - chosen)),
                         tau=float(g['tau'].values[i]), rho=float(g['rho'].values[i]))
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--datasets', nargs='+', default=['intent_act_device_early_prelim', 'intent_act_device_late_prelim'])
    ap.add_argument('--exclude-list', default='lists/pilot.jsonl')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    run = Path(args.run)
    seen = set(json.loads((run / 'device/device_folds.json').read_text()).get('seen_seeds', []))
    if args.exclude_list:
        seen |= {json.loads(l)['seed'] for l in (run / args.exclude_list).read_text().splitlines() if l.strip()}
    D = pd.concat([load(run / 'datasets' / d, seen)[0] for d in args.datasets], ignore_index=True)
    G = pd.DataFrame(gesture_rows(D))
    rng = np.random.default_rng(0)
    res = []
    for th, g in G.groupby('level'):
        c = g[g['crossed']]
        ep = c['ep'].values
        f = c['flip'].values.astype(float)
        w = c['wrong'].values.astype(float)
        res.append(dict(level=float(th), gestures=int(len(g)), crossed=float(g['crossed'].mean()),
                        flip=float(f.mean()), flip_ci=boot(f, ep, rng=rng), wrong=float(w.mean()), wrong_ci=boot(w, ep, rng=rng),
                        drop_median=float(c['drop'].median()), drop_p90=float(c['drop'].quantile(0.9)),
                        tau_median=float(c['tau'].median()), rho_median=float(c['rho'].median()),
                        rho_iqr=[float(c['rho'].quantile(0.25)), float(c['rho'].quantile(0.75))]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'device_stability.json').write_text(json.dumps(dict(n_gestures=int(D['seg'].nunique()),
                                                               n_episodes=int(D['ep'].nunique()), levels=res), indent=1))
    L = [f"# Belief stability after a level is first reached (simulation, {D['seg'].nunique()} directed gestures, "
         f"{D['ep'].nunique()} episodes outside the device pool)\n",
         '| level | reached | wrong at the frame | flips afterwards | largest fall afterwards, median / p90 | '
         'τ median [s] | share of the gesture seen, median [IQR] |', '|---|---|---|---|---|---|---|']
    for r in res:
        L.append(f"| {r['level']:.2f} | {r['crossed']:.2f} | {r['wrong']:.3f} [{r['wrong_ci'][0]:.3f}, {r['wrong_ci'][1]:.3f}] | "
                 f"{r['flip']:.3f} [{r['flip_ci'][0]:.3f}, {r['flip_ci'][1]:.3f}] | {r['drop_median']:.3f} / {r['drop_p90']:.3f} | "
                 f"{r['tau_median']:.2f} | {100 * r['rho_median']:.0f}% [{100 * r['rho_iqr'][0]:.0f}, {100 * r['rho_iqr'][1]:.0f}] |")
    (out / 'device_stability.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
