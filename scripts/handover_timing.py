"""Handover timing of T2 episodes (docs/discussions/handover_som_va_intent_sai.md §4): is the robot ready when the
human's open hand is? Reads evaluation runs (scripts/evaluate.py episode records) and reports, per run:
  HRM            t(hand_ready) - t(robot_at_handover): > 0 the robot was ready first (the human did not wait);
                 = - Hoffman's functional delay of this event (Evaluating Fluency in Human-Robot Collaboration, 2019,
                 Eq. 9)
  human wait     max(0, -HRM); robot wait max(0, HRM) (waiting times, Ortenzi et al. 2021)
  stages         the expert's plan stages (trace extras.expert_stage) relative to the start of the reach-out
  safety         the episode's minimum human-robot distance
  success        episode success, and the handover done (robot_at_handover logged)
With two runs on the same episode list (--paired A B): per-seed differences of HRM, with 95 % bootstrap intervals
over episodes.

  ./run.sh -m scripts.handover_timing --runs outputs/handover_trial/2026-10-18/expert_staging_wait \
      outputs/handover_trial/2026-10-18/expert_anticipate --paired --out outputs/handover_trial/2026-10-18
"""
import argparse
import glob
import gzip
import json
from pathlib import Path

import numpy as np

STAGES = ('to_staging', 'await_hand', 'await_palm', 'hedge', 'over_hand', 'settle_over_hand', 'time_descent', 'to_handover')


def first(r, typ, key=None, val=None):
    return next((e['timestamp'] for e in r['events'] if e['event_type'] == typ
                 and (key is None or (e.get('payload') or {}).get(key) == val)), None)


def episode(r):
    ro = first(r, 'human_motion_start', 'label', 'reach_out')
    hr = first(r, 'protocol_step_complete', 'step', 'hand_ready')
    ra = first(r, 'protocol_step_complete', 'step', 'robot_at_handover')
    st, prev = {}, None
    for s in r.get('trace', []):
        g = (s.get('extras') or {}).get('expert_stage')
        if g != prev and g not in st:
            st[g] = s['t']
        prev = g
    return dict(seed=r['seed'], success=bool(r['success']), failure=r.get('failure'), reach=ro, ready=hr, at=ra,
                hrm=None if hr is None or ra is None else hr - ra, mind=r.get('min_human_robot_distance'),
                stages={k: (st[k] - ro) for k in STAGES if k in st and ro is not None}, duration=r['duration_s'])


def load(run):
    return {e['seed']: e for e in (episode(json.load(gzip.open(f)))
                                   for f in sorted(glob.glob(f'{run}/episodes/default/T2/*.json.gz')))}


def boot(x, n=5000, rng=np.random.default_rng(0)):
    x = np.asarray(x, float)
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--paired', action='store_true', help='per-seed differences of the second run minus the first')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    R = {Path(r).name: load(r) for r in args.runs}
    L = ['# Handover timing (T2)\n', '| run | episodes | success | handover done | HRM median [95 % CI] | P(HRM >= 0) | '
         'human wait median (when > 0) | robot wait median (when > 0) | min human-robot distance median [m] |',
         '|---|---|---|---|---|---|---|---|---|']
    for name, E in R.items():
        v = list(E.values())
        h = np.array([e['hrm'] for e in v if e['hrm'] is not None])
        hw, rw = np.maximum(0, -h), np.maximum(0, h)
        md = [e['mind'] for e in v if e['mind'] is not None]
        ci = boot(h) if len(h) > 1 else (np.nan, np.nan)
        L.append(f"| {name} | {len(v)} | {np.mean([e['success'] for e in v]):.2f} | {len(h)} | "
                 f"{np.median(h):+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] | {(h >= 0).mean():.2f} | "
                 f"{np.median(hw[hw > 0]) if (hw > 0).any() else 0:.2f} s ({(hw > 0).sum()}) | "
                 f"{np.median(rw[rw > 0]) if (rw > 0).any() else 0:.2f} s ({(rw > 0).sum()}) | "
                 f"{np.median(md) if md else float('nan'):.3f} |")
    L += ['', '## Stages, seconds after the reach-out starts (median)\n',
          '| run | ' + ' | '.join(STAGES) + ' | robot ready | hand ready |', '|---|' + '---|' * (len(STAGES) + 2)]
    for name, E in R.items():
        v = list(E.values())
        cells = []
        for k in STAGES:
            x = [e['stages'][k] for e in v if k in e['stages']]
            cells.append(f'{np.median(x):+.2f}' if x else '—')
        ra = [e['at'] - e['reach'] for e in v if e['at'] is not None and e['reach'] is not None]
        hr = [e['ready'] - e['reach'] for e in v if e['ready'] is not None and e['reach'] is not None]
        L.append(f"| {name} | " + ' | '.join(cells) + f" | {np.median(ra):+.2f} | {np.median(hr):+.2f} |")
    fails = {name: sorted({str(e['failure']) for e in E.values() if not e['success']}) for name, E in R.items()}
    L += ['', f'Failures: {fails}']
    if args.paired and len(R) == 2:
        (na, A), (nb, B) = R.items()
        seeds = sorted(s for s in A if s in B and A[s]['hrm'] is not None and B[s]['hrm'] is not None)
        d = np.array([B[s]['hrm'] - A[s]['hrm'] for s in seeds])
        if len(d):
            ci = boot(d)
            L += ['', f'## Paired ({nb} - {na}), {len(seeds)} seeds with both handovers done\n',
                  f'HRM difference: median {np.median(d):+.2f} s [95 % CI {ci[0]:+.2f}, {ci[1]:+.2f}]; '
                  f'{nb} earlier in {(d > 0).sum()}/{len(d)} episodes.']
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'handover_timing.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
