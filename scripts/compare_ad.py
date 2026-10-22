"""A (late, no intent) vs D (early, intent) on the same evaluation cells (trial of 2026-10-18, scripts/trial_t235.sh).

Per run and task: CSR, failure reasons, CT, AM, WCR, CRsp (benchmark/metrics.py, HRIBench-style) and the functional
delays of *Evaluating Fluency in Human-Robot Collaboration* (Hoffman 2019, Eq. 9; F-DEL = start of the waiting
party's dependent action - end of the other party's action, < 0: the robot was there first):
  pick      robot_target_commit (first correct object commit) - end of the human's cue  = -AM          (T2, T3, T5)
  handover  robot_at_handover - hand_ready (T2 and T5 on T2) = -HRM; the human waits max(0, F-DEL)
  cup       cup_at_support - interaction_window_start (T3, T5 on T3: the human holds the cube and waits for the cup)
Paired differences A - D per seed (> 0: D earlier / shorter), median with a 95 % bootstrap interval, only over the
seeds where both runs have the value; CSR as the paired counts (both / A only / D only / neither).

  ./run.sh -m scripts.compare_ad --runs A=outputs/trial_t235/2026-10-18/eval/A D=outputs/trial_t235/2026-10-18/eval/D \
      --out outputs/trial_t235/2026-10-18/compare_ad
"""
import argparse
import glob
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

TASKS = ('T2', 'T3', 'T5')


def t_of(events, etype, **payload):
    for e in events:
        if e['event_type'] == etype and all((e.get('payload') or {}).get(k) == v for k, v in payload.items()):
            return float(e['timestamp'])
    return None


def num(x):
    return float(x) if isinstance(x, (int, float)) and x is not None and np.isfinite(x) else None


def episode(r) -> dict:
    ev, m = r['events'], r['metrics']
    hand = t_of(ev, 'protocol_step_complete', step='hand_ready')
    at = t_of(ev, 'protocol_step_complete', step='robot_at_handover')
    wait = t_of(ev, 'interaction_window_start', kind='assist')
    cup = t_of(ev, 'protocol_step_complete', step='cup_at_support')
    am = num(m.get('AM'))
    return dict(task=r.get('list_task') or r['task'], seed=int(r['seed']), success=bool(r['success']), failure=r.get('failure'),
                ct=num(m.get('CT')) if r['success'] else None, am=am, pick=None if am is None else -am,
                wcr=num(m.get('WCR')), crsp=num(m.get('CRsp')),
                handover=None if hand is None or at is None else at - hand,
                cup=None if wait is None or cup is None else cup - wait,
                min_dist=num(r.get('min_human_robot_distance')))


def load(run: Path) -> dict:
    out = {}
    for f in sorted(glob.glob(str(run / 'episodes' / 'default' / '*' / '*.json.gz'))):
        e = episode(json.load(gzip.open(f)))
        out[(e['task'], e['seed'])] = e
    return out


def boot_median(x, n=5000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def med(xs):
    xs = [x for x in xs if x is not None]
    return (float(np.median(xs)), len(xs)) if xs else (None, 0)


def fmt(v, digits=2, sign=False):
    return '—' if v is None or (isinstance(v, float) and np.isnan(v)) else (f'{v:+.{digits}f}' if sign else f'{v:.{digits}f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs=2, required=True, metavar='NAME=DIR', help='reference first (A), then D')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    names, R = [], {}
    for item in args.runs:
        name, path = item.split('=', 1)
        names.append(name)
        R[name] = load(Path(path))
    a, d = names
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    L = [f'# {a} vs {d} on the same evaluation cells\n',
         'Medians over episodes where the value exists (n). F-DEL < 0: the robot was there before the human finished.\n',
         '## Per run and task\n',
         '| run | task | episodes | CSR | failures | CT median [s] (n) | F-DEL pick median [s] (n) | F-DEL handover (n) | '
         'F-DEL cup (n) | WCR | CRsp median (n) |', '|---|---|---|---|---|---|---|---|---|---|---|']
    res = dict(runs={}, paired={})
    for name in names:
        for t in TASKS + ('all',):
            E = [e for (tk, _), e in R[name].items() if t == 'all' or tk == t]
            if not E:
                continue
            ct, pk, ho, cu, cr = (med([e[k] for e in E]) for k in ('ct', 'pick', 'handover', 'cup', 'crsp'))
            fails = Counter(e['failure'] for e in E if not e['success'])
            csr = float(np.mean([e['success'] for e in E]))
            wcr = float(np.mean([e['wcr'] for e in E if e['wcr'] is not None])) if any(e['wcr'] is not None for e in E) else None
            res['runs'][f'{name}/{t}'] = dict(n=len(E), csr=csr, failures=dict(fails), ct=ct, pick=pk, handover=ho,
                                              cup=cu, wcr=wcr, crsp=cr)
            L.append(f"| {name} | {t} | {len(E)} | {csr:.2f} | {', '.join(f'{k} {v}' for k, v in fails.most_common()) or '—'} | "
                     f"{fmt(ct[0])} ({ct[1]}) | {fmt(pk[0], sign=True)} ({pk[1]}) | {fmt(ho[0], sign=True)} ({ho[1]}) | "
                     f"{fmt(cu[0], sign=True)} ({cu[1]}) | {fmt(wcr)} | {fmt(cr[0])} ({cr[1]}) |")
    L += ['', f'## Paired by seed: {a} - {d} (> 0: {d} earlier / shorter)\n',
          '| task | CSR: both / only ' + a + ' / only ' + d + ' / neither | CT (n) | F-DEL pick (n) | F-DEL handover (n) | '
          'F-DEL cup (n) | CRsp (n) |', '|---|---|---|---|---|---|---|']
    for t in TASKS + ('all',):
        keys = [k for k in R[a] if k in R[d] and (t == 'all' or k[0] == t)]
        if not keys:
            continue
        c = Counter((R[a][k]['success'], R[d][k]['success']) for k in keys)
        row, res['paired'][t] = [f"{c[(True, True)]} / {c[(True, False)]} / {c[(False, True)]} / {c[(False, False)]}"], {}
        for m in ('ct', 'pick', 'handover', 'cup', 'crsp'):
            diff = [R[a][k][m] - R[d][k][m] for k in keys if R[a][k][m] is not None and R[d][k][m] is not None]
            mm = float(np.median(diff)) if diff else None
            ci = boot_median(diff) if diff else (np.nan, np.nan)
            res['paired'][t][m] = dict(median=mm, ci=ci, n=len(diff))
            row.append('—' if mm is None else f'{mm:+.2f} [{fmt(ci[0], sign=True)}; {fmt(ci[1], sign=True)}] ({len(diff)})')
        L.append(f'| {t} | ' + ' | '.join(row) + ' |')
    (out / 'compare.md').write_text('\n'.join(L) + '\n')
    (out / 'compare.json').write_text(json.dumps(res, indent=1, default=str))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
