"""Functional delay (F-DEL) of evaluation runs: the main "earlier" measure of the thesis deployment doc (§1.2, §7),
from *Evaluating Fluency in Human-Robot Collaboration* (Hoffman 2019, Eq. 9), per event and signed:
  F-DEL = start of the waiting party's dependent action - end of the other party's action  (< 0: anticipated)
Two events are measured from the logged benchmark events of scripts/evaluate.py episode records:
  pick      robot_target_commit (first correct, object stage) - end of the human's cue (the scenario's
            `am_cue_complete` step) = -AM (benchmark/metrics.py), T1, T2, T3, T5
  handover  robot_at_handover - hand_ready (T2) = -HRM; human idle H-IDLE = max(0, F-DEL)
"Earlier" = F-DEL(reference) - F-DEL(run), paired by seed: median of the per-seed differences and a 95 % bootstrap
interval over the paired episodes. Also CSR, CT, WCR (benchmark/metrics.py).

  ./run.sh -m scripts.fdel_report --root outputs/prelim/2026-10-13_slow/eval --runs A B D D_f1 D_film_vdrop \
      expert_early --ref B --out outputs/prelim/2026-10-13_slow/fdel_2026-10-18
"""
import argparse
import glob
import gzip
import json
from pathlib import Path

import numpy as np

from intent_policy.benchmark.metrics import AM, CT, WCR, EpisodeContext

TASKS = ('T1', 'T2', 'T3', 'T5')


def first_step(events, step):
    return next((float(e['timestamp']) for e in events if e['event_type'] == 'protocol_step_complete'
                 and (e.get('payload') or {}).get('step') == step), None)


def episode(r) -> dict:
    sc = r['scenario_config']
    ev = r['events']
    ctx = EpisodeContext(protocol_steps=sc.get('protocol_steps') or [], params=sc.get('metric_params') or {})
    am = AM().episode_value(ev, ctx)
    hr, ra = first_step(ev, 'hand_ready'), first_step(ev, 'robot_at_handover')
    return dict(seed=r['seed'], task=r['task'], success=bool(r['success']),
                pick=None if am is None else -am,
                handover=None if hr is None or ra is None else ra - hr,
                ct=CT().episode_value(ev, ctx), wcr=WCR().episode_value(ev, ctx))


def load(run: Path) -> dict:
    out = {}
    for f in sorted(glob.glob(str(run / 'episodes' / 'default' / '*' / '*.json.gz'))):
        e = episode(json.load(gzip.open(f)))
        if e['task'] in TASKS:
            out[(e['task'], e['seed'])] = e
    return out


def boot_median(x, n=5000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def fmt(v, ci=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    s = f'{v:+.2f}'
    return s if ci is None or np.isnan(ci[0]) else f'{s} [{ci[0]:+.2f}, {ci[1]:+.2f}]'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, help='folder holding the evaluation runs')
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--ref', required=True, help='reference run of the paired "earlier" differences')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    R = {name: load(Path(args.root) / name) for name in args.runs}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    L = ['# F-DEL (Hoffman 2019) of the evaluation runs\n',
         f'Runs under `{args.root}`. F-DEL in seconds, median over episodes where the event happened '
         '(< 0: the robot acted before the human finished). Paired "earlier" = F-DEL(ref) - F-DEL(run), '
         'median of per-seed differences with a 95 % bootstrap interval.\n',
         '## Per run\n',
         '| run | episodes | CSR | WCR | CT median [s] | F-DEL pick, median (n) | F-DEL handover T2, median (n) | '
         'H-IDLE T2, median |', '|---|---|---|---|---|---|---|---|']
    summary = {}
    for name, E in R.items():
        v = list(E.values())
        pick = [e['pick'] for e in v if e['pick'] is not None]
        ho = [e['handover'] for e in v if e['handover'] is not None]
        ct = [e['ct'] for e in v if e['ct'] is not None]
        summary[name] = dict(n=len(v), csr=float(np.mean([e['success'] for e in v])) if v else None,
                             wcr=float(np.mean([e['wcr'] for e in v])) if v else None,
                             ct=float(np.median(ct)) if ct else None,
                             pick=float(np.median(pick)) if pick else None, n_pick=len(pick),
                             handover=float(np.median(ho)) if ho else None, n_handover=len(ho),
                             hidle=float(np.median(np.maximum(0, ho))) if ho else None)
        s = summary[name]
        ct_s = '—' if s['ct'] is None else f"{s['ct']:.2f}"
        L.append(f"| {name} | {s['n']} | {s['csr']:.2f} | {s['wcr']:.2f} | {ct_s} | {fmt(s['pick'])} ({s['n_pick']}) | "
                 f"{fmt(s['handover'])} ({s['n_handover']}) | {fmt(s['hidle'])} |")
    L += ['', '## Per task: F-DEL pick, median (n)\n', '| run | ' + ' | '.join(TASKS) + ' |',
          '|---|' + '---|' * len(TASKS)]
    for name, E in R.items():
        cells = []
        for t in TASKS:
            x = [e['pick'] for (tk, _), e in E.items() if tk == t and e['pick'] is not None]
            cells.append(f'{fmt(float(np.median(x)) if x else None)} ({len(x)})')
        L.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    ref = R[args.ref]
    L += ['', f'## Paired "earlier" than {args.ref}: F-DEL({args.ref}) - F-DEL(run)\n',
          '| run | pick, all tasks (n) | pick T1 (n) | pick T2 (n) | handover T2 (n) |', '|---|---|---|---|---|']
    paired = {}
    for name, E in R.items():
        if name == args.ref:
            continue
        row, paired[name] = [], {}
        for key, tasks in (('pick', TASKS), ('pick', ('T1',)), ('pick', ('T2',)), ('handover', ('T2',))):
            d = [ref[k][key] - E[k][key] for k in ref if k in E and k[0] in tasks
                 and ref[k][key] is not None and E[k][key] is not None]
            med = float(np.median(d)) if d else None
            ci = boot_median(d) if d else (np.nan, np.nan)
            paired[name][f"{key}_{'_'.join(tasks) if len(tasks) == 1 else 'all'}"] = dict(median=med, ci=ci, n=len(d))
            row.append(f'{fmt(med, ci)} ({len(d)})')
        L.append(f'| {name} | ' + ' | '.join(row) + ' |')
    (out / 'fdel.md').write_text('\n'.join(L) + '\n')
    (out / 'fdel.json').write_text(json.dumps(dict(root=args.root, ref=args.ref, summary=summary, paired=paired),
                                              indent=1))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
