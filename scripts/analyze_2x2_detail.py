"""Detailed 2x2 analysis against the hypotheses (docs/requirements/INTENT_ACT_v3.2.md §0, §7).

  goal  "có intent thì hành động sớm mới hiệu quả": the interaction (D - B) - (C - A) is the main result
  H1    B commits wrongly more than D (WCR, WCD)
  H2    the AM of D ~ the early expert's > A ~ C
  H3    (D - B) > (C - A) in CT and in commit correctness (WCR)

Reads evaluation runs on the same episode list (scripts/evaluate.py: episodes/<split>/<task>/*.json.gz): the models
A (late), B (early), C (late + intent), D (early + intent) and, as references, the two experts on the same episodes.
Reports, per list task, over the intent tasks (T1, T2, T3, T5) and over all:
  means     the HRIBench-style metrics (CSR, CT, Rsp, WCR, WCD, AM, CRsp, DSR; benchmark/metrics.py)
  contrasts as benefits (the sign of lower-is-better metrics flipped: > 0 always means better):
            cost of acting early without intent B vs A, gain of acting early with intent D vs C, intent effect late
            C vs A and early D vs B, interaction (D vs B) - (C vs A); 95 % paired bootstrap intervals over the episodes
            the compared runs share, the bootstrap probability that the benefit is > 0, per-episode sign counts
  control   T4 / T4neg: the late and the early expert act identically there, so their B vs A and D vs C measure the
            training noise of the networks
  mechanism the first robot_target_commit of each episode on the intent tasks (its time relative to the end of the
            human's cue, its correctness) and the wrong commits before / after the cue ended: a decomposition of AM and
            WCR, not a new metric; the failure reasons
Verdicts: "supported" when the 95 % interval excludes 0 in the predicted direction, "contradicted" when it excludes 0
in the other direction, else "not decided".

  ./run.sh -m scripts.analyze_2x2_detail --runs A=<eval> B=<eval> C=<eval> D=<eval> \
      --experts late=<eval> early=<eval> --output <report.md>
"""
import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from intent_policy.experts.scripted_expert import CUE_COMPLETE_STEP
from intent_policy.utils import resolve

METRICS = ('CSR', 'CT', 'Rsp', 'WCR', 'WCD', 'AM', 'CRsp', 'DSR')       # HRIBench-style (benchmark/metrics.py)
LOWER_IS_BETTER = {'CT', 'Rsp', 'WCR', 'WCD', 'CRsp'}
UNITS = {'CT': 's', 'Rsp': 's', 'WCD': 's', 'AM': 's', 'CRsp': 's'}
TASKS = ('T1', 'T2', 'T3', 'T4', 'T4neg', 'T5')
INTENT_TASKS = ('T1', 'T2', 'T3', 'T5')             # a target to choose from the human's gesture
CONTROL_TASKS = ('T4', 'T4neg')                     # the late and the early expert act identically
CONTRASTS = {                                       # name: (better run, baseline run) or interaction
    'B vs A: early without intent': ('B', 'A'),
    'D vs C: early with intent': ('D', 'C'),
    'C vs A: intent when late': ('C', 'A'),
    'D vs B: intent when early': ('D', 'B'),
    'interaction (D vs B) - (C vs A)': 'interaction',
}
EXPECTED = {'B vs A: early without intent': -1, 'D vs C: early with intent': +1, 'D vs B: intent when early': +1,
            'interaction (D vs B) - (C vs A)': +1}
N_BOOT = 10_000


def cue_step(rec: dict) -> str | None:
    task = rec.get('task')
    return CUE_COMPLETE_STEP.get(task if task in CUE_COMPLETE_STEP else (rec.get('spec') or {}).get('task'))


def mechanism(rec: dict) -> dict:
    """First robot_target_commit relative to the end of the cue; wrong commits before / after it."""
    ev = rec['events']
    step = cue_step(rec)
    cue = next((e['timestamp'] for e in ev if e['event_type'] == 'protocol_step_complete'
                and e['payload'].get('step') == step), None) if step else None
    com = [e for e in ev if e['event_type'] == 'robot_target_commit']
    first = com[0] if com else None
    wrong = [e['timestamp'] for e in com if not e['payload'].get('correct')]
    return dict(cue_t=cue, first_t=None if first is None else first['timestamp'],
                first_correct=None if first is None else bool(first['payload'].get('correct')),
                lead=None if first is None or cue is None else cue - first['timestamp'],
                wrong_before_cue=None if cue is None else sum(t < cue for t in wrong),
                wrong_after_cue=None if cue is None else sum(t >= cue for t in wrong))


def load_run(run) -> dict[str, dict]:
    out = {}
    for f in sorted((resolve(run) / 'episodes').glob('*/*/*.json.gz')):
        with gzip.open(f, 'rt') as fh:
            rec = json.load(fh)
        m = rec.get('metrics') or {}
        vals = {k: float(v) for k, v in m.items() if k in METRICS and isinstance(v, (int, float))}
        out[rec['episode_id']] = dict(task=rec.get('list_task') or f.parent.name, success=bool(rec.get('success')),
                                      failure=rec.get('failure'), metrics=vals, mech=mechanism(rec))
    return out


def benefit(metric: str, x: np.ndarray) -> np.ndarray:
    return -x if metric in LOWER_IS_BETTER else x


def contrast_values(runs: dict, metric: str, name, ids) -> tuple[list[str], np.ndarray]:
    """Per-episode benefit of a contrast on the episodes where every run it needs has the metric."""
    need = 'ABCD' if name == 'interaction' else name
    keep = [e for e in ids if all(e in runs[k] and metric in runs[k][e]['metrics'] for k in need)]
    v = lambda k: np.array([runs[k][e]['metrics'][metric] for e in keep])
    if not keep:
        return keep, np.array([])
    if name == 'interaction':
        return keep, benefit(metric, (v('D') - v('B')) - (v('C') - v('A')))
    a, b = name
    return keep, benefit(metric, v(a) - v(b))


def bootstrap(x: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """95 % interval of the mean and P(mean > 0) over paired episode resamples."""
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float((means > 0).mean())


def verdict(lo: float, hi: float, expected: int) -> str:
    if lo > 0:
        return 'supported' if expected > 0 else 'contradicted'
    if hi < 0:
        return 'supported' if expected < 0 else 'contradicted'
    return 'not decided'


def scope_ids(runs: dict, scope) -> list[str]:
    tasks = scope if isinstance(scope, tuple) else (scope,)
    ids = set.intersection(*(set(r) for r in runs.values()))
    return sorted(e for e in ids if runs['A'][e]['task'] in tasks)


def fmt(v, unit='') -> str:
    return '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.3f}{unit}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs=4, required=True, help='A=<eval> B=<eval> C=<eval> D=<eval>')
    ap.add_argument('--experts', nargs='*', default=[], help='late=<eval> early=<eval> (references, same episodes)')
    ap.add_argument('--output', required=True, help='markdown report (+ .json next to it)')
    args = ap.parse_args()
    runs = {k: load_run(v) for k, v in (s.split('=', 1) for s in args.runs)}
    experts = {k: load_run(v) for k, v in (s.split('=', 1) for s in args.experts)}
    allruns = {**runs, **{f'expert {k}': v for k, v in experts.items()}}
    scopes = [*TASKS, INTENT_TASKS, TASKS]
    label = lambda sc: 'intent tasks (T1 T2 T3 T5)' if sc == INTENT_TASKS else 'all' if sc == TASKS else sc
    md, out = ['# 2×2 (preliminary): detailed analysis', ''], dict(means={}, contrasts={}, mechanism={}, failures={})
    n_all = len(scope_ids(runs, TASKS))
    md += [f"Runs: {', '.join(f'{k} = {v}' for k, v in (s.split('=', 1) for s in args.runs + args.experts))}", '',
           f'{n_all} episodes shared by A-D. Benefits: > 0 = better (lower-is-better metrics sign-flipped). '
           f'95 % intervals: paired bootstrap over episodes ({N_BOOT} resamples).', '']

    # ---- hypotheses first
    def contrast(metric, cname, scope):
        name = CONTRASTS[cname]
        keep, x = contrast_values(runs, metric, name, scope_ids(runs, scope))
        if not len(x):
            return None
        lo, hi, p = bootstrap(x)
        return dict(n=len(x), mean=float(x.mean()), lo=lo, hi=hi, p_gt0=p, better=int((x > 0).sum()),
                    worse=int((x < 0).sum()), tie=int((x == 0).sum()))
    md += ['## Hypotheses', '']
    rows = []
    for title, metric, cname, scope in (
            ('Goal / H3: interaction on CT (intent shortens the task more when acting early)', 'CT', 'interaction (D vs B) - (C vs A)', INTENT_TASKS),
            ('Goal / H3: interaction on WCR (intent removes more wrong commits when acting early)', 'WCR', 'interaction (D vs B) - (C vs A)', INTENT_TASKS),
            ('Goal: interaction on WCD', 'WCD', 'interaction (D vs B) - (C vs A)', INTENT_TASKS),
            ('Goal: interaction on CSR', 'CSR', 'interaction (D vs B) - (C vs A)', INTENT_TASKS),
            ('H1: D commits wrongly less than B (WCR)', 'WCR', 'D vs B: intent when early', INTENT_TASKS),
            ('H1: D is committed to wrong targets for less time than B (WCD)', 'WCD', 'D vs B: intent when early', INTENT_TASKS),
            ('cost of acting early without intent: B vs A (WCR)', 'WCR', 'B vs A: early without intent', INTENT_TASKS),
            ('cost of acting early without intent: B vs A (CSR)', 'CSR', 'B vs A: early without intent', INTENT_TASKS),
            ('gain of acting early with intent: D vs C (CT)', 'CT', 'D vs C: early with intent', INTENT_TASKS),
            ('gain of acting early with intent: D vs C (AM)', 'AM', 'D vs C: early with intent', INTENT_TASKS)):
        c = contrast(metric, cname, scope)
        if c is None:
            rows.append(f'| {title} | — | — | — | not observable |')
            continue
        v = verdict(c['lo'], c['hi'], EXPECTED.get(cname, +1))
        rows.append(f"| {title} | {c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] | {c['p_gt0']:.2f} | "
                    f"{c['better']} / {c['worse']} / {c['tie']} (n {c['n']}) | **{v}** |")
    md += ['| check | benefit [95 % CI] | P(benefit > 0) | episodes better / worse / tie | verdict |',
           '|---|---|---|---|---|'] + rows + ['']
    # H2: AM against the experts
    am = {}
    for k, r in allruns.items():
        vals = [x['metrics']['AM'] for e, x in r.items() if x['task'] in INTENT_TASKS and 'AM' in x['metrics']]
        am[k] = (float(np.mean(vals)) if vals else None, len(vals))
    md += ['H2 (AM, intent tasks; positive = the robot committed to the right target before the human finished):', '',
           '| run | AM mean [s] | episodes with a correct commit and a completed cue |', '|---|---|---|']
    md += [f'| {k} | {fmt(m)} | {n} |' for k, (m, n) in am.items()] + ['']
    out['am'] = am

    # ---- means
    md += ['## Means per task', '']
    for sc in scopes:
        md += [f'### {label(sc)}', '', '| metric | ' + ' | '.join(allruns) + ' |', '|---|' + '---|' * len(allruns)]
        tasks = sc if isinstance(sc, tuple) else (sc,)
        for m in METRICS:
            cells, any_ = [], False
            for k, r in allruns.items():
                vals = [x['metrics'][m] for x in r.values() if x['task'] in tasks and m in x['metrics']]
                any_ |= bool(vals)
                cells.append(f"{np.mean(vals):.3f} (n {len(vals)})" if vals else '—')
                out['means'].setdefault(label(sc), {}).setdefault(m, {})[k] = float(np.mean(vals)) if vals else None
            if any_:
                md.append(f"| {m}{' ↓' if m in LOWER_IS_BETTER else ''} | " + ' | '.join(cells) + ' |')
        md.append('')

    # ---- contrasts
    md += ['## Contrasts (benefit > 0 = better)', '']
    for sc in scopes:
        md += [f'### {label(sc)}', '', '| metric | contrast | benefit [95 % CI] | P(> 0) | better / worse / tie | n |',
               '|---|---|---|---|---|---|']
        for m in METRICS:
            for cname in CONTRASTS:
                c = contrast(m, cname, sc)
                if c is None:
                    continue
                out['contrasts'].setdefault(label(sc), {}).setdefault(m, {})[cname] = c
                md.append(f"| {m} | {cname} | {c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] | {c['p_gt0']:.2f} | "
                          f"{c['better']} / {c['worse']} / {c['tie']} | {c['n']} |")
        md.append('')

    # ---- control
    md += ['## Control: T4 / T4neg (the experts act identically: these differences are training noise)', '',
           '| metric | B vs A | D vs C | n |', '|---|---|---|---|']
    for m in METRICS:
        ba, dc = contrast(m, 'B vs A: early without intent', CONTROL_TASKS), contrast(m, 'D vs C: early with intent', CONTROL_TASKS)
        if ba or dc:
            f_ = lambda c: '—' if c is None else f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"
            md.append(f"| {m} | {f_(ba)} | {f_(dc)} | {(ba or dc)['n']} |")
    md.append('')

    # ---- mechanism
    md += ['## Mechanism on the intent tasks: the first commitment (decomposes AM and WCR)', '',
           '| run | episodes with a commit | first commit correct | first commit before the cue ended | lead of the first '
           'commit over the cue end [s] (median) | wrong commits before / after the cue ended |', '|---|---|---|---|---|---|']
    for k, r in allruns.items():
        mech = [x['mech'] for x in r.values() if x['task'] in INTENT_TASKS]
        withc = [x for x in mech if x['first_t'] is not None]
        lead = [x['lead'] for x in withc if x['lead'] is not None]
        before = [x for x in withc if x['lead'] is not None and x['lead'] > 0]
        wb = sum(x['wrong_before_cue'] or 0 for x in mech)
        wa = sum(x['wrong_after_cue'] or 0 for x in mech)
        corr = [x for x in withc if x['first_correct']]
        out['mechanism'][k] = dict(n=len(mech), with_commit=len(withc), first_correct=len(corr), before_cue=len(before),
                                   lead_median=float(np.median(lead)) if lead else None, wrong_before=wb, wrong_after=wa)
        md.append(f"| {k} | {len(withc)}/{len(mech)} | {len(corr)}/{len(withc)} | {len(before)}/{len(withc)} | "
                  f"{fmt(float(np.median(lead)) if lead else None)} | {wb} / {wa} |")
    md += ['', '## Failures', '', '| run | ' + ' | '.join(TASKS) + ' |', '|---|' + '---|' * len(TASKS)]
    for k, r in allruns.items():
        cells = []
        for t in TASKS:
            c = Counter(x['failure'] for x in r.values() if x['task'] == t and not x['success'])
            n = sum(1 for x in r.values() if x['task'] == t)
            cells.append(f"{n - sum(c.values())}/{n} ok" + (': ' + ', '.join(f'{f} {v}' for f, v in c.most_common()) if c else ''))
        out['failures'][k] = cells
        md.append(f'| {k} | ' + ' | '.join(cells) + ' |')
    Path(args.output).write_text('\n'.join(md) + '\n')
    Path(args.output).with_suffix('.json').write_text(json.dumps(out, indent=1, default=str))
    print('\n'.join(md[:40]))


if __name__ == '__main__':
    main()
