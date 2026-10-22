"""Belief readiness vs correct / wrong movement over time, held-out eval episodes (research_collab.md [C16], [X12]).

T1-T3 object decision, τ after the human's cue onset. Device: the share of episodes whose belief (logged
`p_target_values` of a v3.2 run) ranks the human's object above the alternative, and the share >= 0.7 on the wrong one
(calibration of the early rule). Runs: the share whose end-effector leans towards the human's object / the
alternative by more than the controller tolerance 0.005 m (progress difference since the onset). Not the same
quantity as the belief accuracy (movement includes the policy, the controller and the arm's dynamics). 95 % bootstrap
intervals over episodes; paired differences between two runs on the same episodes.

  ./run.sh -m scripts.belief_motion_timing --belief-run <eval> --run B=<eval> --run D_f1=<eval> --pair D_f1:B --output <md>
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from intent_policy.benchmark.counterfactual import alternative_partners
from intent_policy.intent.labels import load_schema
from intent_policy.utils import resolve

TAUS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5)
TOL_M = 0.005                                   # benchmark/counterfactual.correct_moves(tol)
N_BOOT = 10_000


def records(run: str) -> dict:
    out = {}
    for f in sorted((resolve(run) / 'episodes').glob('*/*/*.json.gz')):
        with gzip.open(f, 'rt') as fh:
            r = json.load(fh)
        if r.get('list_task') in ('T1', 'T2', 'T3'):
            out[r['episode_id']] = r
    return out


def step_at(tr, t):
    ts = np.array([s['state']['t'] for s in tr])
    return tr[int(np.clip(np.searchsorted(ts, t), 0, len(tr) - 1))]


def onset(r):
    return next((e['timestamp'] for e in r['events'] if e['event_type'] == 'human_cue_onset'), None)


def belief_rows(recs: dict, T: list) -> dict:
    out = {}
    for e, r in recs.items():
        sp = r['spec']
        alt = alternative_partners(sp)[sp['target']]
        ih, ia = T.index(sp['layout'][sp['target']]), T.index(sp['layout'][alt])
        right, wrong_high = [], []
        for tau in TAUS:
            v = step_at(r['trace'], onset(r) + tau)['extras']['intent']['p_target_values']
            tot = v[ih] + v[ia]
            share = v[ih] / tot if tot > 0 else 0.5
            right.append(float(v[ih] > v[ia]))
            wrong_high.append(float(1 - share >= 0.7))
        out[e] = (np.array(right), np.array(wrong_high))
    return out


def motion_rows(recs: dict) -> dict:
    out = {}
    for e, r in recs.items():
        sp = r['spec']
        alt = alternative_partners(sp)[sp['target']]
        tr, on = r['trace'], onset(r)

        def dist(t):
            st = step_at(tr, t)['state']
            ee = np.asarray(st['robot']['ee_pos'][:2])
            return {k: np.linalg.norm(ee - np.asarray(st['objects'][k]['position'][:2])) for k in (sp['target'], alt)}
        d0 = dist(on)
        right, wrong = [], []
        for tau in TAUS:
            d = dist(on + tau)
            diff = (d0[alt] - d[alt]) - (d0[sp['target']] - d[sp['target']])
            right.append(float(diff < -TOL_M))
            wrong.append(float(diff > TOL_M))
        out[e] = (np.array(right), np.array(wrong))
    return out


def ci(x: np.ndarray, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cell(x: np.ndarray) -> str:
    lo, hi = ci(x)
    return f'{x.mean():.2f} [{lo:.2f}, {hi:.2f}]'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--belief-run', required=True, help='a v3.2 eval run that logged p_target_values')
    p.add_argument('--run', action='append', required=True, metavar='NAME=EVAL')
    p.add_argument('--pair', action='append', default=[], metavar='A:B', help='paired difference A - B')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    T = load_schema()['targets']
    bel = belief_rows(records(args.belief_run), T)
    runs = {n: motion_rows(records(path)) for n, path in (x.split('=', 1) for x in args.run)}
    head = '| | n | ' + ' | '.join(f'τ {t:.2f} s' for t in TAUS) + ' |'
    sep = '|---|---|' + '---|' * len(TAUS)
    L = ['# Belief readiness vs movement (held-out eval, T1-T3, object decision)', '',
         'Share of episodes with 95 % bootstrap intervals over episodes. Belief and movement are different quantities '
         '(movement includes the policy, the controller and the arm dynamics).', '', head, sep]
    ids = sorted(bel)
    L.append('| device belief ranks the human\'s object first | ' + f'{len(ids)} | ' +
             ' | '.join(cell(np.array([bel[e][0][i] for e in ids])) for i in range(len(TAUS))) + ' |')
    L.append('| device belief >= 0.7 on the alternative | ' + f'{len(ids)} | ' +
             ' | '.join(cell(np.array([bel[e][1][i] for e in ids])) for i in range(len(TAUS))) + ' |')
    for n, rows in runs.items():
        ids = sorted(rows)
        for k, lab in ((0, 'leans to the human\'s object'), (1, 'leans to the alternative')):
            L.append(f'| {n}: {lab} | {len(ids)} | ' +
                     ' | '.join(cell(np.array([rows[e][k][i] for e in ids])) for i in range(len(TAUS))) + ' |')
    if args.pair:
        L += ['', '## Paired differences (same episodes)', '', head, sep]
        for pr in args.pair:
            a, b = pr.split(':')
            ids = sorted(set(runs[a]) & set(runs[b]))
            for k, lab in ((0, 'leans right'), (1, 'leans wrong')):
                L.append(f'| {a} - {b}: {lab} | {len(ids)} | ' + ' | '.join(
                    '{:+.2f} [{:+.2f}, {:+.2f}]'.format(d.mean(), *ci(d))
                    for d in (np.array([runs[a][e][k][i] - runs[b][e][k][i] for e in ids]) for i in range(len(TAUS)))) + ' |')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
