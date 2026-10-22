"""Intent swap test: is the policy's decision driven by its intent? (INTENT_ACT_v3.2.md §5.3)

Counterfactual intervention on the condition with the scene unchanged (LangGap, 2026; LIBERO-PRO, 2025; the
counterfactual audit of Zhang et al., 2026): in closed loop on T1-T3, scripts/evaluate.py --intent-swap gives the
policy p_target / c_target of the alternative the human did not choose (the target's identical twin, the other cup,
P1 <-> P2, H1 <-> H2), while the cameras still show the human's true cue. Per decision of an episode (the object; in
T1 also the place zone; the T2 hand is not a commit event, so T2 scores the object only) the robot's first
robot_target_commit of that kind is classified
  human   the human's target            alt     the alternative the intent pointed to under the swap
  other   any other target              none    no commit of that kind
A policy whose decisions follow its intent moves from `human` to `alt` under the swap; one that ignores it does not
move (sensitivity alone is not enough: the commits must go where the intent says, Zhang et al. 2026). B (no intent
input) is the negative control: its `alt` share is the base rate of confusing the twins. Paired over the decisions:
swap vs true intent, 95 % bootstrap intervals (scripts/analyze_2x2_detail.py).

The irreversible commit comes ~3 s after the early decision, when the cue is already evident and even the expert
follows it; the early decision itself shows in where the arm starts to go. `heading`: for the object decision, at
times τ after the human's cue onset, the progress of the end effector (horizontal distance at onset minus at τ)
towards the alternative minus that towards the human's target, from the episode traces (> 0: heading for the
alternative). Paired swap - true per episode, per τ.
`heading from t_intent` (research_collab.md K6, for evals that log the given belief): τ counted from t_intent, the first
step at which the belief the policy was given ranks one of the two candidates strictly above the other (the input, fixed
before looking at the actions); per episode the side the arm heads for (`alt` / `human`) or `uncommitted` when the
progress difference is below the controller tolerance 0.005 m (benchmark/counterfactual.correct_moves), and the number
of rank changes between the two candidates after t_intent.

  ./run.sh -m scripts.intent_swap_test --run D0/true=<eval> --run D0/swap=<eval> --run B/true=<eval> --output <md>
"""
import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from intent_policy.benchmark.counterfactual import SWAP_TASKS, alternative_partners
from intent_policy.utils import resolve
from scripts.analyze_2x2_detail import bootstrap

KINDS = {'place': ('P1', 'P2'), 'hand': ('H1', 'H2')}
CLASSES = ('human', 'alt', 'other', 'none')


def kind_of(target: str) -> str:
    return next((k for k, v in KINDS.items() if target in v), 'object')


def decisions(rec: dict) -> dict[str, str]:
    """{decision: class} of one episode: the first commit to an object and, in T1, to a place zone."""
    spec = rec['spec']
    truth = {'object': spec['target']}
    if rec['list_task'] == 'T1':
        truth['place'] = spec['place']
    alt = alternative_partners(spec)
    first = {}
    for e in rec['events']:
        if e['event_type'] == 'robot_target_commit':
            first.setdefault(kind_of(e['payload']['target']), e['payload']['target'])
    out = {}
    for kind, true in truth.items():
        got = first.get(kind)
        out[kind] = 'none' if got is None else 'human' if got == true else 'alt' if got == alt.get(true) else 'other'
    return out


def heading(rec: dict, taus) -> dict[float, float] | None:
    """{τ: progress towards the alternative - progress towards the human's object} (m), the object decision."""
    spec, tr = rec['spec'], rec.get('trace') or []
    alt = alternative_partners(spec).get(spec['target'])
    onset = next((e['timestamp'] for e in rec['events'] if e['event_type'] == 'human_cue_onset'), None)
    if alt is None or onset is None or not tr:
        return None
    t = np.array([s['state']['t'] for s in tr])
    def at(time):
        st = tr[int(np.clip(np.searchsorted(t, time), 0, len(tr) - 1))]['state']
        ee = np.asarray(st['robot']['ee_pos'][:2])
        return {k: np.linalg.norm(ee - np.asarray(st['objects'][k]['position'][:2])) for k in (spec['target'], alt)}
    d0 = at(onset)
    out = {}
    for tau in taus:
        d = at(onset + tau)
        out[tau] = float((d0[alt] - d[alt]) - (d0[spec['target']] - d[spec['target']]))
    return out


TOL_M = 0.005                                      # benchmark/counterfactual.correct_moves(tol)


def heading_from_intent(rec: dict, taus) -> dict | None:
    """{'t_intent', 'flips', τ: (progress difference m, side)} from the logged given belief, the object decision."""
    spec, tr = rec['spec'], rec.get('trace') or []
    alt = alternative_partners(spec).get(spec['target'])
    if alt is None or not tr or 'p_target_values' not in ((tr[0].get('extras') or {}).get('intent') or {}):
        return None
    from intent_policy.intent.labels import load_schema
    T = load_schema()['targets']
    lay = spec['layout']
    ih, ia = T.index(lay[spec['target']]), T.index(lay[alt])
    ranks = []
    for s in tr:
        v = ((s.get('extras') or {}).get('intent') or {}).get('p_target_values')
        ranks.append(None if v is None or round(v[ih], 3) == round(v[ia], 3) else ('human' if v[ih] > v[ia] else 'alt'))
    i0 = next((i for i, r in enumerate(ranks) if r is not None), None)
    if i0 is None:
        return None
    seq = [r for r in ranks[i0:] if r is not None]
    t = np.array([s['state']['t'] for s in tr])
    def at(time):
        st = tr[int(np.clip(np.searchsorted(t, time), 0, len(tr) - 1))]['state']
        ee = np.asarray(st['robot']['ee_pos'][:2])
        return {k: np.linalg.norm(ee - np.asarray(st['objects'][k]['position'][:2])) for k in (spec['target'], alt)}
    d0 = at(t[i0])
    out = {'t_intent': float(t[i0]), 'flips': int(sum(a != b for a, b in zip(seq, seq[1:])))}
    for tau in taus:
        d = at(t[i0] + tau)
        diff = float((d0[alt] - d[alt]) - (d0[spec['target']] - d[spec['target']]))
        out[tau] = (diff, 'uncommitted' if abs(diff) < TOL_M else 'alt' if diff > 0 else 'human')
    return out


def load(run: str, taus=()) -> dict:
    out = {}
    for f in sorted((resolve(run) / 'episodes').glob('*/*/*.json.gz')):
        with gzip.open(f, 'rt') as fh:
            rec = json.load(fh)
        if rec.get('list_task') in SWAP_TASKS:             # T5 specs carry their base task: filter on the list task
            out[rec['episode_id']] = dict(task=rec['list_task'], success=bool(rec.get('success')), dec=decisions(rec),
                                          heading=heading(rec, taus) if taus else None,
                                          heading_ti=heading_from_intent(rec, taus) if taus else None)
    return out


def shares(run: dict) -> tuple[Counter, int]:
    c = Counter(cls for ep in run.values() for cls in ep['dec'].values())
    return c, sum(c.values())


def paired(a: dict, b: dict, cls: str) -> np.ndarray:
    """Per decision: [a's class == cls] - [b's class == cls] on the decisions both runs have."""
    return np.array([float(a[e]['dec'][k] == cls) - float(b[e]['dec'][k] == cls)
                     for e in sorted(set(a) & set(b)) for k in a[e]['dec'] if k in b[e]['dec']])


def ci(x: np.ndarray) -> str:
    if len(x) < 2:
        return f'n {len(x)}'
    lo, hi, _ = bootstrap(x)
    return f'{x.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] (n {len(x)})'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', required=True, metavar='MODEL/CONDITION=EVAL',
                   help='conditions: true, masked, swap, swap_nohuman (any name; swap* are compared with true)')
    p.add_argument('--taus', default='0.5,1.0,1.5,2.0,2.5,3.0', help='times after the cue onset (s) for `heading`')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    taus = [float(x) for x in args.taus.split(',') if x]
    runs = defaultdict(dict)
    for spec in args.run:
        label, path = spec.split('=', 1)
        model, cond = label.split('/', 1)
        runs[model][cond] = load(path, taus)

    lines = ['# Intent swap test (T1-T3)', '',
             "First commit per decision (object; T1 also the place): to the human's target, to the alternative the "
             'swapped intent points to, elsewhere, or none. Under the swap, `alt` is the share of decisions the intent '
             'steered.', '',
             "| model | condition | decisions | → human's target | → alternative | other | none | CSR |", '|---|---|---|---|---|---|---|---|']
    summary = {}
    for model, conds in runs.items():
        for cond, run in conds.items():
            c, n = shares(run)
            csr = np.mean([ep['success'] for ep in run.values()]) if run else float('nan')
            lines.append(f'| {model} | {cond} | {n} | ' + ' | '.join(f'{c[k] / max(n, 1):.3f}' for k in CLASSES) +
                         f' | {csr:.3f} |')
            summary[f'{model}/{cond}'] = dict(n=n, **{k: c[k] / max(n, 1) for k in CLASSES}, csr=float(csr))
    lines += ['', '## Swap vs true intent (paired over decisions)', '',
              '| model | condition | Δ → alternative (steered) | Δ → human\'s target |', '|---|---|---|---|']
    for model, conds in runs.items():
        if 'true' not in conds:
            continue
        for cond, run in conds.items():
            if cond == 'true':
                continue
            d_alt, d_hum = paired(run, conds['true'], 'alt'), paired(run, conds['true'], 'human')
            lines.append(f'| {model} | {cond} | {ci(d_alt)} | {ci(d_hum)} |')
            summary[f'{model}/{cond} vs true'] = dict(alt=d_alt.tolist(), human=d_hum.tolist())
    lines += ['', '## Heading: progress towards the alternative minus towards the human\'s object (m), object decision', '',
              '| model | condition | n | ' + ' | '.join(f'τ {t:.1f} s' for t in taus) + ' |', '|---|---|---|' + '---|' * len(taus)]
    for model, conds in runs.items():
        for cond, run in conds.items():
            h = [ep['heading'] for ep in run.values() if ep['heading']]
            lines.append(f'| {model} | {cond} | {len(h)} | ' + ' | '.join(f'{np.mean([x[t] for x in h]):+.3f}' for t in taus) + ' |')
    lines += ['', 'Swap - true, paired per episode (m), 95 % bootstrap:', '']
    for model, conds in runs.items():
        if 'true' not in conds:
            continue
        for cond, run in conds.items():
            if cond == 'true':
                continue
            ids = [e for e in run if e in conds['true'] and run[e]['heading'] and conds['true'][e]['heading']]
            cells = [ci(np.array([run[e]['heading'][t] - conds['true'][e]['heading'][t] for e in ids])) for t in taus]
            lines.append(f'- {model}/{cond}: ' + '; '.join(f'τ {t:.1f}: {c}' for t, c in zip(taus, cells)))
            summary[f'{model}/{cond} heading vs true'] = {
                str(t): [run[e]['heading'][t] - conds['true'][e]['heading'][t] for e in ids] for t in taus}
    lines += ['', '## Heading from t_intent (K6): progress difference (m) and the side the arm heads for', '',
              '| model | condition | n | ' + ' | '.join(f'Δ {t:.1f} s: mean / alt / human / uncommitted' for t in taus) +
              ' | rank changes after t_intent (mean) |', '|---|---|---|' + '---|' * (len(taus) + 1)]
    for model, conds in runs.items():
        for cond, run in conds.items():
            h = [ep['heading_ti'] for ep in run.values() if ep.get('heading_ti')]
            if not h:
                continue
            cells = []
            for t in taus:
                sides = Counter(x[t][1] for x in h)
                cells.append(f"{np.mean([x[t][0] for x in h]):+.3f} / {sides['alt']} / {sides['human']} / {sides['uncommitted']}")
            lines.append(f'| {model} | {cond} | {len(h)} | ' + ' | '.join(cells) + f" | {np.mean([x['flips'] for x in h]):.1f} |")
    lines += ['', 'Swap - true from t_intent, paired per episode (m), 95 % bootstrap:', '']
    for model, conds in runs.items():
        if 'true' not in conds:
            continue
        for cond, run in conds.items():
            if cond == 'true':
                continue
            ids = [e for e in run if e in conds['true'] and run[e].get('heading_ti') and conds['true'][e].get('heading_ti')]
            if not ids:
                continue
            cells = [ci(np.array([run[e]['heading_ti'][t][0] - conds['true'][e]['heading_ti'][t][0] for e in ids])) for t in taus]
            lines.append(f'- {model}/{cond}: ' + '; '.join(f'Δ {t:.1f}: {c}' for t, c in zip(taus, cells)))
            summary[f'{model}/{cond} heading_ti vs true'] = {
                str(t): [run[e]['heading_ti'][t][0] - conds['true'][e]['heading_ti'][t][0] for e in ids] for t in taus}
    lines += ['', '## Per task (share → alternative / → human)', '']
    for model, conds in runs.items():
        for cond, run in conds.items():
            per = defaultdict(Counter)
            for ep in run.values():
                per[ep['task']].update(ep['dec'].values())
            lines.append(f'- {model}/{cond}: ' + '; '.join(
                f"{t} {per[t]['alt'] / max(sum(per[t].values()), 1):.2f} / {per[t]['human'] / max(sum(per[t].values()), 1):.2f}"
                f" (n {sum(per[t].values())})" for t in SWAP_TASKS if t in per))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text('\n'.join(lines) + '\n')
    Path(args.output).with_suffix('.json').write_text(json.dumps(summary, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
