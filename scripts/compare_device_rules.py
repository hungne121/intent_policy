"""Late vs early device expert per task, from two scripts/check_expert.py runs on the same list
(docs/requirements/INTENT_ACT_v3.2.md §4.2g; the user's report, 2026-10-12):
  commit gap   first object commit, late - early, on the episodes both experts committed in: median, and the episodes
               with a gap >= 0.1 s
  success, episodes with a wrong commit, and the HRIBench-style metrics of every successful episode (mean): CT, Rsp,
  AM, WCR, WCD (intent_policy/benchmark/metrics.py; no new metric)

  ./run.sh -m scripts.compare_device_rules --late outputs/device/<date>/check_late.json \
      --early outputs/device/<date>/check_early.json --output outputs/device/<date>/compare
Or from two collected datasets of the same list (meta/hri_episodes.jsonl: kept episodes with their metrics and expert
commits; meta/hri_rejected.jsonl: the expert's failures), as the preliminary 2x2 does (scripts/prelim_2x2.sh):
  ./run.sh -m scripts.compare_device_rules --late-dataset <late root> --early-dataset <early root> --output <out>
"""
import argparse
import json

import numpy as np

from intent_policy.utils import resolve

METRICS = ('CT', 'Rsp', 'AM', 'WCR', 'WCD')


def rows_by_task(path) -> dict:
    return {r['task']: {x['index']: x for x in r['rows']} for r in json.loads(resolve(path).read_text())}


def wrong_commits(m: dict) -> int:
    """Expert commits to a target the human did not want then (as scripts/check_expert.expert_wrong_commits): objects
    vs the selection (the old target before a change, the new one after), zones vs the episode's zone."""
    spec, var = m['spec'], m['variation']
    change = (m.get('intention_changes') or [None])[0]
    objects, zone = set(spec.get('layout') or {}), var.get('target_zone')
    n = 0
    for c in m.get('expert_commits') or []:
        if c['target'] in objects:
            want = change['previous'] if change is not None and c['t'] < change['t'] else var['target_object']
            n += c['target'] != want
        elif zone is not None:
            n += c['target'] != zone
    return n


def rows_from_dataset(root) -> dict:
    """{list task: {list index: row}} of a collected dataset: kept episodes and the expert's rejected failures."""
    root = resolve(root)
    out: dict = {}
    for m in map(json.loads, (root / 'meta/hri_episodes.jsonl').read_text().splitlines()):
        out.setdefault(m['list_task'], {})[m['list_index']] = dict(
            index=m['list_index'], task=m['list_task'], success=True, wrong_commits=wrong_commits(m),
            metrics=m.get('metrics') or {}, expert_commits=m.get('expert_commits') or [])
    rej = root / 'meta/hri_rejected.jsonl'
    for r in map(json.loads, rej.read_text().splitlines() if rej.exists() else []):
        e = r['entry']
        out.setdefault(e['task'], {})[e['index']] = dict(index=e['index'], task=e['task'], success=False, wrong_commits=0,
                                                         metrics={}, expert_commits=[], failure=r['failure'])
    return out


def first_commit(row) -> float | None:
    return next((c['t'] for c in row.get('expert_commits') or []), None)


def mean(xs) -> str:
    xs = [x for x in xs if isinstance(x, (int, float)) and np.isfinite(x)]
    return f'{np.mean(xs):.2f}' if xs else '—'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--late', help='check_expert.py json of the late rule')
    ap.add_argument('--early', help='check_expert.py json of the early rule')
    ap.add_argument('--late-dataset', help='instead: the late dataset collected on the list')
    ap.add_argument('--early-dataset', help='instead: the early dataset collected on the same list')
    ap.add_argument('--output', help='write <output>.md and <output>.json')
    args = ap.parse_args()
    if args.late_dataset and args.early_dataset:
        L, E = rows_from_dataset(args.late_dataset), rows_from_dataset(args.early_dataset)
    elif args.late and args.early:
        L, E = rows_by_task(args.late), rows_by_task(args.early)
    else:
        raise SystemExit('give --late / --early (expert checks) or --late-dataset / --early-dataset')
    md = ['| task | success late / early | wrong-commit ep. late / early | commit gap late − early: median [s] '
          '| gap ≥ 0.1 s | ' + ' | '.join(f'{m} late / early' for m in METRICS) + ' |',
          '|---|---|---|---|---|' + '---|' * len(METRICS)]
    out = {}
    for task in [t for t in L if t in E]:
        late, early = L[task], E[task]
        both = sorted(set(late) & set(early))
        gaps = [first_commit(late[i]) - first_commit(early[i]) for i in both
                if first_commit(late[i]) is not None and first_commit(early[i]) is not None]
        ok = lambda rs: [r for r in rs.values() if r['success']]
        met = {m: (mean([r['metrics'].get(m) for r in ok(late)]), mean([r['metrics'].get(m) for r in ok(early)])) for m in METRICS}
        out[task] = dict(episodes=len(both), gaps=[round(g, 3) for g in gaps], metrics=met,
                         success=(len(ok(late)), len(ok(early))),
                         wrong=(sum(r['wrong_commits'] > 0 for r in late.values()), sum(r['wrong_commits'] > 0 for r in early.values())))
        md.append(f"| {task} | {len(ok(late))}/{len(late)} / {len(ok(early))}/{len(early)} | {out[task]['wrong'][0]} / {out[task]['wrong'][1]} "
                  f"| {np.median(gaps) if gaps else float('nan'):.2f} | {sum(g >= 0.1 for g in gaps)}/{len(gaps)} | "
                  + ' | '.join(f'{a} / {b}' for a, b in met.values()) + ' |')
    print('\n'.join(md))
    if args.output:
        o = resolve(args.output)
        o.parent.mkdir(parents=True, exist_ok=True)
        (o.parent / f'{o.name}.md').write_text('# Late vs early device expert\n\n' + '\n'.join(md) + '\n')
        (o.parent / f'{o.name}.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
