"""Late vs early device expert per task, from two scripts/check_expert.py runs on the same list
(docs/requirements/INTENT_ACT_v3.2.md §4.2g; the user's report, 2026-10-12):
  commit gap   first object commit, late - early, on the episodes both experts committed in: median, and the episodes
               with a gap >= 0.1 s
  success, episodes with a wrong commit, and the HRIBench-style metrics of every successful episode (mean): CT, Rsp,
  AM, WCR, WCD (intent_policy/benchmark/metrics.py; no new metric)

  ./run.sh -m scripts.compare_device_rules --late outputs/device/<date>/check_late.json \
      --early outputs/device/<date>/check_early.json --output outputs/device/<date>/compare
"""
import argparse
import json

import numpy as np

from intent_policy.utils import resolve

METRICS = ('CT', 'Rsp', 'AM', 'WCR', 'WCD')


def rows_by_task(path) -> dict:
    return {r['task']: {x['index']: x for x in r['rows']} for r in json.loads(resolve(path).read_text())}


def first_commit(row) -> float | None:
    return next((c['t'] for c in row.get('expert_commits') or []), None)


def mean(xs) -> str:
    xs = [x for x in xs if isinstance(x, (int, float)) and np.isfinite(x)]
    return f'{np.mean(xs):.2f}' if xs else '—'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--late', required=True)
    ap.add_argument('--early', required=True)
    ap.add_argument('--output', help='write <output>.md and <output>.json')
    args = ap.parse_args()
    L, E = rows_by_task(args.late), rows_by_task(args.early)
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
