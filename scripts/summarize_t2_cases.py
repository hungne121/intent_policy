"""Per-case summary of a T2 deep-task evaluation (docs/PIF.md §4, §7): success, completion time, the human's waiting
time (the open hand out -> the robot at the hand-over point, the last request), failures, per case C1-C4 and overall.
  ./run.sh -m scripts.summarize_t2_cases outputs/eval/t2deep/<run A> outputs/eval/t2deep/<run D> --names A D-tok
"""
import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load(run: Path) -> list[dict]:
    return [json.load(gzip.open(f)) for f in sorted(run.glob('episodes/*/*/*.json.gz'))]


def wait_s(rec: dict) -> float | None:
    """The last request: hand_ready -> robot_at_handover (None if the robot never got there)."""
    steps = [(e['timestamp'], e['payload'].get('step')) for e in rec['events'] if e['event_type'] == 'protocol_step_complete']
    ready = [t for t, s in steps if s == 'hand_ready']
    at = [t for t, s in steps if s == 'robot_at_handover' and ready and t >= ready[-1]]
    return round(at[0] - ready[-1], 3) if at else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--names', nargs='+')
    args = ap.parse_args()
    names = args.names or [Path(r).name for r in args.runs]
    fmt = lambda v: f'{np.mean(v):.2f} ± {np.std(v):.2f} (n={len(v)})' if v else '—'
    lines = ['| model | case | success | completion time [s] (successes) | human wait [s] (successes) | failures |',
             '|---|---|---|---|---|---|']
    for name, run in zip(names, args.runs):
        recs = load(Path(run))
        by = defaultdict(list)
        for r in recs:
            by[r['human_trajectory_variant'].get('case', 'C1')].append(r)
            by['all'].append(r)
        for case in sorted(k for k in by if k != 'all') + ['all']:
            rs = by[case]
            ok = [r for r in rs if r['success']]
            fails = Counter(r['failure'] for r in rs if not r['success'])
            lines.append(f"| {name} | {case} | {len(ok)}/{len(rs)} ({len(ok) / len(rs):.0%}) | "
                         f"{fmt([r['duration_s'] for r in ok])} | {fmt([w for r in ok if (w := wait_s(r)) is not None])} | "
                         f"{', '.join(f'{k} {v}' for k, v in fails.most_common()) or '—'} |")
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
