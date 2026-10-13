"""Episode lists of the preliminary 2x2 on the laptop (docs/requirements/INTENT_ACT_v3.2.md §4.2i, scripts/prelim_2x2.sh):
per list task, N demonstrations and M evaluation episodes spread evenly over the task's design cells (the list order
follows the design strata, as the val split of scripts/make_episode_list_v5.py), leaving out every cell whose seed the
single device model was trained or validated on (device_folds.json `seen_seeds`).
  <out>/demo.jsonl   from configs/episode_lists/v5/demo_list.jsonl (DART per cell as listed; both rules collect it)
  <out>/eval.jsonl   from configs/episode_lists/v5/eval_list.jsonl (held-out seeds, no DART)

  ./run.sh -m scripts.make_prelim_lists --folds-file outputs/device/current/device_folds.json --n-demo 20 --n-eval 8 \
      --out outputs/prelim/<date>/lists
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from intent_policy.utils import resolve


def spread(entries: list[dict], n: int) -> list[dict]:
    """n entries evenly spaced over the list (all when there are fewer)."""
    if len(entries) <= n:
        return list(entries)
    return [entries[int(i)] for i in np.unique(np.linspace(0, len(entries) - 1, n).round())]


def pick(path, n: int, seen: set[int]) -> tuple[list[dict], dict]:
    by = defaultdict(list)
    for e in map(json.loads, resolve(path).read_text().splitlines()):
        if int(e['seed']) not in seen:
            by[e['task']].append(e)
    out = [e for task in sorted(by) for e in spread(by[task], n)]
    return out, {t: len([e for e in out if e['task'] == t]) for t in sorted(by)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--folds-file', default='outputs/device/current/device_folds.json')
    ap.add_argument('--demo-list', default='configs/episode_lists/v5/demo_list.jsonl')
    ap.add_argument('--eval-list', default='configs/episode_lists/v5/eval_list.jsonl')
    ap.add_argument('--n-demo', type=int, default=20)
    ap.add_argument('--n-eval', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    seen = set(json.loads(resolve(args.folds_file).read_text()).get('seen_seeds') or [])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, path, n in (('demo', args.demo_list, args.n_demo), ('eval', args.eval_list, args.n_eval)):
        rows, counts = pick(path, n, seen)
        (out / f'{name}.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in rows))
        dart = sum(bool(e.get('dart')) for e in rows)
        print(f'{name}: {len(rows)} episodes {counts}, DART {dart}, seen seeds left out: {len(seen)}')


if __name__ == '__main__':
    main()
