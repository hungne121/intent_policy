"""Episode lists of the 2026-10-18 trial (A late vs D early on T2, T3, T5 of T2 / T3; configs/experiments/
intent_act_t235_{late,early}.yaml). Same rules as scripts/make_prelim_lists.py: cells spread evenly over each task's
design cells, leaving out every seed the single device model was trained or validated on (device_folds.json
`seen_seeds`), so every episode is one the device never saw.
  demo   from configs/episode_lists/v5/demo_list.jsonl: --n-demo cells per task (DART per cell as listed). The demo list
         holds only 34 unseen T5 cells (every non-DART T5 cell trained the device), so T5 is topped up with copies of
         those cells under new seeds (seed + 3 ... seed + 9: the v5 seed selection used seed_base + 0..2, seed_base
         steps by 10, so these seeds are in no list); the episode's random draws (timings, jitter) change, the design
         cell does not. Collection drops failed expert episodes.
  eval   from configs/episode_lists/v5/eval_list.jsonl (held-out seeds, no DART): --n-eval cells per task.

  ./run.sh -m scripts.make_t235_lists --n-demo 60 --n-eval 20 --out outputs/trial_t235/<date>/lists
"""
import argparse
import json
from pathlib import Path

from intent_policy.utils import resolve
from scripts.make_prelim_lists import spread

TASKS = ('T2', 'T3', 'T5')
BASES = ('T2', 'T3')                    # T5 on T2 / T3 only


def ok(e, seen) -> bool:
    return int(e['seed']) not in seen and e['base'] in BASES


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--folds-file', default='outputs/device/current/device_folds.json')
    ap.add_argument('--demo-list', default='configs/episode_lists/v5/demo_list.jsonl')
    ap.add_argument('--eval-list', default='configs/episode_lists/v5/eval_list.jsonl')
    ap.add_argument('--n-demo', type=int, default=60)
    ap.add_argument('--n-eval', type=int, default=20)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    seen = set(json.loads(resolve(args.folds_file).read_text()).get('seen_seeds') or [])
    demo = [json.loads(l) for l in resolve(args.demo_list).read_text().splitlines()]
    ev = [json.loads(l) for l in resolve(args.eval_list).read_text().splitlines()]
    used = {int(e['seed']) for e in demo + ev} | seen
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in TASKS:
        cand = [e for e in demo if e['task'] == t and ok(e, seen)]
        if t == 'T5' and len(cand) < args.n_demo:
            extra, k = [], 3
            while len(cand) + len(extra) < args.n_demo and k <= 9:
                for e in cand:
                    s = int(e['seed']) + k
                    if s not in used and len(cand) + len(extra) < args.n_demo:
                        extra.append(dict(e, seed=s, cell_id=f"{e['cell_id']}+s{k}", index=10_000 + len(extra)))
                        used.add(s)
                k += 1
            cand = cand + extra
        rows += spread(cand, args.n_demo)
    (out / 'demo.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in rows))
    erows = [e for t in TASKS for e in spread([e for e in ev if e['task'] == t and ok(e, seen)], args.n_eval)]
    (out / 'eval.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in erows))
    for name, r in (('demo', rows), ('eval', erows)):
        by = {t: sum(e['task'] == t for e in r) for t in TASKS}
        base = {b: sum(e['task'] == 'T5' and e['base'] == b for e in r) for b in ('T2', 'T3')}
        print(f"{name}: {len(r)} {by}, T5 bases {base}, DART {sum(bool(e.get('dart')) for e in r)}, "
              f"new seeds {sum('+s' in e['cell_id'] for e in r)}")


if __name__ == '__main__':
    main()
