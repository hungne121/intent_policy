"""How far apart are the early and the late commit on the measuring device? (docs/requirements/INTENT_ACT_v3.2.md §4.2g)

Runs the late device expert (it waits motionless until its first commit, so the human's first gesture does not
depend on the rule) on list entries and logs, per control frame, the best candidate object's belief and whether the
gesture is complete (DeviceBelief.complete). Per episode: the early decision (belief >= early_star, 0.7), the late
decision (gesture complete and belief >= late_star, 0.95: the late rule, §0 / §4.2j) and the belief's own 0.95
crossing, whether each chosen candidate is the human's target, and gap = t(late) - t(early); also the device's first
output (pre-roll check: at t = 0, class none). Per task: median gap and the episodes with gap >= 0.1 s.

  ./run.sh -m scripts.device_gap --list configs/episode_lists/v5/expert_check_fold0.jsonl --per-task 6 \
      --output outputs/device/<date>/gap_fold0
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from intent_policy.benchmark.runner import ExpertAgent, run_episode
from intent_policy.intent.labels import load_schema
from intent_policy.scenarios.config import load_scenario_list
from intent_policy.scenarios.scenario_registry import make_scenario, scenario_overrides
from intent_policy.utils import load_yaml, resolve


class Stop(Exception):
    pass


def crossing(log, level):
    return next(((t, k) for t, k, v in log if k is not None and v >= level), (None, None))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', default='configs/episode_lists/v5/expert_check_fold0.jsonl')
    ap.add_argument('--config', default='configs/experiments/intent_act_device_late.yaml')
    ap.add_argument('--tasks', nargs='+', default=['T1', 'T2', 'T3'])
    ap.add_argument('--per-task', type=int, default=6)
    ap.add_argument('--tmax', type=float, default=5.0, help='seconds of each episode (the first gesture)')
    ap.add_argument('--output', help='write <output>.md and <output>.json')
    args = ap.parse_args()
    dev = load_schema()['device']
    early, late = float(dev['early_star']), float(dev['late_star'])
    exp = load_yaml(args.config)
    entries = [e for e in load_scenario_list(args.list) if e['task'] in args.tasks]
    rows, cache, seen = [], {}, defaultdict(int)
    for e in entries:
        if seen[e['task']] >= args.per_task:
            continue
        seen[e['task']] += 1
        sid = e['scenario_id']
        sc = cache.setdefault(sid, make_scenario(sid, scenario_overrides(exp, sid, {'expert': {'trigger': 'device'}})))
        agent, log, first, done = ExpertAgent(), [], {}, []

        def on_frame(obs, decision, target, s):
            b = agent.expert.belief
            if s.time > args.tmax:
                raise Stop
            if b.probs is not None and 'out' not in first:
                first['out'] = (round(s.time, 2), b.top)
            objs = b.objects()
            best = max(objs, key=objs.get) if objs else None
            log.append((round(s.time, 2), best, objs.get(best, 0.0)))
            done.append(b.complete('object'))
        try:
            run_episode(sc, agent, int(e['seed']), None, keep_trace=False, on_frame=on_frame, spec=e['spec'])
        except Stop:
            pass
        truth = e['spec']['target']
        (t7, k7), (t95, k95) = crossing(log, early), crossing(log, late)
        tl, kl = next(((t, k) for (t, k, v), c in zip(log, done) if c and k is not None and v >= late), (None, None))
        rows.append(dict(task=e['task'], index=e['index'], seed=e['seed'], onset_s=sc.human.params.get('cue_onset_s'),
                         first_output=first.get('out'), t_early=t7, early_target=k7, early_correct=k7 == truth,
                         t_belief95=t95, belief95_correct=k95 == truth, t_late=tl, late_target=kl, late_correct=kl == truth,
                         gap_s=None if t7 is None or tl is None else round(tl - t7, 3)))
        r = rows[-1]
        print(f"{r['task']} {r['index']:4d} first output {r['first_output']} | >= {early}: {t7} {k7} "
              f"{'' if r['early_correct'] else 'X'} | belief >= {late}: {t95} | complete and >= {late}: {tl} {kl} "
              f"{'' if r['late_correct'] else 'X'} | gap {r['gap_s']}", flush=True)
    for sc in cache.values():
        sc.close()
    md = [f'# Early (belief >= {early}) vs late (gesture complete and belief >= {late}), {args.list}', '',
          '| task | episodes | gap median [s] | gap >= 0.1 s | wrong at early | wrong at late | first output (t, class) |',
          '|---|---|---|---|---|---|---|']
    for task in args.tasks:
        rs = [r for r in rows if r['task'] == task]
        g = [r['gap_s'] for r in rs if r['gap_s'] is not None]
        md.append(f"| {task} | {len(rs)} | {np.median(g) if g else float('nan'):.2f} | {sum(x >= 0.1 for x in g)}/{len(g)} "
                  f"| {sum(not r['early_correct'] for r in rs if r['t_early'] is not None)} "
                  f"| {sum(not r['late_correct'] for r in rs if r['t_late'] is not None)} "
                  f"| {sorted({str(tuple(r['first_output'] or ())) for r in rs})} |")
    print('\n'.join(md))
    if args.output:
        out = resolve(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        (out.parent / f'{out.name}.md').write_text('\n'.join(md) + '\n')
        (out.parent / f'{out.name}.json').write_text(json.dumps(rows, indent=1, default=str))


if __name__ == '__main__':
    main()
