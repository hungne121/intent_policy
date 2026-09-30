"""Check the scripted expert on a scenario list before collecting demonstrations.

Runs the privileged expert without rendering on the list entries (per task T1 / T2 / T3 / T4 / T4neg / T5) for one
or more expert triggers and reports success, failure reasons, completion time, cue -> first robot motion, minimum
human-robot distance, safety events and, for change-of-mind episodes (T5), when the change happened and how long
the robot took to head for the new target.

  ./run.sh -m scripts.check_expert --list configs/scenario_lists/demo_v1.jsonl --per-task 20 \
      --triggers cue_complete evidence --output outputs/expert_check/demo_v1
"""
import argparse
import json
from collections import Counter, defaultdict
import numpy as np
from intent_policy.benchmark.runner import ExpertAgent, run_episode
from intent_policy.scenarios.config import load_scenario_list
from intent_policy.scenarios.scenario_registry import make_scenario
from intent_policy.utils import resolve

HEAD_FOR_NEW_TARGET_M = 0.02    # robot "responds" once its horizontal distance to the new target shrank by this much
TASKS = ('T1', 'T2', 'T3', 'T4', 'T4neg', 'T5')


def first_time(events, event_type, after=-1.0):
    ts = [e['timestamp'] for e in events if e['event_type'] == event_type and e['timestamp'] >= after]
    return ts[0] if ts else None


def change_response(rec) -> dict | None:
    """Time from the change of mind until the TCP clearly heads for the new target."""
    change = next((e for e in rec['events'] if e['event_type'] == 'human_intention_change'), None)
    if change is None:
        return None
    new, t_c = change['payload']['current'], change['timestamp']
    trace = [s['state'] for s in rec['trace'] if s['state']['t'] >= t_c - 1e-9]
    target = lambda st: np.asarray(st['objects'][new]['position'])[:2]
    dist = lambda st: float(np.linalg.norm(np.asarray(st['robot']['ee_pos'])[:2] - target(st)))
    d0 = dist(trace[0]) if trace else None
    for st in trace:
        if d0 - dist(st) >= HEAD_FOR_NEW_TARGET_M:
            return dict(t_change=round(t_c, 3), timing=change['payload'].get('timing'), response_s=round(st['t'] - t_c, 3))
    return dict(t_change=round(t_c, 3), timing=change['payload'].get('timing'), response_s=None)


def run_entry(sc, e) -> dict:
    rec = run_episode(sc, ExpertAgent(), int(e['seed']), keep_trace=True, spec=e['spec'])
    ev = rec['events']
    cue = first_time(ev, 'human_cue_onset')
    move = first_time(ev, 'robot_motion_start', after=cue if cue is not None else 0.0)
    kinds = Counter(x['event_type'] for x in ev)
    return dict(index=e['index'], task=e['task'], seed=e['seed'], success=rec['success'], failure=rec['failure'],
                duration_s=rec['duration_s'], ct=rec['metrics'].get('CT'),
                cue_to_motion_s=None if cue is None or move is None else move - cue,
                min_dist=rec['min_human_robot_distance'], contacts=kinds['human_robot_contact'],
                sdv=kinds['safety_distance_violation'], change=change_response(rec))


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return (round(float(np.mean(xs)), 3), round(float(np.std(xs)), 3), len(xs)) if xs else (None, None, 0)


def summarize(rows) -> dict:
    changes = [r['change'] for r in rows if r['change']]
    return dict(episodes=len(rows), success=sum(r['success'] for r in rows),
                failures=dict(Counter(r['failure'] for r in rows if not r['success'])),
                failed=[dict(index=r['index'], seed=r['seed'], failure=r['failure']) for r in rows if not r['success']],
                ct=mean([r['ct'] for r in rows if r['success']]), cue_to_motion_s=mean([r['cue_to_motion_s'] for r in rows]),
                min_dist_m=round(min(r['min_dist'] for r in rows), 4), episodes_with_contact=sum(r['contacts'] > 0 for r in rows),
                episodes_with_sdv=sum(r['sdv'] > 0 for r in rows), changes=len(changes),
                change_response_s=mean([c['response_s'] for c in changes]),
                change_no_response=sum(c['response_s'] is None for c in changes))


def fmt(m):
    return '—' if m[0] is None else f'{m[0]:.2f} ± {m[1]:.2f}'


def markdown(results) -> str:
    lines = ['| task | expert | success | CT [s] (succ.) | cue→motion [s] | min dist [m] | contact ep. | SDV ep. '
             '| changes | robot heads to new target after [s] |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        s = r['summary']
        lines.append(f"| {r['task']} | {r['trigger']} | {s['success']}/{s['episodes']} | {fmt(s['ct'])} | {fmt(s['cue_to_motion_s'])} "
                     f"| {s['min_dist_m']:.3f} | {s['episodes_with_contact']} | {s['episodes_with_sdv']} | {s['changes']} "
                     f"| {fmt(s['change_response_s'])}" + (f" ({s['change_no_response']} none)" if s['change_no_response'] else '') + ' |')
    failures = [f"- {r['task']} / {r['trigger']}: {r['summary']['failures']} (list indices "
                f"{[f['index'] for f in r['summary']['failed']]})" for r in results if r['summary']['failures']]
    return '\n'.join(lines + ['', '**Failures:**'] + (failures or ['- none'])) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--list', default='configs/scenario_lists/demo_v1.jsonl')
    p.add_argument('--tasks', nargs='*', default=list(TASKS))
    p.add_argument('--per-task', type=int, help='first N entries of each task')
    p.add_argument('--triggers', nargs='*', default=['cue_complete'], help='expert triggers to check')
    p.add_argument('--output', help='write <output>.md and <output>.json')
    args = p.parse_args()
    by_task = defaultdict(list)
    for e in load_scenario_list(args.list):
        if e['task'] in args.tasks and (args.per_task is None or len(by_task[e['task']]) < args.per_task):
            by_task[e['task']].append(e)
    results = []
    for trigger in args.triggers:
        cache = {}
        for task in [t for t in TASKS if t in by_task]:
            rows = []
            for e in by_task[task]:
                sid = e['scenario_id']
                cache.setdefault(sid, make_scenario(sid, {'expert': {'trigger': trigger}}))
                rows.append(run_entry(cache[sid], e))
            s = summarize(rows)
            results.append(dict(task=task, trigger=trigger, summary=s, rows=rows))
            print(f"[{trigger}] {task}: {s['success']}/{s['episodes']} ok, failures={s['failures']}, CT={fmt(s['ct'])}, "
                  f"min dist={s['min_dist_m']:.3f}", flush=True)
        for sc in cache.values():
            sc.close()
    md = markdown(results)
    print(md)
    if args.output:
        out = resolve(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix('.md').write_text(f'# Expert check — {args.list}\n\n' + md)
        out.with_suffix('.json').write_text(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    main()
