"""Check the scripted expert on a scenario list before collecting demonstrations.

Runs the privileged expert without rendering on the list entries (per task T1 / T2 / T3 / T4 / T4neg / T5) for one
or more expert triggers and reports success, failure reasons, completion time, cue -> first robot motion, minimum
human-robot distance, safety events and, for change-of-mind episodes (T5), when the change happened and how long
the robot took to head for the new target.

  ./run.sh -m scripts.check_expert --list configs/scenario_lists/demo_v1.jsonl --per-task 20 \
      --triggers cue_complete evidence --output outputs/expert_check/demo_v1

With `--config` (an experiment yaml) the episodes run as they are collected: the experiment's scenario overrides,
controller and DART per list entry; T5 entries also need the change to have happened (plan v5 seed rule). The device
trigger (docs/requirements/INTENT_ACT_v3.2.md §4.2e-g) is reported with its commits: wrong commits (robot_target_commit
events with correct = false) and the lead of the first object commit over the late trigger's cue-complete step; each
episode also keeps its HRIBench-style metrics (CT, Rsp, AM, WCR, WCD, CRsp).

  ./run.sh -m scripts.check_expert --list outputs/expert_check_fold0_list.jsonl --per-task 6 \
      --config configs/experiments/intent_act_device_late.yaml --triggers device --output outputs/expert_check/device_late_f0
"""
import argparse
import json
from collections import Counter, defaultdict
import numpy as np
from intent_policy.benchmark.runner import ExpertAgent, NoisyExpertAgent, run_episode
from intent_policy.experts.scripted_expert import CUE_COMPLETE_STEP
from intent_policy.scenarios.config import load_scenario_list
from intent_policy.scenarios.scenario_registry import make_scenario, scenario_overrides
from intent_policy.utils import controller_config, load_yaml, resolve

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


def commit_lead(rec, expert) -> float | None:
    """Seconds by which the expert's first object commit precedes the late trigger's cue-complete step."""
    step = CUE_COMPLETE_STEP.get(expert.code)
    done = next((e['timestamp'] for e in rec['events'] if e['event_type'] == 'protocol_step_complete'
                 and e['payload'].get('step') == step), None)
    objects = set(rec['spec'].get('layout') or {})
    first = next((c['t'] for c in expert.commit_log if c['target'] in objects), None)
    return None if done is None or first is None else round(done - first, 3)


def expert_wrong_commits(rec, expert, sc) -> int:
    """Expert commits to a target the human did not want at that moment: objects vs the selection (the old target of a
    change before the change, the new one after), zones vs the episode's zone."""
    spec, var = rec['spec'], rec['variation']
    change = next((e for e in rec['events'] if e['event_type'] == 'human_intention_change'), None)
    objects, zone = set(spec.get('layout') or {}), var.get('hand_zone') or var.get('target_zone')   # T3 new: the hand zone
    wrong = 0
    for c in expert.commit_log:
        if c['target'] in objects:
            want = change['payload']['previous'] if change is not None and c['t'] < change['timestamp'] else var['target_object']
            wrong += c['target'] != want
        elif zone is not None:
            wrong += c['target'] != zone
    return wrong


def run_entry(sc, e, ctrl=None, noise=None) -> dict:
    dart = bool(e.get('dart')) and noise is not None
    agent = NoisyExpertAgent(noise['burst_prob'], noise['burst_ticks']) if dart else ExpertAgent()
    rec = run_episode(sc, agent, int(e['seed']), ctrl, keep_trace=True, spec=e['spec'])
    ev = rec['events']
    cue = first_time(ev, 'human_cue_onset')
    move = first_time(ev, 'robot_motion_start', after=cue if cue is not None else 0.0)
    kinds = Counter(x['event_type'] for x in ev)
    commits = [x['payload'] for x in ev if x['event_type'] == 'robot_target_commit']
    valid = rec['success']
    if e['spec'].get('change'):                 # plan v5: a T5 episode counts only if the change happened
        valid = valid and sc.change_t is not None and 'change_dropped' not in sc.human.extras
    return dict(index=e['index'], task=e['task'], seed=e['seed'], success=rec['success'], failure=rec['failure'],
                valid=bool(valid), dart=dart, duration_s=rec['duration_s'], ct=rec['metrics'].get('CT'),
                cue_to_motion_s=None if cue is None or move is None else move - cue,
                min_dist=rec['min_human_robot_distance'], contacts=kinds['human_robot_contact'],
                sdv=kinds['safety_distance_violation'], change=change_response(rec),
                wrong_commits=expert_wrong_commits(rec, agent.expert, sc),
                detector_wrong=sum(not c.get('correct', True) for c in commits),
                metrics={k: rec['metrics'].get(k) for k in ('CT', 'Rsp', 'AM', 'WCR', 'WCD', 'CRsp')},   # HRIBench-style
                lead_s=commit_lead(rec, agent.expert), expert_commits=list(agent.expert.commit_log))


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return (round(float(np.mean(xs)), 3), round(float(np.std(xs)), 3), len(xs)) if xs else (None, None, 0)


def summarize(rows) -> dict:
    changes = [r['change'] for r in rows if r['change']]
    return dict(episodes=len(rows), success=sum(r['success'] for r in rows), valid=sum(r.get('valid', r['success']) for r in rows),
                episodes_with_wrong_commit=sum(r.get('wrong_commits', 0) > 0 for r in rows),
                episodes_with_detector_wrong=sum(r.get('detector_wrong', 0) > 0 for r in rows),
                lead_s=mean([r.get('lead_s') for r in rows if r['success']]),
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
    lines = ['| task | expert | success | valid | expert wrong-commit ep. | detector wrong-commit ep. | commit lead [s] '
             '| CT [s] (succ.) | cue→motion [s] | min dist [m] | contact ep. | SDV ep. | changes | robot heads to new target after [s] |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        s = r['summary']
        lines.append(f"| {r['task']} | {r['trigger']} | {s['success']}/{s['episodes']} | {s['valid']} | {s['episodes_with_wrong_commit']} "
                     f"| {s['episodes_with_detector_wrong']} "
                     f"| {fmt(s['lead_s'])} | {fmt(s['ct'])} | {fmt(s['cue_to_motion_s'])} "
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
    p.add_argument('--config', help='experiment yaml: scenario overrides, controller and DART as in collection')
    p.add_argument('--output', help='write <output>.md and <output>.json')
    args = p.parse_args()
    by_task = defaultdict(list)
    for e in load_scenario_list(args.list):
        if e['task'] in args.tasks and (args.per_task is None or len(by_task[e['task']]) < args.per_task):
            by_task[e['task']].append(e)
    exp = load_yaml(args.config) if args.config else {}
    ctrl = controller_config(exp) if exp else None
    noise = (exp.get('data') or {}).get('noise') or None
    results = []
    for trigger in args.triggers:
        cache = {}
        extra = {'expert': {'trigger': trigger}}
        for task in [t for t in TASKS if t in by_task]:
            rows = []
            for e in by_task[task]:
                sid = e['scenario_id']
                cache.setdefault(sid, make_scenario(sid, scenario_overrides(exp, sid, extra)))
                rows.append(run_entry(cache[sid], e, ctrl, noise))
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
        named = lambda ext: out.parent / f'{out.name}{ext}'       # not with_suffix: names like device_late_f0.5
        named('.md').write_text(f'# Expert check — {args.list}\n\n' + md)
        named('.json').write_text(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    main()
