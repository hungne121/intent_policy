"""Check the scripted expert of an experiment config (per split) before collecting demonstrations.

Runs the privileged expert without rendering on N seeds per scenario and split and reports success,
failure reasons, completion time, cue -> first robot motion, minimum human-robot distance, safety
events and, for intention-change episodes, when the change happened and how long the robot took to
head for the new target. `--compare-cue-complete` also runs the Phase-1 expert timing on the
nominal split for comparison.

  ./run.sh -m scripts.check_expert --config configs/experiments/phase2_scenario_preview.yaml \
      --seeds 60 --compare-cue-complete --output outputs/viz/hri_phase2_scenario_preview/expert_check
"""
import argparse
import json
from collections import Counter
import numpy as np
from intent_policy.benchmark.runner import ExpertAgent, run_episode
from intent_policy.scenarios.config import _deep_merge
from intent_policy.scenarios.scenario_registry import make_scenario, scenario_overrides
from intent_policy.utils import load_yaml, resolve, controller_config

HEAD_FOR_NEW_TARGET_M = 0.02    # robot "responds" once its horizontal distance to the new target shrank by this much


def first_time(events, event_type, after=-1.0):
    ts = [e['timestamp'] for e in events if e['event_type'] == event_type and e['timestamp'] >= after]
    return ts[0] if ts else None


def change_response(rec, sc) -> dict | None:
    """Time from the intention change until the TCP clearly heads for the new target."""
    change = next((e for e in rec['events'] if e['event_type'] == 'human_intention_change'), None)
    if change is None:
        return None
    new, t_c = change['payload']['current'], change['timestamp']
    trace = [s['state'] for s in rec['trace'] if s['state']['t'] >= t_c - 1e-9]
    target = lambda st: np.asarray(st['objects'][new]['position'])[:2] if new in st['objects'] else sc.region_pos(new)[:2]
    d0 = np.linalg.norm(np.asarray(trace[0]['robot']['ee_pos'])[:2] - target(trace[0])) if trace else None
    for st in trace:
        if d0 - np.linalg.norm(np.asarray(st['robot']['ee_pos'])[:2] - target(st)) >= HEAD_FOR_NEW_TARGET_M:
            return dict(t_change=round(t_c, 3), kind=change['payload']['kind'], response_s=round(st['t'] - t_c, 3))
    return dict(t_change=round(t_c, 3), kind=change['payload']['kind'], response_s=None)


def run_split(exp, split, sid, seeds, ctrl_cfg, extra=None) -> dict:
    ov = _deep_merge(scenario_overrides(exp, sid),
                     scenario_overrides({'scenario_overrides': split.get('scenario_overrides')}, sid, extra))
    sc = make_scenario(sid, ov)
    rows = []
    try:
        for seed in seeds:
            rec = run_episode(sc, ExpertAgent(), seed, ctrl_cfg, keep_trace=True)
            ev = rec['events']
            cue = first_time(ev, 'human_cue_onset')
            move = first_time(ev, 'robot_motion_start', after=cue if cue is not None else 0.0)
            kinds = Counter(e['event_type'] for e in ev)
            rows.append(dict(seed=seed, success=rec['success'], failure=rec['failure'], duration_s=rec['duration_s'],
                             ct=rec['metrics'].get('CT'), cue_to_motion_s=None if cue is None or move is None else move - cue,
                             min_dist=rec['min_human_robot_distance'], contacts=kinds['human_robot_contact'],
                             sdv=kinds['safety_distance_violation'], change=change_response(rec, sc),
                             change_skipped=sc.human.extras.get('intention_change_skipped_t')))
    finally:
        sc.close()
    return summarize(rows)


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x is not None]
    return (round(float(np.mean(xs)), 3), round(float(np.std(xs)), 3), len(xs)) if xs else (None, None, 0)


def summarize(rows) -> dict:
    changes = [r['change'] for r in rows if r['change']]
    return dict(episodes=len(rows), success=sum(r['success'] for r in rows),
                failures=dict(Counter(r['failure'] for r in rows if not r['success'])),
                ct=mean([r['ct'] for r in rows if r['success']]), cue_to_motion_s=mean([r['cue_to_motion_s'] for r in rows]),
                min_dist_m=round(min(r['min_dist'] for r in rows), 4), episodes_with_contact=sum(r['contacts'] > 0 for r in rows),
                episodes_with_sdv=sum(r['sdv'] > 0 for r in rows), changes=len(changes),
                changes_skipped=sum(r['change_skipped'] is not None for r in rows),
                change_time_s=mean([c['t_change'] for c in changes]),
                change_response_s=mean([c['response_s'] for c in changes]),
                change_no_response=sum(c['response_s'] is None for c in changes), rows=rows)


def fmt(m):
    return '—' if m[0] is None else f'{m[0]:.2f} ± {m[1]:.2f}'


def markdown(results) -> str:
    lines = ['| split | scenario | expert | success | CT [s] (succ.) | cue→motion [s] | min dist [m] | contact ep. | SDV ep. '
             '| changes (skipped) | change at t [s] | robot heads to new target after [s] |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        s = r['summary']
        lines.append(f"| {r['split']} | {r['scenario']} | {r['trigger']} | {s['success']}/{s['episodes']} | {fmt(s['ct'])} "
                     f"| {fmt(s['cue_to_motion_s'])} | {s['min_dist_m']:.3f} | {s['episodes_with_contact']} | {s['episodes_with_sdv']} "
                     f"| {s['changes']} ({s['changes_skipped']}) | {fmt(s['change_time_s'])} | {fmt(s['change_response_s'])}"
                     + (f" ({s['change_no_response']} none)" if s['change_no_response'] else '') + ' |')
    failures = [f"- {r['split']} / {r['scenario']} / {r['trigger']}: {r['summary']['failures']}"
                for r in results if r['summary']['failures']]
    lines += ['', '**Failures:**'] + (failures or ['- none'])
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/experiments/phase2_scenario_preview.yaml')
    p.add_argument('--seeds', type=int, default=60)
    p.add_argument('--seed-start', type=int, default=300)
    p.add_argument('--scenarios', nargs='*')
    p.add_argument('--compare-cue-complete', action='store_true')
    p.add_argument('--output', help='write <output>.md and <output>.json')
    args = p.parse_args()
    exp = load_yaml(args.config)
    ctrl_cfg = controller_config(exp)
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    splits = exp['data'].get('splits') or [dict(name='default')]
    results = []
    for split in splits:
        for sid in split.get('scenarios') or exp['benchmark']['scenarios']:
            if args.scenarios and sid not in args.scenarios:
                continue
            variants = [None]
            if args.compare_cue_complete and split['name'] in ('nominal', 'default'):
                variants.append({'expert': {'trigger': 'cue_complete', 'yield_prediction_horizon_s': 0.0}})
            for extra in variants:
                summary = run_split(exp, split, sid, seeds, ctrl_cfg, extra)
                trigger = 'cue_complete (Phase 1)' if extra else \
                    _deep_merge(scenario_overrides(exp, sid), {}).get('expert', {}).get('trigger', 'cue_complete')
                results.append(dict(split=split['name'], scenario=sid, trigger=trigger, summary=summary))
                s = summary
                print(f"[{split['name']}] {sid} ({trigger}): {s['success']}/{s['episodes']} ok, failures={s['failures']}, "
                      f"CT={fmt(s['ct'])}, changes={s['changes']}, response={fmt(s['change_response_s'])}", flush=True)
    md = markdown(results)
    print(md)
    if args.output:
        out = resolve(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix('.md').write_text(f'# Expert check — {exp["experiment"]} (seeds {seeds.start}–{seeds.stop - 1})\n\n' + md)
        out.with_suffix('.json').write_text(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    main()
