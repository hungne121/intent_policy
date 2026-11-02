"""Gated expert pilot without recording (docs/PIF.md §8.12): pass rate of the collection gate per task / case / rule.

Same scenario overrides and gate as scripts/collect_pairs.py (configs/experiments/ten_task_v7.yaml); no images, no
dataset. One JSON line per episode: success, the gate's first violation (bodies, stages, depth), protected-object
movement, the motion sources that were not HY-Motion. Diagnosis only, never demonstrations.
"""
import argparse
import json
import time
from pathlib import Path
from intent_policy.benchmark.runner import run_episode, ExpertAgent
from intent_policy.scenarios.config import _deep_merge
from intent_policy.scenarios.scenario_registry import make_scenario
from intent_policy.sim.collection_gate import CollectionGate
from intent_policy.utils import controller_config, load_yaml
from scripts.collect_pairs import chunk_overrides
from intent_policy.benchmark.collection_integrity import motion_source_hash


def gate_config(exp, task):
    gate = _deep_merge(_deep_merge({}, exp['gates']['default']), task['gate'])
    gate['scene_coverage_reviewed'] = task['scene_coverage_reviewed']
    return gate


def run_one(exp, task_id, case, split, rule, seed):
    task = exp['tasks'][task_id]
    overrides = chunk_overrides(task, task['cases'][case], split, rule)
    if exp['data'].get('paired_source_selection'):
        overrides = _deep_merge(overrides, {'human_behavior': {'motion_bank': {'paired_source_selection': True}}})
    sc = make_scenario(task['scenario'], overrides)
    agent, gates = ExpertAgent(), []
    t0 = time.time()
    try:
        rec = run_episode(sc, agent, seed, controller_config(exp), keep_trace=False,
                          on_reset=lambda s, a: gates.append(CollectionGate(s, a, gate_config(exp, task))))
        q = gates[0].finish(rec['events'])
        v = q['geometry']['first_violation']
        if v and 'geom_a' in v:
            m = sc.env.model
            for k in ('geom_a', 'geom_b'):
                g = m.geom(v[k]).id
                v[k + '_body'] = m.body(m.geom_bodyid[g]).name
                v[k] = m.geom(g).name or int(g)
        src = [(e['payload']['label'], (e['payload'].get('motion_source') or {}).get('kind'))
               for e in rec['events'] if e['event_type'] == 'human_motion_start']
        return dict(task=task_id, case=case, split=split, rule=rule, seed=seed, wall_s=round(time.time() - t0, 1),
                    sim_s=round(sc.time, 2), motion_source_sha256=motion_source_hash(rec),
                    motion_sources=[dict(label=e['payload'].get('label'), source=e['payload'].get('motion_source'))
                                    for e in rec['events'] if e['event_type'] == 'human_motion_start'],
                    changes=[e for e in rec['events'] if e['event_type'] == 'human_intention_change'], success=rec['success'], failure=rec['failure'], gate=q['passed'],
                    violation=v, moved=q['movement_violation'], missing_hymotion=q['missing_hymotion'],
                    not_hymotion=[l for l, k in src if k != 'hymotion'])
    finally:
        sc.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/experiments/ten_task_v7.yaml')
    p.add_argument('--task', required=True)
    p.add_argument('--case', default='nominal')
    p.add_argument('--split', default='train')
    p.add_argument('--rules', default='observed,early')
    p.add_argument('--seeds', required=True, help='a,b,c or start:stop')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    exp = load_yaml(a.config)
    seeds = range(*map(int, a.seeds.split(':'))) if ':' in a.seeds else [int(s) for s in a.seeds.split(',')]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('a') as f:
        for seed in seeds:
            for rule in a.rules.split(','):
                r = run_one(exp, a.task, a.case, a.split, rule, seed)
                f.write(json.dumps(r, default=str) + '\n')
                f.flush()
                v = r['violation']
                print(seed, rule, r['sim_s'], r['success'], r['failure'], r['gate'],
                      v and (v.get('geom_a_body'), v.get('geom_b_body'), round(v['distance_m'] * 1e3, 2), round(v['t'], 2),
                             v.get('human_stage'), v.get('robot_stage')),
                      r['moved'] and r['moved']['object'], 'not_hymotion', len(r['not_hymotion']), flush=True)


if __name__ == '__main__':
    main()
