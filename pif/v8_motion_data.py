"""Record the mannequin's arm keypoints in the v8 ten-task scenarios for fine-tuning ManiCast again (docs/PIF.md §8.12
"Thiết kế lại theo trần dự báo": the v6 checkpoint never saw the HY-Motion ten-task library nor left-arm requests).
Same format as pif/v6_motion_data.py, so pif/forecast_finetune.py reads it unchanged. Every control tick of every
episode (observed expert, no forecaster involved); the HY-Motion clip split = the data split (train / val / test clips
of outputs/human_motion/.../splits.json), so test clips stay unseen. Cases with a forecast fault are skipped (the
human moves as in nominal).
  ./run.sh -m pif.v8_motion_data --tasks I1,C1 --out outputs/human_motion/v8_sim_motion/part_a
"""
from __future__ import annotations
import argparse
import json

import numpy as np

from pif.utils import log_run, resolve
from pif.v6_motion_data import ARM

SPLIT_OFFSET = {'train': 0, 'val': 5000, 'test': 7000}     # disjoint from scripts/collect_pairs.py seeds (base 3000000)


def episode(sc, seed: int, ccfg) -> tuple[np.ndarray, list]:
    from intent_policy.benchmark.runner import ExpertAgent
    from intent_policy.sim.restricted_action import RestrictedActionMapper
    sc.reset(seed)
    mapper = RestrictedActionMapper(sc.env, ccfg)
    ag = ExpertAgent()
    ag.reset(sc, mapper)
    kp, done = [], False
    while not done:
        cmd = mapper.map(ag.act(None).action_id)
        for _ in range(cmd.execution_ticks):
            kp.append(sc.human_keypoints(ARM))
            done = sc.step(cmd.joint_target)
            if done:
                break
    moves = [(e['timestamp'], (e.get('payload') or {}).get('duration', 0.0), (e.get('payload') or {}).get('label', ''))
             for e in sc.get_events() if e['event_type'] == 'human_motion_start']
    return np.asarray(kp, np.float32), moves


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/experiments/ten_task_v8.yaml')
    ap.add_argument('--tasks', required=True)
    ap.add_argument('--episodes', default='train:30,val:6,test:6', help='per task, spread over its cases')
    ap.add_argument('--seed-base', type=int, default=6000000)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    from intent_policy.scenarios.scenario_registry import make_scenario
    from intent_policy.utils import controller_config, load_yaml
    from scripts.collect_pairs import chunk_overrides
    from intent_policy.scenarios.config import _deep_merge
    exp = load_yaml(args.config)
    ccfg = controller_config(exp)
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_run(out, 'pif.v8_motion_data')
    order = list(exp['tasks'])
    stats = {}
    for item in args.episodes.split(','):
        split, n = item.split(':')
        arrays, seeds = {}, []
        for tid in args.tasks.split(','):
            task = exp['tasks'][tid]
            cases = [c for c in task['cases'] if 'fault' not in c]
            for ci, case in enumerate(cases):
                k = int(n) // len(cases) + (1 if ci < int(n) % len(cases) else 0)
                base = args.seed_base + 100000 * order.index(tid) + 10000 * ci + SPLIT_OFFSET[split]
                ov = chunk_overrides(task, task['cases'][case], split, 'observed')
                if exp['data'].get('paired_source_selection'):     # the clip choice of the collection (collect_pairs)
                    ov = _deep_merge(ov, {'human_behavior': {'motion_bank': {'paired_source_selection': True}}})
                sc = make_scenario(task['scenario'], ov)
                rate = 1.0 / sc.dt
                rec = [episode(sc, base + i, ccfg) for i in range(k)]
                sc.close()
                key = f'{tid}_{case}'
                arrays[key] = np.empty(len(rec), object)
                arrays[key][:] = [r[0] for r in rec]
                arrays[f'{key}__moves'] = np.empty(len(rec), object)
                arrays[f'{key}__moves'][:] = [np.array(r[1], dtype=object) for r in rec]
                seeds += [base + i for i in range(k)]
                stats[f'{split}/{key}'] = dict(episodes=k, seconds=round(sum(len(r[0]) for r in rec) / rate, 1))
                print(split, key, stats[f'{split}/{key}'], flush=True)
        np.savez_compressed(out / f'{split}.npz', joints=np.array(ARM), rate_hz=np.array(rate), seeds=np.array(seeds),
                            **arrays)
    (out / 'stats.json').write_text(json.dumps(stats, indent=1))
    print('DATA_DONE', out)


if __name__ == '__main__':
    main()
