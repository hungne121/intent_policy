"""Replay a recorded HRC episode (QĐ 53/QĐ-DL §8, §9): rebuild the scene and the human scenario from the episode
metadata, feed the RECORDED actions instead of the expert, and compare the TCP path and final object positions
with what was recorded. The human reacts to the robot exactly as during collection, so equal actions must
reproduce the episode.

    ./run.sh -m tools.replay_episode --mode forecast --episode 0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from intent_policy.shared.sim_env import ROOT
from intent_policy.shared.episode import Episode
from intent_policy.scenarios import scenario

CFG = yaml.safe_load((ROOT / 'configs/hrc/collection.yaml').read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=CFG['modes'], required=True)
    p.add_argument('--episode', type=int, required=True)
    p.add_argument('--root', type=Path, default=ROOT / CFG['datasets']['root'])
    a = p.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    name = CFG['datasets'][a.mode]
    side = {e['episode_index']: e for e in (json.loads(l) for l in (a.root / name / 'meta/hrc_episodes.jsonl').read_text().splitlines() if l.strip())}
    meta = side[a.episode]
    ds = LeRobotDataset(f'local/{name}', root=a.root / name, episodes=[a.episode])
    actions = np.stack([np.asarray(ds[i]['action']) for i in range(len(ds))])
    tcp_rec = np.stack([np.asarray(ds[i]['observation.tcp_pose'])[:3] for i in range(len(ds))])
    sc = scenario(meta['task_name'])
    scene, positions, xml, info = (sc.build(meta['seed'], meta['task_variant']) if hasattr(sc, 'VARIANTS') else sc.build(meta['seed']))
    variation = meta.get('variation')                     # 59/QĐ-DL N1: same robot start + human timing
    # The scripted human may wait on expert state (I1 `robot_prepared`), so the expert runs in lockstep on the same
    # observations (same predictor, same N5 fault); the RECORDED action is applied and the expert's own command is
    # compared with it (equal => the episode is reproduced from seed + config + code).
    from intent_policy.shared.manicast_adapter import load_predictor
    from intent_policy.experts import expert_class
    injector = None
    if meta.get('fault_parameters'):
        from intent_policy.shared.forecast_fault import FaultSpec, FaultInjector
        injector = FaultInjector(FaultSpec(**meta['fault_parameters']), CFG['n5'])
    ep = Episode(meta['task_name'], meta['expert_mode'], meta['seed'], scene, positions, xml, load_predictor(), out=None,
                 max_ticks=len(actions), variant=meta['task_variant'], variation=variation, injector=injector)
    expert = expert_class(meta['task_name'])(ep)
    robot = expert.run()
    human = sc.human(ep, info)
    if variation:
        from intent_policy.shared.variation import timed_human
        human = timed_human(human, variation, {})
    tcp_rep, cmd_dev, last = [], [], None
    for act in actions:
        try:
            last = next(human)
        except StopIteration:
            pass
        ep.apply_human(last)
        tcp_rep.append(ep.tcp())
        try:
            cmd = np.asarray(next(robot), float)
            cmd_dev.append(float(np.abs(cmd - act).max()))
        except StopIteration:
            robot = iter(())
        ep.step(act)
    dev = np.linalg.norm(np.asarray(tcp_rep) - tcp_rec, axis=1)
    final = {k: float(np.linalg.norm(ep.pos(k) - np.asarray(v))) for k, v in meta['final_positions'].items()}
    ep.env.close()
    print(json.dumps({'episode_id': meta['episode_id'], 'frames': len(actions), 'tcp_max_dev_m': float(dev.max()),
                      'expert_command_max_dev': max(cmd_dev) if cmd_dev else None,
                      'final_object_max_dev_m': max(final.values()), 'final_object_dev_m': final}, indent=2))


if __name__ == '__main__':
    main()
