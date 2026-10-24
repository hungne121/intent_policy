"""Record the mannequin's arm keypoints in the v6 scenarios for fine-tuning ManiCast (configs/pif/pif.yaml
forecast.data; docs/sim/scenarios_v6_proposal.md M5). Every control tick (20 Hz) of every episode (scripted expert,
the human from its v6 behaviour incl. the HY-Motion library): l/r shoulder, elbow, wrist in the world frame.
  ./run.sh -m pif.v6_motion_data  -> <out>/<split>.npz: per scenario an object array of [T, 6, 3] episodes and one of
  their hand motions (start, duration, label: the transitions of ManiCast's fine-tuning), + seeds
"""
from __future__ import annotations
import argparse
import json

import numpy as np

from pif.config import load_config
from pif.utils import log_run, resolve

ARM = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist']


def episode(sc, seed: int) -> np.ndarray:
    from intent_policy.benchmark.runner import ExpertAgent
    from intent_policy.sim.restricted_action import RestrictedActionConfig, RestrictedActionMapper
    sc.reset(seed)
    mapper = RestrictedActionMapper(sc.env, RestrictedActionConfig.load())
    ag = ExpertAgent()
    ag.reset(sc, mapper)
    kp = []
    while not sc.done:
        kp.append(sc.human_keypoints(ARM))
        sc.step(mapper.map(ag.act(None).action_id).joint_target)
    moves = [(e['timestamp'], (e.get('payload') or {}).get('duration', 0.0), (e.get('payload') or {}).get('label', ''))
             for e in sc.get_events() if e['event_type'] == 'human_motion_start']
    return np.asarray(kp, np.float32), moves


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/pif/pif.yaml')
    ap.add_argument('--set', nargs='*', default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    d = cfg['forecast']['data']
    out = resolve(d['out'])
    out.mkdir(parents=True, exist_ok=True)
    log_run(out, 'pif.v6_motion_data')
    from intent_policy.scenarios.scenario_registry import make_scenario
    stats = {}
    for split, (a, b) in d['seeds'].items():
        arrays = {}
        for sid in d['scenarios']:
            sc = make_scenario(sid)
            rec = [episode(sc, s) for s in range(a, b)]
            sc.close()
            eps = [r[0] for r in rec]
            arrays[sid] = np.empty(len(eps), object)
            arrays[sid][:] = eps
            arrays[f'{sid}__moves'] = np.empty(len(eps), object)    # (start s, duration s, label) of every hand motion
            arrays[f'{sid}__moves'][:] = [np.array(r[1], dtype=object) for r in rec]
            stats[f'{split}/{sid}'] = dict(episodes=len(eps), seconds=round(sum(len(e) for e in eps) / 20.0, 1))
            print(split, sid, stats[f'{split}/{sid}'], flush=True)
        np.savez_compressed(out / f'{split}.npz', joints=np.array(ARM), rate_hz=np.array(20.0),
                            seeds=np.arange(a, b), **arrays)
    (out / 'stats.json').write_text(json.dumps(stats, indent=1))
    print('DATA_DONE', out)


if __name__ == '__main__':
    main()
