"""Phase 3 (28/QĐ-HL items 1, 3): run the I2 expert in the sim and log, per tick, the gripper opening, which finger
touches the cake, the attach tick and the cake height. Diagnosis only (expert, not a policy claim).

    ./run.sh -m tools.probe_i2_attach --seeds 300820 300821 --mode forecast --out outputs/diag/phase3_1120/i2_attach
"""
import argparse
import json
from pathlib import Path

import numpy as np

from intent_policy.shared.manicast_adapter import load_predictor
from intent_policy.shared.episode import Episode
from intent_policy.scenarios import scenario
from intent_policy.experts import expert_class
from tools.preview import CFG


def run(seed, mode, predictor, hook_tag=''):
    sc = scenario('I2')
    scene, positions, xml, info = sc.build(seed)
    ep = Episode('I2', mode, seed, scene, positions, xml, predictor, out=None, max_ticks=sc.MAX_TICKS,
                 review_camera=CFG['review_camera'], review_hw=CFG['review_hw'], variant='nominal')
    env, cake = ep.env, ep.m.body('cake').id
    log = []
    orig = env.step

    def step(action):
        s = orig(action)
        log.append(dict(t=round(ep.tick * 0.05, 2), cmd=float(action[6]), gap=float(env.data.qpos[env.gid]),
                        touch=sorted(env.finger_contacts(cake)), attached=env.attached == cake,
                        cake_z=float(env.data.xpos[cake][2]), tcp_z=float(s.tcp[2]) if hasattr(s, 'tcp') else None,
                        phase=ep.phase))
        return s
    env.step = step
    expert = expert_class('I2')(ep)
    ep.event('task_active')
    human = sc.human(ep, info)
    ep.run(human, expert.run(), done=lambda: sc.human_done(ep))
    ok, reason = expert.success()
    if ok and hasattr(sc, 'check'):
        ok, reason = sc.check(ep, info)
    return dict(seed=seed, plate=info['plate'], success=bool(ok), reason=reason, log=log)


def summarize(r):
    L = r['log']
    close = [i for i, x in enumerate(L) if x['phase'] == 'close']
    if not close:
        return dict(seed=r['seed'], plate=r['plate'], success=r['success'], close=None)
    i0 = close[0]
    first_touch = next((i for i in close if L[i]['touch']), None)
    both = next((i for i in close if len(L[i]['touch']) == 2), None)
    att = next((i for i in close if L[i]['attached']), None)
    first = lambda i: None if i is None else round((i - i0) * 0.05, 2)
    sides = None if first_touch is None else L[first_touch]['touch']
    lift = next((i for i, x in enumerate(L) if x['phase'] == 'lift'), None)
    return dict(seed=r['seed'], plate=r['plate'], success=r['success'], close_start_t=L[i0]['t'], t_first_touch=first(first_touch),
                first_touch_side=sides, t_both_touch=first(both), t_attach=first(att), t_lift=first(lift),
                gap_at_close=round(L[i0]['gap'], 4), gap_at_attach=None if att is None else round(L[att]['gap'], 4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--mode', default='forecast')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pred = load_predictor()
    for s in a.seeds:
        r = run(s, a.mode, pred)
        (a.out / f'{a.mode}_{s}.json').write_text(json.dumps(r))
        print(json.dumps(summarize(r)), flush=True)


if __name__ == '__main__':
    main()
