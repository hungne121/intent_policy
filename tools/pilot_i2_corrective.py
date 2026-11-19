"""Pilot of the I2 corrective expert on the laptop (28/QĐ-HL Phase 3 item 6): per seed table (success, physics, contacts,
attempts) + optional review videos. Nothing is written to a dataset.

    ./run.sh -m tools.pilot_i2_corrective --variant recover --seeds 300820 300821 --out outputs/diag/phase3_1120/pilot_i2 [--video]
"""
import argparse
import json
from pathlib import Path

from intent_policy.shared.manicast_adapter import load_predictor
from intent_policy.shared.episode import Episode
from intent_policy.scenarios import scenario
from intent_policy.experts.i2_corrective_expert import make
from tools.preview import CFG


def run(seed, variant, mode, pred, out=None, rest_lean_free=False):
    sc = scenario('I2')
    scene, positions, xml, info = sc.build(seed)
    scene['human']['rest_lean_free'] = rest_lean_free     # 29/QĐ-HL (pilot only)
    ep = Episode('I2', mode, seed, scene, positions, xml, pred, out=out, max_ticks=sc.MAX_TICKS,
                 review_camera=CFG['review_camera'], review_hw=CFG['review_hw'], variant='nominal',
                 name_suffix=f'corrective_{variant}_{seed}_')
    expert = make(variant)(ep)
    ep.event('task_active', plate_evaluator_only=info['plate'])
    ep.run(sc.human(ep, info), expert.run(), done=lambda: sc.human_done(ep))
    ok, reason = expert.success()
    if ok and hasattr(sc, 'check'):
        ok, reason = sc.check(ep, info)
    if ep.stalled:
        ok, reason = False, 'motion_stalled'
    s = ep.finish(ok, reason, {'reading': expert.reading, **expert.extra()})
    s.pop('_arrays', None)
    return info, s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['original', 'nominal_dwell', 'recover'], required=True)
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--mode', default='forecast')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--video', action='store_true')
    p.add_argument('--rest-lean-free', action='store_true', help='29/QĐ-HL: resting human does not lean')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pred = load_predictor()
    for seed in a.seeds:
        info, s = run(seed, a.variant, a.mode, pred, out=a.out / 'video' if a.video else None, rest_lean_free=a.rest_lean_free)
        viol = {k: v.get('category') for k, v in s.get('contact_classes', {}).items() if v.get('violation')}
        row = dict(variant=a.variant, seed=seed, plate=info['plate'], success=s['success'], reason=s['failure_reason'],
                   physics_valid=s.get('physics_valid'), robot_limits_ok=s.get('robot_limits_ok'), violations=viol,
                   early_hold_s=s.get('early_hold_s'), dwell_s=s.get('dwell_s'), attempts=s.get('recover_attempts'),
                   dur_s=s['duration_s'], events=[e['event'] for e in s['events'] if e['event'] in ('injected_early_lift', 'grasp_not_attached', 'robot_grasp', 'grasp_failed')])
        with open(a.out / f'pilot_{a.variant}.jsonl', 'a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
