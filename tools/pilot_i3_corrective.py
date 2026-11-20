"""Pilot of I3 corrective injections on the laptop (28/QĐ-HL Phase 3): per-seed table + optional review video.
    ./run.sh -m tools.pilot_i3_corrective --variant s2_wrong_neighbour --seeds 300820 --out outputs/diag/phase3_1120/pilot_i3 [--video]
"""
import argparse, json
from pathlib import Path
import numpy as np, yaml
from intent_policy.shared.sim_env import ROOT
from intent_policy.shared.manicast_adapter import load_predictor
from intent_policy.shared.episode import Episode
from intent_policy.scenarios import scenario
from intent_policy.experts import expert_class
from tools.preview import CFG
SPEC = yaml.safe_load((ROOT / 'configs/hrc/i3_corrective.yaml').read_text())['variants']


def run(seed, variant, mode, pred, out=None, rest_lean_free=True, task_variant='switch_before_commit'):
    sc = scenario('I3'); scene, positions, xml, info = sc.build(seed, task_variant)
    scene['human']['rest_lean_free'] = rest_lean_free
    ep = Episode('I3', mode, seed, scene, positions, xml, pred, out=out, max_ticks=sc.MAX_TICKS, review_camera=CFG['review_camera'],
                 review_hw=CFG['review_hw'], variant=task_variant, name_suffix=f'corrective_{variant}_{seed}_')
    ex = expert_class('I3')(ep); spec = SPEC[variant]
    ex.inject = spec['inject']; ex.safe_switch_lift = bool(spec.get('safe_switch_lift', False)); ex.inject_open_ticks = int(yaml.safe_load((ROOT / 'configs/hrc/i3_corrective.yaml').read_text())['open_ticks'])
    if spec.get('hold_s'):
        ex.inject_hold_ticks = int(round(float(np.random.default_rng(seed + 7919).uniform(*spec['hold_s'])) / 0.05))
    ep.event('task_active', **{k + '_evaluator_only': v for k, v in info.items() if k != 'variant'})
    ep.run(sc.human(ep, info), ex.run(), done=lambda: sc.human_done(ep))
    ok, reason = ex.success()
    ok_v2, reason_v2 = ok, reason
    if ok and hasattr(sc, 'check'):
        ok, reason = sc.check(ep, info, retry_ok=False)        # old evaluator (31/QĐ-HL: kept for comparison)
        ok_v2, reason_v2 = sc.check(ep, info, retry_ok=True)   # retry-aware evaluator
    if ep.stalled: ok, reason = False, 'motion_stalled'
    s = ep.finish(ok, reason, {'reading': ex.reading, **ex.extra()}); s.pop('_arrays', None)
    s['_v2'] = (bool(ok_v2), reason_v2)
    return info, s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=list(SPEC), required=True)
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--mode', default='forecast'); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--video', action='store_true'); p.add_argument('--keep-rest-lean', action='store_true')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True); pred = load_predictor()
    keep = ('injected_close_after_switch', 'injected_wrong_commit', 'robot_grasp', 'grasp_failed', 'switch_observed', 'human_intention_change')
    for seed in a.seeds:
        info, s = run(seed, a.variant, a.mode, pred, out=a.out / 'video' if a.video else None, rest_lean_free=not a.keep_rest_lean)
        viol = {k: v.get('category') for k, v in s.get('contact_classes', {}).items() if v.get('violation')}
        rest = [r for r in (s['failure_reason'] or '').split('; ') if r and not r.startswith('grasp sequence')]
        row = dict(variant=a.variant, seed=seed, A=info['A'], B=info['B'], success=s['success'], ok_v2=s['_v2'][0], reason_v2=s['_v2'][1], reason=s['failure_reason'],
                   corrective_ok=bool(s['success'] or (not rest and a.variant != 'original' and s['failure_reason'] and s['failure_reason'].startswith('grasp sequence'))),
                   physics_valid=s.get('physics_valid'), robot_limits_ok=s.get('robot_limits_ok'), violations=viol, dur_s=s['duration_s'],
                   events=[(e['event'], e.get('object') or e.get('committed') or e.get('item')) for e in s['events'] if e['event'] in keep])
        with open(a.out / f'pilot_{a.variant}.jsonl', 'a') as f: f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
