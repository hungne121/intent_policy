"""Review previews (QĐ 53/QĐ-DL §9): run observed and forecast rollouts of the same seed / human scenario and
write one review-camera video (collection.yaml `review_camera`), a summary and a trace per mode. Not dataset collection.

    ./run.sh -m tools.preview --tasks I1 C2 --out previews
"""
import argparse
import json
from pathlib import Path

import yaml

from intent_policy.shared.sim_env import ROOT
from intent_policy.shared.episode import Episode
from intent_policy.shared.manicast_adapter import load_predictor
from intent_policy.experts import TASKS, expert_class
from intent_policy.scenarios import scenario

CFG = yaml.safe_load((ROOT / 'configs/hrc/collection.yaml').read_text())


def variants(task):
    return getattr(scenario(task), 'VARIANTS', ('nominal',))


def run_episode(task, mode, seed, predictor, out=None, recorder=None, review=True, variant='nominal',
                variation=None, fault=None, name_suffix=''):
    """One rollout. `variation`: N1 spec (shared/variation.py) or None; `fault`: N5 FaultSpec or None (59/QĐ-DL)."""
    sc = scenario(task)
    scene, positions, xml, info = sc.build(seed, variant) if hasattr(sc, 'VARIANTS') else sc.build(seed)
    injector = None
    if fault is not None:
        from intent_policy.shared.forecast_fault import FaultInjector
        injector = FaultInjector(fault, CFG['n5'])
    ep = Episode(task, mode, seed, scene, positions, xml, predictor, out=out if review else None,
                 max_ticks=sc.MAX_TICKS, review_camera=CFG['review_camera'], review_hw=CFG['review_hw'],
                 recorder=recorder, variant=info.get('variant', 'nominal'), injector=injector, variation=variation,
                 name_suffix=name_suffix)
    if recorder is not None:
        recorder.start_episode(ep)
    expert = expert_class(task)(ep)
    ep.event('task_active', **{k + '_evaluator_only': v for k, v in info.items() if k != 'variant'})
    human = sc.human(ep, info)
    if variation:
        from intent_policy.shared.variation import timed_human
        ep.variation_log['human'] = {}
        human = timed_human(human, variation, ep.variation_log['human'])
    ep.run(human, expert.run(), done=lambda: sc.human_done(ep))
    ok, reason = expert.success()
    if ok and hasattr(sc, 'check'):                 # evaluator-only predicate using the human plan
        ok, reason = sc.check(ep, info)
    if ep.stalled:
        ok, reason = False, 'motion_stalled'
    elif ep.human_failed:
        ok, reason = False, 'human_reach_failed'
    elif not ok and ep.tick >= ep.max_ticks:
        reason = 'time_limit'
    return ep.finish(ok, reason, {'reading': expert.reading, **expert.extra(),
                                 'human_plan_evaluator_only': {k: v for k, v in info.items()}})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tasks', nargs='+', default=list(TASKS))
    p.add_argument('--modes', nargs='+', default=CFG['modes'])
    p.add_argument('--seed', type=int, default=None, help='default: configs/hrc/collection.yaml preview_seeds')
    p.add_argument('--out', type=Path, default=ROOT / 'previews')
    a = p.parse_args()
    predictor = load_predictor()          # both modes compute (and record) the forecast; only `forecast` uses it
    for task in a.tasks:
        seed = a.seed if a.seed is not None else CFG['preview_seeds'][task]
        for variant in variants(task):
            for mode in a.modes:
                s = run_episode(task, mode, seed, predictor, out=a.out, variant=variant); s.pop('_arrays', None)
                print(json.dumps({k: s[k] for k in ('task', 'variant', 'mode', 'seed', 'success', 'failure_reason',
                                                    'duration_s', 'physics_valid')}), flush=True)


if __name__ == '__main__':
    main()
