"""22/QĐ-HL 5 (read-only): the scripted C2 expert on the same val seeds as the policy evaluations (forecast mode, N1 variation,
as in outputs/diag/audit_env_1119/replay_expert_vs_env.py): success, duration, key events, TCP path, idle time.
    ./run.sh -m tools.c2_expert_reference --seeds 300810 300811 300812 300813 300814 --out outputs/diag/c2_audit_1119/expert_c2.json"""
import argparse, json
from pathlib import Path
import numpy as np
import intent_policy.envs.hrc_env as H
from intent_policy.shared.episode import Episode
from intent_policy.scenarios import scenario
from intent_policy.experts import expert_class
from intent_policy.shared.variation import sample_variation, timed_human
from tools.collect import task_cfg

ap = argparse.ArgumentParser(); ap.add_argument('--seeds', nargs='+', type=int, required=True); ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--task', default='C2'); a = ap.parse_args()
H.policy_observation = lambda ep, cams, hw: {'pixels': {}, 'agent_pos': np.zeros(13, np.float32), 'human_forecast': np.zeros(180, np.float32), 'forecast_valid': np.zeros(1, np.float32)}
res = []
for seed in a.seeds:
    task = a.task
    sc = scenario(task)
    scene, positions, xml, info = sc.build(seed, 'nominal') if hasattr(sc, 'VARIANTS') else sc.build(seed)
    variation = sample_variation(seed, task_cfg('n1', task))
    ep = Episode(task, 'forecast', seed, scene, positions, xml, H.predictor(), out=None, max_ticks=sc.MAX_TICKS, review_camera=H.CFG['review_camera'],
                 review_hw=H.CFG['review_hw'], recorder=None, variant=info.get('variant', 'nominal'), injector=None, variation=variation)
    expert = expert_class(task)(ep)
    ep.event('task_active', **{k + '_evaluator_only': v for k, v in info.items() if k != 'variant'})
    human = sc.human(ep, info)
    ep.variation_log['human'] = {}
    human = timed_human(human, variation, ep.variation_log['human'])
    ep.run(human, expert.run(), done=lambda: sc.human_done(ep))
    ok, reason = expert.success()
    if ok and hasattr(sc, 'check'):
        ok, reason = sc.check(ep, info)
    if ep.stalled: ok, reason = False, 'motion_stalled'
    elif ep.human_failed: ok, reason = False, 'human_reach_failed'
    elif not ok and ep.tick >= ep.max_ticks: reason = 'time_limit'
    s = ep.finish(ok, reason, {'reading': expert.reading})
    tcp = np.array([r['tcp'] for r in ep.trace]); t = np.array([r['t'] for r in ep.trace])
    sp = np.r_[0, np.linalg.norm(np.diff(tcp, axis=0), axis=1) / 0.05]
    res.append(dict(seed=seed, success=bool(ok), reason=reason, duration_s=round(float(t[-1]), 2), time_limit_s=round(ep.max_ticks * 0.05, 1),
                    path_m=round(float(sp.sum() * 0.05), 2), idle_frac=round(float((sp < 0.02).mean()), 3),
                    events=[(e['t'], e['event'], e.get('object') or e.get('cue')) for e in ep.events if e['event'] in ('human_grasp', 'robot_grasp', 'human_cue', 'human_release', 'robot_release', 'wipe_start', 'wipe_end')],
                    tcp_at=[np.round(tcp[int(np.argmin(abs(t - x)))], 2).tolist() for x in (5, 10, 20, 30)]))
    print(seed, ok, reason, res[-1]['duration_s'], res[-1]['idle_frac'], flush=True)
a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))
