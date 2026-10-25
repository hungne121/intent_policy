"""Gripper command over an episode (docs/reports/ACT_v6_failure_analysis.md): the policy's commanded opening, the
grasp / release events, and the assisted-grasp thresholds (configs/scene/common.yaml assisted_grasp).
  ./run.sh scripts/check_gripper_trace.py <checkpoint> <scenario> <seed> [temporal_ensemble_coeff|none]
"""
import sys
import numpy as np
from intent_policy.benchmark.runner import run_episode
from intent_policy.policies.policy_agent import PolicyAgent
from intent_policy.scenarios.scenario_registry import make_scenario, scenario_overrides
from intent_policy.utils import controller_config, load_yaml, observation_config

ckpt, sid, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
te = None if len(sys.argv) > 4 and sys.argv[4] == 'none' else float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
exp = load_yaml('configs/experiments/v6_early.yaml')
ctrl = controller_config(exp)
agent = PolicyAgent(ckpt, 'cuda', None, None, ctrl, te)
sc = make_scenario(sid, scenario_overrides(exp, sid))
rec = run_episode(sc, agent, seed, ctrl, observation_config(exp), keep_trace=True)
print('result', rec['success'], rec['failure'], 'assisted grasp: close_below', sc.env.close_below, 'open_above', sc.env.open_above)
g = np.array([t['mapped_controller_command']['joint_target'][6] for t in rec['trace']])
tt = np.array([t['t'] for t in rec['trace']])
for e in rec['events']:
    if e['event_type'] in ('object_grasp', 'object_release', 'task_failure') or e['payload'].get('step') in ('hand_ready', 'work_done'):
        k = int(np.searchsorted(tt, e['timestamp']))
        lo = max(0, k - 10)
        print(f"{e['timestamp']:6.2f}s {e['event_type']:24s} {e['entity_id']:6s} {e['payload'].get('step', '')}  "
              f"gripper cmd over the last 0.5 s: {np.round(g[lo:k + 1], 3).tolist()}")
sc.close()
