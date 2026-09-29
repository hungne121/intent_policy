"""Re-run an episode from its saved record (configuration + seed) and verify it reproduces.

The scenario is rebuilt from the *saved* scenario configuration (not the current YAML files),
the same agent is re-created (scripted expert, or the recorded policy checkpoint) and the
new episode is compared with the saved one: variation, event stream, selected actions,
robot/human/object states and outcome.
"""
import argparse
import sys

import numpy as np
import torch

from benchmark.runner import ExpertAgent, NoisyExpertAgent, ObservationConfig, load_record, run_episode
from controllers.restricted_action import RestrictedActionConfig
from scenarios.scenario_registry import make_scenario


def rebuild_agent(record: dict, checkpoint: str | None = None, device: str | None = None):
    policy = record['policy']
    if policy.get('type') == 'scripted_expert':
        if policy.get('noise') == 'dart':
            return NoisyExpertAgent(policy['burst_prob'], policy['burst_ticks'])
        return ExpertAgent()
    from policies.policy_agent import PolicyAgent
    return PolicyAgent(checkpoint or policy['checkpoint'], device=device or policy.get('device'), oracle_condition=policy.get('oracle_spec'),
                       eval_seeds=policy.get('eval_seeds'), controller_cfg=RestrictedActionConfig(**record['controller']),
                       temporal_ensemble_coeff=policy.get('temporal_ensemble_coeff'))


def compare(a: dict, b: dict, atol: float = 1e-6) -> list[str]:
    problems = []
    if a['variation'] != b['variation']:
        problems.append('variation differs')
    ea = [(e['event_type'], round(e['timestamp'], 6), e['entity_id']) for e in a['events']]
    eb = [(e['event_type'], round(e['timestamp'], 6), e['entity_id']) for e in b['events']]
    if ea != eb:
        first = next((i for i, (x, y) in enumerate(zip(ea, eb)) if x != y), min(len(ea), len(eb)))
        problems.append(f'event stream differs at index {first}: {ea[first:first + 1]} vs {eb[first:first + 1]}')
    if (a['success'], a['failure']) != (b['success'], b['failure']):
        problems.append(f"outcome differs: {(a['success'], a['failure'])} vs {(b['success'], b['failure'])}")
    ta, tb = a.get('trace') or [], b.get('trace') or []
    if ta and tb:
        if [s['selected_action'] for s in ta] != [s['selected_action'] for s in tb]:
            problems.append('selected action sequence differs')
        n = min(len(ta), len(tb))
        worst = 0.0
        for sa, sb in zip(ta[:n], tb[:n]):
            for path in (('robot', 'q'), ('human', 'hand_position')):
                va, vb = sa['state'][path[0]][path[1]], sb['state'][path[0]][path[1]]
                worst = max(worst, float(np.max(np.abs(np.subtract(va, vb)))))
        if worst > atol:
            problems.append(f'state trajectories differ (max abs deviation {worst:.3g})')
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('record')
    p.add_argument('--checkpoint', help='override the checkpoint path stored in the record')
    p.add_argument('--device', help='override the inference device stored in the record')
    args = p.parse_args()
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    rec = load_record(args.record)
    sc = make_scenario(rec['scenario_config'])
    obs = rec['observation']
    new = run_episode(sc, rebuild_agent(rec, args.checkpoint, args.device), rec['seed'], RestrictedActionConfig(**rec['controller']),
                      ObservationConfig(tuple(obs['cameras']), obs['width'], obs['height']),
                      keep_trace=bool(rec.get('trace')), episode_id=rec['episode_id'])
    sc.close()
    problems = compare(rec, new)
    print(f"episode {rec['episode_id']} ({rec['scenario_id']}, seed {rec['seed']}): "
          f"saved success={rec['success']} / re-run success={new['success']}, events {len(rec['events'])}/{len(new['events'])}")
    if problems:
        print('NOT REPRODUCED:\n  - ' + '\n  - '.join(problems))
        sys.exit(1)
    print('REPRODUCED: identical variation, events, actions and states')


if __name__ == '__main__':
    main()
