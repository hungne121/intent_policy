"""Watch a trained policy live (MuJoCo viewer): one episode per list task from an episode list, run exactly as
scripts/evaluate.py runs it (experiment scenario overrides, controller / observation config, temporal ensembling),
with the viewer open. Nothing is saved; the outcome of each episode is printed.

  ./run.sh -m scripts.live_policy --config configs/experiments/intent_act_device_early.yaml \
      --checkpoint <run>/train/D_f1/checkpoints/last/pretrained_model --list <run>/lists/eval.jsonl \
      --tasks T1 T2 T3 T4 T5
"""
import argparse
import time

from intent_policy.benchmark.runner import run_episode
from intent_policy.policies.policy_agent import PolicyAgent
from intent_policy.scenarios.config import load_scenario_list
from intent_policy.scenarios.scenario_registry import make_scenario, scenario_overrides
from intent_policy.utils import controller_config, load_yaml, observation_config, resolve


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--list', required=True)
    p.add_argument('--tasks', nargs='+', default=['T1', 'T2', 'T3', 'T4', 'T5'])
    p.add_argument('--nth', type=int, default=0, help='which episode of each task in the list (default: the first)')
    p.add_argument('--temporal-ensemble-coeff', type=float, default=0.01)
    p.add_argument('--pause', type=float, default=2.0, help='seconds to keep the viewer open after an episode')
    args = p.parse_args()
    exp = load_yaml(args.config)
    obs_cfg, ctrl_cfg = observation_config(exp), controller_config(exp)
    entries = load_scenario_list(resolve(args.list))
    agent = PolicyAgent(resolve(args.checkpoint), None, None, None, ctrl_cfg, args.temporal_ensemble_coeff)
    for task in args.tasks:
        es = [e for e in entries if e['task'] == task]
        if len(es) <= args.nth:
            print(f'{task}: no episode {args.nth} in the list')
            continue
        e = es[args.nth]
        ov = scenario_overrides(exp, e['scenario_id'])
        sc = make_scenario(e['scenario_id'], ov, gui=True)
        try:
            print(f'{task}: {e["scenario_id"]} seed {e["seed"]} spec {e["spec"]}', flush=True)
            rec = run_episode(sc, agent, int(e['seed']), ctrl_cfg, obs_cfg, keep_trace=False, spec=e['spec'])
            print(f'  -> success={rec["success"]} failure={rec["failure"]} T={rec["duration_s"]:.1f}s', flush=True)
            time.sleep(args.pause)
        finally:
            sc.close()


if __name__ == '__main__':
    main()
