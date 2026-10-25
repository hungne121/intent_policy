"""Watch a trained policy (or the scripted expert) live in the MuJoCo viewer, with the same agent, controller,
observation, inference setting and scenario overrides as scripts/evaluate.py; events and the outcome are printed as
they happen. The policy's cameras are rendered off-screen as in the evaluation (MUJOCO_GL, EGL by default).
  DISPLAY=:0 ./run.sh -m scripts.watch_policy --config configs/experiments/v6_early.yaml \\
      --checkpoint outputs/train_v6/2026-10-25_pod/run/B/checkpoints/006650/pretrained_model \\
      --scenario t2_fetch_return --seeds 400000 400001 --temporal-ensemble-coeff 0.01
  (D: add --intent-online manicast; A: --config configs/experiments/v6_late.yaml; --expert: the scripted expert)
Close the viewer window to stop.
"""
import argparse
import time

from intent_policy.benchmark.runner import Agent, ExpertAgent, run_episode
from intent_policy.scenarios.scenario_registry import make_scenario, scenario_overrides
from intent_policy.sim.viewer_lifecycle import ViewerClosed
from intent_policy.utils import controller_config, load_yaml, observation_config, resolve


class Paced(Agent):
    """Wraps an agent: prints new scenario events and keeps the episode at `speed` x real time."""

    def __init__(self, agent, speed: float):
        self.agent, self.speed = agent, speed
        self.action_mode, self.needs_observation = agent.action_mode, agent.needs_observation

    def reset(self, scenario, mapper):
        self.sc, self.seen, self.t0 = scenario, 0, time.monotonic()
        self.agent.reset(scenario, mapper)

    def act(self, obs):
        d = self.agent.act(obs)
        events = self.sc.logger.events
        for e in events[self.seen:]:
            print(f"  {e.timestamp:6.2f}s  {e.event_type}  {e.entity_id}  {e.payload or ''}", flush=True)
        self.seen = len(events)
        lag = self.sc.time / self.speed - (time.monotonic() - self.t0)
        if lag > 0:
            time.sleep(lag)
        return d


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', help='pretrained_model dir; omit with --expert')
    p.add_argument('--expert', action='store_true')
    p.add_argument('--scenario', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[400000])
    p.add_argument('--temporal-ensemble-coeff', type=float, default=None)
    p.add_argument('--intent-online', choices=['manicast'])
    p.add_argument('--speed', type=float, default=1.0, help='playback speed (1 = real time)')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    exp = load_yaml(resolve(args.config))
    ctrl, obs_cfg = controller_config(exp), observation_config(exp)
    if args.expert:
        agent = ExpertAgent()
    else:
        from intent_policy.policies.policy_agent import PolicyAgent
        te = 'config' if args.temporal_ensemble_coeff is None else args.temporal_ensemble_coeff
        agent = PolicyAgent(resolve(args.checkpoint), args.device, None, None, ctrl, te, intent_online=args.intent_online)
    sc = make_scenario(args.scenario, scenario_overrides(exp, args.scenario), gui=True)
    try:
        for seed in args.seeds:
            print(f'== {args.scenario} seed {seed}', flush=True)
            rec = run_episode(sc, Paced(agent, args.speed), seed, ctrl, obs_cfg, keep_trace=False)
            print(f"== result: {'SUCCESS' if rec['success'] else 'FAIL ' + str(rec['failure'])}  "
                  f"t={rec.get('duration_s') or 0:.2f}s", flush=True)
            time.sleep(2.0)
    except ViewerClosed:
        print('viewer closed')
    finally:
        sc.close()


if __name__ == '__main__':
    main()
