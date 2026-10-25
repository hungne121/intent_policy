"""Expert check of the T2 deep-task cases C1-C4 (docs/PIF.md §4): per episode the case, success / failure, the
expert's aborts (commit_log 'abort'), the human's case times (hesitate / own reach / cancel), and whether the robot had
really started towards the hand before it had to correct (closest TCP-palm xy distance before the abort).
  ./run.sh -m scripts.check_t2_cases --config configs/experiments/t2c1_early.yaml --seeds 300000 300016
"""
import argparse
from collections import Counter, defaultdict

import numpy as np

from intent_policy.benchmark.runner import ExpertAgent, run_episode
from intent_policy.scenarios.scenario_registry import make_scenario, scenario_overrides
from intent_policy.utils import controller_config, load_yaml, observation_config, resolve


class Watch(ExpertAgent):
    """The expert, also logging the TCP - palm xy distance per decision."""
    needs_observation = False

    def reset(self, scenario, mapper):
        super().reset(scenario, mapper)
        self.sc, self.dist = scenario, []

    def act(self, obs):
        sc, h = self.sc, self.sc.human
        palm = h.palm_at_zone(sc.human_context()) if hasattr(h, 'palm_at_zone') else h.pos
        self.dist.append((sc.time, float(np.linalg.norm(sc.tcp()[:2] - np.asarray(palm)[:2]))))
        return super().act(obs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--seeds', type=int, nargs=2, default=[300000, 300016])
    args = ap.parse_args()
    exp = load_yaml(resolve(args.config))
    sid = 't2_fetch_return'
    sc = make_scenario(sid, scenario_overrides(exp, sid))
    ctrl, obs = controller_config(exp), observation_config(exp)
    by = defaultdict(Counter)
    for seed in range(*args.seeds):
        agent = Watch()
        rec = run_episode(sc, agent, seed, ctrl, obs, keep_trace=False)
        case = sc.human.params.get('case')
        aborts = [c['t'] for c in agent.expert.commit_log if c['target'] == 'abort']
        ex = sc.human.extras
        before = [d for t, d in agent.dist if not aborts or t <= aborts[0]]
        started = round(min(before), 3) if before else None
        print(f"seed {seed} {case}: {'OK  ' if rec['success'] else 'FAIL ' + str(rec['failure'])} t={rec['duration_s']:.1f}s "
              f"aborts={aborts} cancel_t={ex.get('cancel_t')} own_reach_t={ex.get('own_reach_t')} "
              f"hesitate_t={ex.get('hesitate_t')} closest_tcp_palm_before_abort={started}", flush=True)
        by[case]['success' if rec['success'] else str(rec['failure'])] += 1
        by[case]['aborted'] += bool(aborts)
    sc.close()
    for case in sorted(by):
        print(case, dict(by[case]))


if __name__ == '__main__':
    main()
