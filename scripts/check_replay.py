"""Replay test (docs/sim/PIPELINE_v6.md §6): a demo's recorded `action` (joint targets), executed open loop through the policy's continuous path in
the same scenario / seed / spec. Success = the action labels + simulator reproduce the demo.
  ./run.sh scripts/check_replay.py outputs/datasets/v6_early configs/experiments/v6_early.yaml <episodes per task>
"""
import json, sys
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from intent_policy.benchmark.runner import Agent, Decision, run_episode
from intent_policy.scenarios.config import _deep_merge
from intent_policy.scenarios.scenario_registry import make_scenario, scenario_overrides
from intent_policy.utils import controller_config, load_yaml, observation_config

root, exp_path, per = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
exp = load_yaml(exp_path)
meta = [json.loads(l) for l in open(root / 'meta/hri_episodes.jsonl')]
tab = pq.read_table(sorted((root / 'data').rglob('*.parquet'))[0], columns=['episode_index', 'frame_index', 'action']).to_pandas()

class Replay(Agent):
    action_mode, needs_observation = 'continuous', False
    def __init__(self, actions): self.a, self.k = actions, 0
    def reset(self, scenario, mapper): self.k = 0
    def act(self, obs):
        a = self.a[min(self.k, len(self.a) - 1)]; self.k += 1
        return Decision(joint_target=np.asarray(a, float))

ctrl = controller_config(exp)
by_sid = {}
for m in meta:
    by_sid.setdefault(m['scenario_id'], []).append(m)
for sid, ms in by_sid.items():
    sc = make_scenario(sid, scenario_overrides(exp, sid))
    res = []
    for m in ms[:per]:
        g = tab[tab.episode_index == m['episode_index']].sort_values('frame_index')
        acts = np.stack(g['action'].to_numpy())
        rec = run_episode(sc, Replay(acts), m['seed'], ctrl, observation_config(exp), keep_trace=False, spec=m['spec'])
        res.append((m['seed'], rec['success'], rec['failure'], len(acts), round(rec.get('duration_s') or 0, 2)))
    sc.close()
    print(sid, f"{sum(r[1] for r in res)}/{len(res)} replays succeed", [r for r in res if not r[1]][:4], flush=True)
