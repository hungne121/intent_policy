"""Diagnosis: one expert episode of a scenario, printed (success, duration, failure reasons, commits, recoveries, the
human's cues / changes and contacts). Not a collection.
  ./run.sh scripts/diag/smoke_one.py <scenario> <rule> <seed> ['{"overrides": ...}']
"""
import json
import sys

import numpy as np

from intent_policy.benchmark.runner import ExpertAgent, run_episode
from intent_policy.scenarios.scenario_registry import make_scenario
from intent_policy.utils import controller_config, load_yaml

scen, rule, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
ov = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
ov = {**ov, 'expert': {**ov.get('expert', {}), 'rule': rule}}
exp = load_yaml('configs/experiments/ten_task_v9.yaml')
sc = make_scenario(scen, ov)
agent = ExpertAgent()
LAT = {}                                    # I3: the largest sideways wrist displacement per human motion label
if scen == 'i3_direct_sort':
    from intent_policy.experts import i3_expert
    _lat = i3_expert.I3Expert._lateral
    def _logged(self):
        out = _lat(self)
        seg = self.sc.human.segment
        if out is not None and seg is not None:
            LAT[seg.label] = max(LAT.get(seg.label, 0.0), abs(out[0]))
        return out
    i3_expert.I3Expert._lateral = _logged
rec = run_episode(sc, agent, seed, controller_config(exp), keep_trace=False)
if LAT:
    print('  max |sideways wrist displacement| per motion (m):', {k: round(v, 3) for k, v in LAT.items()})
print(f"{scen} {rule} seed {seed}: success {rec['success']} duration {rec['duration_s']:.2f} s "
      f"failure {rec.get('failure_reasons') or rec.get('failure')}")
KEEP = ('human_cue_onset', 'human_intention_change', 'protocol_step_complete', 'human_yield_to_robot', 'contact',
        'human_robot_contact', 'failure')
for e in rec['events']:
    if e['event_type'] in KEEP or 'contact' in e['event_type'] or 'fail' in e['event_type']:
        pl = {k: v for k, v in (e.get('payload') or {}).items() if k != 'motion_source'}
        print(f"  {e['timestamp']:7.2f} {e['event_type']} {pl}")
ex = agent.expert
print('  commits', getattr(ex, 'commit_log', None))
print('  recoveries', getattr(sc, 'log_recovery', None))
sc.close()
