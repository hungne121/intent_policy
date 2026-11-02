"""Diagnosis: I3 expert state over a time window (stage, armed, side, sideways wrist now / ahead, human motion).
  ./run.sh scripts/diag/trace_i3.py <rule> <seed> <t0> <t1> ['{overrides}']"""
import sys
from intent_policy.benchmark.runner import ExpertAgent, run_episode
from intent_policy.scenarios.scenario_registry import make_scenario
from intent_policy.utils import controller_config, load_yaml
from intent_policy.experts import i3_expert
import json
rule, seed, t0, t1 = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
ov = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
_act = i3_expert.I3Expert.act
last = [-1.0]
def act(self):
    out = _act(self)
    t = self.sc.time
    if t0 <= t <= t1 and t - last[0] >= 0.2:
        last[0] = t
        lat = self._lateral()
        seg = self.sc.human.segment
        print(f"{t:6.2f} moving {self._belief().moving} stage {self.stage:12s} ref {self.ref_x} side {self.side} item {self.item} "
              f"grasped {bool(self.sc.grasp_state.get(self.item))} lat {None if lat is None else tuple(round(x, 3) for x in lat)} "
              f"human {None if seg is None else seg.label} prev {self.prev} basket {[self.sc.basket_of(k) for k in self.sc.items]}", flush=True)
    return out
i3_expert.I3Expert.act = act
sc = make_scenario('i3_direct_sort', {**ov, 'expert': {'rule': rule}, 'timing': {'timeout_s': t1 + 1}})
rec = run_episode(sc, ExpertAgent(), seed, controller_config(load_yaml('configs/experiments/ten_task_v9.yaml')), keep_trace=False)
sc.close()
