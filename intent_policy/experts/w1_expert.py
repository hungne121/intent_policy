"""W1 expert (Cowork): put one can in each receiving tray. 57/QĐ-DL: the expert only lists the valid remaining
operations (take the next can; put the held can in any tray still without one); when, where and in which order is
decided by the shared Predictive Minimum-Interruption Avoidance (`shared/predictive_avoidance.py`)."""
import numpy as np

from intent_policy.experts.base_expert import BaseExpert
from intent_policy.scenarios.w1 import CAN_SPOT, CAN_Z, in_tray
from intent_policy.shared.sim_env import TRAVEL_Z, GRASP_DZ
from intent_policy.shared.predictive_avoidance import PredictiveAvoider, pick_op, place_op

HOME = np.array([.40, .05, .82])


class Expert(BaseExpert):
    task, reading = 'W1', 'avoid'

    def __init__(self, ep):
        super().__init__(ep)
        self.avoid = PredictiveAvoider(ep)

    def has_can(self, tray):
        return any(in_tray(self.ep, f'can{k}', tray) for k in range(2))

    def operations(self):
        ep = self.ep
        free = [t for t in ('L', 'R') if not self.has_can(t)]
        held = ep.attached_name()
        if held is not None:
            return [place_op(ep, f'place:{held}:{t}', np.r_[CAN_SPOT[t], CAN_Z + .024], TRAVEL_Z,
                             {'name': 'can_placed', 'can': held, 'tray': t}, claim=f'tray_{t}') for t in free]
        cans = [f'can{k}' for k in range(2) if not any(in_tray(ep, f'can{k}', t) for t in ('L', 'R'))]
        return [pick_op(ep, cans[0], TRAVEL_Z, GRASP_DZ)] if cans and free else []

    def safe_pose(self):
        return HOME

    def run(self):
        ep = self.ep
        yield from self.avoid.run(self)
        yield from ep.goto(HOME, 'home')
        ep.phase = 'done'

    def success(self):
        ep = self.ep
        cans = sorted(t for t in ('L', 'R') if sum(in_tray(ep, f'can{k}', t) for k in range(2)) == 1)
        cakes = all(any(in_tray(ep, f'cake{i}', t) for t in ('L', 'R')) for i in range(4))
        ok = cans == ['L', 'R'] and cakes and ep.attached_name() is None
        return ok, None if ok else 'cans_or_cakes_misplaced'

    def extra(self):
        return {'avoid_decisions_ticks': dict(self.avoid.decisions)}
