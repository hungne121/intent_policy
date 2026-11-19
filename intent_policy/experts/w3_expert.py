"""W3 expert (Cowork): pack the items within the robot's reach into the robot's box. 57/QĐ-DL: the expert only
lists the valid remaining operations (take any reachable unpacked item the person does not hold; put the held item
in the next free slot of the robot's box); the shared Predictive Minimum-Interruption Avoidance chooses."""
import numpy as np

from intent_policy.experts.base_expert import BaseExpert
from intent_policy.scenarios.w3 import BOX, SLOTS, ITEM_Z, packed
from intent_policy.shared.sim_env import TRAVEL_Z, GRASP_DZ, robot_reaches
from intent_policy.shared.predictive_avoidance import PredictiveAvoider, pick_op, place_op

HOME = np.array([.30, .02, .82])


class Expert(BaseExpert):
    task, reading = 'W3', 'avoid'

    def __init__(self, ep):
        super().__init__(ep)
        self.avoid = PredictiveAvoider(ep)
        self.n = 0

    def operations(self):
        ep = self.ep
        held = ep.attached_name()
        if held is not None:
            self.n = sum(1 for e in ep.events if e['event'] == 'robot_packed')     # next free slot of the robot's box
            return [place_op(ep, f'place:{held}', np.r_[BOX['robot'] + SLOTS[self.n], ITEM_Z + .024], TRAVEL_Z,
                             {'name': 'robot_packed', 'item': held}, claim=held)]
        return [pick_op(ep, f'item{i}', TRAVEL_Z, GRASP_DZ) for i in range(6)
                if robot_reaches(ep.pos(f'item{i}')) and not packed(ep, i) and f'item{i}' not in ep.held_by_human]

    def safe_pose(self):
        return HOME

    def run(self):
        ep = self.ep
        yield from self.avoid.run(self)
        yield from ep.goto(HOME, 'home')
        ep.phase = 'done'

    def success(self):
        ok = all(packed(self.ep, i) for i in range(6)) and self.ep.attached_name() is None
        return ok, None if ok else 'items_not_packed'

    def extra(self):
        return {'avoid_decisions_ticks': dict(self.avoid.decisions)}
