"""W2 expert (Cowork): sort the colour blocks within the robot's reach into their trays. 57/QĐ-DL: the expert only
lists the valid remaining operations (take any reachable unsorted block the person does not hold; put the held
block in its colour tray); the shared Predictive Minimum-Interruption Avoidance chooses among them."""
import numpy as np

from intent_policy.experts.base_expert import BaseExpert
from intent_policy.scenarios.w2 import TRAY_SLOTS, BLOCK_Z, color, sorted_block, free_slot
from intent_policy.shared.sim_env import TRAVEL_Z, GRASP_DZ, robot_reaches
from intent_policy.shared.predictive_avoidance import PredictiveAvoider, pick_op, place_op

HOME = np.array([.28, .05, .82])


class Expert(BaseExpert):
    task, reading = 'W2', 'avoid'

    def __init__(self, ep):
        super().__init__(ep)
        self.avoid = PredictiveAvoider(ep)

    def operations(self):
        ep = self.ep
        held = ep.attached_name()
        if held is not None:
            i = int(held[5:])
            return [place_op(ep, f'place:{held}', np.r_[free_slot(ep, color(i), TRAY_SLOTS[::-1], held), BLOCK_Z + .024], TRAVEL_Z,
                             {'name': 'robot_placed', 'block': held}, claim=held)]
        return [pick_op(ep, f'block{i}', TRAVEL_Z, GRASP_DZ) for i in range(6)
                if robot_reaches(ep.pos(f'block{i}')) and not sorted_block(ep, i) and f'block{i}' not in ep.held_by_human]

    def safe_pose(self):
        return HOME

    def run(self):
        ep = self.ep
        ep.claimed = None
        yield from self.avoid.run(self)
        yield from ep.goto(HOME, 'home')
        ep.phase = 'done'

    def success(self):
        ep = self.ep
        ok = all(sorted_block(ep, i) for i in range(6)) and ep.attached_name() is None
        return ok, None if ok else 'blocks_not_sorted'

    def extra(self):
        return {'avoid_decisions_ticks': dict(self.avoid.decisions)}
