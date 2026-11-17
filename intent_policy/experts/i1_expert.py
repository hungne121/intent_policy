"""I1 expert (Instructor, FOLLOW reading). Each round: park above the nearest remaining cup (prepared, observable);
when the person's hand starts moving, grasp and lift the cup (both modes). Observed: hold the cup until the coaster
is down and the hand 15 cm away, then place. Forecast: while the coaster is carried / lowered, follow the 1 s
forecast end point + 8 cm towards the robot at human pace (safety on the observed person); place once the
coaster is down and the hand has left (observed)."""
import numpy as np

from intent_policy.experts.base_expert import BaseExpert
from intent_policy.scenarios.i1 import CUPS, COASTER_T
from intent_policy.shared.sim_env import TRAVEL_Z, GRIP_CLOSED, GRASP_DZ

FOLLOW_Z = .80
FOLLOW_CLEAR_Z = .84


class Expert(BaseExpert):
    task, reading = 'I1', 'follow'

    def run(self):
        ep = self.ep
        ep.robot_prepared = False; ep.coaster_down = [False] * 3
        remaining = list(CUPS)
        for r in range(3):
            cup = min(remaining, key=lambda c: np.linalg.norm(ep.pos(c)[:2] - ep.tcp()[:2])); remaining.remove(cup)
            co = f'coaster{2 - r}'
            yield from ep.goto(np.r_[ep.pos(cup)[:2], TRAVEL_Z], 'approach')
            ep.robot_prepared = True; ep.event('robot_prepared', cup=cup)
            yield from ep.wait(lambda: ep.hand_speed > .05, 'wait_hand_motion')     # observable cue
            ep.robot_prepared = False; ep.event('hand_motion_observed', round=r)
            yield from ep.pick(cup)
            yield from ep.goto(np.r_[ep.tcp()[:2], FOLLOW_Z], 'lift', GRIP_CLOSED)
            ep.event('robot_holds_cup', cup=cup)
            while not ep.coaster_down[r]:
                f = self.plan.follow_point(offset=(0., .08, 0.)) if ep.human_phase.startswith(('carry_', 'lower_')) else None
                if f is not None:
                    yield self.plan.follow_step(np.r_[f[:2], FOLLOW_CLEAR_Z], 'follow', GRIP_CLOSED)
                else:
                    yield ep.hold(GRIP_CLOSED)
            target = lambda: ep.pos(co)
            # wait for the hand to leave the coaster; yield (back off smoothly) if the withdrawing arm comes close
            while np.linalg.norm(np.asarray(ep.pose['palm'])[:2] - target()[:2]) <= .15:
                yield self.plan.follow_step(ep.tcp(), 'wait_hand_clear', GRIP_CLOSED)
            ep.event('hand_clear', round=r)
            place_z = target()[2] + COASTER_T / 2 + .037 + .008 + GRASP_DZ
            yield from ep.goto(lambda: np.r_[target()[:2], TRAVEL_Z], 'move', GRIP_CLOSED)
            yield from ep.goto(lambda: np.r_[target()[:2], place_z], 'lower', GRIP_CLOSED)
            yield from ep.open_gripper(); ep.event('cup_on_coaster', round=r, cup=cup)
            yield from ep.goto(lambda: np.r_[target()[:2], TRAVEL_Z], 'retreat')
        yield from ep.goto(np.array([.28, .05, .82]), 'home')
        ep.phase = 'done'

    def cups_on(self):
        ep = self.ep; n = 0
        for r in range(3):
            co = ep.pos(f'coaster{2 - r}')
            if any(np.linalg.norm(ep.pos(c)[:2] - co[:2]) < .02 and abs(ep.pos(c)[2] - (co[2] + COASTER_T / 2 + .037 + .008)) < .01 for c in CUPS):
                n += 1
        return n

    def success(self):
        n = self.cups_on()
        ok = n == 3 and self.ep.attached_name() is None and self.ep.phase == 'done'
        return ok, None if ok else f'cups_on_coasters_{n}_of_3'

    def extra(self):
        return {'cups_on_coasters': self.cups_on(), 'human_motion_source': getattr(self.ep, 'human_motion_source', None)}
