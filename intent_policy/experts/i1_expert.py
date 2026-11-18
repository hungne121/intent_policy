"""I1 expert (Instructor, FOLLOW reading). Each round: park above the nearest remaining cup (prepared, observable);
when the person's hand starts moving, grasp and lift the cup (both modes). Observed: hold the cup until the coaster
is down and the hand 15 cm away, then place. Forecast (68/QĐ-DL B.2, 54/QĐ-DL): once the coaster is off the stack
(observed), the free spot nearest the forecast palm (1 s wrist end point + observed palm-wrist offset, within
target_radius) is the target; the robot moves in one diagonal move to a ready point on its side of that spot and
re-targets whenever the forecast points at another spot (safety on the observed person, follow_step). Once the coaster
is down (observed) it keeps closing in on the ready point of the real coaster, lowered to just above the place height,
while the hand withdraws; once the hand has left (observed), one move to just above the coaster, then lower."""
import numpy as np

from intent_policy.experts.base_expert import BaseExpert
from intent_policy.scenarios.i1 import CUPS, COASTER_T, SPOTS
from intent_policy.shared.sim_env import TRAVEL_Z, GRIP_CLOSED, GRASP_DZ
from intent_policy.shared.forecast_planner import TARGET_RADIUS, FOLLOW_GOAL_CLEAR

FOLLOW_Z = .80
FOLLOW_CLEAR_Z = .84
READY_DY = FOLLOW_GOAL_CLEAR       # 68/QĐ-DL B.2: ready point this far on the robot side (+y) of the forecast spot
READY_Z = FOLLOW_CLEAR_Z           # same height as the former forecast following
PLACE_PRE_DZ = .03                 # forecast placement: diagonal move ends this far above the place height, then lower


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
            if self.forecast_mode:
                yield from self.prepare(r, co)
            else:
                while not ep.coaster_down[r]:
                    yield ep.hold(GRIP_CLOSED)
            target = lambda: ep.pos(co)
            # wait for the hand to leave the coaster; yield (back off smoothly) if the withdrawing arm comes close.
            # Forecast: keep closing in on the ready point of the coaster now observed down, lowered to just above
            # the place height (same safety)
            while np.linalg.norm(np.asarray(ep.pose['palm'])[:2] - target()[:2]) <= .15:
                goal = ep.tcp() if not self.forecast_mode else self.ready_point(
                    target(), target()[2] + COASTER_T / 2 + .037 + .008 + GRASP_DZ + PLACE_PRE_DZ)
                yield self.plan.follow_step(goal, 'wait_hand_clear', GRIP_CLOSED)
            ep.event('hand_clear', round=r)
            place_z = target()[2] + COASTER_T / 2 + .037 + .008 + GRASP_DZ
            if self.forecast_mode:          # from the ready point: one move to just above the coaster
                yield from ep.goto(lambda: np.r_[target()[:2], place_z + PLACE_PRE_DZ], 'move', GRIP_CLOSED)
            else:
                yield from ep.goto(lambda: np.r_[target()[:2], TRAVEL_Z], 'move', GRIP_CLOSED)
            yield from ep.goto(lambda: np.r_[target()[:2], place_z], 'lower', GRIP_CLOSED)
            yield from ep.open_gripper(); ep.event('cup_on_coaster', round=r, cup=cup)
            yield from ep.goto(lambda: np.r_[target()[:2], TRAVEL_Z], 'retreat')
        yield from ep.goto(np.array([.28, .05, .82]), 'home')
        ep.phase = 'done'

    def free_spots(self, carried):
        """Coaster spots with no coaster on them yet (observed positions of the coasters other than the carried one)."""
        ep = self.ep
        down = [ep.pos(f'coaster{i}')[:2] for i in range(3) if f'coaster{i}' != carried]
        return {i: s for i, s in enumerate(SPOTS) if all(np.linalg.norm(c - s) > .03 for c in down)}

    def forecast_spot(self, free):
        """FOLLOW reading as a target cue (54/QĐ-DL): free spot nearest the forecast palm (1 s wrist end point + the
        observed palm-wrist offset; the coaster is under the palm), if within target_radius; None if no forecast."""
        ep = self.ep
        f = self.plan.follow_point()
        if f is None or ep.pose is None or not free:
            return None
        palm = f + (np.asarray(ep.pose['palm'], float) - ep.wrist())
        d = {i: float(np.linalg.norm(palm[:2] - s)) for i, s in free.items()}
        i = min(d, key=d.get)
        return i if d[i] < TARGET_RADIUS else None

    def prepare(self, r, co):
        """Forecast mode, until the coaster of round `r` is down: from the moment it leaves the stack (observed),
        go to the ready point of the forecast spot (one straight move at human pace, observed-person safety in
        follow_step); a new forecast spot re-targets, no forecast keeps the last ready point."""
        ep = self.ep
        spot = None
        while not ep.coaster_down[r]:
            if np.linalg.norm(ep.pos(co) - ep.initial[co]) > .01:          # coaster off the stack: being carried
                guess = self.forecast_spot(self.free_spots(co))
                if guess is not None and guess != spot:
                    spot = guess
                    ep.event('ready_target', round=r, spot=SPOTS[spot].round(3).tolist())
            if spot is None:
                yield ep.hold(GRIP_CLOSED)
            else:
                yield self.plan.follow_step(self.ready_point(SPOTS[spot]), 'prepare', GRIP_CLOSED)

    @staticmethod
    def ready_point(xy, z=READY_Z):
        """Ready point of a coaster spot: READY_DY on the robot side (+y) of it, at height `z`."""
        return np.r_[xy[0], xy[1] + READY_DY, z]

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
