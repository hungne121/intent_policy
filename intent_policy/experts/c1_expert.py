"""C1 expert (Collaborator, FOLLOW reading). Waits at the neutral point between the trays. Observed: when a cloth
is seen lifted, take that tray's cup, lift it clear towards the robot; return it once the cloth is spread and
the hand is back at rest. Forecast: as soon as the hand moves, drift along the 1 s forecast end point (x clipped
between the remaining cups, y -0.10, cruise height) at human pace with safety on the observed person; the cup is
fixed only when a cloth is seen lifted; the return starts when the forecast says the hand is withdrawing from the
spread cloth (user 16/11: accepted when correct). Before lowering the cup the observed hand must be 19 cm away."""
import numpy as np

from intent_policy.experts.base_expert import BaseExpert
from intent_policy.scenarios.c1 import NEUTRAL, CRUISE_Z
from intent_policy.shared.sim_env import GRIP_CLOSED, GRIP_OPEN, STATION


class Expert(BaseExpert):
    task, reading = 'C1', 'follow'

    def run(self):
        ep = self.ep
        # C1 work moves (approach, descend, lift, return, replace) have no wait-for-hand rule, as in the reviewed
        # 16/11 version: the person waits for the cup to be lifted, so a 4 cm wait deadlocks round 2 (16/11 port).
        # The forecast-following phase keeps its 0.16 m safety on the observed person (follow_step).
        ep.keep_clear = 0.
        done = []
        for _ in range(2):
            available = [n for n in ('cupL', 'cupR') if n not in done]
            def lifted():
                # Observed approach to the cloth is enough to prepare the matching cup.
                return [n for n in available if ep.pos('cloth' + n[-1])[2] > ep.initial['cloth' + n[-1]][2] + .018
                        or (ep.hand_speed > .03 and np.linalg.norm(np.asarray(ep.pose['palm']) - ep.pos('cloth' + n[-1]))
                            < STATION['task_tuning']['C1']['reach_cue_m'])]
            yield from ep.goto(NEUTRAL, 'prepare')
            if self.forecast_mode:
                started = False
                while not lifted():
                    f = self.plan.follow_point()
                    if f is None:
                        yield ep.hold(); ep.phase = 'wait_person'; continue
                    if not started:
                        started = True; ep.event('forecast_follow_started')
                    xs = [ep.initial[n][0] for n in available]
                    yield self.plan.follow_step(np.array([np.clip(f[0], min(xs), max(xs)), -.10, CRUISE_Z]), 'forecast_approach')
            else:
                yield from ep.wait(lambda: bool(lifted()), 'wait_person')
            cup = min(lifted(), key=lambda n: np.linalg.norm(ep.wrist() - ep.pos('cloth' + n[-1])))
            cloth = 'cloth' + cup[-1]
            ep.event('observed_cloth_target', cup=cup)
            xy = ep.initial[cup][:2]
            yield from ep.goto(np.r_[ep.pos(cup)[:2], CRUISE_Z], 'approach', tol=.020)
            yield from ep.goto(np.r_[xy, .688], 'descend')
            yield from ep.close_on(cup)
            yield from ep.goto(np.array([xy[0], .08, .84]), 'lift', GRIP_CLOSED, tol=.008)
            # hold until the cloth is spread and the hand is back (observed) or forecast to be withdrawing
            ep.phase = 'hold'; prev = None
            while True:
                cp = ep.pos(cloth)
                placed = np.linalg.norm(cp[:2] - xy) < .035 and cp[2] < .65
                wd = float(np.linalg.norm(ep.wrist() - cp))
                hand_back = (np.linalg.norm(np.asarray(ep.pose['palm']) - ep.human.hanging_palm(ep.human.active)) < .045
                             and ep.hand_speed < .04 and wd > .19)
                f = self.plan.follow_point()
                withdraw = (f is not None and prev is not None and wd > prev + .0005
                            and float(np.linalg.norm(f - cp)) > wd + .008)
                prev = wd
                palm_clear = np.linalg.norm(np.asarray(ep.pose['palm'])[:2] - xy) > STATION['task_tuning']['C1']['return_palm_clear_xy_m']
                if placed and cloth not in ep.held_by_human and palm_clear and (hand_back or withdraw):
                    ep.event('return_started', cup=cup, source='forecast_withdraw' if withdraw and not hand_back else 'hand_at_rest')
                    break
                yield ep.hold(GRIP_CLOSED)
            yield from ep.goto(np.r_[xy, CRUISE_Z], 'return_above', GRIP_CLOSED, tol=.020)
            yield from ep.wait(lambda: np.linalg.norm(ep.wrist() - ep.pos(cloth)) > .19, 'wait_hand_clear', GRIP_CLOSED)
            yield from ep.goto(np.r_[xy, .696], 'replace', GRIP_CLOSED)
            yield from ep.open_gripper('release', ticks=12)
            yield from ep.goto(np.r_[xy, CRUISE_Z], 'retreat', GRIP_OPEN, tol=.025)
            done.append(cup)
        yield from ep.goto(NEUTRAL, 'prepare')
        ep.phase = 'done'

    def success(self):
        ep = self.ep
        ok = all(np.linalg.norm(ep.pos('cloth' + s)[:2] - ep.initial['cup' + s][:2]) < .035
                 and np.linalg.norm(ep.pos('cup' + s)[:2] - ep.initial['cup' + s][:2]) < .03 for s in 'LR')
        ok = ok and ep.attached_name() is None
        return ok, None if ok else 'cloth_or_cup_misplaced'
