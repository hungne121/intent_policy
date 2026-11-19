"""C2 expert (Collaborator, FOLLOW reading). Takes the brush from the open hand, sweeps the dust under each lifted
object, gives the brush back. Forecast mode: heads for the handover point when the 1 s forecast says the hand goes
for the brush and follows the offering hand at grasp height; once the hand is free and the forecast heads for the
next object it waits at the robot-side standoff of that object (>= 0.16 m from the observed person)."""
import numpy as np

from intent_policy.experts.base_expert import BaseExpert
from intent_policy.scenarios.c2 import (HANDOVER, PALM_TO_CLOTH, ROBOT_GRIP_OFF, BRISTLE_HALF, SPOTS, STANDBY,
                                         SHELF_FLOOR, TOOL_YAW_DEG)
from intent_policy.shared.sim_env import TRAVEL_Z, GRIP_CLOSED, GRASP_DZ, STATION
from intent_policy.shared.episode import marks_left
from intent_policy.shared.forecast_planner import FOLLOW_KEEP_CLEAR, SAFETY_BAND


class Expert(BaseExpert):
    task, reading = 'C2', 'follow'

    def __init__(self, ep):
        super().__init__(ep)
        ep.env.cfg['tool_yaw_deg'] = TOOL_YAW_DEG

    def run(self):
        ep = self.ep
        self.wiped = {}          # expert FSM only; the person reads the dust marks (events.py, 09/QĐ-KT)
        # 1. take the cloth where the human offers it (observed cloth pose; forecast of the offering hand in Intent)
        if self.forecast_mode:
            # 16/11 (user): the robot must be ready when the person is. Start as soon as the forecast says the hand is
            # heading for the brush on its stand (observable: brush pose + ManiCast), pre-position above the handover
            # point at work speed, then follow the offering hand until it is still.
            # FOLLOW reading (54/QĐ-DL): the 1 s forecast end point heads for the brush on its stand
            yield from ep.wait(lambda: self.plan.heads_to(ep.pos('cloth')) or ep.human_phase in ('offer_cloth', 'wait_robot_grip'), 'wait_offer')
            ep.event('forecast_handover_prep')
            yield from ep.goto(HANDOVER + [*ROBOT_GRIP_OFF, GRASP_DZ + .06], 'approach', keep_clear=FOLLOW_KEEP_CLEAR,
                               until=lambda: ep.human_phase in ('offer_cloth', 'wait_robot_grip'))
            # follow the offering hand (1 s end point + hand-to-handle offset) just above the handle height, so only a
            # 3 cm descent remains when the hand stops; this is the deliberate handover approach (no keep-clear)
            grip_off = (0., PALM_TO_CLOTH[1] + ROBOT_GRIP_OFF[1], 0.)
            def handover_goal():
                f = self.plan.follow_point(grip_off)
                xy = f[:2] if f is not None else ep.pos('cloth')[:2] + ROBOT_GRIP_OFF
                return np.r_[xy, ep.pos('cloth')[2] + GRASP_DZ + .03]
            yield from ep.goto(handover_goal, 'approach', following=True,
                               until=lambda: ep.human_phase == 'wait_robot_grip', keep_clear=0.)
        else:
            yield from ep.wait(lambda: ep.human_phase in ('offer_cloth', 'wait_robot_grip'), 'wait_offer')
        yield from ep.wait(lambda: ep.human_phase == 'wait_robot_grip', 'wait_offer_still')
        cloth = lambda: ep.pos('cloth')
        # taking the brush from the open hand is a deliberate close approach: no keep-clear rule in these two moves
        if not self.forecast_mode:
            yield from ep.goto(lambda: np.r_[cloth()[:2] + ROBOT_GRIP_OFF, cloth()[2] + .06], 'approach', keep_clear=0.)
        yield from ep.goto(lambda: np.r_[cloth()[:2] + ROBOT_GRIP_OFF, cloth()[2] + GRASP_DZ], 'descend', keep_clear=0.)
        ok = yield from ep.close_on('cloth')
        yield from ep.wait(lambda: 'cloth' not in ep.held_by_human, 'wait_human_release', GRIP_CLOSED, timeout=40)
        yield from ep.goto(STANDBY, 'retreat', GRIP_CLOSED)
        for _ in range(2):
            # 2. which object is being lifted: observed lift, or in Intent the forecast hand heading
            def hand_free():
                # observable: no object is off its mark (the hand is not carrying anything)
                return all(ep.pos(o)[2] < ep.initial[o][2] + .02 for o in ('cup', 'clock'))
            def lifted():
                for o in ('cup', 'clock'):
                    if o not in self.wiped and ep.pos(o)[2] > ep.initial[o][2] + .07: return o
                return None
            if self.forecast_mode:
                # FOLLOW reading: while the hand is free, wait at the robot-side standoff (y +0.10, z 0.74; bristles at
                # y ~ +0.025, the hand never comes to +y) of the object nearest the 1 s forecast end point, re-evaluated
                # every tick; safety from the observed person in parallel (follow_step, >= 0.16 m)
                # 58/QĐ-DL: a robot at rest (standby or the standoff) stays put when the person comes near the brush
                # (the person keeps clear); the stop / back-off rule applies only while it is moving
                hover, resting = None, False
                while lifted() is None:
                    cand = {o: ep.pos(o) for o in ('cup', 'clock') if o not in self.wiped} if hand_free() else {}
                    h = self.plan.nearest(cand)
                    if h is not None and h != hover:
                        hover, resting = h, False; ep.event('forecast_prep', object=h)
                    if hover is not None and not resting:
                        goal = np.array([ep.pos(hover)[0], STATION['task_tuning']['C2']['prep_y_m'], STATION['task_tuning']['C2']['prep_z_m']])
                        dist, _ = ep.validator.robot_near_human(include_held=True)
                        still = float(np.linalg.norm(ep.controller.follower.velocity)) < .01
                        if ep.reached(goal, .015):
                            resting = True; ep.event('robot_rests', object=hover)
                        elif still and dist < FOLLOW_KEEP_CLEAR + SAFETY_BAND:
                            # at rest (standby) with the person near the brush: stay put, the person keeps clear
                            if ep.phase != 'rest_person_near':
                                ep.event('robot_stays_for_person', distance_m=round(dist, 4))
                            ep.phase = 'rest_person_near'
                        else:
                            yield self.plan.follow_step(goal, 'approach', GRIP_CLOSED); continue
                    yield ep.hold(GRIP_CLOSED)
            yield from ep.wait(lambda: lifted() is not None, 'wait_lift', GRIP_CLOSED)
            obj = lifted()
            spots = [np.array(s) for s in SPOTS[obj]]
            # enter the wipe area only once the lifting hand is clearly out of it (observable), never under the hand
            def hand_clear():
                w = ep.wrist()
                return w[2] > .85 and all(np.linalg.norm(w[:2] - s) > .12 for s in spots)
            yield from ep.wait(hand_clear, 'wait_hand_clear', GRIP_CLOSED)
            ep.event('wipe_start', object=obj)
            wipe_z = SHELF_FLOOR + BRISTLE_HALF[2] + GRASP_DZ + .001   # bristle bottom ~1 mm above the shelf floor
            # Cross the shelf wall while the whole gripper is above it, then lower vertically.
            yield from ep.goto(np.r_[spots[0] + ROBOT_GRIP_OFF, .74], 'move', GRIP_CLOSED)
            yield from ep.goto(np.r_[spots[0] + ROBOT_GRIP_OFF, wipe_z], 'wipe_lower', GRIP_CLOSED)
            # sweep 1.5 cm past both end spots so the outer dust columns pass under the bristles (goto tol 1.2 cm)
            ends = [spots[0] + [-.015, 0.], spots[-1] + [.015, 0.]]
            for s in [ends[0]] + spots + [ends[1]] + spots[::-1][1:] + [ends[0]]:
                # Wipe marks have a 35 mm effect radius; reaching within 12 mm is
                # sufficient and avoids freezing against the shelf's IK boundary.
                for cmd in ep.goto(np.r_[s + ROBOT_GRIP_OFF, wipe_z], 'wipe', GRIP_CLOSED, tol=.012):
                    yield cmd
            self.wiped[obj] = True; ep.event('wipe_end', object=obj, marks_left=marks_left(ep))
            yield from ep.goto(np.r_[spots[-1] + ROBOT_GRIP_OFF, .74], 'lift', GRIP_CLOSED)
            yield from ep.goto(STANDBY, 'retreat', GRIP_CLOSED)
        # 3. return the cloth: present it at the handover point and release when the human holds it
        yield from ep.wait(lambda: ep.human_phase in ('rest', 'wait_cloth') and ep.pos('clock')[2] < ep.initial['clock'][2] + .02
                           and ep.pos('cup')[2] < ep.initial['cup'][2] + .02, 'wait_objects_back', GRIP_CLOSED)
        yield from ep.goto(HANDOVER + [*ROBOT_GRIP_OFF, GRASP_DZ + .06], 'move', GRIP_CLOSED)
        # handing the towel back is a deliberate close approach to the open hand: no keep-clear rule here
        yield from ep.goto(HANDOVER + [*ROBOT_GRIP_OFF, GRASP_DZ], 'present_cloth', GRIP_CLOSED, keep_clear=0.)
        yield from ep.wait(lambda: 'cloth' in ep.held_by_human and ep.human_phase == 'wait_robot_release', 'present_cloth', GRIP_CLOSED)
        yield from ep.open_gripper(); ep.event('return_handover')
        # back off upwards and towards the robot side (the straight-up point was at the arm's reach limit)
        yield from ep.goto(HANDOVER + [0, .10, GRASP_DZ + .06], 'retreat', keep_clear=0.)
        yield from ep.goto(np.array([.25, .05, .82]), 'home')
        ep.phase = 'done'

    def success(self):
        ep = self.ep
        back = all(np.linalg.norm(ep.pos(o)[:2] - ep.initial[o][:2]) < .02 and abs(ep.pos(o)[2] - ep.initial[o][2]) < .01
                   for o in ('cup', 'clock'))
        on_stand = np.linalg.norm(ep.pos('cloth')[:2] - ep.initial['cloth'][:2]) < .03 and abs(ep.pos('cloth')[2] - ep.initial['cloth'][2]) < .02
        ok = back and marks_left(ep) == 0 and on_stand and ep.attached_name() is None
        return ok, None if ok else 'dust_objects_or_brush'

    def extra(self):
        return {'marks_left': marks_left(self.ep)}
