"""I2 expert (Instructor, FOLLOW reading). Both modes first stage above the cake. Observed: pick after a stable
pointing ray (5 frames) with a slowing hand. Forecast: pick as soon as the forecast is active (hand moving), then
while lifting follow the 1 s forecast end point mapped from the hand (shoulder frame) into the plate row. Both:
carry to the plate confirmed by the observed pointing ray once the hand is 22 cm from that plate, lower, release."""
import numpy as np

from intent_policy.experts.base_expert import BaseExpert
from intent_policy.scenarios.i2 import PLATES
from intent_policy.shared.sim_env import GRIP_OPEN, GRIP_CLOSED
from intent_policy.shared.episode import select_pointed


class Expert(BaseExpert):
    task, reading = 'I2', 'follow'

    def __init__(self, ep):
        super().__init__(ep)
        self.selected, self.observed, self.ready = None, [], 0
        self.anchor = None

    def observe(self):
        """Observed pointing: the same plate on the index ray for 5 frames; hand slowing for 3 frames."""
        ep = self.ep
        arm = ep.human.active
        ray = ep.human.keypoints(ep.pose, [arm + '_index_base', arm + '_index_tip'])
        visible = select_pointed(ray[0], ray[1], PLATES, max_distance_m=.065, margin_m=.01)
        self.observed = (self.observed + [visible])[-5:]
        if self.selected is None and len(self.observed) == 5 and self.observed[0] is not None and all(x == self.observed[0] for x in self.observed):
            self.selected = self.observed[0]; ep.event('target_observed', plate=self.selected)
        self.ready = self.ready + 1 if self.selected is not None and ep.hand_speed < .045 else 0

    def follow_goal(self):
        """1 s end point of the wrist in the shoulder frame, mapped from the anchor into the plate row."""
        ep = self.ep
        f = ep.forecast_for_expert()
        if f is None or not self.plan.active():
            return None
        shoulders = ep.kp[:2].mean(axis=0)
        future_rel = f[-1, ep.wrist_index()] - f[-1, :2].mean(axis=0)
        if self.anchor is None:
            self.anchor = (ep.wrist() - shoulders, ep.tcp()); ep.event('forecast_follow_started')
        g = self.anchor[1] + np.r_[(future_rel - self.anchor[0])[:2], 0.]
        xs = [v[0] for v in PLATES.values()]
        return np.array([np.clip(g[0], min(xs), max(xs)), np.clip(g[1], -.11, .08), .80])

    def wait(self, cond, phase, grip):
        while not cond():
            self.observe(); self.ep.phase = phase
            yield self.ep.hold(grip)
        self.observe()

    def goto(self, goal, phase, grip, follow=False):
        ep = self.ep
        while True:
            self.observe()
            g = self.follow_goal() if follow else None
            if g is None:
                g = goal
                if ep.reached(g):
                    return
                yield ep.command_to(g, phase, grip)
            else:
                yield self.plan.follow_step(g, phase, grip)
                if self.selected is not None and np.linalg.norm(ep.wrist() - PLATES[self.selected]) > .22:
                    return

    def run(self):
        ep = self.ep
        src = ep.pos('cake')
        yield from self.goto(np.r_[src[:2], .80], 'above_cake', GRIP_OPEN)
        if self.forecast_mode:
            yield from self.wait(lambda: self.plan.active() or (self.selected is not None and self.ready >= 3), 'wait_cue', GRIP_OPEN)
            ep.event('pick_start', basis='forecast' if self.selected is None else 'observed')
        else:
            yield from self.wait(lambda: self.selected is not None and self.ready >= 3, 'wait_cue', GRIP_OPEN)
            ep.event('pick_start', basis='observed')
        yield from self.goto(np.r_[src[:2], .667], 'descend', GRIP_OPEN)
        for cmd in ep.close_on('cake'):
            self.observe(); yield cmd
        yield from self.goto(np.r_[src[:2], .80], 'lift', GRIP_CLOSED, follow=self.forecast_mode)
        yield from self.wait(lambda: self.selected is not None and np.linalg.norm(ep.wrist() - PLATES[self.selected]) > .22,
                             'wait_target', GRIP_CLOSED)
        plate = PLATES[self.selected]; ep.event('transport_start', plate=self.selected)
        yield from self.goto(np.r_[plate[:2], .80], 'move_plate', GRIP_CLOSED)
        yield from self.goto(np.r_[plate[:2], .674], 'lower_plate', GRIP_CLOSED)
        for cmd in ep.open_gripper('release', ticks=20):
            self.observe(); yield cmd
        yield from self.goto(np.r_[plate[:2], .80], 'retreat', GRIP_OPEN)
        ep.phase = 'done'

    def success(self):
        ep = self.ep
        cake = ep.pos('cake')
        ok = (self.selected is not None and np.all(np.abs(cake[:2] - PLATES[self.selected][:2]) < [.035, .03])
              and ep.attached_name() is None and .63 < cake[2] < .71)
        return bool(ok), None if ok else 'cake_not_on_pointed_plate'

    def extra(self):
        return {'selected_from_observed_pointing': self.selected, 'human_motion_source': getattr(self.ep, 'human_motion_source', None)}
