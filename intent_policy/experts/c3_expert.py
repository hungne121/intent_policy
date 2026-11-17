"""C3 expert (Collaborator, FOLLOW reading). Observed: grasp the robot handle when the person's hand holds the
other handle; co-lift / lower tracking the observed hand height. Forecast: move above the robot handle when the
1 s forecast end point heads for the person's handle; co-lift tracking the 1 s end-point height at human pace."""
import numpy as np

from intent_policy.experts.base_expert import BaseExpert
from intent_policy.scenarios.c3 import HANDLE, TRAY_XY, LIFT, HANDLE_Z

TOOL_YAW_DEG = -90.
from intent_policy.shared.sim_env import GRIP_OPEN, GRIP_CLOSED, GRASP_DZ
from intent_policy.shared.episode import marks_left


class Expert(BaseExpert):
    task, reading = 'C3', 'follow'

    def __init__(self, ep):
        super().__init__(ep)
        # 55/QĐ-DL: close the fingers ACROSS the 2.4 cm grip bar (world Y), not along its 10 cm length (world X)
        ep.env.cfg['tool_yaw_deg'] = TOOL_YAW_DEG

    def run(self):
        ep = self.ep
        hxy = HANDLE['robot']
        handle_tcp = lambda: np.r_[ep.pos('tray')[:2] + (hxy - TRAY_XY), ep.pos('tray')[2] + (HANDLE_Z - .624) + GRASP_DZ]
        above = lambda: handle_tcp() + [0, 0, .10]
        near_handle = lambda: np.linalg.norm(ep.wrist()[:2] - HANDLE['human']) < .06 and ep.hand_speed < .03
        if self.forecast_mode:
            # FOLLOW reading (QD-DL-54): the 1 s forecast end point heads for the human handle -> move above the
            # robot handle at human pace, safety from the observed person in parallel (follow_step)
            yield from ep.wait(lambda: self.plan.heads_to(HANDLE['human']) or near_handle(), 'observe')
            ep.event('forecast_prep')
            while not near_handle() and not ep.reached(above(), .004):
                yield self.plan.follow_step(above(), 'approach')
        yield from ep.wait(lambda: near_handle() and 'tray' in ep.held_by_human, 'wait_human_grasp')
        ep.event('human_grasp_observed')
        yield from ep.goto(above, 'approach')
        yield from ep.goto(handle_tcp, 'descend')
        ok = yield from ep.close_on('tray')
        z0 = ep.tcp()[2]; palm0 = ep.wrist()[2]; w0 = ep.wrist()[:2].copy()
        # co-lift: track the human handle height. Observed follows the measured hand; Intent follows the 1 s
        # forecast end point (shared FOLLOW reading, QD-DL-54) at human pace (<= 0.8 x hand speed).
        def target():
            # 55/QĐ-DL: track the person's handle in xy as well as height, so the two grips never pull the tray apart
            f = self.plan.follow_point()
            p = f if f is not None else ep.wrist()
            dxy = np.clip(ep.wrist()[:2] - w0, -.03, .03)
            return np.r_[hxy + dxy, z0 + float(np.clip(p[2] - palm0, -.01, LIFT + .01))]
        while ep.human_phase != 'settle':
            yield ep.command_to(target(), 'colift', GRIP_CLOSED, following=self.forecast_mode)
        yield from ep.wait(lambda: 'tray' not in ep.held_by_human, 'settle', GRIP_CLOSED)
        yield from ep.open_gripper(ticks=25); ep.event('robot_release_handle')
        # rise straight off the grip bar where it actually is (55/QĐ-DL: a half-open gripper dragged the tray)
        up = np.r_[ep.tcp()[:2], ep.tcp()[2] + .12]
        yield from ep.goto(up, 'lift', GRIP_OPEN)
        yield from ep.goto(np.array([.25, .15, .82]), 'home')
        ep.phase = 'done'

    def success(self):
        ep = self.ep
        tray = ep.pos('tray')
        ok = (np.linalg.norm(tray[:2] - TRAY_XY) < .03 and abs(tray[2] - ep.initial['tray'][2]) < .01
              and marks_left(ep) == 0 and ep.attached_name() is None)
        return ok, None if ok else 'tray_or_marks'

    def extra(self):
        return {'marks_left': marks_left(self.ep)}
