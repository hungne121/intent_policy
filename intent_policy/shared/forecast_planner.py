"""Forecast-aware Cartesian planner shared by every expert (QĐ 53/QĐ-DL §3, QĐ 54/QĐ-DL).

One forecast, two readings ("tuy hai mà một", user 16/11):
  * FOLLOW (Instructor, Collaborator): the TCP reference tracks the 1 s end point of the forecast wrist
    trajectory plus a fixed task offset, re-evaluated every tick, at <= 0.8 x the observed hand speed.
    Safety runs in parallel on the OBSERVED person: hold while any robot part is within follow_keep_clear.
  * AVOID (Cowork): while working, a robot work point must lie outside the tube made of the observed wrist
    and the whole 0-1 s forecast wrist trajectory.
The FSM still confirms phase changes only from observed events. In the `observed` mode the planner gets no
forecast: FOLLOW readings return None and AVOID tests only the observed wrist.
"""
import numpy as np

from intent_policy.shared.sim_env import INTENT, BASE_XY

MIN_HAND_SPEED = float(INTENT['min_hand_speed_m_s'])
TARGET_RADIUS = float(INTENT['target_radius_m'])
FOLLOW_KEEP_CLEAR = float(INTENT['follow_keep_clear_m'])
AVOID_RADIUS = float(INTENT['avoid_radius_m'])
FOLLOW_INDEX = int(INTENT['follow_point_index'])
SAFETY_BAND = float(INTENT['safety_band_m'])
FOLLOW_GOAL_CLEAR = float(INTENT['follow_goal_clear_m'])
EMERGENCY_CLEAR = float(INTENT['emergency_clear_m'])


class ForecastPlanner:
    def __init__(self, episode):
        self.ep = episode

    # ---------------------------------------------------------------- inputs
    def active(self):
        """Forecast usable by the expert: forecast mode, valid forecast, observed hand moving. Reads only the
        validity, not the content, so it does not count as a forecast-use tick (59/QĐ-DL N5 trigger)."""
        return self.ep.forecast_available() is not None and self.ep.hand_speed > MIN_HAND_SPEED

    def wrist_trajectory(self):
        """[10, 3] forecast WORLD positions of the active wrist, or None."""
        f = self.ep.forecast_for_expert()
        return None if f is None else f[:, self.ep.wrist_index()]

    # ---------------------------------------------------------------- FOLLOW reading
    def follow_point(self, offset=(0., 0., 0.)):
        """1 s end point of the forecast wrist trajectory + a fixed task offset; None if inactive."""
        if not self.active():
            return None
        return self.wrist_trajectory()[FOLLOW_INDEX] + np.asarray(offset, float)

    def heads_to(self, xy, radius=TARGET_RADIUS):
        """FOLLOW reading as a cue: the 1 s end point lies within `radius` (xy) of `xy`."""
        f = self.follow_point()
        return f is not None and float(np.linalg.norm(f[:2] - np.asarray(xy, float)[:2])) < radius

    def nearest(self, candidates, radius=TARGET_RADIUS):
        """Key of the candidate xy nearest to the 1 s end point, if within `radius`."""
        f = self.follow_point()
        if f is None or not candidates:
            return None
        d = {k: float(np.linalg.norm(f[:2] - np.asarray(v, float)[:2])) for k, v in candidates.items()}
        k = min(d, key=d.get)
        return k if d[k] < radius else None

    def follow_step(self, goal, phase='follow', grip=None):
        """One FOLLOW command towards `goal` at human pace, with safety on the OBSERVED person (55/QĐ-DL: smooth).
        Closer than follow_keep_clear: back off along the arm-to-robot separation; inside the next 2 cm band: come
        to rest smoothly at the current position. The Cartesian follower is never reset, so no stop-go jerk."""
        ep = self.ep
        grip = ep.grip_open if grip is None else grip
        dist, pair = ep.validator.robot_near_human(include_held=True)
        if pair is not None and dist < FOLLOW_KEEP_CLEAR + SAFETY_BAND:
            if ep.phase != phase + '_safety':
                ep.hand_holds += 1
                ep.event('robot_yields_to_person', phase=phase, distance_m=round(dist, 4), pair=list(pair))
            if dist < EMERGENCY_CLEAR:
                # back off horizontally towards the robot base (always the side away from the person at this
                # station; a straight-up escape runs into the arm's reach limit), at work pace
                tcp = ep.tcp(); v = np.r_[np.asarray(BASE_XY) - tcp[:2], 0.]
                v /= max(float(np.linalg.norm(v)), 1e-9)
                return ep.command_to(tcp + v * (EMERGENCY_CLEAR - dist + .03), phase + '_safety', grip)
            return ep.command_to(ep.tcp(), phase + '_safety', grip, following=True)
        return ep.command_to(self.clear_goal(goal), phase, grip, following=True)

    def clear_goal(self, goal):
        """Shift a FOLLOW goal horizontally towards the robot base until it is FOLLOW_GOAL_CLEAR from the observed
        wrist, so the robot waits as close as the safety rule allows instead of approaching and backing off."""
        goal = np.asarray(goal, float).copy()
        w = self.ep.wrist()[:2]
        v = np.asarray(BASE_XY) - goal[:2]
        v /= max(float(np.linalg.norm(v)), 1e-9)
        for _ in range(40):
            if np.linalg.norm(goal[:2] - w) >= FOLLOW_GOAL_CLEAR:
                break
            goal[:2] += v * .01
        return goal

    # ---------------------------------------------------------------- AVOID reading
    def tube_hits(self, xy, radius=AVOID_RADIUS):
        """`xy` within `radius` of the observed wrist or (forecast mode) of any 0-1 s forecast wrist point."""
        pts = [self.ep.wrist()[:2]]
        if self.active():
            pts += list(self.wrist_trajectory()[:, :2])
        xy = np.asarray(xy, float)[:2]
        return any(float(np.linalg.norm(p - xy)) < radius for p in pts)
