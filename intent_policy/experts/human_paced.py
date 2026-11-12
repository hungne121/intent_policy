"""Causal speed estimate for preview experts; not a task or case label."""

import numpy as np


def observed_hand_speed(history: list[np.ndarray], wrist_index: int, dt: float) -> float:
    """Median recent wrist speed in m/s from available observations."""
    if len(history) < 2:
        return 0.0
    wrist = np.asarray(history[-5:], dtype=float)[:, wrist_index]
    return float(np.median(np.linalg.norm(np.diff(wrist, axis=0), axis=1) / dt))


def forecast_hand_speed(displacements: np.ndarray, wrist_index: int, horizons_s: np.ndarray) -> float:
    """Median predicted wrist speed from ManiCast's successive future positions."""
    future = np.asarray(displacements, dtype=float)[:, wrist_index]
    times = np.asarray(horizons_s, dtype=float)
    if len(times) < 2 or future.shape[0] != len(times) or np.any(np.diff(times) <= 0):
        raise ValueError('forecast and strictly increasing horizons must align')
    return float(np.median(np.linalg.norm(np.diff(future, axis=0), axis=1) / np.diff(times)))


# Independent robot work (approach after a confirmed cue, grasp, lift, transport, place,
# retreat, undo after a change of mind) runs at the same fixed pace in BOTH experts.
# Horizontal pace matches the I1 preview (continuous_motion tcp_speed 0.39 m/s, accel 1.0);
# vertical contact motion stays slower (0.34 m/s with a held cake overlapped the finger 6 mm,
# docs/MANICAST_SPEED_CHECK.md). Human-paced following is reserved for the Intent expert
# while it tracks a ManiCast forecast before the cue is confirmed (user, 2026-11-13 04:47).
WORK_SPEED = {'horizontal': .36, 'vertical': .18}
WORK_ACCEL = 1.0
VERTICAL_PHASES = ('descend', 'lower_plate', 'return_lower', 'lower', 'replace', 'place_lower', 'wipe_lower')


def work_speed(phase: str) -> float:
    """Fixed independent pace for a robot phase (same in Observed and Intent)."""
    return WORK_SPEED['vertical'] if phase in VERTICAL_PHASES else WORK_SPEED['horizontal']


def robot_travel_speed(human_speed_m_s: float, max_speed_m_s: float = 0.34) -> float:
    """Bounded arm transport pace, matched approximately to a human hand."""
    return float(np.clip(0.10 + 0.85 * max(0.0, human_speed_m_s), 0.14, max_speed_m_s))


def travel_ticks(distance_m: float, human_speed_m_s: float, dt: float,
                 minimum: int = 8, max_speed_m_s: float = 0.34) -> int:
    return max(minimum, int(np.ceil(max(0.0, distance_m) /
                                     (robot_travel_speed(human_speed_m_s, max_speed_m_s) * dt))))


class HumanPace:
    """Causal pace shared by preview experts. Retain recent pace after the cue ends.

    During observed motion the goal speed is 80% of filtered hand speed. Forecast
    speed may anticipate the trend but cannot raise it above observed hand speed.
    Once the human holds/withdraws, keep a bounded recent work pace so placement
    does not freeze merely because the human has stopped giving the instruction.
    """
    def __init__(self, dt=.05, ratio=.8):
        self.dt = float(dt)
        self.ratio = float(ratio)
        self.filtered = 0.
        self.recent = .16
        self.active = False

    def update(self, observed, forecast=0.):
        observed = max(0., float(observed))
        forecast = max(0., float(forecast))
        estimate = min(observed, .8 * observed + .2 * forecast) if forecast > 0 else observed
        self.filtered += (1. - np.exp(-self.dt / .12)) * (estimate - self.filtered)
        self.active = observed > .015
        if observed > .06:
            self.recent = float(np.clip(.85 * self.recent + .15 * self.ratio * self.filtered, .14, .24))
        return self.speed()

    def speed(self, cap=.24, following=False):
        desired = self.ratio * self.filtered if following and self.active else self.recent
        return float(np.clip(desired, .008 if following else .04, cap))


class CartesianFollower:
    """Continuous Cartesian setpoint with bounded speed and acceleration."""
    def __init__(self, dt=.05, acceleration=.65):
        self.dt = float(dt)
        self.acceleration = float(acceleration)
        self.velocity = np.zeros(3)
        self.position = None

    def step(self, current, goal, speed):
        actual, goal = np.asarray(current, float), np.asarray(goal, float)
        if self.position is None:
            self.position = actual.copy()
        lag = self.position - actual
        if np.linalg.norm(lag) > .04:
            self.position = actual + lag * (.04 / np.linalg.norm(lag))
        current = self.position
        delta = goal - current
        distance = float(np.linalg.norm(delta))
        if distance < .0005:
            self.velocity *= .5
            self.position = goal.copy()
            return self.position.copy()
        target_speed = min(max(0., float(speed)), np.sqrt(2 * self.acceleration * distance))
        desired = delta / distance * target_speed
        change = desired - self.velocity
        norm = np.linalg.norm(change)
        if norm > self.acceleration * self.dt:
            change *= self.acceleration * self.dt / norm
        self.velocity += change
        step = self.velocity * self.dt
        if np.linalg.norm(step) >= distance and np.dot(step, delta) > 0:
            self.velocity = delta / self.dt
            self.position = goal.copy()
            return self.position.copy()
        self.position = current + step
        return self.position.copy()
