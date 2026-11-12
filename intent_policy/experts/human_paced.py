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


def robot_travel_speed(human_speed_m_s: float, max_speed_m_s: float = 0.34) -> float:
    """Bounded arm transport pace, matched approximately to a human hand."""
    return float(np.clip(0.10 + 0.85 * max(0.0, human_speed_m_s), 0.14, max_speed_m_s))


def travel_ticks(distance_m: float, human_speed_m_s: float, dt: float,
                 minimum: int = 8, max_speed_m_s: float = 0.34) -> int:
    return max(minimum, int(np.ceil(max(0.0, distance_m) /
                                     (robot_travel_speed(human_speed_m_s, max_speed_m_s) * dt))))
