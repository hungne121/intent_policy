"""Controlled C3 intervention on ManiCast motion tokens only (not a natural error).

Rotate a predicted arm around its predicted shoulder, gradually over the forecast
horizon. Bone lengths of each forecast pose are preserved. Raw observations,
human goals and human trajectories are never arguments to mutate. The schedule
is experiment metadata, never an input feature of ACT.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ForecastFault:
    start_s: float
    end_s: float
    yaw_deg: float
    arm: str
    ramp_s: float

    def __post_init__(self):
        if self.arm not in ('left', 'right', 'both'):
            raise ValueError('arm must be left, right or both')
        if not np.isfinite([self.start_s, self.end_s, self.yaw_deg, self.ramp_s]).all():
            raise ValueError('Fault parameters must be finite')
        if self.start_s < 0 or self.end_s <= self.start_s or self.ramp_s <= 0:
            raise ValueError('Fault needs a positive duration and ramp')


def corrupt_forecast(xi, current_keypoints, timestamps, horizons, fault: ForecastFault):
    """[T,M,6,3] displacements -> same shape, deterministic offline/online.

    The current joints follow MANICAST order. No confidence, goal label or fault
    flag is appended. `timestamps` are episode times, not window-local indices.
    This tests recovery from a controlled wrong direction, not physical feasibility
    of the entire forecast or empirically measured ManiCast error severity.
    """
    x = np.asarray(xi)
    now = np.asarray(current_keypoints)
    tt, hh = np.asarray(timestamps), np.asarray(horizons)
    if x.ndim != 4 or x.shape[2:] != (6, 3) or now.shape != (x.shape[0], 6, 3):
        raise ValueError('Expected xi [T,M,6,3] and current joints [T,6,3]')
    if tt.shape != (len(x),) or hh.shape != (x.shape[1],):
        raise ValueError('Timestamp/horizon length mismatch')
    if not all(np.isfinite(v).all() for v in (x, now, tt, hh)) or len(hh) == 0:
        raise ValueError('Nonfinite or empty forecast')
    if hh[0] <= 0 or np.any(np.diff(hh) <= 0):
        raise ValueError('Forecast horizons must be positive and increasing')
    strength = np.clip(np.minimum((tt - fault.start_s) / fault.ramp_s,
                                 (fault.end_s - tt) / fault.ramp_s), 0, 1)
    angle = np.deg2rad(fault.yaw_deg) * strength[:, None] * hh[None] / hh[-1]
    co, si = np.cos(angle), np.sin(angle)
    future = x.astype(np.float64) + now[:, None]
    result = future.copy()
    arms = [(0, 2, 4)] if fault.arm == 'left' else [(1, 3, 5)] if fault.arm == 'right' else [(0, 2, 4), (1, 3, 5)]
    for shoulder, elbow, wrist in arms:
        for joint in (elbow, wrist):
            rel = future[:, :, joint] - future[:, :, shoulder]
            result[:, :, joint, 0] = future[:, :, shoulder, 0] + co * rel[:, :, 0] - si * rel[:, :, 1]
            result[:, :, joint, 1] = future[:, :, shoulder, 1] + si * rel[:, :, 0] + co * rel[:, :, 1]
    out = (result - now[:, None]).astype(x.dtype)
    out[strength == 0] = x[strength == 0]  # bitwise unchanged outside the intervention
    return out
