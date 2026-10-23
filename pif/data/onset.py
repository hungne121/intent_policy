"""Movement onset / offset labels from a keypoint speed series (eval only: HABIT has no onset / t_clear labels,
CORRECTIONS #10).

Speed = |d position / dt| after zero-phase Savitzky-Golay smoothing (Savitzky & Golay 1964; the sim labels use the
same filter, configs/intent_schema.yaml `savgol`). A labelling pass may use the future: these are evaluation labels,
never model inputs (handoff §0.3-0.4).
Threshold = baseline mean + k x baseline SD, the baseline being a stretch where the person is known to wait (a
baseline-statistics criterion, one family of the onset methods compared by Teasdale et al. 1993, J. Motor Behavior
25(2) 97-106). A movement is a run above the threshold lasting at least `min_frames`; its onset / offset are the first /
last frame of the run.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter


def smoothed_speed(pos: np.ndarray, fps: float, window: int = 7, order: int = 2) -> np.ndarray:
    """[T] speed of a [T, D] position series (D = 2 image px or 3 metres); NaN frames are interpolated first."""
    pos = np.asarray(pos, float).copy()
    t = np.arange(len(pos))
    for d in range(pos.shape[1]):
        ok = np.isfinite(pos[:, d])
        if ok.sum() < 2:
            return np.full(len(pos), np.nan)
        pos[:, d] = np.interp(t, t[ok], pos[ok, d])
    if len(pos) >= window:
        vel = savgol_filter(pos, window, order, deriv=1, axis=0) * fps
    else:
        vel = np.gradient(pos, axis=0) * fps
    return np.linalg.norm(vel, axis=1)


def movements(speed: np.ndarray, baseline: slice, k: float = 3.0, min_frames: int = 3) -> tuple[list[tuple[int, int]], float]:
    """Runs of speed > mean + k * SD of speed[baseline] lasting >= min_frames: [(onset, offset)], threshold."""
    b = speed[baseline]
    thr = float(np.nanmean(b) + k * np.nanstd(b))
    above = np.r_[False, speed > thr, False]
    edges = np.flatnonzero(above[1:] != above[:-1])
    runs = [(int(a), int(e) - 1) for a, e in zip(edges[::2], edges[1::2]) if e - a >= min_frames]
    return runs, thr
