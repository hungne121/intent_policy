"""Unified intent state (TRIAL, 2026-10-18; docs/discussions/de_xuat_intent_thong_nhat.md, handover_som_va_intent_sai.md).

Not used by the pipeline yet: a new module so that the trial can be dropped by deleting it.

One distribution over what the human currently wants the robot to serve, built only from the measuring device's
outputs (intent/device.py; recorded as intent_pd.* fields):
  u(hold) = π            π = p_gesture(palm_forward): the human signals 'wait' now (T5 change of mind)
  u(g)    = (1 - π) b(g) b = p_target, the device's FETCH-POMDP belief over the schema targets (incl. `none`), which
                         keeps the last directed gesture's decision once the hand has stopped (decision lock)
The two events are exclusive: a 'wait' signal suspends the pending request (chain rule P(g) = P(not wait) P(g | not
wait)); summing to 1 holds by construction.

Time to ready (handover, T2): seconds until the open hand is in place, from the device's own onset of the palm-up
gesture (first frame with p_gesture(palm_up) >= 0.5). `ready_prior` is the prior term of the phase estimate of
Maeda et al. (IJRR 2017: phase ratio α ~ N(μ, σ²) learned from demonstrations): with no likelihood term the
remaining time is the prior duration minus the elapsed time.
"""
from __future__ import annotations

import numpy as np

HOLD = 'hold'


def unified_belief(p_target: np.ndarray, p_gesture: np.ndarray, classes: list[str]) -> np.ndarray:
    """(..., K) target belief + (..., C) device class probabilities -> (..., K + 1): targets then `hold`."""
    pi = np.asarray(p_gesture)[..., classes.index('palm_forward')]
    b = np.asarray(p_target, float)
    b = b / np.clip(b.sum(-1, keepdims=True), 1e-12, None)
    return np.concatenate([(1.0 - pi)[..., None] * b, pi[..., None]], -1)


def ready_prior(elapsed_s: float, durations_s: np.ndarray) -> tuple[float, float]:
    """Remaining time to ready [s] (mean, std) from the prior over gesture durations (the onset-to-ready times of
    other demonstrations), given the time elapsed since the device's onset."""
    d = np.asarray(durations_s, float)
    rem = d[d > elapsed_s] - elapsed_s                  # durations still compatible with the elapsed time
    if not len(rem):
        return 0.0, 0.0
    return float(np.median(rem)), float(np.std(rem))
