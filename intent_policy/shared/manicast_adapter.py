"""ManiCast adapter shared by every task (QĐ 53/QĐ-DL §3, §5; 59/QĐ-DL §D).

Input: the last 9 frames (0.4 s at 20 Hz) of the 6 ManiCast joints (both shoulders, elbows, wrists) in world
coordinates. Output: H_t in R^{10 x 6 x 3}, the WORLD positions of all 6 joints at 0.1 ... 1.0 s ahead, plus a
validity flag, the time the forecast was computed and its age. Never a ground-truth future.

`raw_forecast` / `raw_offsets` are ManiCast's own output (world positions / displacements from the current pose).
`forecast` / `offsets` are the EFFECTIVE forecast: equal to the raw one, except inside an N5 fault window where the
shared injector (`shared/forecast_fault.py`) replaces it (59/QĐ-DL). Everything downstream (planner, experts,
recorder) reads the effective forecast, so the policy sees exactly what the expert saw.

The forecast is computed in both modes so both datasets share one schema; only the `forecast` mode hands it to
the expert (`Episode.forecast_for_expert`). The observed expert never reads it.
"""
import numpy as np

from intent_policy.shared.sim_env import HORIZONS, DT, JOINTS

HISTORY_FRAMES = 9                  # 0.4 s of history at 20 Hz (ManiCast input)


def check_joint_order():
    """The adapter's joint order must be ManiCast's (intent/manicast_intent.py MANICAST)."""
    from intent_policy.intent.manicast_intent import MANICAST
    if list(JOINTS) != list(MANICAST):
        raise RuntimeError(f'joint order {JOINTS} != ManiCast {MANICAST}')


class ManiCastAdapter:
    def __init__(self, predictor, injector=None):
        self.predictor = predictor
        self.injector = injector    # N5 fault injector or None
        self.raw_forecast = None    # [10, 6, 3] ManiCast world positions
        self.raw_offsets = None     # [10, 6, 3] ManiCast displacement from the current pose (model output)
        self.forecast = None        # [10, 6, 3] effective world positions handed on
        self.offsets = None
        self.fault_strength = 0.    # 0 outside a fault window
        self.computed_at = None     # episode time (s) of the last valid forecast
        self.valid = False
        if predictor is not None:
            check_joint_order()

    def update(self, history, t, hand_speed=0., arm='r', used_last_tick=False):
        """history: list of [6, 3] world keypoints, newest last. Recomputed once per control tick (20 Hz)."""
        self.valid = False
        self.fault_strength = 0.
        raw = None
        if self.predictor is not None and len(history) >= HISTORY_FRAMES:
            from intent_policy.intent.manicast_intent import forecast_xi
            hist = np.stack(history[-HISTORY_FRAMES:])
            offsets = np.asarray(forecast_xi(hist, self.predictor, 1. / DT, HORIZONS)[-1], float)
            if offsets.shape == (len(HORIZONS), hist.shape[1], 3) and np.isfinite(offsets).all():
                raw = hist[-1][None] + offsets
                self.raw_offsets, self.raw_forecast = offsets, raw
        current = np.asarray(history[-1], float) if history else None
        eff = raw
        if self.injector is not None and current is not None:
            eff, self.fault_strength = self.injector.apply(raw, current, t, used_last_tick, arm)
        if eff is None:
            return None
        self.forecast = eff
        self.offsets = self.raw_offsets if eff is raw else eff - current[None]   # bit-identical without a fault
        self.computed_at = float(t)
        self.valid = True
        return self.forecast

    def age(self, t):
        return None if self.computed_at is None else float(t) - self.computed_at


def load_predictor():
    from intent_policy.intent.manicast_intent import load_forecaster
    return load_forecaster()
