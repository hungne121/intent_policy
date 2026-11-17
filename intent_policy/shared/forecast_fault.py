"""N5 forecast-fault injector, shared by every task (59/QĐ-DL §E N5). Sits right after ManiCast:

    H~_t = F(H^_t; delta_t)

The human trajectory, the scenario and the physics are untouched; only the forecast handed to the expert changes,
inside one fault window, and the true forecast comes back after it. The expert runs on H~_t, so the recorded actions
are the expert's real reaction (never nominal actions with a swapped forecast tensor). `observation.human_forecast`
stores H~_t; the raw H^_t, the fault schedule and the parameters go to the episode sidecar.

Three fault types, all on the ACTIVE arm only (the arm whose wrist the planner reads), growing with the horizon so
the 0.1 s step stays consistent with the observed pose, ramped in / out over `ramp_s` for continuity across ticks:
  * spatial_fault   - the forecast wrist path is shifted horizontally by A * h / 1 s along a direction phi;
                      the elbow is re-solved on the shoulder-wrist circle (2-link, bone lengths of the observed
                      pose, swivel nearest the ManiCast elbow); the shoulder is unchanged.
  * temporal_fault  - forecast progress is scaled by s < 1 (the forecast lags the real motion): every joint is
                      re-sampled along [current pose, 0.1 ... 1.0 s] at h * s; the elbow is re-solved.
  * direction_fault - the arm is rotated about the forecast shoulder by yaw * h / 1 s (bone lengths kept exactly):
                      `intent_policy.intent.forecast_fault.corrupt_forecast` (C3 v9 intervention), reused as is.

Trigger (online, causal): the fault starts `delay_s` after the k-th onset of forecast USE, i.e. the first tick of a
run of ticks in which the expert / planner read the forecast CONTENT while the observed hand moved (validity-only
checks such as `ForecastPlanner.active()` do not count), after >= `rearm_s`
without such reads (pilot 17/11: a trigger on hand-motion onsets hit I2 at 0.2 s and 4.1 s, before its expert reads
the forecast at 5.4 s, so the fault changed nothing). The injector sees only the past tick's read flag, never the
future or the human plan. k and all parameters are drawn from the episode seed before the rollout.

Checks per faulted tick (reported, and a violation rejects the episode): finite; bone lengths within
`bone_tol_m` of the observed pose beyond the raw forecast's own deviation; 0.1 s step within `start_tol_m` of the
raw forecast; the fault offset changes by at most `jump_tol_m` per tick.
"""
from dataclasses import dataclass, asdict, field

import numpy as np

from intent_policy.shared.sim_env import HORIZONS, DT, INTENT

FAULT_TYPES = ('spatial_fault', 'temporal_fault', 'direction_fault')
ARMS = {'l': (0, 2, 4), 'r': (1, 3, 5)}          # (shoulder, elbow, wrist) indices in JOINTS order


@dataclass
class FaultSpec:
    fault_type: str
    onset_index: int            # which observed hand-motion onset triggers the fault
    delay_s: float              # start = onset time + delay
    duration_s: float           # window length (strength > 0)
    ramp_s: float
    amplitude_m: float = 0.     # spatial: wrist shift at 1 s
    direction_deg: float = 0.   # spatial: horizontal direction of the shift (world yaw)
    progress_scale: float = 1.  # temporal: forecast progress factor
    yaw_deg: float = 0.         # direction: arm rotation at 1 s
    params: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def sample_fault(fault_type, seed, cfg):
    """Draw one fault schedule from the seed (independent stream from the scenario's)."""
    if fault_type not in FAULT_TYPES:
        raise ValueError(f'fault_type must be one of {FAULT_TYPES}')
    rng = np.random.default_rng([int(seed), 59, FAULT_TYPES.index(fault_type)])
    c = cfg[fault_type]
    spec = FaultSpec(fault_type, onset_index=int(rng.integers(cfg['onset_index'][0], cfg['onset_index'][1] + 1)),
                     delay_s=float(rng.uniform(*cfg['delay_s'])), duration_s=float(rng.uniform(*cfg['duration_s'])),
                     ramp_s=float(cfg['ramp_s']))
    if fault_type == 'spatial_fault':
        spec.amplitude_m = float(rng.uniform(*c['amplitude_m'])); spec.direction_deg = float(rng.uniform(0., 360.))
    elif fault_type == 'temporal_fault':
        spec.progress_scale = float(rng.uniform(*c['progress_scale']))
    else:
        spec.yaw_deg = float(rng.uniform(*c['yaw_deg'])) * float(rng.choice([-1., 1.]))
    return spec


def solve_elbow(shoulder, wrist, l1, l2, hint):
    """Elbow on the shoulder-wrist circle with bone lengths l1, l2, nearest `hint`. Returns (elbow, wrist, clamped)."""
    s, w = np.asarray(shoulder, float), np.asarray(wrist, float).copy()
    v = w - s; d = float(np.linalg.norm(v)); clamped = False
    lo, hi = abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4
    if d > hi or d < lo:                              # unreachable: keep the direction, clamp the distance
        d = float(np.clip(d, lo, hi)); w = s + v / max(np.linalg.norm(v), 1e-9) * d; clamped = True
        v = w - s
    u = v / d
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    c = s + a * u
    r = float(np.sqrt(max(l1 * l1 - a * a, 0.)))
    p = np.asarray(hint, float) - c
    p = p - (p @ u) * u
    if np.linalg.norm(p) < 1e-6:
        p = np.cross(u, [0., 0., 1.]) if abs(u[2]) < .9 else np.cross(u, [1., 0., 0.])
    return c + r * p / np.linalg.norm(p), w, clamped


class FaultInjector:
    """Per-episode injector. `apply(raw, current, t, hand_speed, arm)` -> effective forecast [10, 6, 3]."""

    def __init__(self, spec: FaultSpec, cfg):
        self.spec = spec
        self.rearm_ticks = max(1, int(round(float(cfg['rearm_s']) / DT)))
        self.bone_tol, self.start_tol, self.jump_tol = (float(cfg['checks'][k]) for k in ('bone_tol_m', 'start_tol_m', 'jump_tol_m'))
        self.onsets, self._still, self._moving = [], self.rearm_ticks, False
        self.start = self.end = None
        self.arm = None
        self._last_offset = None
        self.log = {'active_ticks': 0, 'max_strength': 0., 'max_wrist_shift_1s_m': 0., 'clamped_ticks': 0,
                    'max_bone_dev_m': 0., 'max_start_err_m': 0., 'max_jump_m': 0., 'violations': []}

    # ------------------------------------------------------------------ schedule
    def _track_onsets(self, t, used):
        if used:
            if not self._moving and self._still >= self.rearm_ticks:
                self.onsets.append(float(t))
            self._moving, self._still = True, 0
        else:
            self._moving = False; self._still += 1
        if self.start is None and len(self.onsets) > self.spec.onset_index:
            self.start = self.onsets[self.spec.onset_index] + self.spec.delay_s
            self.end = self.start + self.spec.duration_s

    def strength(self, t):
        if self.start is None:
            return 0.
        r = self.spec.ramp_s
        return float(np.clip(min((t - self.start) / r, (self.end - t) / r), 0., 1.))

    @property
    def applied(self):
        return self.log['active_ticks'] > 0

    # ------------------------------------------------------------------ fault forms
    def _spatial(self, raw, cur, k, arm):
        s_i, e_i, w_i = ARMS[arm]
        phi = np.radians(self.spec.direction_deg)
        dvec = np.array([np.cos(phi), np.sin(phi), 0.]) * self.spec.amplitude_m
        out = raw.copy(); clamped = 0
        l1 = float(np.linalg.norm(cur[e_i] - cur[s_i])); l2 = float(np.linalg.norm(cur[w_i] - cur[e_i]))
        for j, h in enumerate(HORIZONS):
            w = raw[j, w_i] + k * dvec * h / HORIZONS[-1]
            out[j, e_i], out[j, w_i], c = solve_elbow(raw[j, s_i], w, l1, l2, raw[j, e_i]); clamped += c
        return out, clamped

    def _temporal(self, raw, cur, k, arm):
        s_i, e_i, w_i = ARMS[arm]
        scale = 1. - k * (1. - self.spec.progress_scale)
        times = np.r_[0., HORIZONS]; traj = np.concatenate([cur[None], raw])
        out = raw.copy(); clamped = 0
        l1 = float(np.linalg.norm(cur[e_i] - cur[s_i])); l2 = float(np.linalg.norm(cur[w_i] - cur[e_i]))
        for j, h in enumerate(HORIZONS):
            q = h * scale
            for jt in (s_i, e_i, w_i):
                out[j, jt] = [np.interp(q, times, traj[:, jt, a]) for a in range(3)]
            out[j, e_i], out[j, w_i], c = solve_elbow(out[j, s_i], out[j, w_i], l1, l2, out[j, e_i]); clamped += c
        return out, clamped

    def _direction(self, raw, cur, k, arm, t):
        from intent_policy.intent.forecast_fault import ForecastFault, corrupt_forecast
        # corrupt_forecast ramps by time itself; in a local clock (0 = start, ramp 1) its strength at k equals k
        ff = ForecastFault(start_s=0., end_s=k + 1., yaw_deg=self.spec.yaw_deg,
                           arm='left' if arm == 'l' else 'right', ramp_s=1.)
        xi = (raw - cur[None])[None]
        out = corrupt_forecast(xi, cur[None], np.array([k]), HORIZONS, ff)[0] + cur[None]
        return out, 0

    # ------------------------------------------------------------------ main
    def apply(self, raw, current, t, used_last_tick, arm):
        """raw: ManiCast WORLD forecast [10, 6, 3] (or None); current: observed pose [6, 3]; used_last_tick: the
        expert read the forecast (hand moving) on the previous tick."""
        self._track_onsets(t, bool(used_last_tick))
        k = self.strength(t)
        if raw is None or k <= 0.:
            self._last_offset = None if raw is None else np.zeros_like(raw)
            return raw, 0.
        if self.arm is None:
            self.arm = arm                                   # the active arm at the fault start stays faulted
        raw = np.asarray(raw, float); cur = np.asarray(current, float)
        f = self.spec.fault_type
        if f == 'spatial_fault':
            eff, clamped = self._spatial(raw, cur, k, self.arm)
        elif f == 'temporal_fault':
            eff, clamped = self._temporal(raw, cur, k, self.arm)
        else:
            eff, clamped = self._direction(raw, cur, k, self.arm, t)
        self._check(raw, eff, cur, t, clamped)
        return eff, k

    def _check(self, raw, eff, cur, t, clamped):
        lg = self.log
        lg['active_ticks'] += 1; lg['clamped_ticks'] += int(clamped > 0)
        s_i, e_i, w_i = ARMS[self.arm]
        off = eff - raw
        lg['max_strength'] = max(lg['max_strength'], self.strength(t))
        lg['max_wrist_shift_1s_m'] = max(lg['max_wrist_shift_1s_m'], float(np.linalg.norm(off[-1, w_i])))
        bad = []
        if not np.isfinite(eff).all():
            bad.append('non_finite')
        l_obs = np.array([np.linalg.norm(cur[e_i] - cur[s_i]), np.linalg.norm(cur[w_i] - cur[e_i])])

        def dev(x):
            ln = np.stack([np.linalg.norm(x[:, e_i] - x[:, s_i], axis=1), np.linalg.norm(x[:, w_i] - x[:, e_i], axis=1)], 1)
            return float(np.abs(ln - l_obs).max())
        bone = dev(eff) - dev(raw)
        start = float(np.linalg.norm(off[0], axis=1).max())
        jump = 0. if self._last_offset is None else float(np.linalg.norm(off - self._last_offset, axis=2).max())
        self._last_offset = off
        lg['max_bone_dev_m'] = max(lg['max_bone_dev_m'], bone)
        lg['max_start_err_m'] = max(lg['max_start_err_m'], start)
        lg['max_jump_m'] = max(lg['max_jump_m'], jump)
        if bone > self.bone_tol:
            bad.append(f'bone {bone:.3f}')
        if start > self.start_tol:
            bad.append(f'start {start:.3f}')
        if jump > self.jump_tol:
            bad.append(f'jump {jump:.3f}')
        if bad and len(lg['violations']) < 20:
            lg['violations'].append({'t': round(t, 3), 'checks': bad})

    def summary(self):
        return {'fault_type': self.spec.fault_type, 'fault_parameters': self.spec.to_dict(),
                'fault_start': None if self.start is None else round(self.start, 3),
                'fault_end': None if self.end is None else round(self.end, 3),
                'fault_applied': self.applied, 'fault_arm': self.arm, 'use_onsets_s': [round(x, 3) for x in self.onsets[:8]],
                'use_onsets_total': len(self.onsets),
                **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.log.items()}}
