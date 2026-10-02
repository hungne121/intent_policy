"""Intent sources (INTENT_ACT_GUIDE_v2.md §3, §9): every source turns (human keypoints, robot state) into the same
per-frame contract (intent/contract.py), so the policy cannot tell them apart.

  HindsightSource          ceiling: ground-truth segments and the true future (NOT causal)
  PerfectPerceptionSource  causal pipeline on clean keypoints (pointing ray, reach end point, kinematic gesture)
  PredictedSource          the same pipeline on noisy keypoints with a learned gesture classifier

All sources share IntentTracker (c_who / c_target) and HoldingDetector (RobotState.holding from the gripper).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from intent_policy.intent.contract import confidence, one_hot
from intent_policy.intent.labels import future_waypoints, hand_indices
from intent_policy.intent.tracker import IntentTracker, RobotState


@dataclass
class HumanObs:
    t: int                     # frame index
    keypoints: np.ndarray      # (J, 3) world / robot-base frame [m]


class IntentSource(Protocol):
    def reset(self, episode_meta: dict) -> None: ...
    def step(self, obs: HumanObs, robot: RobotState) -> dict: ...


def occupancy(kp: np.ndarray, xi: np.ndarray, hands: dict, margin: float, conf: float) -> np.ndarray:
    """Box around the moving hand's keypoints now and along xi, grown by `margin`: [cx, cy, cz, sx, sy, sz, conf]."""
    moving = max(hands.values(), key=lambda idx: float(np.abs(xi[:, idx]).max()))
    pts = np.concatenate([kp[moving], (kp[None, moving] + xi[:, moving]).reshape(-1, 3)])
    lo, hi = pts.min(0) - margin, pts.max(0) + margin
    return np.r_[(lo + hi) / 2, hi - lo, conf].astype(np.float32)


def finish(schema: dict, p_who, p_target, c: dict, kp, xi, tte, tte_std, phase, human_conf: float) -> dict:
    """Assemble one contract frame."""
    p_who, p_target = np.asarray(p_who, np.float32), np.asarray(p_target, np.float32)
    occ = occupancy(kp, xi, hand_indices(schema), float(schema.get('occupancy_margin_m', 0.05)), human_conf)
    tmax = float(schema['tte_max_s'])
    return dict(p_who=p_who, p_target=p_target, c_who=c['c_who'], c_target=c['c_target'], occupancy=occ,
                tte=np.float32([np.clip(tte, 0.0, tmax)]), tte_std=np.float32([np.clip(tte_std, 0.0, tmax)]),
                phase=np.asarray(phase, np.float32), xi=np.asarray(xi, np.float32),
                confidence=confidence(p_target, p_who))


class HindsightSource:
    """Ground truth (§3.1): one-hot target / who / gesture over [t_onset, t_end], phases from the true milestones,
    tte to the gesture's goal pose (t_clear) with tte_std = 0, xi = the true future. Knows the future: ceiling only.
    episode_meta: segments, keypoints (T, J, 3)."""
    name = 'hindsight'

    def __init__(self, schema: dict):
        self.schema = schema
        self.tracker = IntentTracker.from_schema(schema)
        s = schema
        self.K, self.W, self.P = len(s['targets']), len(s['who']), len(s['phases'])
        self.ph = {p: i for i, p in enumerate(s['phases'])}
        self.hands = hand_indices(schema)
        self.hand_idx = sorted(i for idx in self.hands.values() for i in idx)

    def reset(self, episode_meta: dict) -> None:
        s = self.schema
        self.segments = sorted(episode_meta['segments'], key=lambda g: g['t_onset'])
        sg = s.get('savgol') or {}
        self.xi = future_waypoints(episode_meta['keypoints'], s['fps'], s['horizon_s'], s['n_waypoints'],
                                   sg.get('window', 7), sg.get('order', 2))
        self.tracker.reset()

    def _segment(self, t: int) -> dict | None:
        return next((g for g in reversed(self.segments) if g['t_onset'] <= t <= g['t_end']), None)

    def _phase(self, t: int, seg: dict | None) -> str:
        if seg is not None:
            return 'prepare' if t < seg['t_evident'] else 'stroke' if t < seg['t_clear'] else 'hold'
        if any(g.get('t_retract') is not None and g['t_retract_start'] <= t <= g['t_retract'] for g in self.segments):
            return 'retract'
        return 'rest'

    def step(self, obs: HumanObs, robot: RobotState) -> dict:
        s, t = self.schema, obs.t
        seg = self._segment(t)
        if seg is None:
            p_target, p_who, p_g = one_hot(0, self.K), one_hot(0, self.W), one_hot(s['gestures'].index('rest'), len(s['gestures']))
            tte = s['tte_max_s']
        else:
            p_target = one_hot(s['targets'].index(seg['target']), self.K)
            p_who = one_hot(s['who'].index(seg['who']), self.W)
            p_g = one_hot(s['gestures'].index(seg['gesture']), len(s['gestures']))
            tte = max(seg['t_clear'] - t, 0) / s['fps']
        c = self.tracker.step(p_target, p_who, p_g, robot, obs.keypoints[self.hand_idx])
        human = float(seg is not None and seg['who'] == 'human')
        return finish(s, p_who, p_target, c, obs.keypoints, self.xi[t], tte, 0.0,
                      one_hot(self.ph[self._phase(t, seg)], self.P), human)
