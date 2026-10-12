"""Active hand and causal motion onset of the tracked human keypoints (intent contract v3.2 kinematic fields,
intent/source_v32.py). Thresholds: schema `perfect` (onset_speed_m_s, still_speed_m_s)."""
from __future__ import annotations
import numpy as np
from intent_policy.intent.labels import hand_indices


class HandKinematics:
    """Active hand and causal motion onset (as PerfectPerceptionSource): the onset fires when the hand's speed rises
    above `onset_speed` after it was below `still_speed`."""

    def __init__(self, schema: dict, onset_speed: float, still_speed: float):
        self.hands = hand_indices(schema)
        self.fps = float(schema['fps'])
        self.onset_speed, self.still_speed = float(onset_speed), float(still_speed)
        self.reset()

    def reset(self) -> None:
        self.rest = None
        self.prev: list[np.ndarray] = []
        self.still = True
        self.onset_t: int | None = None
        self.speed = 0.0
        self.hand = next(iter(self.hands))

    def step(self, t: int, kp: np.ndarray) -> None:
        kp = np.asarray(kp, np.float64)
        if self.rest is None:
            self.rest = kp.copy()
        dist = {h: float(np.linalg.norm(kp[i].mean(0) - self.rest[i].mean(0))) for h, i in self.hands.items()}
        self.hand = max(dist, key=dist.get)
        idx = self.hands[self.hand]
        self.speed = float(np.linalg.norm(kp[idx].mean(0) - self.prev[-1][idx].mean(0)) * self.fps) if self.prev else 0.0
        self.prev = (self.prev + [kp])[-3:]
        if self.speed < self.still_speed:
            self.still = True
        elif self.speed > self.onset_speed and self.still:
            self.onset_t, self.still = t, False

    @property
    def moving(self) -> bool:
        return self.speed > self.onset_speed
