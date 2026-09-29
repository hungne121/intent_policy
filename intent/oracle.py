"""Oracle intention information from simulator ground truth, under the information-timing contract.

The oracle is what an ideal intention predictor could know at time t — no more:
  * spatial / target: the object / interaction region the human targets, known only from the
    moment the target became predictable from the observable cue (`human_intention_evident`,
    scheduled by the scripted human; ScriptedHuman.known_evidence);
  * motion: the current hand state and the future hand position/velocity at fixed horizons, from
    ScriptedHuman.predicted_hand_position (planned motion only after its evidence point).
`lead_s` shifts the contract (information `lead_s` earlier, for the lead dose-response test).

Records are kept in the world frame; `intent.representation` turns them into policy features
relative to the current TCP. No semantic label (e.g. RECEIVE/WAIT) is ever part of a record.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
import numpy as np

HORIZONS = (0.5, 1.0)


def horizon_tag(h: float) -> str:
    return f'{h:.1f}'.replace('.', 'p')


@dataclass
class OracleRecord:
    t: float
    object_key: str | None
    object_pos: np.ndarray | None
    region_key: str | None
    region_pos: np.ndarray | None
    hand_pos: np.ndarray
    hand_vel: np.ndarray
    future_pos: dict = field(default_factory=dict)      # horizon -> [3]
    future_vel: dict = field(default_factory=dict)

    def copy(self) -> 'OracleRecord':
        return copy.deepcopy(self)

    def compact(self) -> dict:
        r = lambda a: None if a is None else np.round(np.asarray(a, float), 4).tolist()
        return dict(t=round(self.t, 4), object=self.object_key, object_pos=r(self.object_pos), region=self.region_key,
                    region_pos=r(self.region_pos), hand_pos=r(self.hand_pos),
                    future_pos={horizon_tag(h): r(p) for h, p in self.future_pos.items()})


def empty_record(t: float, hand_pos, hand_vel, horizons=HORIZONS) -> OracleRecord:
    hand_pos, hand_vel = np.asarray(hand_pos, float), np.asarray(hand_vel, float)
    return OracleRecord(t, None, None, None, None, hand_pos.copy(), hand_vel.copy(),
                        {h: hand_pos.copy() for h in horizons}, {h: np.zeros(3) for h in horizons})


class OracleIntentProvider:
    """Ground-truth oracle for one episode (reads the scenario; knows nothing about policies)."""

    def __init__(self, horizons=HORIZONS, lead_s: float = 0.0):
        self.horizons = tuple(float(h) for h in horizons)
        self.lead_s = float(lead_s)

    def record(self, scenario) -> OracleRecord:
        human = scenario.human
        t = human.t_last
        slots = scenario.oracle_slots(human.known_evidence(t + self.lead_s))
        pos = lambda k: None if k is None else np.asarray(scenario.oracle_position(k), float).copy()
        dt = scenario.dt
        pred = lambda h: human.predicted_hand_position(h, self.lead_s)
        return OracleRecord(
            t=float(t), object_key=slots['object'], object_pos=pos(slots['object']),
            region_key=slots['region'], region_pos=pos(slots['region']),
            hand_pos=human.pos.copy(), hand_vel=human.vel.copy(),
            future_pos={h: pred(h) for h in self.horizons},
            future_vel={h: (pred(h + dt) - pred(max(h - dt, 0.0))) / (h + dt - max(h - dt, 0.0)) for h in self.horizons})
