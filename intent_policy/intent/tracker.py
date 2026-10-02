"""Committed intentions (INTENT_ACT_GUIDE_v2.md §3.4, §9.5): `IntentTracker` keeps what the human asked the robot to
do until the robot has done it — the memory channel c_who / c_target. Causal, shared by every intent source.

Grammar (stable gesture + confident target -> queue item):
  point at an object place      -> pick (robot)      replaces a pick that is not grasped yet (change of mind)
  point at a place zone         -> place (robot)     only with a pick queued or an object held; replaces a queued place
  palm_up at a hand zone        -> handover (joint)  replaces a queued handover
  reach                         -> nothing (only p_*)       palm_forward (cancel) is disabled: no cancel gesture
An intention is committed once per gesture: the same (gesture, target) is not committed again until the hand has
been at rest (no re-commit of a handover the robot just completed while the palm is still held out).
Completion, from the robot state only (never the simulator):
  pick      holding and the TCP lift_m above where the grasp closed
  place     released (holding -> not holding) with the TCP within place_radius_m (xy) of the place
  handover  released while a human hand keypoint is within hand_radius_m of the hand zone
T3 (reach for a cube) and T4 (hand into the robot zone) have no committing gesture: c_* stays `none` there.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from intent_policy.intent.contract import one_hot
from intent_policy.intent.places import target_positions


@dataclass
class RobotState:
    tcp: np.ndarray            # (3,) world / robot-base frame [m]
    gripper: float             # measured gripper opening [m]
    holding: bool = False      # from HoldingDetector (gripper opening), never the simulator's grasp state


class HoldingDetector:
    """Object held <=> the opening is inside (open_min_m, open_max_m) and has settled (it changed less than steady_m
    per frame for steady_frames frames); an opening gripper (increase > steady_m) releases at once."""

    def __init__(self, open_min_m: float, open_max_m: float, steady_m: float = 0.001, steady_frames: int = 2):
        self.lo, self.hi, self.steady_m, self.steady_frames = float(open_min_m), float(open_max_m), float(steady_m), int(steady_frames)
        self.reset()

    def reset(self) -> None:
        self.prev, self.still, self.holding = None, 0, False

    def step(self, opening: float) -> bool:
        d = 0.0 if self.prev is None else opening - self.prev
        self.prev = opening
        self.still = self.still + 1 if abs(d) < self.steady_m else 0
        inside = self.lo < opening < self.hi
        if not inside or d > self.steady_m:
            self.holding = False
        elif self.still >= self.steady_frames:
            self.holding = True
        return self.holding


@dataclass
class Item:
    kind: str                  # pick | place | handover
    target: int                # index in schema['targets']
    who: int                   # index in schema['who']
    grasped: bool = False      # pick: the grasp has closed (no replacement after this)
    grasp_z: float | None = None


@dataclass
class IntentTracker:
    schema: dict
    commit_thr: float = 0.8
    hold_frames: int = 5
    lift_m: float = 0.03
    place_radius_m: float = 0.08
    hand_radius_m: float = 0.10
    cancel: bool = False
    queue: list = field(default_factory=list)

    def __post_init__(self):
        s = self.schema
        self.targets, self.who_names, self.gestures = list(s['targets']), list(s['who']), list(s['gestures'])
        kinds = s['target_kinds']
        self.kind_of = {self.targets.index(k): kind for kind, ks in kinds.items() for k in ks}
        self.pos = {self.targets.index(k): p for k, p in target_positions(s).items()}
        self.reset()

    @classmethod
    def from_schema(cls, schema: dict) -> 'IntentTracker':
        return cls(schema, **(schema.get('tracker') or {}))

    def reset(self) -> None:
        self.queue = []
        self.run_key, self.run_len, self.committed = None, 0, set()
        self.prev_holding = False

    # ------------------------------------------------------------------ commit
    def _commit(self, gesture: str, target: int, holding: bool) -> None:
        kind = self.kind_of.get(target)
        robot, joint = self.who_names.index('robot'), self.who_names.index('joint')
        find = lambda k: next((it for it in self.queue if it.kind == k), None)
        if gesture == 'point' and kind == 'object':
            pending = next((it for it in self.queue if it.kind == 'pick' and not it.grasped), None)
            if pending is not None:
                pending.target = target                                   # change of mind before the grasp
            elif not holding and not any(it.kind == 'pick' for it in self.queue):
                self.queue.insert(0, Item('pick', target, robot))
        elif gesture == 'point' and kind == 'place':
            if any(it.kind == 'pick' for it in self.queue) or holding:
                place = find('place')
                if place is not None:
                    place.target = target
                else:
                    self.queue.append(Item('place', target, robot))
        elif gesture == 'palm_up' and kind == 'hand':
            ho = find('handover')
            if ho is not None:
                ho.target = target
            else:
                self.queue.append(Item('handover', target, joint))
        elif gesture == 'palm_forward' and self.cancel and self.queue:
            self.queue.pop(0)

    # ------------------------------------------------------------------ completion
    def _complete(self, robot: RobotState, hands: np.ndarray | None) -> None:
        if not self.queue:
            return
        head = self.queue[0]
        released = self.prev_holding and not robot.holding
        tcp = np.asarray(robot.tcp, float)
        if head.kind == 'pick':
            if robot.holding:
                if head.grasp_z is None:
                    head.grasp_z, head.grasped = float(tcp[2]), True
                if tcp[2] - head.grasp_z > self.lift_m:
                    self.queue.pop(0)
            else:
                head.grasp_z = None                                       # slipped before the lift: retry
        elif head.kind == 'place':
            if released and np.linalg.norm(tcp[:2] - self.pos[head.target][:2]) <= self.place_radius_m:
                self.queue.pop(0)
        elif head.kind == 'handover':
            if released and hands is not None and \
                    np.min(np.linalg.norm(np.asarray(hands, float) - self.pos[head.target], axis=-1)) <= self.hand_radius_m:
                self.queue.pop(0)

    def step(self, p_target: np.ndarray, p_who: np.ndarray, p_gesture: np.ndarray, robot: RobotState,
             hands: np.ndarray | None = None) -> dict:
        """One frame. hands: human hand keypoints (n, 3) for the handover check. Returns c_who, c_target, queue."""
        t, g = int(np.argmax(p_target)), int(np.argmax(p_gesture))
        key = (g, t)
        self.run_len = self.run_len + 1 if key == self.run_key else 1
        self.run_key = key
        if self.gestures[g] == 'rest':
            self.committed = set()
        if t != 0 and p_target[t] >= self.commit_thr and self.run_len >= self.hold_frames and key not in self.committed:
            self._commit(self.gestures[g], t, robot.holding)
            self.committed.add(key)
        self._complete(robot, hands)
        self.prev_holding = robot.holding
        head = self.queue[0] if self.queue else None
        K, W = len(self.targets), len(self.who_names)
        return dict(c_target=one_hot(head.target if head else 0, K), c_who=one_hot(head.who if head else 0, W),
                    queue=[(it.kind, self.targets[it.target]) for it in self.queue])
