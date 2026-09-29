"""Simulator-independent robot state and joint-position command contract (SI units)."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass
class RobotState:
    """Robot-only proprioception: q/dq [6], TCP xyz [3], quaternion XYZW [4], gripper [m]."""
    q: np.ndarray
    dq: np.ndarray
    ee_pos: np.ndarray
    ee_quat: np.ndarray
    gripper: float
    timestamp: float

    def policy_vector(self) -> np.ndarray:
        """Return observation.state [10]: q [6], gripper opening [1], TCP xyz [3]."""
        return np.concatenate([self.q, [self.gripper], self.ee_pos]).astype(np.float32)

    def to_dict(self) -> dict:
        return dict(q=self.q.tolist(), dq=self.dq.tolist(), ee_pos=self.ee_pos.tolist(),
                    ee_quat=self.ee_quat.tolist(), gripper=self.gripper, timestamp=self.timestamp)


class RobotInterface(Protocol):
    """Hardware adapter needs only robot state and a bounded command API."""
    def observe(self) -> RobotState: ...
    def step(self, action: np.ndarray) -> RobotState:
        """Execute [7] absolute arm joint targets (rad) + gripper opening command (m)."""
        ...
    def hold(self) -> np.ndarray: ...
    def close(self) -> None: ...
