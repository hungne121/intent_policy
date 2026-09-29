"""Restricted 9-action motor interface and its deterministic mapping to controller commands.

Actions are generic short-horizon motor primitives, not semantic skills. The mapper keeps a
Cartesian set-point: MOVE_* shifts it by a configured step in the configured control frame,
HOLD leaves it unchanged (no translation command), OPEN/CLOSE_GRIPPER only change the
gripper target. The set-point is converted to joint targets with the simulator IK.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
import numpy as np
import yaml
from env.base_env import ROOT


class RestrictedAction(IntEnum):
    HOLD = 0
    MOVE_FORWARD = 1
    MOVE_BACKWARD = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    MOVE_UP = 5
    MOVE_DOWN = 6
    OPEN_GRIPPER = 7
    CLOSE_GRIPPER = 8


NUM_RESTRICTED_ACTIONS = len(RestrictedAction)
# Unit direction in the control frame for every translation action.
DIRECTIONS = {
    RestrictedAction.MOVE_FORWARD: (0, +1), RestrictedAction.MOVE_BACKWARD: (0, -1),
    RestrictedAction.MOVE_LEFT: (1, +1), RestrictedAction.MOVE_RIGHT: (1, -1),
    RestrictedAction.MOVE_UP: (2, +1), RestrictedAction.MOVE_DOWN: (2, -1),
}


@dataclass
class RestrictedActionConfig:
    control_frame: str = 'base_link'
    translation_step: dict = field(default_factory=lambda: {'x': 0.01, 'y': 0.01, 'z': 0.01})
    execution_ticks: int = 1
    gripper: dict = field(default_factory=lambda: {'open': 0.085, 'closed': 0.0})
    limits: dict = field(default_factory=lambda: {'x': [0.12, 0.56], 'y': [-0.32, 0.32], 'z': [0.635, 0.95]})
    orientation: str = 'top_down'
    max_ik_error: float = 0.004      # [m] set-points the IK cannot reach within this are refused

    @classmethod
    def load(cls, path: str | Path = ROOT / 'configs/controller/restricted_action.yaml') -> 'RestrictedActionConfig':
        return cls(**yaml.safe_load(Path(path).read_text())['restricted_action'])

    def step_vector(self) -> np.ndarray:
        return np.array([self.translation_step[a] for a in 'xyz'], float)

    def to_dict(self) -> dict:
        return dict(control_frame=self.control_frame, translation_step=dict(self.translation_step),
                    execution_ticks=self.execution_ticks, gripper=dict(self.gripper), limits=dict(self.limits),
                    orientation=self.orientation, max_ik_error=self.max_ik_error)


@dataclass
class ControllerCommand:
    action_id: int
    action_name: str
    delta_frame: np.ndarray          # translation in the control frame [m]
    delta_world: np.ndarray          # same translation in the world frame [m]
    cartesian_target: np.ndarray     # TCP set-point, world frame [m]
    gripper_target: float            # [m]
    joint_target: np.ndarray         # [7] = 6 arm joint targets (rad) + gripper (m), sent to env.step
    execution_ticks: int
    rejected: bool = False           # translation refused: set-point outside workspace / IK reach

    def to_dict(self) -> dict:
        return dict(action_id=self.action_id, action_name=self.action_name,
                    delta_frame=self.delta_frame.round(6).tolist(), delta_world=self.delta_world.round(6).tolist(),
                    cartesian_target=self.cartesian_target.round(6).tolist(), gripper_target=self.gripper_target,
                    joint_target=self.joint_target.round(6).tolist(), execution_ticks=self.execution_ticks,
                    rejected=self.rejected)


def validate_action(action_id) -> RestrictedAction:
    if isinstance(action_id, (bool, np.bool_)) or not isinstance(action_id, (int, np.integer)):
        raise TypeError(f'restricted action id must be an int, got {type(action_id).__name__}')
    if not 0 <= int(action_id) < NUM_RESTRICTED_ACTIONS:
        raise ValueError(f'restricted action id {action_id} outside [0, {NUM_RESTRICTED_ACTIONS - 1}]')
    return RestrictedAction(int(action_id))


class RestrictedActionMapper:
    """Stateful (set-point) but deterministic mapping from action ids to controller commands."""

    def __init__(self, env, config: RestrictedActionConfig | None = None):
        self.env = env
        self.cfg = config or RestrictedActionConfig.load()
        self.step = self.cfg.step_vector()
        self.lo = np.array([self.cfg.limits[a][0] for a in 'xyz'])
        self.hi = np.array([self.cfg.limits[a][1] for a in 'xyz'])
        self.reset()

    def frame_rotation(self) -> np.ndarray:
        """World-from-control-frame rotation (3x3)."""
        body = self.env.model.body(self.cfg.control_frame).id
        return self.env.data.xmat[body].reshape(3, 3).copy()

    def reset(self) -> None:
        """Re-anchor the set-point at the measured TCP and the current gripper command."""
        self.setpoint = self.env.observe().ee_pos.copy()
        self.gripper_target = float(self.env.data.ctrl[6])
        self._arm_cache: tuple[np.ndarray, np.ndarray] | None = None

    def translation(self, action: RestrictedAction) -> np.ndarray:
        delta = np.zeros(3)
        if action in DIRECTIONS:
            axis, sign = DIRECTIONS[action]
            delta[axis] = sign * self.step[axis]
        return delta

    def preview(self, action_id) -> ControllerCommand:
        """The command `map` would return, without changing the mapper state (set-point, gripper, cache)."""
        saved = (self.setpoint.copy(), self.gripper_target, self._arm_cache, getattr(self.env, 'last_ik_error', None))
        try:
            return self.map(action_id)
        finally:
            self.setpoint, self.gripper_target, self._arm_cache, last = saved
            if last is not None:
                self.env.last_ik_error = last

    def map(self, action_id) -> ControllerCommand:
        action = validate_action(action_id)
        delta_frame = self.translation(action)
        delta_world = self.frame_rotation() @ delta_frame
        if action is RestrictedAction.OPEN_GRIPPER:
            self.gripper_target = float(self.cfg.gripper['open'])
        elif action is RestrictedAction.CLOSE_GRIPPER:
            self.gripper_target = float(self.cfg.gripper['closed'])
        rejected = False
        if np.any(delta_world):
            candidate = np.clip(self.setpoint + delta_world, self.lo, self.hi)
            arm = self.env.solve_ik(candidate)
            if np.array_equal(candidate, self.setpoint) or self.env.last_ik_error > self.cfg.max_ik_error:
                rejected = True               # workspace box or kinematic reach limit: stay put
            else:
                self.setpoint = candidate
                self._arm_cache = (candidate.copy(), arm)
        if self._arm_cache is None or not np.array_equal(self._arm_cache[0], self.setpoint):
            self._arm_cache = (self.setpoint.copy(), self.env.solve_ik(self.setpoint))
        arm = self._arm_cache[1]              # unchanged set-point => identical joint target
        return ControllerCommand(int(action), action.name, delta_frame, delta_world, self.setpoint.copy(),
                                 self.gripper_target, np.r_[arm, self.gripper_target], int(self.cfg.execution_ticks),
                                 rejected)
