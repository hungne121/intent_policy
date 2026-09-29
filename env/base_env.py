"""MuJoCo UR3e + SusGrip simulator: joint-target control, IK, grasp contacts, rendering.

The environment knows only the robot. Scene objects, humans and task logic are added by
`env.scene_builder` and `scenarios/`; this module never contains scenario behaviour.
"""
from pathlib import Path
import json
import time
import numpy as np
import mujoco
import yaml
from scipy.spatial.transform import Rotation
from env.robot_interface import RobotState
from env.viewer_lifecycle import ViewerClosed, launch_viewer, close_viewer

ROOT = Path(__file__).resolve().parents[1]
ARM = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
       'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
GRIPPER_OPEN = 0.085


class ManipulationEnv:
    """20 Hz joint-target interface; physics runs at 500 Hz.

    `model_xml` (string) takes precedence over `model_path`; both default to the base scene.
    """
    def __init__(self, config: str | Path = ROOT / 'configs/robot.yaml', gui: bool = False,
                 model_path: str | Path | None = None, model_xml: str | None = None):
        self.cfg = yaml.safe_load(Path(config).read_text())
        if model_xml is not None:
            self.model = mujoco.MjModel.from_xml_string(model_xml)
        else:
            model_path = Path(model_path or self.cfg['model'])
            self.model = mujoco.MjModel.from_xml_path(str(model_path if model_path.is_absolute() else ROOT / model_path))
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.arm_ids = np.array([self.model.joint(n).id for n in ARM])
        self.qids = self.model.jnt_qposadr[self.arm_ids]
        self.dids = self.model.jnt_dofadr[self.arm_ids]
        self.gid = self.model.joint('gripper_joint').qposadr[0]
        self.tcp = self.model.site('tcp').id
        # Only joints flagged `limited` have a meaningful range (UR3e wrist_3 is unlimited and
        # its range is [0, 0] in the asset: clipping to it would lock the gripper yaw).
        rng = self.model.jnt_range[self.arm_ids].copy()
        unlimited = self.model.jnt_limited[self.arm_ids] == 0
        rng[unlimited] = [-np.inf, np.inf]
        self.arm_limits = rng
        self.dt = 1.0 / self.cfg['control_hz']
        self.substeps = round(self.dt / self.model.opt.timestep)
        if not np.isclose(self.substeps * self.model.opt.timestep, self.dt):
            raise ValueError('control_hz must divide the physics rate')
        self.mimics = json.loads((ROOT / 'assets/provenance.json').read_text())['mimics']
        self.finger_bodies = {side: {i for i in range(self.model.nbody)
                                     if f'_{side[0]}_' in self.model.body(i).name
                                     and ('pad' in self.model.body(i).name or 'finger' in self.model.body(i).name)}
                              for side in ('left', 'right')}
        ag = self.cfg.get('assisted_grasp', {})
        self.assisted = bool(ag.get('enabled', False))
        self.close_below = float(ag.get('close_below', 0.03))
        self.open_above = float(ag.get('open_above', 0.05))
        self.last_ik_error = 0.0                   # position residual of the last solve_ik [m]
        self.grasp_welds: dict[int, int] = {}      # body id -> weld equality id (set by scenario)
        self.attached: int | None = None           # body currently attached by assisted grasp
        self.attach_opening = 0.0                  # finger gap when the attachment was made
        self.viewer = None
        self.viewer_thread = None
        self.renderer = None
        self.render_size = None
        self.reset()
        if gui:
            self.viewer, self.viewer_thread = launch_viewer(self.model, self.data)

    def reset(self, home: np.ndarray | None = None) -> RobotState:
        """Reset physics, put the arm at `home` (default config home) with the gripper open.

        Scene objects keep their XML default pose; scenarios place them after this call.
        """
        mujoco.mj_resetData(self.model, self.data)
        self.attached = None
        home = np.asarray(self.cfg['home'] if home is None else home, dtype=float)
        self.data.qpos[self.qids] = home
        self.set_gripper_qpos(GRIPPER_OPEN)
        self.data.ctrl[:] = np.r_[home, GRIPPER_OPEN]
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def set_gripper_qpos(self, opening: float) -> None:
        self.data.qpos[self.gid] = opening
        for name, _, offset, mult in self.mimics:
            self.data.qpos[self.model.joint(name).qposadr[0]] = offset + mult * opening

    def settle(self, ticks: int) -> None:
        """Advance physics while holding the current command, then zero the clock."""
        for _ in range(ticks * self.substeps):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0

    def finger_contacts(self, body_id: int) -> set[str]:
        """Return finger sides ('left'/'right') touching `body_id` with positive normal force."""
        sides = set()
        force = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if body_id not in (b1, b2):
                continue
            other = b2 if b1 == body_id else b1
            mujoco.mj_contactForce(self.model, self.data, i, force)
            if force[0] <= 0.01:
                continue
            for side, bodies in self.finger_bodies.items():
                if other in bodies:
                    sides.add(side)
        return sides

    def register_graspable(self, body_id: int, weld_id: int) -> None:
        """Declare a graspable body and its (inactive) gripper weld for assisted grasping."""
        self.grasp_welds[body_id] = weld_id

    def is_grasped(self, body_id: int) -> bool:
        """Assisted mode: body attached to the gripper; otherwise both fingers push on it."""
        if self.assisted and body_id in self.grasp_welds:
            return self.attached == body_id
        return self.finger_contacts(body_id) == {'left', 'right'}

    def _attach(self, body_id: int) -> None:
        """Activate the gripper weld keeping the current relative pose (no jump)."""
        m, d = self.model, self.data
        eid = self.grasp_welds[body_id]
        g = m.eq_obj1id[eid]
        rot_g = d.xmat[g].reshape(3, 3)
        m.eq_data[eid, 0:3] = 0
        m.eq_data[eid, 3:6] = rot_g.T @ (d.xpos[body_id] - d.xpos[g])
        q = np.zeros(4)
        mujoco.mju_negQuat(q, d.xquat[g])
        rel = np.zeros(4)
        mujoco.mju_mulQuat(rel, q, d.xquat[body_id])
        m.eq_data[eid, 6:10] = rel
        d.eq_active[eid] = True
        self.attached = body_id
        self.attach_opening = float(d.qpos[self.gid])

    def _release(self) -> None:
        if self.attached is not None:
            self.data.eq_active[self.grasp_welds[self.attached]] = False
            self.attached = None

    def _update_assisted_grasp(self, command: float) -> None:
        if not self.assisted:
            return
        if command > self.open_above:
            self._release()
        elif command < self.close_below and self.attached is None:
            for body_id in self.grasp_welds:
                if self.finger_contacts(body_id) == {'left', 'right'}:
                    self._attach(body_id)
                    break

    def observe(self) -> RobotState:
        """Robot proprioception; quaternions at this boundary are XYZW."""
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.data.site_xmat[self.tcp])
        return RobotState(self.data.qpos[self.qids].copy(), self.data.qvel[self.dids].copy(),
                          self.data.site_xpos[self.tcp].copy(), q[[1, 2, 3, 0]],
                          float(self.data.qpos[self.gid]), float(self.data.time))

    def hold(self) -> np.ndarray:
        """Stop queued arm motion while preserving current gripper target [7]."""
        return np.r_[self.data.qpos[self.qids], self.data.ctrl[6]]

    def step(self, action: np.ndarray) -> RobotState:
        """Apply finite [7] target with joint/velocity bounds for one control tick."""
        if self.viewer is not None and not self.viewer.is_running():
            raise ViewerClosed()
        started_at = time.monotonic()
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError('Action must be finite shape (7,)')
        limits = self.arm_limits
        arm = np.clip(action[:6], limits[:, 0], limits[:, 1])
        arm = self.data.qpos[self.qids] + np.clip(arm - self.data.qpos[self.qids],
            -self.cfg['max_joint_speed'] * self.dt, self.cfg['max_joint_speed'] * self.dt)
        if self.assisted and action[6] > self.open_above:
            self._release()                       # opening command frees the object immediately
        grip = np.clip(action[6], 0, GRIPPER_OPEN)
        if self.attached is not None:
            # Hold the fingers at the contact width: the weld carries the load, the fingers do not
            # keep squeezing (squeezing a small handle twists the object about the vertical axis).
            grip = max(grip, self.attach_opening - 0.002)
        grip = self.data.ctrl[6] + np.clip(grip - self.data.ctrl[6],
            -self.cfg['max_gripper_speed'] * self.dt, self.cfg['max_gripper_speed'] * self.dt)
        # Linearly ramp the position targets across the physics substeps (first-order hold)
        # instead of a step input: avoids an acceleration spike every control tick.
        previous, target = self.data.ctrl.copy(), np.r_[arm, grip]
        for i in range(self.substeps):
            self.data.ctrl[:] = previous + (target - previous) * (i + 1) / self.substeps
            mujoco.mj_step(self.model, self.data)
        self._update_assisted_grasp(float(action[6]))
        if self.viewer:
            self.viewer.sync()
            time.sleep(max(0., self.dt - (time.monotonic() - started_at)))
        return self.observe()

    def solve_ik(self, position: np.ndarray, quaternion: np.ndarray | None = None) -> np.ndarray:
        """Damped least-squares IK returns arm [6]; never changes live physics state.

        Default orientation is the top-down grasp pose (fingers close along world Y).
        """
        d = self.ik_data
        d.qpos[:] = self.data.qpos
        target_rot = Rotation.from_euler('xyz', [np.pi, 0, -np.pi/2]).as_matrix() if quaternion is None else Rotation.from_quat(quaternion).as_matrix()
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        limits = self.arm_limits
        for _ in range(100):
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            ep = np.asarray(position) - d.site_xpos[self.tcp]
            er = Rotation.from_matrix(target_rot @ d.site_xmat[self.tcp].reshape(3, 3).T).as_rotvec()
            error = np.r_[ep, er * 0.3]
            if np.linalg.norm(error) < 0.0003:
                break
            mujoco.mj_jacSite(self.model, d, jp, jr, self.tcp)
            jac = np.vstack([jp[:, self.dids], jr[:, self.dids] * 0.3])
            delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 1e-4, error)
            d.qpos[self.qids] += np.clip(delta, -0.12, 0.12)
            d.qpos[self.qids] = np.clip(d.qpos[self.qids], limits[:, 0], limits[:, 1])
        mujoco.mj_kinematics(self.model, d)
        self.last_ik_error = float(np.linalg.norm(np.asarray(position) - d.site_xpos[self.tcp]))
        return d.qpos[self.qids].copy()

    def render(self, camera: str = 'front', width: int = 640, height: int = 480) -> np.ndarray:
        """Render uint8 RGB [height,width,3]; renderer allocated on first request."""
        if self.renderer is None or self.render_size != (width, height):
            if self.renderer is not None:
                self.renderer.close()
            self.model.vis.global_.offwidth = max(width, self.model.vis.global_.offwidth)
            self.model.vis.global_.offheight = max(height, self.model.vis.global_.offheight)
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
            self.render_size = (width, height)
            self.render_option = mujoco.MjvOption()
            self.render_option.sitegroup[:] = 0          # camera images: no debug sites (e.g. the red TCP marker)
        self.renderer.update_scene(self.data, camera=camera, scene_option=self.render_option)
        return self.renderer.render().copy()

    def close(self) -> None:
        """Release viewer and offscreen GL resources."""
        if self.viewer:
            close_viewer(self.viewer, self.viewer_thread)
            self.viewer = None
            self.viewer_thread = None
        if self.renderer:
            self.renderer.close()
            self.renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
