"""Common scenario runtime: scene construction, scripted human, benchmark events, predicates.

Scenario code never contains policy code. It consumes low-level controller commands
([6] joint targets + gripper opening) and exposes:
  * policy observations (robot proprioception + camera images, no privileged fields),
  * ground-truth human state / intention information (for later oracle phases and experts),
  * an ordered benchmark event stream and success/failure predicates.
"""
from __future__ import annotations
from collections import deque
import numpy as np
import mujoco
from intent_policy.benchmark.events import EventType as E
from intent_policy.benchmark.logger import EpisodeEventLogger
from intent_policy.sim.base_env import ManipulationEnv, ROOT
from intent_policy.sim.scene_builder import build_scene_xml, object_half_height, weld_name, robot_weld_name
from intent_policy.sim.scripted_human import make_human
from intent_policy.sim.human_body import HumanBody
from intent_policy.scenarios.config import ScenarioConfig, sample_variation, layout_of, is_cup

LIFT_HEIGHT = 0.03          # [m] above rest height => object counts as lifted by the robot
GRASP_DEBOUNCE = 2          # consecutive ticks required before grasp/release events
# robot_target_commit (agent-independent; docs/intent_act/INTENT_ACT_v3.2.md §7, user decision 2026-10-09): the
# irreversible part of acting on a target has begun: the TCP is within COMMIT_RADIUS (horizontally) of a candidate's
# commit point (where the TCP grasps / places) and at most COMMIT_DZ above it, i.e. it is descending onto that object /
# into that zone. Moving at travel height (hedging between candidates, passing over one) is not a commit.
COMMIT_RADIUS = 0.05        # [m] slots are 12 cm apart, place zones 12 x 12 cm
COMMIT_DZ = 0.06            # [m] cubes: travel height is ~10 cm above the grasp point, so the TCP must descend;
                            # cups (grasped at the rim, 4 cm below travel height): hovering right above one counts
GRASP_DZ = 0.012            # cube grasp: TCP above the object centre (fingertips reach ~3 cm below the TCP)
CUP_GRASP_DZ = 0.005        # cup grasp: TCP above the rim (fingertips ~2.5 cm down the wall)


def quat_z_to(direction: np.ndarray) -> np.ndarray:
    """WXYZ quaternion rotating +Z onto `direction`."""
    d = direction / (np.linalg.norm(direction) + 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, d)
    s, c = np.linalg.norm(axis), float(np.dot(z, d))
    if s < 1e-9:
        return np.array([1.0, 0, 0, 0]) if c > 0 else np.array([0.0, 1, 0, 0])
    angle = np.arctan2(s, c)
    return np.r_[np.cos(angle / 2), axis / s * np.sin(angle / 2)]


class BaseScenario:
    """Subclasses define `robot_target`, `human_context_extras`, `update_task`, `success_predicate`.

    Every catalog object of the scenario exists in the model; an episode puts 3-4 of them on the table (variation
    `on_table`) and parks the rest out of sight. Zones S/P/H/A/U and the robot zone come from configs/layout.yaml.
    """

    def __init__(self, cfg: ScenarioConfig, robot_config=ROOT / 'configs/robot.yaml', gui: bool = False):
        self.cfg = cfg
        self.scene = cfg.scene
        self.env = ManipulationEnv(robot_config, gui=gui, model_xml=build_scene_xml(self.scene))
        if 'grasp_hold_squeeze_m' in self.scene:
            self.env.hold_squeeze = float(self.scene['grasp_hold_squeeze_m'])
        self.env.attach_substep = bool(self.scene.get('grasp_attach_substep', False))
        m = self.env.model
        self.table_z = float(self.scene['table_top_z'])
        self.objects = list(self.scene.get('objects', {}))
        self.graspables = list(self.objects)
        self.layout = layout_of(self.scene)
        self.zones = self.layout['zones']
        rz = self.layout['robot_zone']
        self.robot_zone_lo = np.array([rz['x'][0], rz['y'][0], self.table_z], float)
        self.robot_zone_hi = np.array([rz['x'][1], rz['y'][1], self.table_z + rz['z_top']], float)
        self.radius = {k: self._radius(o) for k, o in self.scene['objects'].items()}
        self.body_id = {k: m.body(k).id for k in self.graspables}
        self.qadr = {k: int(m.joint(f'{k}_free').qposadr[0]) for k in self.graspables}
        self.dadr = {k: int(m.joint(f'{k}_free').dofadr[0]) for k in self.graspables}
        self.weld_id = {k: m.equality(weld_name(k)).id for k in self.graspables}
        for k in self.graspables:
            self.env.register_graspable(self.body_id[k], m.equality(robot_weld_name(k)).id)
        self.body_model = HumanBody({**self.scene['human'], 'floor_z': self.scene.get('floor_z', 0.0)})
        self.body_model.bind(m)
        self.mocap = {k: int(m.body(k).mocapid[0]) for k in ['human_hand', 'human_grasp_anchor']}
        root = m.body('robot_mount').id
        self.robot_geoms = []
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            while b > 0 and b != root:
                b = int(m.body_parentid[b])
            if b == root and m.geom_contype[g]:
                self.robot_geoms.append(g)
        self.human_geoms = list(self.body_model.proxy_geoms)
        self.object_geom_ids = {k: [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.body_id[k]] for k in self.objects}
        self.geom_object = {g: k for k, gs in self.object_geom_ids.items() for g in gs}
        self.robot_geom_set = set(self.robot_geoms)
        self.robot_chain = [m.body(n).id for n in ('base_link', 'shoulder_link', 'forearm_link', 'wrist_1_link',
                                                    'wrist_2_link', 'wrist_3_link', 'gripper_link')]
        # rest pose: the TCP at the nominal home configuration (the robot returns here at the end of a task)
        self.env.reset(np.asarray(self.env.cfg['home']))
        mujoco.mj_forward(m, self.env.data)
        self.rest_tcp = self.tcp()
        safety = dict(self.scene['safety'])
        safety.update({k: v for k, v in cfg.safety_constraints.items() if k in safety})
        self.safety = safety
        self.motion_cfg = self.scene['robot_motion']
        self.logger = EpisodeEventLogger(cfg.id, cfg.role)
        self.variation: dict = {}
        self.human = None

    # ------------------------------------------------------------------ properties / helpers
    @property
    def time(self) -> float:
        return float(self.env.data.time)

    @property
    def dt(self) -> float:
        return self.env.dt

    def pos(self, key: str) -> np.ndarray:
        return self.env.data.xpos[self.body_id[key]].copy()

    def quat(self, key: str) -> np.ndarray:
        return self.env.data.xquat[self.body_id[key]].copy()

    def speed(self, key: str) -> float:
        d = self.dadr[key]
        return float(np.linalg.norm(self.env.data.qvel[d:d + 3]))

    @staticmethod
    def _radius(obj: dict) -> float:
        """Horizontal half-extent of an object (cube half size / cup outer radius)."""
        return float(obj['size'][0])

    def zone_pos(self, key: str) -> np.ndarray:
        """Zone centre [x, y, table + zone height] (hand zones: where the receiving palm is)."""
        z = self.zones[key]
        return np.array([*z['xy'], self.table_z + float(z.get('height', 0.0))])

    def inside_zone(self, p: np.ndarray, zone: str, margin: float = 0.0) -> bool:
        """p (xy) inside the zone's tape square shrunk by `margin`."""
        half = np.broadcast_to(self.zones[zone].get('size', self.layout['marker_size']), 2) / 2 - margin
        return bool(np.all(np.abs(np.asarray(p)[:2] - np.asarray(self.zones[zone]['xy'])) <= half))

    def place_dz(self, key: str) -> float:
        """Height of a place location above the table where an object is set down there (v6: a coaster)."""
        return 0.0

    def in_robot_zone(self, p: np.ndarray, margin: float = 0.0) -> bool:
        return bool(np.all(np.asarray(p) >= self.robot_zone_lo - margin) and np.all(np.asarray(p) <= self.robot_zone_hi + margin))

    def robot_at_rest(self) -> bool:
        tol = float(self.cfg.success_conditions.get('rest_tolerance_m', 0.04))
        return bool(np.linalg.norm(self.tcp() - self.rest_tcp) < tol and not any(self.grasp_state.values()))

    def tcp(self) -> np.ndarray:
        return self.env.data.site_xpos[self.env.tcp].copy()

    def log(self, event_type, entity_id: str = 'scene', t: float | None = None, **payload):
        return self.logger.log(event_type, self.time if t is None else t, entity_id=entity_id, **payload)

    def protocol(self, step: str, entity_id: str = 'scene', **payload) -> None:
        if step not in self.protocol_done:
            self.protocol_done.append(step)
            self.log(E.PROTOCOL_STEP_COMPLETE, entity_id, step=step, **payload)

    def fail(self, reason: str) -> None:
        if self.failure is None and not self.success:
            self.failure = reason

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None, episode_id: str | None = None, spec: dict | None = None) -> None:
        """`spec`: discrete episode spec from a scenario list (None: drawn from the seed)."""
        seed = self.cfg.seed if seed is None else int(seed)
        self.seed = seed
        self.episode_id = episode_id or f'{self.cfg.id}__seed{seed:05d}'
        self.variation = var = self.sample_episode_variation(seed, spec)
        env, m, d = self.env, self.env.model, self.env.data
        env.reset(np.asarray(env.cfg['home']) + var['robot_home_offset'])
        self.on_table = list(var['on_table'])
        park = self.scene['parking']
        for i, k in enumerate(self.objects):
            obj = self.scene['objects'][k]
            q = self.qadr[k]
            if k in self.on_table:
                yaw = var['object_yaw'][k]
                dz = float((var.get('object_dz') or {}).get(k, 0.0))      # standing on static decor (a tray)
                d.qpos[q:q + 3] = [*var['object_xy'][k], self.table_z + dz + object_half_height(obj) + 0.001]
                d.qpos[q + 3:q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            else:                                         # parked on the floor, out of every camera view
                d.qpos[q:q + 3] = [park['x'], park['y0'] + i * park['dy'], float(self.scene.get('floor_z', 0.0)) + object_half_height(obj) + 0.001]
                d.qpos[q + 3:q + 7] = [1, 0, 0, 0]
        for eid in self.weld_id.values():
            d.eq_active[eid] = False
        if hasattr(self.body_model, 'default_active'):  # a reused scenario starts like a new one (I2 ends on either arm;
            self.body_model.set_active(self.body_model.default_active)    # collect_pairs reuses it, 2026-10-31)
        self.human = make_human(self.scene, self.cfg.human_behavior, var)
        mb = self.cfg.human_behavior.get('motion_bank')
        if mb:                                           # generated hand motions (sim/motion_bank.py), loaded once
            if getattr(self, '_bank', None) is None:
                from intent_policy.sim.motion_bank import MotionBank
                J = self.body_model.J
                height = J['Head'][2] - min(J[k][2] for k in ('L_Ankle', 'R_Ankle', 'L_Foot', 'R_Foot'))
                self._bank = MotionBank(mb['folder'], J['Pelvis'], float(self.scene['human'].get('facing_deg', 180.0)),
                                        float(self.scene.get('floor_z', 0.0)), float(height),
                                        hand=mb.get('hand', 'right'), split_manifest=mb.get('split_manifest'),
                                        split=mb.get('split'))
            self.human.bank, self.human.bank_rng = self._bank, np.random.default_rng([int(seed), 7])
            self.human.bank_table_top = self.table_z
        self.body_model.reset_shape()
        self.human.reach = self.body_model.clamp          # the scripted palm never aims beyond the body's reach
        length = float(self.cfg.human_behavior.get('point_length_m', 0.35))
        self.human.point_pose = lambda target: self.body_model.pointing_palm(target, length)
        self.human.shoulder = np.asarray(self.body_model.J[f'{self.body_model.active.upper()}_Shoulder'], float)
        self.max_reach_error = 0.0
        self._apply_human_pose()
        mujoco.mj_forward(m, d)
        env.settle(10)
        self.rest_height = {k: float(self.pos(k)[2]) for k in self.graspables}
        self.start_pos = {k: self.pos(k) for k in self.on_table}
        # objects the robot must neither move nor touch: everything on a slot except the (final) target and, with a
        # change of mind, the old target (the robot may hold it when the human changes; it must then put it back,
        # checked at the end: put_back_max_m)
        self.change_old = (var.get('change') or {}).get('old')
        self.watch = [k for k in self.on_table if k not in (var['target_object'], self.change_old)
                      and k not in (var['spec'].get('u_blocks') or [])]
        self.change_t: float | None = None
        self.change_robot_state: str | None = None      # approach | grasp | holding | delivered (plan v5 metadata)
        # Trackers
        self.protocol_done: list[str] = []
        self.success = False
        self.failure: str | None = None
        self.stable = 0
        self.grasp_state = {k: False for k in self.graspables}
        self.grasp_count = {k: 0 for k in self.graspables}
        self.robot_lifted: str | None = None
        self.ever_lifted: set[str] = set()
        self.robot_moving = False
        self.moving_now = False
        self.arm_moving_now = False
        self.idle_since: float | None = None
        self.prev_tcp = self.tcp()
        self.prev_grip = float(d.qpos[env.gid])
        self.tcp_speed = 0.0
        self.contact_active = False
        self.violation_active = False
        self.min_human_distance = np.inf
        self.human_distance = self._human_distance()
        self.intention_changes: list[dict] = []     # payloads of human_intention_change events (with time t)
        self.evidence_events: list[dict] = []       # payloads of human_intention_evident events (with time t)
        self.cue_t: float | None = None
        self.commit_stage_name: str | None = None
        self.committed: str | None = None
        self.done = False
        self.logger.start(self.episode_id, self.time, seed=seed, variation=var)
        self.realized_forecast_fault = None          # C3 anchored fault window applied in this episode (metadata)
        self.on_reset()

    def sample_episode_variation(self, seed, spec=None):
        return sample_variation(self.cfg, seed, spec)

    def on_reset(self) -> None:
        """Subclass hook for task-specific trackers."""

    # ------------------------------------------------------------------ human
    def _apply_human_pose(self) -> None:
        """Pose the mannequin from the scripted palm position (lean + arm IK) and move the distance proxies."""
        aim = self.human.point_at
        progress = getattr(self.human, 'gesture_progress', 1.0)     # a pointing hand forms with the arm motion
        bi = getattr(self.human, 'bimanual', None)            # v7: both hands on one object
        pose = self.body_model.apply(self.env.data, self.human.pos, aim, self.human.hand_shape,
                                     aim_blend=progress, shape_from=getattr(self.human, 'shape_from', None),
                                     shape_progress=getattr(self.human, 'shape_progress', None),
                                     palm_up=self.human.palm_up, smooth=True,
                                     idle_palm=self.human.idle_palm() if bi else None,
                                     idle_shape=bi.get('shape', 'grasp') if bi else
                                     ('relaxed' if getattr(self.human, 'uses_bimanual', False) else None))
        self.human_pose = pose
        if getattr(self.body_model, 'held_object_frame', 'world') == 'hand':
            self._carry_held_objects()
        self.human.torso = pose['chest']
        self.human_lean = pose['lean']
        self.max_reach_error = max(self.max_reach_error, pose['reach_error'])

    def hand_clearance_lift(self, path, shapes, palm_up: float = 0.0, point_at=None, ignore=(), holding=None,
                            margin: float = 0.003, held_margin: float = 0.0005, touch=()) -> np.ndarray:
        """Vertical lift [N] each palm sample of `path` needs so that the whole hand (every vertex of each rendered
        shape in `shapes`, posed by the mannequin's own IK) and the object it holds stay above the table top and the
        tops of the objects on it (except `ignore`, e.g. the object the motion reaches for). Exact mesh geometry, not
        the palm point: the hand reaches 85-130 mm below the palm depending on its shape (measured 2026-10-30).
        Compiled hand variants and a rigid grasp (held_object_frame: hand) only; otherwise zeros."""
        bm, m = self.body_model, self.env.model
        path = np.asarray(path, float)
        geoms = getattr(bm, 'compiled_hand_geoms', {}).get(bm.active)
        if not geoms or getattr(bm, 'held_object_frame', 'world') != 'hand':
            return np.zeros(len(path))
        if not hasattr(self, '_hand_vertices'):
            self._hand_vertices = {}
            for shape, g in geoms.items():
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, m.geom_quat[g])
                self._hand_vertices[shape] = m.geom_pos[g] + v @ R.reshape(3, 3).T     # in the hand segment frame
            (x0, x1), (y0, y1) = self.layout['table']['x'], self.layout['table']['y']
            self._table_rect = (x0, x1, y0, y1)
        verts = [self._hand_vertices[s] for s in dict.fromkeys(shapes) if s in self._hand_vertices]
        if not verts:
            return np.zeros(len(path))
        verts = np.concatenate(verts)
        held = None
        if holding is not None:
            # the grasp pose in the anchor frame from the current poses, exactly as attach_to_hand sets it (a motion
            # planned in the tick of the grasp runs before the attach request is applied: the weld data is stale)
            d = self.env.data
            anchor = self.mocap['human_grasp_anchor']
            Ra, Ro = np.zeros(9), np.zeros(9)
            mujoco.mju_quat2Mat(Ra, d.mocap_quat[anchor])
            mujoco.mju_quat2Mat(Ro, self.quat(holding))
            Ra, Ro = Ra.reshape(3, 3), Ro.reshape(3, 3)
            r, hh = float(self.radius[holding]), float(object_half_height(self.scene['objects'][holding]))
            a = np.linspace(0, 2 * np.pi, 48, endpoint=False)          # rim points ~6 mm apart
            ring = np.concatenate([np.c_[r * np.cos(a), r * np.sin(a), np.full(48, -hh)], [[0.0, 0.0, -hh]]])
            held = (Ra.T @ (self.pos(holding) - d.mocap_pos[anchor]), Ra.T @ Ro, ring)
        obstacles = []
        for k in self.on_table:
            if k == holding or k in ignore:
                continue                     # the object the hand takes / holds (it closes around it)
            p = self.pos(k)
            # footprint inflated by 1 cm: the hand vertices are 5-16 mm apart and the held object ring ~12 mm, so an
            # edge between two samples cannot clip an object that a sample just misses
            pad = 0.0 if k in touch else 0.01     # `touch`: worked on from above (its top is the limit, fingertips on it)
            obstacles.append((p[:2], float(self.radius[k]) + pad, float(p[2] + object_half_height(self.scene['objects'][k]))))
        touchable = [k in touch for k in self.on_table if k != holding and k not in ignore]
        reserved = self.reserved_obstacles() + self.decor_obstacles()
        obstacles += reserved
        touchable += [False] * len(reserved)
        x0, x1, y0, y1 = self._table_rect

        boxes = [(np.asarray(d['xy'], float), np.asarray(d['half'], float), float(d['height']), float(d['wall']))
                 for d in self.scene.get('decor', []) or [] if d.get('type') == 'open_box']

        def need(points, gap):
            req = np.full(len(points), -np.inf)
            on = (points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)
            req[on] = self.table_z + gap
            for xy, half, h, w in boxes:        # an open box: over its walls the top, inside it the floor
                rel = np.abs(points[:, :2] - xy)
                outer = np.all(rel <= half + 0.01, axis=1)
                inner = np.all(rel <= half - w - 0.01, axis=1)
                req[outer & ~inner] = np.maximum(req[outer & ~inner], self.table_z + h + gap)
                req[inner] = np.maximum(req[inner], self.table_z + w + gap)
            for (xy, rad, top), touch in zip(obstacles, touchable):
                inside = np.linalg.norm(points[:, :2] - xy, axis=1) <= rad
                req[inside] = np.maximum(req[inside], top + (0.0 if touch else gap))
            return float(np.max(req - points[:, 2], initial=0.0))

        lift = np.zeros(len(path))
        ups = palm_up if isinstance(palm_up, (tuple, list)) else (palm_up,)     # a hand turning: both orientations
        bi = getattr(self.human, 'bimanual', None)
        idle_verts = None
        if bi:                                   # both hands on one object: the left hand's mesh too
            ig = getattr(bm, 'compiled_hand_geoms', {}).get(bm.idle, {}).get(bi.get('shape', 'grasp'))
            if ig is not None:
                mid = m.geom_dataid[ig]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
                Rg = np.zeros(9)
                mujoco.mju_quat2Mat(Rg, m.geom_quat[ig])
                idle_verts = m.geom_pos[ig] + v @ Rg.reshape(3, 3).T
            offset = self.human.idle_palm() - self.human.pos
        for i, palm in enumerate(path):
            for up in ups:
                pose = bm.pose(palm, point_at, up, idle_palm=palm + offset if bi else None)
                pos, R = pose['segments'][f'{bm.active}_hand']
                lift[i] = max(lift[i], need(pos + verts @ np.asarray(R).T, margin))
                if idle_verts is not None:
                    ipos, iR = pose['segments'][f'{bm.idle}_hand']
                    lift[i] = max(lift[i], need(ipos + idle_verts @ np.asarray(iR).T, margin))
                if held is not None:
                    rel, Rrel, ring = held
                    c = palm + np.asarray(R) @ rel
                    lift[i] = max(lift[i], need(c + ring @ (np.asarray(R) @ Rrel).T, held_margin))
        return np.maximum(lift, 0.0)

    def decor_obstacles(self) -> list:
        """Static decor boxes (trays) as obstacles of the hand clearance: (centre xy, half diagonal, top z)."""
        if not hasattr(self, '_decor_obs'):
            self._decor_obs = []
            for d in self.scene.get('decor', []) or []:
                if d.get('type', 'box') == 'box':
                    hx, hy, hz = d['size']
                    self._decor_obs.append((np.asarray(d['xy'], float), float(np.hypot(hx, hy)), self.table_z + 2 * float(hz)))
                elif d.get('type') == 'open_box':      # handled as walls + floor (open_boxes below)
                    continue
        return list(self._decor_obs)

    def reserved_obstacles(self) -> list:
        """Volumes the human's hand keeps clear of besides the objects: [(xy, radius, top z)] (task hook)."""
        return []

    def _two_hand_frame(self):
        """(origin, R) of an object held with both hands: origin between the palms, x from the left palm to the right
        one (horizontal), z up: the object stays level and between the hands (it does not turn with one wrist)."""
        right, left = np.asarray(self.human.pos, float), np.asarray(self.human.idle_palm(), float)
        x = right - left
        x[2] = 0.0
        x /= max(float(np.linalg.norm(x)), 1e-9)
        z = np.array([0.0, 0.0, 1.0])
        return (right + left) / 2, np.column_stack([x, np.cross(z, x), z])

    def _carry_held_objects(self) -> None:
        """A held object is part of the hand's kinematic chain: put it at its grasp pose in the (just moved) anchor
        frame, at rest, so it does not trail the 50 ms hand steps through the physics substeps (the weld then holds it
        there). Rigid-grasp mode only."""
        m, d = self.env.model, self.env.data
        anchor = self.mocap['human_grasp_anchor']
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, d.mocap_quat[anchor])
        bi = getattr(self.human, 'bimanual', None)
        for k, eid in self.weld_id.items():
            if not d.eq_active[eid]:
                continue
            q, v = self.qadr[k], self.dadr[k]
            if bi and getattr(self, '_two_hand_rel', {}).get(k) is not None:
                o, Rf = self._two_hand_frame()
                rel_p, rel_R = self._two_hand_rel[k]
                d.qpos[q:q + 3] = o + Rf @ rel_p
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, np.ascontiguousarray(Rf @ rel_R).ravel())
                d.qpos[q + 3:q + 7] = quat
                d.qvel[v:v + 6] = 0.0
                # keep the weld consistent with the anchor (it holds the object between ticks)
                Ra = R.reshape(3, 3)
                m.eq_data[eid, 3:6] = Ra.T @ (d.qpos[q:q + 3] - d.mocap_pos[anchor])
                inv, rel = np.zeros(4), np.zeros(4)
                mujoco.mju_negQuat(inv, d.mocap_quat[anchor])
                mujoco.mju_mulQuat(rel, inv, quat)
                m.eq_data[eid, 6:10] = rel
                continue
            d.qpos[q:q + 3] = d.mocap_pos[anchor] + R.reshape(3, 3) @ m.eq_data[eid, 3:6]
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, d.mocap_quat[anchor], m.eq_data[eid, 6:10])
            level = self.human.behavior.get('level_on_put') or {}
            seg = self.human.segment
            if seg is not None and seg.label in level.get('labels', ()):
                # setting the object down: its tilt is taken out (yaw kept) over the end of the lowering, about its centre
                u = (self.human.t_last - seg.start_t) / seg.duration
                a0 = float(level['from_progress'])
                alpha = float(np.clip((u - a0) / max(1.0 - a0, 1e-6), 0.0, 1.0))
                Ro = np.zeros(9)
                mujoco.mju_quat2Mat(Ro, quat)
                z = Ro.reshape(3, 3)[:, 2]
                axis = np.cross(z, [0.0, 0.0, 1.0])
                angle = float(np.arccos(np.clip(z[2], -1.0, 1.0)))
                if np.linalg.norm(axis) > 1e-9 and alpha > 0:
                    qa, out = np.zeros(4), np.zeros(4)
                    mujoco.mju_axisAngle2Quat(qa, axis / np.linalg.norm(axis), alpha * angle)
                    mujoco.mju_mulQuat(out, qa, quat)
                    quat = out
            d.qpos[q + 3:q + 7] = quat
            d.qvel[v:v + 6] = 0.0

    def attach_to_hand(self, key: str) -> None:
        m, d = self.env.model, self.env.data
        eid = self.weld_id[key]
        m.eq_data[eid, 0:3] = 0
        anchor = self.mocap['human_grasp_anchor']
        if getattr(self.body_model, 'held_object_frame', 'world') == 'hand':
            # weld relpose = the object's pose in the (rotating) anchor frame: a rigid grasp
            qa = d.mocap_quat[anchor]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, qa)
            m.eq_data[eid, 3:6] = R.reshape(3, 3).T @ (self.pos(key) - d.mocap_pos[anchor])
            inv, rel = np.zeros(4), np.zeros(4)
            mujoco.mju_negQuat(inv, qa)
            mujoco.mju_mulQuat(rel, inv, self.quat(key))
            m.eq_data[eid, 6:10] = rel
        else:
            m.eq_data[eid, 3:6] = self.pos(key) - d.mocap_pos[anchor]
            m.eq_data[eid, 6:10] = self.quat(key)
        d.eq_active[eid] = True

    def detach_from_hand(self, key: str) -> None:
        self.env.data.eq_active[self.weld_id[key]] = False

    def human_context(self) -> dict:
        tcp = self.tcp()
        ctx = dict(t=self.time, table_top_z=self.table_z, tcp=tcp,
                   object_positions={k: self.pos(k) for k in self.on_table},
                   object_radius=dict(self.radius), zone_positions={k: self.zone_pos(k) for k in self.zones},
                   object_half_height={k: object_half_height(self.scene['objects'][k]) for k in self.objects},
                   robot_grasping=dict(self.grasp_state), robot_lifted_object=self.robot_lifted,
                   human_robot_distance=self.human_distance, robot_speed=self.tcp_speed, robot_moving=self.robot_moving,
                   robot_at_rest=self.robot_at_rest(), robot_chain=[self.env.data.xpos[b].copy() for b in self.robot_chain],
                   robot_committed=self.committed,                  # candidate the TCP clearly heads for (any agent)
                   protocol_done=tuple(self.protocol_done),
                   tcp_xy_distance=lambda p: float(np.linalg.norm(np.asarray(p)[:2] - tcp[:2])),
                   hand_clearance=self.hand_clearance_lift)
        ctx.update(self.human_context_extras())
        return ctx

    def human_context_extras(self) -> dict:
        return {}

    # ------------------------------------------------------------------ step
    def step(self, command: np.ndarray) -> bool:
        """Advance one control tick with a [7] joint/gripper command. Returns `done`."""
        if self.done:
            raise RuntimeError('episode finished; call reset()')
        t0 = self.time
        self.human.update(t0, self.dt, self.human_context())
        for kind, key in self.human.requests:
            if kind == 'attach':
                self.attach_to_hand(key)
            elif kind == 'detach':
                self.detach_from_hand(key)
            elif kind == 'two_hand':                  # v7: held with both hands: its pose in the two-hand frame
                o, Rf = self._two_hand_frame()
                Ro = np.zeros(9)
                mujoco.mju_quat2Mat(Ro, self.quat(key))
                self._two_hand_rel = getattr(self, '_two_hand_rel', {})
                self._two_hand_rel[key] = (Rf.T @ (self.pos(key) - o), Rf.T @ Ro.reshape(3, 3))
            elif kind == 'one_hand':
                getattr(self, '_two_hand_rel', {}).pop(key, None)
            elif kind == 'switch_arm':               # v7 I2: the other arm becomes the moving one (task hook)
                self._switch_arm(key)
        for event_type, payload in self.human.events:
            if event_type == E.PROTOCOL_STEP_COMPLETE.value:
                self.protocol(payload.pop('step'), 'human', **payload)
            else:
                if event_type == E.HUMAN_INTENTION_CHANGE.value:
                    self.intention_changes.append(dict(t=t0, **payload))
                    if self.change_t is None:
                        self.change_t, self.change_robot_state = t0, self.robot_state_wrt(self.change_old)
                elif event_type == E.HUMAN_INTENTION_EVIDENT.value:
                    self.evidence_events.append(dict(t=t0, **payload))
                elif event_type == E.HUMAN_CUE_ONSET.value and self.cue_t is None:
                    self.cue_t = t0
                self.log(event_type, 'human', t=t0, **payload)
        self._apply_human_pose()
        self.env.step(command)
        self._update_robot_events()
        self._update_commitment()
        self._update_safety()
        self._update_other_objects()
        self.update_task()
        self._update_generic_failures()
        if not self.success and self.failure is None:
            self.stable = self.stable + 1 if self.success_predicate() and self.put_back_ok() else 0
            if self.stable * self.dt >= self.cfg.timing['stable_s']:
                self.success = True
                self.log(E.TASK_SUCCESS, 'scene', ct=self.time)
        if self.failure is None and not self.success and self.time >= self.cfg.timing['timeout_s']:
            self.fail('timeout')
        if self.failure is not None:
            self.log(E.TASK_FAILURE, 'scene', reason=self.failure)
        if self.success or self.failure is not None:
            self.done = True
            self.logger.end(self.time, success=self.success, failure=self.failure)
        return self.done

    def _update_robot_events(self) -> None:
        t = self.time
        tcp = self.tcp()
        grip = float(self.env.data.qpos[self.env.gid])
        self.tcp_speed = float(np.linalg.norm(tcp - self.prev_tcp) / self.dt)
        grip_speed = abs(grip - self.prev_grip) / self.dt
        self.prev_tcp, self.prev_grip = tcp, grip
        moving = self.tcp_speed > self.motion_cfg['tcp_speed_threshold'] or grip_speed > self.motion_cfg['gripper_speed_threshold']
        self.moving_now = moving              # instantaneous; robot_moving has hysteresis (events)
        self.arm_moving_now = self.tcp_speed > self.motion_cfg['tcp_speed_threshold']   # separation monitoring
        if moving:
            self.idle_since = None
            if not self.robot_moving:
                self.robot_moving = True
                self.log(E.ROBOT_MOTION_START, 'robot', tcp_speed=self.tcp_speed)
        elif self.robot_moving:
            self.idle_since = t if self.idle_since is None else self.idle_since
            if t - self.idle_since >= self.motion_cfg['min_idle_s'] - 1e-9:
                self.robot_moving = False
                self.log(E.ROBOT_MOTION_END, 'robot', stopped_at=self.idle_since)
        for k in self.graspables:
            g = self.env.is_grasped(self.body_id[k])
            self.grasp_count[k] = self.grasp_count[k] + 1 if g != self.grasp_state[k] else 0
            if self.grasp_count[k] >= GRASP_DEBOUNCE:
                self.grasp_state[k] = g
                self.grasp_count[k] = 0
                self.log(E.OBJECT_GRASP if g else E.OBJECT_RELEASE, 'robot', object=k,
                         position=self.pos(k).round(4).tolist())
        lifted = [k for k in self.graspables if self.grasp_state[k] and self.pos(k)[2] > self.rest_height[k] + LIFT_HEIGHT]
        self.robot_lifted = lifted[0] if lifted else None
        if self.robot_lifted:
            self.ever_lifted.add(self.robot_lifted)
        # the old target of a change of mind may be held (also picked before the robot could see the change); it must
        # end up back where it stood (put_back_ok)
        if self.robot_lifted is not None and self.robot_lifted not in (self.robot_target(), self.change_old):
            self.fail('wrong_object_manipulated')

    def commit_stage(self) -> tuple[str, dict, str] | None:
        """(stage name, {candidate: position}, correct candidate) while a target choice is pending (subclasses)."""
        return None

    def _update_commitment(self) -> None:
        """Log robot_target_commit when the TCP starts descending onto one candidate (whatever the agent is)."""
        stage = self.commit_stage()
        if stage is None or len(stage[1]) < 2:
            self.commit_stage_name = None
            return
        name, candidates, correct = stage
        if name != self.commit_stage_name:
            self.commit_stage_name, self.committed = name, None
        tcp = self.tcp()
        for k, p in candidates.items():
            p = np.asarray(p, float)
            if k != self.committed and np.linalg.norm(tcp[:2] - p[:2]) <= COMMIT_RADIUS and tcp[2] <= p[2] + COMMIT_DZ:
                self.committed = k
                self.log(E.ROBOT_TARGET_COMMIT, 'robot', target=k, correct=k == correct, stage=name)
                break

    def _pre_grasp_stage(self, stage: str, keys: list[str], correct: str):
        """Target choice among graspables: from the human cue until the robot grasps anything."""
        if self.cue_t is None or any(self.grasp_state.values()) or self.ever_lifted:
            return None
        return stage, {k: self.commit_point(k) for k in keys}, correct

    def commit_point(self, key: str) -> np.ndarray:
        """Where the robot goes to grasp `key` (commitment is measured towards these points)."""
        return self.grasp_point(key)

    def grasp_point(self, key: str) -> np.ndarray:
        """TCP position for a top-down grasp of `key`: cubes across two faces, cups by the wall on the -X side
        of the rim (the fingers straddle the wall, they close along world X)."""
        obj, p = self.scene['objects'][key], self.pos(key)
        if is_cup(obj):
            r, h = obj['size']
            return p + [-(r - float(obj.get('wall', 0.004)) / 2), 0.0, h / 2 + CUP_GRASP_DZ]
        return p + [0.0, 0.0, float(obj.get('grasp_dz', GRASP_DZ))]   # v6: tall objects (the can) near their top

    def resting(self, key: str) -> bool:
        """Released by the robot, still, standing on the table."""
        sc = self.cfg.success_conditions
        return (not self.grasp_state[key] and self.speed(key) < sc.get('object_speed_max', 0.04)
                and self.pos(key)[2] < self.rest_height[key] + 0.01)

    def _human_distance(self) -> float:
        m, d = self.env.model, self.env.data
        best = 1.0
        for r in self.robot_geoms:
            for h in self.human_geoms:
                best = min(best, float(mujoco.mj_geomDistance(m, d, r, h, 1.0, None)))
        return best

    def _update_safety(self) -> None:
        dist = self._human_distance()
        self.human_distance = dist
        self.min_human_distance = min(self.min_human_distance, dist)
        contact = dist <= self.safety['contact_distance_m']
        if contact and not self.contact_active:
            self.log(E.HUMAN_ROBOT_CONTACT, 'robot', distance=dist)
            if self.cfg.safety_constraints.get('forbid_human_contact', True):
                self.fail('human_robot_contact')
        self.contact_active = contact
        # Speed & separation monitoring concerns the arm: a gripper-only motion (the hand-over release into a
        # hand that holds the object) does not count; contact is always counted.
        violation = dist < self.safety['violation_distance_m'] and self.arm_moving_now
        if violation and not self.violation_active:
            self.log(E.SAFETY_DISTANCE_VIOLATION, 'robot', distance=dist, tcp_speed=self.tcp_speed, kind='separation')
        self.violation_active = violation

    def _update_other_objects(self) -> None:
        """'Không chạm vật khác': the robot may not touch or move any watched (non-target) object."""
        limit = float(self.cfg.success_conditions.get('other_object_moved_max_m', 0.02))
        d = self.env.data
        for i in range(d.ncon):
            c = d.contact[i]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                k = self.geom_object.get(int(a))
                if k in self.watch and int(b) in self.robot_geom_set:
                    self.fail('touched_other_object')
                    return
        for k in self.watch:
            if np.linalg.norm(self.pos(k) - self.start_pos[k]) > limit:
                self.fail('other_object_moved')
                return

    def _update_generic_failures(self) -> None:
        for k in self.on_table:
            if self.pos(k)[2] < self.table_z - 0.05:
                self.fail(f'{k}_fell_off_table')
        if not np.isfinite(self.env.data.qpos).all():
            self.fail('simulation_unstable')

    def robot_state_wrt(self, key: str | None) -> str | None:
        """Where the robot stands with respect to `key` (the old target at a change of mind), agent-independent:
        holding it, delivered (moved > 2 cm from its spot, released), grasp (gripper within 4 cm horizontally and
        lowered to < 6 cm above its grasp point) or approach."""
        if key is None:
            return None
        if self.grasp_state.get(key, False):
            return 'holding'
        if np.linalg.norm(self.pos(key)[:2] - self.start_pos[key][:2]) > 0.02:
            return 'delivered'
        tcp, g = self.tcp(), self.grasp_point(key)
        if np.linalg.norm(tcp[:2] - g[:2]) < 0.04 and tcp[2] < g[2] + 0.06:
            return 'grasp'
        return 'approach'

    def put_back_ok(self) -> bool:
        """After a change of mind the old target rests where it stood (also when the robot had to carry it back)."""
        old = self.change_old
        if old is None or not self.human.changed or old == self.robot_target():
            return True
        limit = float(self.cfg.success_conditions.get('put_back_max_m', 0.03))
        return bool(not self.grasp_state.get(old, False) and self.resting(old)
                    and np.linalg.norm(self.pos(old) - self.start_pos[old]) <= limit)

    # ------------------------------------------------------------------ subclass API
    def robot_target(self) -> str:
        """Graspable the robot is supposed to manipulate in this episode."""
        raise NotImplementedError

    def update_task(self) -> None:
        raise NotImplementedError

    def success_predicate(self) -> bool:
        raise NotImplementedError

    def ground_truth_intention_information(self) -> dict:
        """Task-specific privileged fields; extended by subclasses."""
        return {}

    # ------------------------------------------------------------------ oracle slots (Phase 2)
    def oracle_slots(self, evidence: list[dict]) -> dict:
        """{'object': key | None, 'region': key | None} the human is known to target, given the evidence
        entries predictable so far (ScriptedHuman.known_evidence). Subclasses map evidence to slots."""
        return {'object': None, 'region': None}

    def oracle_position(self, key: str) -> np.ndarray:
        """World position of an oracle target key (object, zone or scenario-specific point)."""
        if key in self.body_id:
            return self.pos(key)
        if key in self.zones:
            return self.zone_pos(key)
        raise KeyError(key)

    def oracle_alternatives(self, key: str) -> list[str]:
        """Other candidates of the same slot (for the wrong-information condition)."""
        if key in self.on_table:
            return [k for k in self.on_table if k != key and k not in (self.variation['spec'].get('u_blocks') or [])]
        if key in self.zones:
            return [k for k, z in self.zones.items() if k != key and z['kind'] == self.zones[key]['kind']]
        return []

    # ------------------------------------------------------------------ public interface
    def get_observation(self, cameras=('high', 'wrist'), width: int = 128, height: int = 96, command=None) -> dict:
        """Policy observation: proprioception + RGB only. No privileged/intention fields. `command` (the restricted-action
        mapper): also the controller's own command, set-point xyz and gripper target, appended to observation.state
        [10] -> [14]. The measured TCP lags the set-point by 1-4 cm while moving; the controller knows its command."""
        state = self.env.observe().policy_vector()
        if command is not None:
            state = np.r_[state, command.setpoint, command.gripper_target].astype(np.float32)
        obs = {'observation.state': state}
        for cam in cameras:
            obs[f'observation.images.{cam}'] = self.env.render(cam, width, height)
        return obs

    def human_keypoints(self, names) -> np.ndarray:
        """World positions (J, 3) of the mannequin's keypoints as currently posed (= rendered); ground truth for
        intent labels, never a policy observation."""
        return self.body_model.keypoints(self.human_pose, names)

    def get_human_state(self):
        return self.human.state(self.time)

    def get_ground_truth_intention_information(self) -> dict:
        """Simulator ground truth for later oracle phases. NEVER used by the Phase-1 baseline."""
        t = self.time
        hs = self.human.state(t)
        horizons = (0.5, 1.0)
        plan = self.human.planned_hand_position
        future = {f'{h:.1f}s': plan(t + h).round(5).tolist() for h in horizons}
        future_vel = {f'{h:.1f}s': ((plan(t + h + self.dt) - plan(t + h - self.dt)) / (2 * self.dt)).round(5).tolist()
                      for h in horizons}
        info = dict(selected_object=hs.selected_object, selected_target=hs.selected_target,
                    hand_position=hs.hand_position.round(5).tolist(), hand_velocity=hs.hand_velocity.round(5).tolist(),
                    future_hand_position=future, future_hand_velocity=future_vel,
                    human_interaction_state=hs.interaction_state, frame='world',
                    note='future_* follow the currently scheduled scripted segment (reactive re-planning not included)')
        info.update(self.ground_truth_intention_information())
        return info

    def get_events(self) -> list[dict]:
        return self.logger.to_list()

    def is_success(self) -> bool:
        return self.success

    def is_failure(self) -> bool:
        return self.failure is not None

    def snapshot(self) -> dict:
        """Per-tick state record: robot, human and object states."""
        return dict(t=round(self.time, 4), robot=self.env.observe().to_dict(),
                    human=self.human.state(self.time).to_dict(),
                    objects={k: dict(position=self.pos(k).round(5).tolist(), quat=self.quat(k).round(5).tolist(),
                                     robot_grasped=self.grasp_state[k]) for k in self.on_table},
                    human_robot_distance=round(self.human_distance, 5), tcp_speed=round(self.tcp_speed, 5))

    def close(self) -> None:
        self.env.close()
