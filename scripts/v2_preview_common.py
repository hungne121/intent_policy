"""Shared scaffolding for the V2 C2/C3/C4/W1/W2/W3 paired expert previews.

Scripted experts in MuJoCo, not trained ACT. Both experts share the scene, the
seed and the human clip; Intent additionally receives the real ManiCast forecast
of the past 0.4 s. Independent robot work (approach after a confirmed cue, grasp,
lift, transport, place, retreat) runs at the fixed `work_speed` pace in both
modes; human-paced following is used only while Intent tracks a forecast before
the observed cue confirms it (user decision 2026-11-13 04:47).

The human and the robot are written as generators that yield one frame/command per
20 Hz tick, so each task script reads as a sequence of causal steps with explicit
waits on observable conditions.
"""
from pathlib import Path
import os
import json
import time

import mujoco
import numpy as np

from layout_gallery import make
from i3_collect import make_env, smooth, geom_sets, JOINTS, DT, TOL, smooth_motion_path, motion_deviation
from intent_policy.sim.motion_bank_v2 import to_scene, palm_path, segment_ranges
from i3_self_check import I3SelfCheck
from intent_policy.sim.scene_builder_v2 import weld_name
from intent_policy.sim.demo_contact_audit import DemoContactAudit, PairRule
from intent_policy.benchmark.video import VideoWriter
from intent_policy.experts.human_paced import (observed_hand_speed, forecast_hand_speed, HumanPace,
                                                CartesianFollower, work_speed, WORK_ACCEL)
from contact_context import overlap_tolerance

ROOT = Path(__file__).resolve().parents[1]
import yaml
V2_CONFIG = yaml.safe_load((ROOT / 'configs/previews/v2_common.yaml').read_text())
BASE_XY = tuple(float(v) for v in V2_CONFIG['robot_base_xy'])


def apply_station(scene, xml, stand_xy=None, cam=None):
    """Shared V2 station (configs/previews/v2_common.yaml): robot base across the table facing the human,
    human stand, continuous HY-Motion skin, side overview camera. Returns the edited scene XML."""
    import xml.etree.ElementTree as ET
    from intent_policy.sim.scene_builder_v2 import add_continuous_human_skin, look_at_xyaxes, ROOT as SIM_ROOT
    scene['human']['stand_xy'] = list(stand_xy if stand_xy is not None else V2_CONFIG['human_stand_xy'])
    scene['human']['continuous_skin'] = bool(V2_CONFIG['continuous_skin'])
    root = ET.fromstring(xml)
    if scene['human']['continuous_skin'] and scene['human'].get('model_version') not in ('v3', 'v4') \
            and root.find("./asset/skin[@name='human_surface']") is None:
        add_continuous_human_skin(root, SIM_ROOT / 'assets/human')
    cam = cam or V2_CONFIG['camera_high']
    high = root.find("./worldbody/camera[@name='high']")
    high.set('pos', ' '.join(str(v) for v in cam['pos'])); high.set('xyaxes', look_at_xyaxes(cam['pos'], cam['target'])); high.set('fovy', str(cam['fovy']))
    # 15/11 (user): a second overview from behind the robot looking forward at the person
    back = V2_CONFIG.get('camera_back')
    if back and root.find("./worldbody/camera[@name='back']") is None:
        ET.SubElement(root.find('worldbody'), 'camera', name='back', pos=' '.join(str(v) for v in back['pos']),
                      xyaxes=look_at_xyaxes(back['pos'], back['target']), fovy=str(back['fovy']))
    mount = root.find("./worldbody/body[@name='robot_mount']")
    mount.set('pos', f'{BASE_XY[0]} {BASE_XY[1]} 0.62')
    half = np.radians(float(V2_CONFIG['robot_base_yaw_deg'])) / 2
    mount.set('quat', f'{np.cos(half):.7f} 0 0 {np.sin(half):.7f}')
    return ET.tostring(root, encoding='unicode')
HORIZONS = np.arange(1, 11) / 10
PALM_TAU = .08            # s, palm command smoothing (shared by every V2 task)
GRIP_OPEN, GRIP_CLOSED = .085, 0.
TRAVEL_Z = .80            # free travel height of the TCP
GRASP_DZ = .024           # TCP above an object centre for a top-down grasp (I3 cake .643 -> .667, C1 cup .665 -> .688)


class HumanReachError(RuntimeError):
    """A physical human failed to reach a manipulation target in time."""


# Collision capsules approximate the visible hand; a small gap is allowed for
# the assisted grasp and avoids a prolonged probing motion at the tabletop.
HUMAN_GRASP_GAP_M = .01


def load_predictor(mode):
    if mode != 'intent':
        return None
    from intent_policy.intent.manicast_intent import load_forecaster
    return load_forecaster()


class Preview:
    """One rollout: scene, human mannequin, audits, video, forecast, event log."""

    def __init__(self, task, mode, seed, out, predictor, scene_hook=None, max_ticks=1600, camera=None):
        camera = camera or os.environ.get('PREVIEW_CAMERA', 'high')      # e.g. PREVIEW_CAMERA=back
        self.task, self.mode, self.seed, self.out = task, mode, int(seed), Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        scene, positions, xml = make(task)
        if scene_hook is not None:
            scene, positions, xml = scene_hook(scene, positions, xml)
        self.scene, self.positions = scene, positions
        self.env, self.human = make_env(scene, positions, xml)
        self.m, self.d = self.env.model, self.env.data
        self.initial = {k: np.array(v, float) for k, v in positions.items()}
        self.predictor = predictor
        self.max_ticks = max_ticks
        self.tick = 0
        self.frames_written = 0
        self.history = []
        self.kp = None
        self.forecast = None
        self.hand_speed = 0.
        self.predicted_speed = 0.
        self.pace = HumanPace(DT)
        self.follower = CartesianFollower(DT, acceleration=WORK_ACCEL)
        self.events, self.trace = [], []
        self.phase = 'start'
        self.human_phase = 'start'
        self.held_by_human = set()
        self.robot_selection = None
        self.kinematic = {}          # objects moved kinematically (second hand): name -> xyz
        self._palm_filter = {}       # critically damped 2nd-order filters of the palm commands (shared rule)
        self._is_robot, self.robot_geoms, self.human_geoms, self.world_geoms = geom_sets(self.m, positions)
        self.human_model = scene['human'].get('model_version', 'v2')
        if self.human_model in ('v3', 'v4'):
            # audit the collision geometry of the v3 person (capsules, palm sphere, finger capsule), not the visual
            # hand meshes: a mesh is measured by its convex hull, ~7 cm thick for a curled hand
            self.human_geoms = [g for g in self.human_geoms if self.m.geom_contype[g] or self.m.geom_conaffinity[g]]
        self.human_set = set(self.human_geoms)
        pairs = [PairRule(a, b, 0., 0.) for a in self.human_geoms for b in self.robot_geoms + self.world_geoms]
        pairs += [PairRule(a, b, 0., 0.) for a in self.robot_geoms for b in self.world_geoms]
        self.audit = DemoContactAudit(self.m, pairs)
        self.v3 = self.human_model in ('v3', 'v4')
        from intent_policy.sim.human_physics import physics_config
        self.contact_depth_tolerance = float(physics_config()['max_contact_depth_m'])
        self.contact_measurements = {}
        self.self_check = I3SelfCheck(self.m)
        self.proximity_margin = float(V2_CONFIG.get('proximity_margin_m', 0.))
        self.keep_clear = float(V2_CONFIG.get('expert_keep_clear_m', 0.))
        self.hand_holds = 0; self.human_contacts = {}; self.closest_human_m = np.inf
        if self.v3 and self.proximity_margin > 0:
            # proximity contacts: detected within `margin`, no force until penetration (`gap`); measurement only
            for g in self.human_geoms:
                self.m.geom_margin[g] = self.proximity_margin; self.m.geom_gap[g] = self.proximity_margin
        self.violations, self.self_hits = {}, {}
        self.visual_overlap_warnings = {}
        self.physics_contact_samples = 0
        self.env.solver_observers.append(self._audit_physics_contacts)
        name = f'{task}_{"Intent" if mode == "intent" else "Observed"}'
        self.name = name
        self.writer = VideoWriter(self.out / f'{name}.mp4', width=640, height=480)
        self.camera = camera
        self.wall0 = time.time()
        self.pose = None
        self.human_failed = False

    # ------------------------------------------------------------------ helpers
    @property
    def t(self):
        return self.tick * DT

    def pos(self, name):
        return self.d.xpos[self.m.body(name).id].copy()

    def tcp(self):
        return self.d.site_xpos[self.env.tcp].copy()

    def attached_name(self):
        return None if self.env.attached is None else self.m.body(self.env.attached).name

    def event(self, name, **kw):
        self.events.append({'t': round(self.t, 3), 'event': name, **kw})

    def hand_object_distance(self, name):
        """Distance from the active physical palm/finger proxies to an object."""
        body = self.m.body(name).id
        objects = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == body
                   and (self.m.geom_contype[g] or self.m.geom_conaffinity[g])]
        side = self.human.active.upper()
        hands = [self.m.geom(f'human_v3_{side}_Wrist_{part}').id for part in ('palm', 'fingers')]
        if not objects:
            return float('inf')
        return min(mujoco.mj_geomDistance(self.m, self.d, hand, obj, .2, None)
                   for hand in hands for obj in objects)

    # human grasp through the invisible anchor weld (keeps the current relative pose)
    def human_grasp(self, name):
        m, d = self.m, self.d
        if self.human_model == 'v4':
            distance = self.hand_object_distance(name)
            if distance > HUMAN_GRASP_GAP_M:
                raise HumanReachError(f'cannot grasp {name}: hand-object gap {distance:.4f} m')
        eid = m.equality(weld_name(name)).id
        anchor = m.eq_obj1id[eid]
        m.eq_data[eid, :3] = 0
        m.eq_data[eid, 3:6] = d.xmat[anchor].reshape(3, 3).T @ (self.pos(name) - d.xpos[anchor])
        inv, rel = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(inv, d.xquat[anchor])
        mujoco.mju_mulQuat(rel, inv, d.xquat[m.body(name).id])
        m.eq_data[eid, 6:10] = rel
        d.eq_active[eid] = True
        self.held_by_human.add(name)
        self.event('human_grasp', object=name)

    def human_release(self, name):
        self.d.eq_active[self.m.equality(weld_name(name)).id] = False
        self.held_by_human.discard(name)
        self.event('human_release', object=name)

    def place_kinematic(self, name, xyz):
        """Move an object with the second hand (no second anchor): set its pose, zero its velocity."""
        q = int(self.m.joint(name + '_free').qposadr[0]); v = int(self.m.joint(name + '_free').dofadr[0])
        self.d.qpos[q:q + 3] = xyz; self.d.qpos[q + 3:q + 7] = [1, 0, 0, 0]; self.d.qvel[v:v + 6] = 0
        mujoco.mj_forward(self.m, self.d)          # body poses must be current before a weld reads them
        self.kinematic[name] = np.asarray(xyz, float)

    def _smooth_palm(self, key, target, tau=PALM_TAU):
        """Critically damped second-order tracking of a palm command: removes per-tick kinks at
        phase changes (move -> hold -> move) and in the HY-Motion curvature, keeps the path shape."""
        target = np.asarray(target, float)
        st = self._palm_filter.get(key)
        if st is None:
            self._palm_filter[key] = [target.copy(), np.zeros(3)]
            return target
        x, v = st
        w = 1. / tau
        acc = w * w * (target - x) - 2. * w * v
        v = v + acc * DT; x = x + v * DT
        st[0], st[1] = x, v
        return x.copy()

    def apply_human(self, frame):
        """Pose the mannequin from a human frame dict and update speed / forecast estimates."""
        f = dict(frame)
        palm_tau = PALM_TAU
        f['palm'] = self._smooth_palm('active', f['palm'], tau=palm_tau)
        if f.get('idle_palm') is not None:
            f['idle_palm'] = self._smooth_palm('idle', f['idle_palm'], tau=palm_tau)
        self.human_phase = f.get('phase', self.human_phase)
        shape = f.get('shape', 'relaxed')
        pose = self.human.apply(self.d, palm=np.asarray(f['palm'], float), point_at=f.get('point_at'),
                                hand_shape=shape, palm_up=float(f.get('palm_up', 0.)),
                                shape_from=f.get('shape_from'), shape_progress=f.get('progress'),
                                aim_blend=float(f.get('aim_blend', 1.)),
                                idle_palm=f.get('idle_palm'), idle_shape=f.get('idle_shape'),
                                idle_palm_up=float(f.get('idle_palm_up', 0.)))
        self.pose = pose
        for name, xyz in self.kinematic.items():
            self.place_kinematic(name, xyz)
        mujoco.mj_forward(self.m, self.d)
        self.kp = self.human.keypoints(pose, JOINTS)
        self.history = (self.history + [self.kp.copy()])[-9:]
        self.hand_speed = observed_hand_speed(self.history, 5 if self.human.active == 'r' else 4, DT)
        self.predicted_speed = 0.
        if self.mode == 'intent':
            from intent_policy.intent.manicast_intent import forecast_xi
            self.forecast = forecast_xi(np.stack(self.history), self.predictor, 20., HORIZONS)[-1]
            self.predicted_speed = forecast_hand_speed(self.forecast, 5 if self.human.active == 'r' else 4, HORIZONS)
        self.pace.update(self.hand_speed, self.predicted_speed)
        return pose

    def wrist(self):
        return self.kp[5 if self.human.active == 'r' else 4]

    def wrist_future(self):
        """Predicted wrist positions at the 10 horizons (Intent only)."""
        if self.forecast is None:
            return None
        return self.wrist()[None, :] + self.forecast[:, 5 if self.human.active == 'r' else 4, :]

    def idle_palm_pos(self):
        return np.asarray(self.pose['idle_palm'], float)

    # robot command builders
    def command_to(self, goal, phase, grip, following=False):
        self.phase = phase
        speed = self.pace.speed(.24, following=True) if following else work_speed(phase)
        self.last_speed = speed
        target = self.follower.step(self.tcp(), np.asarray(goal, float), speed)
        return np.r_[self.env.solve_ik(target), grip]

    def hold(self, grip=None):
        # a held robot restarts its Cartesian follower from rest: a stale follower velocity made the
        # next command jump (13/11, I1 Intent trace: command 0.24 m/s while the TCP stood still)
        self.follower.velocity[:] = 0.; self.follower.position = None
        cmd = self.env.hold()
        if grip is not None:
            cmd[6] = grip
        return cmd

    def reached(self, goal, tol=.004):
        return bool(np.linalg.norm(self.tcp() - np.asarray(goal, float)) < tol)

    # ------------------------------------------------------------------ robot coroutine helpers
    def goto(self, goal, phase, grip=GRIP_OPEN, following=False, tol=.004, until=None, keep_clear=None):
        """Yield commands until the TCP reaches `goal` (callable or xyz). `until`: stop early.
        `keep_clear` (m, default the config's expert_keep_clear_m): the EXPERT's own rule to wait when any part
        of the robot is closer than this to the person; a deliberate hold, logged as `robot_holds_for_human`."""
        clear = self.keep_clear if keep_clear is None else keep_clear
        holding = False
        while True:
            g = goal() if callable(goal) else goal
            if self.reached(g, tol) or (until is not None and until()):
                self.follower.velocity *= 0.
                return
            if clear > 0 and self.v3:
                dist, pair = self.robot_near_human()
                if pair is not None and dist < clear:
                    if not holding:
                        holding = True; self.hand_holds += 1
                        self.event('robot_holds_for_human', phase=phase, distance_m=round(dist, 4), pair=list(pair))
                    yield self.hold(grip); continue
            holding = False
            yield self.command_to(g, phase, grip, following=callable(following) and following() or following is True)

    def close_on(self, name, phase='close', timeout=60):
        self.phase = phase
        self.robot_selection = name
        for _ in range(timeout):
            if self.env.attached == self.m.body(name).id:
                self.event('robot_grasp', object=name)
                return True
            yield self.hold(GRIP_CLOSED)
        self.event('grasp_failed', object=name)
        return False

    def open_gripper(self, phase='open', ticks=10):
        self.phase = phase
        for _ in range(ticks):
            yield self.hold(GRIP_OPEN)
        self.robot_selection = None

    def wait(self, cond, phase, grip=None, timeout=10**6):
        self.phase = phase
        for _ in range(timeout):
            if cond():
                return
            yield self.hold(grip)

    def pick(self, name, grasp_z=None, above_z=TRAVEL_Z):
        """approach above -> descend -> close -> lift, on the object's current position."""
        xy = self.pos(name)[:2]
        gz = self.pos(name)[2] + GRASP_DZ if grasp_z is None else grasp_z
        yield from self.goto(np.r_[xy, above_z], 'approach')
        yield from self.goto(np.r_[self.pos(name)[:2], gz], 'descend')
        ok = yield from self.close_on(name)
        if ok:
            yield from self.goto(np.r_[xy, above_z], 'lift', GRIP_CLOSED)
        return ok

    def place(self, xyz, above_z=TRAVEL_Z, retreat=True):
        """transport above -> lower -> open -> retreat."""
        xyz = np.asarray(xyz, float)
        yield from self.goto(np.r_[xyz[:2], above_z], 'move', GRIP_CLOSED)
        yield from self.goto(xyz, 'lower', GRIP_CLOSED)
        yield from self.open_gripper()
        self.event('robot_release', xyz=xyz.round(3).tolist())
        if retreat:
            yield from self.goto(np.r_[xyz[:2], above_z], 'retreat')

    # ------------------------------------------------------------------ stepping
    def robot_near_human(self):
        """(min distance, closest pair) over the current robot-human contacts (proximity contacts included)."""
        best = (np.inf, None)
        for c in self.d.contact[:self.d.ncon]:
            a, b = int(c.geom1), int(c.geom2)
            ra, rb = self._is_robot(a), self._is_robot(b)
            if ra == rb: continue
            other = b if ra else a
            if other not in self.human_set: continue
            if c.dist < best[0]: best = (float(c.dist), (self.m.geom(a).name, self.m.geom(b).name))
        return best

    def _audit_physics_contacts(self, model, data):
        """Record human contacts at every physics step; never modify commands."""
        self.physics_contact_samples += 1
        for contact_index, contact in enumerate(data.contact):
            a, b = int(contact.geom1), int(contact.geom2)
            if a not in self.human_set and b not in self.human_set:
                continue
            if b in self.human_set:
                a, b = b, a
            if contact.dist >= 0:
                continue
            ba = model.body(model.geom_bodyid[a]).name
            # Static world geoms belong to `world`, which hides whether the
            # contact was with the table, floor, or a task prop.  Use the geom
            # name as the audit counterpart in that case.
            other_geom = model.geom(b).name or str(b)
            bb = model.body(model.geom_bodyid[b]).name
            if bb == 'world':
                bb = other_geom
            key = f'{ba}|{bb}'
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact_index, force)
            record = self.contact_measurements.setdefault(key, {'samples': 0, 'max_depth_m': 0., 'max_normal_force_n': 0.,
                                                                'first_t_s': round(float(data.time), 4)})
            record['samples'] += 1
            record['max_depth_m'] = max(record['max_depth_m'], -float(contact.dist))
            record['max_normal_force_n'] = max(record['max_normal_force_n'], float(force[0]))
            if -contact.dist > self.contact_depth_tolerance:
                self.violations[key] = max(self.violations.get(key, 0.), -float(contact.dist))
            if b in self.robot_geoms:
                names = f'{model.geom(a).name}|{model.geom(b).name or b}'
                self.human_contacts[names] = min(self.human_contacts.get(names, 0.), float(contact.dist))
                self.closest_human_m = min(self.closest_human_m, float(contact.dist))

    def _audit_geometry(self):
        """Also measure kinematic pairs that MuJoCo does not resolve physically."""
        if self.v3:
            dist, pair = self.robot_near_human()          # measured only; nothing is prevented here
            if pair is not None:
                self.closest_human_m = min(self.closest_human_m, dist)
                if dist < 0:
                    key = f'{pair[0]}|{pair[1]}'
                    self.human_contacts[key] = min(self.human_contacts.get(key, 0.), dist)
        self.audit.observe(self.m, self.d)
        attached = self.attached_name()
        for idx, dist, _, _ in self.audit.step_violations:
            pair = self.audit.rules[idx]
            ba = self.m.body(self.m.geom_bodyid[pair.geom_a]).name; bb = self.m.body(self.m.geom_bodyid[pair.geom_b]).name
            tol = overlap_tolerance(robot_body=ba, other_body=bb, other_geom=self.m.geom(pair.geom_b).name,
                                    attached=attached, selection=self.robot_selection, phase=self.phase,
                                    static_robot=bool(self.m.body_weldid[self.m.geom_bodyid[pair.geom_a]] == 0),
                                    human_pair=ba.startswith('human_seg_'), tolerances=TOL)
            if ba.startswith('human_seg_'):
                tol = self.contact_depth_tolerance
            if tol is not None and -dist > tol:
                key = f'{ba}|{bb}'
                self.violations[key] = max(self.violations.get(key, 0.), -float(dist))
                # A visible mesh may be visual-only, in which case this is a
                # hull-based image-geometry warning rather than solver contact.
                # Preserve the warning instead of claiming physical penetration
                # or silently dropping a potentially visible mismatch.
                if not (self.m.geom_contype[pair.geom_b] or self.m.geom_conaffinity[pair.geom_b]):
                    geom_name = self.m.geom(pair.geom_b).name or str(pair.geom_b)
                    visual_key = f'{ba}|{geom_name}'
                    self.visual_overlap_warnings[visual_key] = max(self.visual_overlap_warnings.get(visual_key, 0.), -float(dist))
        for pair, depth in (self.self_check.measure(self.d).items() if self.self_check is not None else ()):
            if depth > self.contact_depth_tolerance:
                self.self_hits[str(pair)] = max(self.self_hits.get(str(pair), 0.), float(depth))

    def step(self, command):
        # Check immediately after the human pose update, before physics can push an
        # intersected object away and hide the original overlap.
        self._audit_geometry()
        self.env.step(command)
        mujoco.mj_forward(self.m, self.d)
        self._audit_geometry()
        attached = self.attached_name()
        self.trace.append({'t': round(self.t, 3), 'human_phase': self.human_phase, 'robot_phase': self.phase,
                           'wrist': self.wrist().round(4).tolist(),
                           # `palm` is the realised physical point for v4 (not
                           # the requested IK target).  Keeping both its task
                           # phase and trajectory in the trace makes it possible
                           # to reject a scripted transition that outruns a
                           # force-driven arm.
                           'palm': np.asarray(self.pose['palm']).round(4).tolist(),
                           'palm_target': None if getattr(self.human, 'target_pose', None) is None else np.asarray(self.human.target_pose['palm']).round(4).tolist(),
                           'tcp': self.tcp().round(4).tolist(),
                           'attached': attached, 'hand_speed': round(self.hand_speed, 4),
                           'joint_position': self.d.qpos[self.env.qids].tolist(),
                           'joint_velocity': self.d.qvel[self.env.dids].tolist(),
                           'joint_target': np.asarray(command[:6]).tolist(),
                           'actuator_force': self.d.actuator_force[:6].tolist(),
                           'ik_error_m': float(self.env.last_ik_error),
                           'forecast_speed': round(self.predicted_speed, 4) if self.mode == 'intent' else None,
                           'command_speed': round(float(np.linalg.norm(self.follower.velocity)), 4),
                           'forecast_wrist_1s': None if self.forecast is None else self.wrist_future()[-1].round(4).tolist(),
                           'objects': {k: self.pos(k).round(4).tolist() for k in self.positions}})
        self.writer.add(self.env.render(self.camera, 640, 480))
        self.frames_written += 1
        self.tick += 1

    def run(self, human_gen, robot_gen, done):
        """Drive both coroutines until `done()` or the robot coroutine ends (then settle 0.6 s)."""
        last_frame = None
        robot_done_at = None
        try:
            while self.tick < self.max_ticks:
                try:
                    last_frame = next(human_gen)
                except HumanReachError as exc:
                    self.human_failed = True
                    self.event('human_reach_timeout', detail=str(exc))
                    break
                except StopIteration:
                    pass
                if last_frame is None:
                    raise RuntimeError('human script yielded nothing')
                self.apply_human(last_frame)
                if robot_done_at is None:
                    try:
                        command = next(robot_gen)
                    except HumanReachError as exc:
                        self.human_failed = True
                        self.event('human_reach_timeout', detail=str(exc))
                        break
                    except StopIteration:
                        robot_done_at = self.tick; command = self.hold()
                else:
                    command = self.hold()
                self.step(command)
                if self.human_failed:
                    break
                if done() and (robot_done_at is not None or self.phase == 'done') and self.tick - (robot_done_at or self.tick) >= 12:
                    break
                if robot_done_at is not None and self.tick - robot_done_at >= 60:
                    break
        finally:
            self.writer.close()

    def finish(self, success, extra=None):
        summary = {'task': self.task, 'case': 'nominal', 'mode': self.mode, 'seed': self.seed, 'success': bool(success),
                   'duration_s': round(self.t, 3), 'wall_s': round(time.time() - self.wall0, 1),
                   'events': self.events, 'final_positions': {k: self.pos(k).round(4).tolist() for k in self.positions},
                   'contact_violations': self.violations, 'human_self_violations': self.self_hits,
                   'physics_contact_samples': self.physics_contact_samples,
                   'contact_measurements': self.contact_measurements,
                   'visual_overlap_warnings': self.visual_overlap_warnings,
                   'human_contact_depth_tolerance_m': self.contact_depth_tolerance,
                   'geometry_valid': not bool(self.violations or self.self_hits),
                   'human_model': self.human_model, 'expert_holds_for_human': self.hand_holds,
                   'robot_human_penetrations_m': {k: round(-v, 4) for k, v in self.human_contacts.items()},
                   'robot_human_closest_m': None if not np.isfinite(self.closest_human_m) else round(float(self.closest_human_m), 4),
                   'forecast_source': 'ManiCast actual checkpoint' if self.mode == 'intent' else 'none',
                   'dataset_accepted': False,
                   'status': 'visual review preview; scripted expert; not an accepted dataset episode',
                   **(extra or {})}
        (self.out / f'{self.name}_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        (self.out / f'{self.name}_trace.json').write_text(json.dumps(self.trace) + '\n')
        self.env.close()
        return summary


# ---------------------------------------------------------------------- human motion primitives
def move(start, end, duration, lift=0., motion=None, motion_scale=1., **kw):
    """Smooth palm move from start to end over `duration` s with an optional vertical arc.
    `motion`: a HY-Motion palm path (see `hy_motion`); its curvature (endpoint-warped to a zero
    displacement, eased in and out) is added on top of the straight minimum-jerk move."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    n = max(1, int(round(duration / DT)))
    for i in range(1, n + 1):
        u = smooth(i / n)
        p = start + (end - start) * u + np.array([0., 0., lift * np.sin(np.pi * i / n)])
        if motion is not None:
            p = p + motion_scale * motion_deviation(motion, u)
        yield {'palm': p, 'progress': u, **kw}


def move_path(pts, duration, shape='grasp', shape_from=None, phase='', **kw):
    """Smooth, continuous waypoint motion through a sequence of 3D points.
    Unlike chaining multiple move() calls (which comes to a full stop at each waypoint),
    move_path parameterizes the entire polyline with a single minimum-jerk easing,
    ensuring non-zero velocity across intermediate points and eliminating stutter."""
    pts = [np.asarray(p, float) for p in pts]
    dists = [float(np.linalg.norm(pts[k+1] - pts[k])) for k in range(len(pts)-1)]
    total = sum(dists)
    cum = np.cumsum([0] + dists)
    n = max(1, int(round(duration / DT)))
    for i in range(1, n + 1):
        u = smooth(i / n)
        d = u * total
        idx = np.searchsorted(cum, d) - 1
        idx = max(0, min(idx, len(pts) - 2))
        frac = (d - cum[idx]) / max(dists[idx], 1e-9)
        p = pts[idx] + frac * (pts[idx+1] - pts[idx])
        yield {'palm': p, 'progress': u, 'shape': shape, 'shape_from': shape_from, 'phase': phase, **kw}


HY_BANK = ROOT / 'outputs/human_motion/hymotion_ten_tasks_2026-10-30/hymotion_out'


def hy_motion(pv, clip, sample=0, hand='right'):
    """Smoothed HY-Motion palm-path segments of one generated clip, placed in this scene.
    Returns the list of moving segments (each an [L, 3] path) and the clip metadata."""
    src = HY_BANK / f'{clip}.npz'
    with np.load(src, allow_pickle=False) as raw:
        k3d = raw['keypoints3d'][sample].copy(); text = str(raw['text'])
    world = to_scene(k3d, np.array([*pv.scene['human']['stand_xy'], 0.]), pv.scene['human']['facing_deg'], pv.scene['floor_z'])
    path = palm_path(world, hand)
    parts = segment_ranges(path, tails=True)
    segments = [smooth_motion_path(path[a:b]) for a, b in parts]
    return segments, {'clip': clip, 'sample': sample, 'prompt': text, 'segments': parts, 'source': str(src.relative_to(ROOT))}


def hold(palm, duration=None, until=None, **kw):
    """Hold the palm still for `duration` s or until `until()` is true."""
    n = 10**6 if duration is None else max(1, int(round(duration / DT)))
    for _ in range(n):
        if until is not None and until():
            return
        yield {'palm': np.asarray(palm, float), **kw}


def main_pair(task, run_fn, seed_default):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=seed_default)
    p.add_argument('--out', type=Path, default=ROOT / 'previews')
    p.add_argument('--modes', nargs='+', default=['observed', 'intent'])
    a = p.parse_args()
    predictor = load_predictor('intent') if 'intent' in a.modes else None
    for mode in a.modes:
        s = run_fn(mode, a.seed, predictor, a.out)
        print(json.dumps({k: s[k] for k in ('task', 'mode', 'success', 'duration_s')},
                         ) + f" violations={len(s['contact_violations'])} self={len(s['human_self_violations'])}", flush=True)


def carry(pv, obj, dest, start, dz=.035, lift=.08, reach_s=.9, carry_s=1.0, lower_s=.5, shape='grasp', phase='carry'):
    """Human pick-and-place of `obj` to `dest` (object centre), welded to the hand while carried.
    Yields frames; returns the final palm position (above the released object)."""
    start = np.asarray(start, float)
    # The old kinematic hand could target a point inside the tabletop or
    # object.  Leave room for the physical palm and finger volumes.
    top = pv.pos(obj) + np.array([0., 0., max(dz, .065) if pv.human_model == 'v4' else dz])
    def wait_physical(target, label, tolerance=.035):
        if pv.human_model != 'v4':
            return
        for _ in range(int(round(3. / DT))):
            if np.linalg.norm(np.asarray(pv.pose['palm']) - target) < tolerance:
                return
            yield {'palm': np.asarray(target), 'shape': shape, 'phase': label}
        raise HumanReachError(f'{label}: palm did not reach {np.asarray(target).round(3).tolist()}')
    if pv.human_model == 'v4':
        # Traverse above the tabletop before lowering to the object.  Direct
        # diagonals can trap the physical hand against the table edge.
        high = np.r_[start[:2], max(start[2], .83)]
        yield from move(start, high, .8, shape='relaxed', phase=phase + '_reach')
        yield from wait_physical(high, phase + '_settle_high', tolerance=.05)
        above = np.r_[top[:2], max(top[2] + .14, .88)]
        yield from move(high, above, reach_s, shape=shape, shape_from='relaxed', phase=phase + '_reach')
        yield from wait_physical(above, phase + '_settle_above', tolerance=.05)
        yield from move(above, top, .8, shape=shape, phase=phase + '_lower')
    else:
        yield from move(start, top + [0, 0, .05], reach_s, lift=.04, shape=shape, shape_from='relaxed', phase=phase + '_reach')
        yield from move(top + [0, 0, .05], top, .35, shape=shape, phase=phase + '_lower')
    yield from wait_physical(top, phase + '_settle_grasp')
    if pv.human_model == 'v4':
        contact_target = top.copy()
        for _ in range(int(round(4. / DT))):
            if pv.hand_object_distance(obj) <= HUMAN_GRASP_GAP_M:
                break
            contact_target[2] -= .0015
            yield {'palm': contact_target.copy(), 'shape': shape, 'phase': phase + '_contact'}
        else:
            raise HumanReachError(f'{phase}: cannot approach {obj} within {HUMAN_GRASP_GAP_M * 1000:g} mm')
    offset = np.asarray(pv.pose['palm']) - pv.pos(obj) if pv.human_model == 'v4' else np.array([0., 0., dz])
    pv.human_grasp(obj)
    dest = np.asarray(dest, float)
    end = dest + offset
    if pv.human_model == 'v4':
        actual = np.asarray(pv.pose['palm']).copy()
        raised = actual + [0, 0, max(lift, .16)]
        yield from move(actual, raised, .9, shape=shape, phase=phase + '_lift')
        yield from wait_physical(raised, phase + '_settle_lift', tolerance=.04)
        yield from move(raised, end + [0, 0, lift], carry_s, lift=.03, shape=shape, phase=phase + '_move')
    else:
        yield from move(top, end + [0, 0, lift], carry_s, lift=.03, shape=shape, phase=phase + '_move')
    yield from wait_physical(end + [0, 0, lift], phase + '_settle_move')
    yield from move(end + [0, 0, lift], end, lower_s, shape=shape, phase=phase + '_place')
    yield from wait_physical(end, phase + '_settle_place', tolerance=.04)
    pv.human_release(obj)
    yield from move(end, end + [0, 0, .06], .3, shape='relaxed', shape_from=shape, phase=phase + '_release')
    return end + np.array([0., 0., .06])


def shift_decor(xml, name_prefix, dx=0., dy=0.):
    """Move static decor geoms (trays) in the scene XML by (dx, dy)."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    for g in root.iter('geom'):
        if g.get('name', '').startswith('decor_' + name_prefix + '_'):
            p = np.fromstring(g.get('pos'), sep=' '); p[0] += dx; p[1] += dy
            g.set('pos', ' '.join(f'{v:.5f}' for v in p))
    return ET.tostring(root, encoding='unicode')


def dirt_geoms(m):
    return [g for g in range(m.ngeom) if m.geom(g).name.startswith('dirt_preview_')]


def wipe_marks(pv, xy, radius=.03):
    """Hide dirt marks within `radius` of xy (visual proxy of wiping; no fabric physics)."""
    n = 0
    for g in dirt_geoms(pv.m):
        if pv.m.geom_rgba[g, 3] > 0 and np.linalg.norm(pv.m.geom_pos[g, :2] - np.asarray(xy)[:2]) < radius:
            pv.m.geom_rgba[g, 3] = 0.; n += 1
    return n


def marks_left(pv):
    return int(sum(pv.m.geom_rgba[g, 3] > 0 for g in dirt_geoms(pv.m)))
