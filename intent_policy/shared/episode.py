"""One HRC rollout (QĐ 53/QĐ-DL §4): scene, scripted human, task expert, shared controller / planner /
ManiCast adapter / physics validator, review video and an optional LeRobot recorder.

The human and the robot expert are generators that yield one frame / command per 20 Hz tick. Per tick:
human pose -> keypoint history -> ManiCast (both modes, recorded) -> expert command (the forecast is handed to the
expert only in `forecast` mode) -> record frame (observation at t, action applied at t) -> MuJoCo step ->
physics validation. Moved from scripts/v2_preview_common.py::Preview; the control law is unchanged.
"""
from pathlib import Path
import json
import time

import mujoco
import numpy as np

from intent_policy.benchmark.video import VideoWriter
from intent_policy.sim.scene_builder_v2 import weld_name
from intent_policy.sim.motion_bank_v2 import to_scene, palm_path, segment_ranges
from intent_policy.shared.sim_env import (ROOT, STATION, DT, JOINTS, GRIP_OPEN, GRIP_CLOSED, TRAVEL_Z, GRASP_DZ,
                                          PALM_TAU, make_env, ensure_review_cameras, smooth, smooth_motion_path,
                                          motion_deviation)
from intent_policy.shared.cartesian_controller import (CartesianController, HumanPace, observed_hand_speed,
                                                       forecast_hand_speed, work_speed)
from intent_policy.shared.manicast_adapter import ManiCastAdapter
from intent_policy.shared.forecast_planner import ForecastPlanner
from intent_policy.shared.physics_validator import PhysicsValidator

MODES = ('observed', 'forecast')
MIN_HAND_SPEED = float(STATION['intent']['min_hand_speed_m_s'])
HORIZON_TICKS = [int(round(h / DT)) for h in STATION['intent']['horizons_s']]
HUMAN_GRASP_GAP_M = .01
# 68/QĐ-DL B.1: longest wait for the reached palm to come within HUMAN_GRASP_GAP_M of a grasp command. Bound from
# the human model itself: the full waist lean at the lean rate limit (HumanV3 default max 60 deg at 40 deg/s = 1.5 s)
# plus the palm filter settling 13 cm -> 1 cm (critically damped, PALM_TAU 0.08 s: (1 + t/tau) e^(-t/tau) = 0.077 at
# t = 4.3 tau = 0.35 s) = 1.85 s, rounded up to 2 s.
HUMAN_SETTLE_TIMEOUT_S = 2.


class HumanReachError(RuntimeError):
    """A physical human failed to reach a manipulation target in time."""


class MotionStalled(RuntimeError):
    """A robot move made no progress for STALL_TICKS: end the rollout honestly instead of a frozen video."""


STALL_TICKS = 40          # 2 s without 3 mm of progress towards an unchanged goal


class Episode:
    grip_open, grip_closed = GRIP_OPEN, GRIP_CLOSED

    def __init__(self, task, mode, seed, scene, positions, xml, predictor, out=None, max_ticks=1600,
                 review_camera='overhead_oblique', review_hw=(480, 640), recorder=None, variant='nominal', injector=None,
                 variation=None, name_suffix=''):
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        self.task, self.mode, self.seed, self.variant = task, mode, int(seed), variant
        self.out = None if out is None else Path(out)
        if self.out is not None:
            self.out.mkdir(parents=True, exist_ok=True)
        xml = ensure_review_cameras(xml)
        self.scene, self.positions, self.xml = scene, positions, xml
        self.env, self.human = make_env(scene, positions, xml)
        self.m, self.d = self.env.model, self.env.data
        self.initial = {k: np.array(v, float) for k, v in positions.items()}
        self.max_ticks = int(max_ticks)
        self.tick = 0
        self.history, self.kp, self.pose = [], None, None
        self.palm_history = []                          # reached active palm, last 5 ticks (settle_hand)
        self.hand_speed = self.predicted_speed = 0.
        self.pace = HumanPace(DT)
        self.controller = CartesianController(self.env, DT)
        self.manicast = ManiCastAdapter(predictor, injector)      # 59/QĐ-DL: N5 fault right after ManiCast
        self.injector = injector
        self.variation = variation or {}
        self.variation_log = {}
        self.kp_log, self.raw_log, self.fault_log = [], [], []     # per tick, for the sidecar / natural error
        self.forecast_use_ticks = set()                             # ticks where the expert / planner read it
        self.planner = ForecastPlanner(self)
        self.events, self.trace = [], []
        self.phase, self.human_phase = 'start', 'start'
        self.held_by_human, self.kinematic, self._palm_filter = set(), {}, {}
        self.idle_hand_objects = {}
        self.env.physics_controllers.append(self._sync_idle_hand_objects)
        self._palm_up = 0.
        self.robot_selection = None
        self.human_model = scene['human'].get('model_version', 'v2')
        self.validator = PhysicsValidator(self)
        if hasattr(self.human, 'world_geoms'):            # 61/QĐ-DL: the kinematic person does not pass into rigid geoms
            m = self.m
            self.human.world_geoms = [g for g in self.validator.world_geoms if m.geom_contype[g] or m.geom_conaffinity[g]]
            self.human.skip_bodies = lambda: {m.body(n).id for n in set(self.held_by_human) | set(self.idle_hand_objects)}
            # 61/QĐ-DL 08:2x: projecting carried objects with the hand was tried and made W3 worse (24.6 mm / 181 N);
            # removed, carried objects are not part of the hand projection
        self.keep_clear = float(STATION.get('expert_keep_clear_m', 0.))
        self.hand_holds = 0
        self.human_failed = False
        self.stalled = None
        self.name = (f'{task}_' + ('' if variant == 'nominal' else f'{variant}_') + name_suffix
                     + ('Forecast' if mode == 'forecast' else 'Observed'))
        self.review_camera, self.review_hw = review_camera, tuple(review_hw)
        self.writer = None if self.out is None else VideoWriter(
            self.out / f'{self.name}_{review_camera}.mp4', width=self.review_hw[1], height=self.review_hw[0])
        self.recorder = recorder
        if self.variation.get('robot_tcp_offset_m') is not None:        # 59/QĐ-DL N1: valid robot start state
            from intent_policy.shared.variation import apply_robot_start
            self.variation_log['robot_start'] = apply_robot_start(self, self.variation['robot_tcp_offset_m'],
                                                                  self.variation.get('robot_tcp_offset_range_m'))
        self.wall0 = time.time()

    # ---------------------------------------------------------------- state helpers
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

    def wrist_index(self):
        return 5 if self.human.active == 'r' else 4

    def wrist(self):
        return self.kp[self.wrist_index()]

    def forecast_available(self):
        """Internal (pace, trace, recorder): the forecast the expert may receive at this tick, or None."""
        if self.mode != 'forecast' or not self.manicast.valid:
            return None
        return self.manicast.forecast

    def forecast_for_expert(self):
        """[10, 6, 3] world forecast, only in `forecast` mode; the observed expert never receives it. Expert /
        planner API: a read while the observed hand moves marks this tick as a forecast-USE tick (N5 trigger)."""
        f = self.forecast_available()
        if f is not None and self.hand_speed > MIN_HAND_SPEED:
            self.forecast_use_ticks.add(self.tick)
        return f

    def reached(self, goal, tol=.004):
        return bool(np.linalg.norm(self.tcp() - np.asarray(goal, float)) < tol)

    # ---------------------------------------------------------------- human
    def hand_object_distance(self, name):
        body = self.m.body(name).id
        objects = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == body
                   and (self.m.geom_contype[g] or self.m.geom_conaffinity[g])]
        side = self.human.active.upper()
        hands = [self.m.geom(f'human_v3_{side}_Wrist_{part}').id for part in ('palm', 'fingers')]
        if not objects:
            return float('inf')
        return min(mujoco.mj_geomDistance(self.m, self.d, h, o, .2, None) for h in hands for o in objects)

    def settle_hand(self, frame, tol=HUMAN_GRASP_GAP_M, timeout_s=HUMAN_SETTLE_TIMEOUT_S):
        """Generator (68/QĐ-DL B.1), used as `yield from ep.settle_hand(frame)` right before `human_grasp`: keep
        yielding the last commanded grasp frame until the hand has caught up with it (palm filter lag, rate-limited
        waist lean). The weld in `human_grasp` captures the current hand-object offset, so a lagging hand carried
        the object at a distance (C1 seed 300400: reached palm 11 cm from the grip). Settled when
          (a) |reached palm - commanded palm| <= tol, or
          (b) the palm has stopped (median reached-palm speed over the last 5 ticks < MIN_HAND_SPEED, the repo's
              "hand moves" threshold, computed as observed_hand_speed but on the palm; known gap 18/11: at W2 block3
              the palm keeps oscillating at 3.7 cm/s around a 3.1 cm contact offset, so `carry` does not use this) AND the
              arm reaches its own target (pose reach_error <= tol): the remaining offset is then the contact
              correction of the person model (61/66/QĐ-DL, WORLD_CLEAR_M = 0: the hand rests on the object / table
              instead of passing into it). Measured 18/11 (outputs/diag/grasp_gap_1118/probe.jsonl): 1.8-4.5 cm,
              mostly vertical, constant for 2 s at I1 / C1 / W1 / W2 -- the commanded grasp point lies inside a geom.
        Raises HumanReachError (-> `human_reach_timeout`) when neither holds within `timeout_s`."""
        target = np.asarray(frame['palm'], float)
        t0 = self.tick
        while True:
            reached = np.asarray(self.pose['palm'], float) if self.pose is not None else np.full(3, np.inf)
            err = float(np.linalg.norm(reached - target))
            reach_err = float(self.pose.get('reach_error', 0.)) if self.pose is not None else np.inf
            ph = np.asarray(self.palm_history, float)
            palm_speed = float(np.median(np.linalg.norm(np.diff(ph, axis=0), axis=1)) / DT) if len(ph) >= 5 else np.inf
            still = palm_speed < MIN_HAND_SPEED and reach_err <= tol
            if err <= tol or still:
                break
            if (self.tick - t0) * DT >= timeout_s:
                raise HumanReachError(f'hand {err:.3f} m from the grasp command after {timeout_s:.1f} s (palm speed '
                                      f'{palm_speed:.3f} m/s, reach error {reach_err:.3f} m, phase {self.human_phase}, '
                                      f'lean {np.degrees(float(self.pose.get("lean", 0.))):.1f} deg)')
            yield frame
        mem = getattr(self.human, '_push_mem', None) or {}
        push = mem.get(self.human.active) if mem else None
        self.event('human_hand_settled', wait_s=round((self.tick - t0) * DT, 3), error_m=round(err, 4),
                   by='position' if err <= tol else 'still_in_contact',
                   contact_push_m=None if push is None else round(float(np.linalg.norm(push)), 4))

    def human_grasp(self, name):
        m, d = self.m, self.d
        if self.human_model == 'v4' and self.hand_object_distance(name) > HUMAN_GRASP_GAP_M:
            raise HumanReachError(f'cannot grasp {name}')
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

    def set_human_hold_offset(self, name, world_offset):
        """Move a human-held (welded) object to `world_offset` from the grasp anchor, e.g. as the wrist turns."""
        m, d = self.m, self.d
        eid = m.equality(weld_name(name)).id
        anchor = m.eq_obj1id[eid]
        m.eq_data[eid, 3:6] = d.xmat[anchor].reshape(3, 3).T @ np.asarray(world_offset, float)

    def human_release(self, name):
        self.d.eq_active[self.m.equality(weld_name(name)).id] = False
        self.held_by_human.discard(name)
        self.event('human_release', object=name)

    def place_kinematic(self, name, xyz):
        """Second-hand object moved kinematically: set its pose and zero its velocity."""
        q = int(self.m.joint(name + '_free').qposadr[0]); v = int(self.m.joint(name + '_free').dofadr[0])
        self.d.qpos[q:q + 3] = xyz; self.d.qpos[q + 3:q + 7] = [1, 0, 0, 0]; self.d.qvel[v:v + 6] = 0
        mujoco.mj_forward(self.m, self.d)
        self.kinematic[name] = np.asarray(xyz, float)

    def _sync_idle_hand_objects(self, model, data):
        """Keep second-hand kinematic props at the reached palm, including physics substeps."""
        if self.pose is None:
            return
        for name, offset in self.idle_hand_objects.items():
            q = int(model.joint(name + '_free').qposadr[0])
            v = int(model.joint(name + '_free').dofadr[0])
            data.qpos[q:q + 3] = np.asarray(self.pose['idle_palm']) + offset
            data.qpos[q + 3:q + 7] = [1., 0., 0., 0.]
            data.qvel[v:v + 6] = 0.

    def _smooth_palm(self, key, target, tau=PALM_TAU):
        target = np.asarray(target, float)
        st = self._palm_filter.get(key)
        if st is None:
            self._palm_filter[key] = [target.copy(), np.zeros(3)]
            return target
        x, v = st
        w = 1. / tau
        v = v + (w * w * (target - x) - 2. * w * v) * DT; x = x + v * DT
        st[0], st[1] = x, v
        return x.copy()

    def apply_human(self, frame):
        f = dict(frame)
        f['palm'] = self._smooth_palm('active', f['palm'])
        if f.get('idle_palm') is not None:
            f['idle_palm'] = self._smooth_palm('idle', f['idle_palm'])
        self.human_phase = f.get('phase', self.human_phase)
        # palm orientation (0 = down, 1 = up) changes by at most 0.06 per tick (C2 fix 14/11: it used to jump)
        self._palm_up += float(np.clip(float(f.get('palm_up', 0.)) - self._palm_up, -.06, .06))
        self.pose = self.human.apply(self.d, palm=np.asarray(f['palm'], float), point_at=f.get('point_at'),
                                     hand_shape=f.get('shape', 'relaxed'), palm_up=self._palm_up,
                                     shape_from=f.get('shape_from'), shape_progress=f.get('progress'),
                                     aim_blend=float(f.get('aim_blend', 1.)),
                                     idle_palm=f.get('idle_palm'), idle_shape=f.get('idle_shape'),
                                     idle_palm_up=float(f.get('idle_palm_up', 0.)))
        push = getattr(self.human, 'self_push_m', None) or {}
        if push:                                           # 61/QĐ-DL: how far the hand target was moved out of contact
            self.human_push_max_m = max(getattr(self, 'human_push_max_m', 0.), max(push.values()))
        if any(abs(v) > 1e-9 for v in getattr(self.human, 'swivel', {}).values()):
            self.human_swivel_max_deg = max(getattr(self, 'human_swivel_max_deg', 0.), max(abs(v) for v in self.human.swivel.values()))
        self._sync_idle_hand_objects(self.m, self.d)
        for name, xyz in self.kinematic.items():
            self.place_kinematic(name, xyz)
        mujoco.mj_forward(self.m, self.d)
        self.kp = self.human.keypoints(self.pose, JOINTS)
        self.history = (self.history + [self.kp.copy()])[-9:]
        self.palm_history = (self.palm_history + [np.asarray(self.pose['palm'], float).copy()])[-5:]
        self.hand_speed = observed_hand_speed(self.history, self.wrist_index(), DT)
        # both modes: recorded; used only in forecast mode. The injector (N5) sees only observed data.
        self.manicast.update(self.history, self.t, self.hand_speed, self.human.active,
                             used_last_tick=(self.tick - 1) in self.forecast_use_ticks)
        mc = self.manicast
        self.kp_log.append(self.kp.copy())
        self.raw_log.append(None if mc.raw_forecast is None or not mc.valid else mc.raw_forecast.copy())
        self.fault_log.append(float(mc.fault_strength))
        self.predicted_speed = 0.
        if self.forecast_available() is not None:
            self.predicted_speed = forecast_hand_speed(self.manicast.offsets, self.wrist_index(), np.asarray(
                STATION['intent']['horizons_s'], float))
        self.pace.update(self.hand_speed, self.predicted_speed)

    # ---------------------------------------------------------------- robot command builders
    def command_to(self, goal, phase, grip, following=False, scale=1.):
        self.phase = phase
        speed = (self.pace.speed(.24, following=True) if following else work_speed(phase)) * float(scale)
        return self.controller.command(self.tcp(), goal, speed, grip)

    def hold(self, grip=None):
        return self.controller.hold(grip)

    def goto(self, goal, phase, grip=GRIP_OPEN, following=False, tol=.004, until=None, keep_clear=None):
        """Yield commands until the TCP reaches `goal` (callable or xyz) or `until()`. `keep_clear`: the
        expert's own rule to wait while any robot part is closer than this to the observed person."""
        clear = self.keep_clear if keep_clear is None else keep_clear
        holding = False
        best, best_tick, sig = np.inf, self.tick, None
        while True:
            g = goal() if callable(goal) else goal
            if self.reached(g, tol) or (until is not None and until()):
                self.controller.follower.velocity *= 0.
                return
            err = float(np.linalg.norm(self.tcp() - np.asarray(g, float)))
            s = tuple(np.round(np.asarray(g, float), 2))
            if s != sig or err < best - .003 or holding:
                sig, best, best_tick = s, min(err, best) if s == sig else err, self.tick
            elif self.tick - best_tick >= STALL_TICKS:
                raise MotionStalled(f'{phase} target {np.round(g, 3).tolist()}, TCP {self.tcp().round(3).tolist()}')
            if clear > 0:
                dist, pair = self.validator.robot_near_human(include_held=True)
                away = self.validator.away
                # 57/QĐ-DL: hold only if the next step brings the robot closer; a step away from the person goes on
                # (holding still while the person moves into the robot caused the C4 elbow / head contacts)
                leaving = away is not None and float(np.dot(np.asarray(g, float) - self.tcp(), away)) > 0.
                if pair is not None and dist < clear and not leaving:
                    if not holding:
                        holding = True; self.hand_holds += 1
                        self.event('robot_holds_for_human', phase=phase, distance_m=round(dist, 4), pair=list(pair))
                    yield self.hold(grip); continue
            holding = False
            yield self.command_to(g, phase, grip, following=(following() if callable(following) else following))

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
        xy = self.pos(name)[:2]
        gz = self.pos(name)[2] + GRASP_DZ if grasp_z is None else grasp_z
        yield from self.goto(np.r_[xy, above_z], 'approach')
        yield from self.goto(np.r_[self.pos(name)[:2], gz], 'descend')
        ok = yield from self.close_on(name)
        if ok:
            yield from self.goto(np.r_[xy, above_z], 'lift', GRIP_CLOSED)
        return ok

    def place(self, xyz, above_z=TRAVEL_Z, retreat=True):
        xyz = np.asarray(xyz, float)
        yield from self.goto(np.r_[xyz[:2], above_z], 'move', GRIP_CLOSED)
        yield from self.goto(xyz, 'lower', GRIP_CLOSED)
        yield from self.open_gripper()
        self.event('robot_release', xyz=xyz.round(3).tolist())
        if retreat:
            yield from self.goto(np.r_[xyz[:2], above_z], 'retreat')

    # ---------------------------------------------------------------- stepping
    def step(self, command):
        command = np.asarray(command, float)
        self.validator.audit_geometry()                 # before physics can push an overlap away
        if self.recorder is not None:
            self.recorder.add_frame(self, command)      # observation at t + the action applied at t
        self.env.step(command)
        self._sync_idle_hand_objects(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        self.validator.audit_geometry()
        self.validator.audit_robot()
        f = self.forecast_available()
        near, pair = self.validator.robot_near_human()
        self.trace.append({'t': round(self.t, 3), 'human_phase': self.human_phase, 'robot_phase': self.phase,
                           'robot_person_m': None if pair is None else round(near, 4), 'robot_person_pair': pair,
                           'idle_palm': np.asarray(self.pose['idle_palm']).round(4).tolist(),
                           'idle_hand_object_error_m': {n: float(np.linalg.norm(self.pos(n) - np.asarray(self.pose['idle_palm']) - off)) for n, off in self.idle_hand_objects.items()},
                           'wrist': self.wrist().round(4).tolist(), 'palm': np.asarray(self.pose['palm']).round(4).tolist(),
                           'tcp': self.tcp().round(4).tolist(),
                           'tcp_reference': None if self.controller.reference is None else self.controller.reference.round(4).tolist(),
                           'attached': self.attached_name(), 'hand_speed': round(self.hand_speed, 4),
                           'joint_position': self.d.qpos[self.env.qids].tolist(),
                           'joint_velocity': self.d.qvel[self.env.dids].tolist(),
                           'action': command.tolist(), 'ik_error_m': float(self.env.last_ik_error),
                           'command_speed': round(self.controller.speed, 4),
                           'forecast_valid': bool(self.manicast.valid), 'forecast_used': f is not None,
                           'forecast_wrist_1s': None if f is None else f[-1, self.wrist_index()].round(4).tolist(),
                           'fault_strength': round(float(self.manicast.fault_strength), 3),
                           'raw_forecast_wrist_1s': None if self.raw_log[-1] is None else self.raw_log[-1][-1, self.wrist_index()].round(4).tolist(),
                           'objects': {k: self.pos(k).round(4).tolist() for k in self.positions}})
        if self.writer is not None:
            self.writer.add(self.env.render(self.review_camera, self.review_hw[1], self.review_hw[0]))
        self.tick += 1

    def run(self, human_gen, robot_gen, done):
        """Drive both coroutines until `done()` (then settle 0.6 s), the robot ends, or max_ticks."""
        last_frame, robot_done_at = None, None
        try:
            while self.tick < self.max_ticks:
                try:
                    last_frame = next(human_gen)
                except HumanReachError as exc:
                    self.human_failed = True; self.event('human_reach_timeout', detail=str(exc)); break
                except StopIteration:
                    pass
                if last_frame is None:
                    raise RuntimeError('human script yielded nothing')
                self.apply_human(last_frame)
                if robot_done_at is None:
                    try:
                        command = next(robot_gen)
                    except StopIteration:
                        robot_done_at = self.tick; command = self.hold()
                    except MotionStalled as exc:
                        self.stalled = str(exc); self.event('motion_stalled', detail=self.stalled); break
                else:
                    command = self.hold()
                self.step(command)
                if done() and (robot_done_at is not None or self.phase == 'done') and self.tick - (robot_done_at or self.tick) >= 12:
                    break
                if robot_done_at is not None and self.tick - robot_done_at >= 60:
                    break
            else:
                self.event('time_limit', max_ticks=self.max_ticks)
        finally:
            if self.writer is not None:
                self.writer.close()

    def finish(self, success, failure_reason=None, extra=None):
        phys = self.validator.summary()
        valid = [r['forecast_valid'] for r in self.trace]
        used = [r['forecast_used'] for r in self.trace]
        summary = {'task': self.task, 'variant': self.variant, 'mode': self.mode, 'seed': self.seed,
                   'success': bool(success), 'failure_reason': None if success else (failure_reason or 'task_incomplete'),
                   'duration_s': round(self.t, 3), 'wall_s': round(time.time() - self.wall0, 1),
                   'events': self.events, 'final_positions': {k: self.pos(k).round(4).tolist() for k in self.positions},
                   **phys, 'human_model': self.human_model, 'expert_holds_for_human': self.hand_holds,
                   'forecast_validity_summary': {'frames': len(valid), 'valid': int(sum(valid)), 'used_by_expert': int(sum(used)),
                                                 'read_by_expert_while_moving': len(self.forecast_use_ticks)},
                   'variation': self.variation, 'variation_log': self.variation_log,
                   'human_push_max_m': round(getattr(self, 'human_push_max_m', 0.), 4),
                   'human_swivel_max_deg': round(getattr(self, 'human_swivel_max_deg', 0.), 1),
                   'fault': None if self.injector is None else self.injector.summary(),
                   'natural_forecast_error': self.natural_error()[0],
                   **(extra or {})}
        arrays = self.sidecar_arrays()
        if self.out is not None:
            (self.out / f'{self.name}_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            (self.out / f'{self.name}_trace.json').write_text(json.dumps(self.trace) + '\n')
        self.env.close()
        summary['_arrays'] = arrays           # popped by the collector into the episode sidecar (not JSON)
        return summary

    def natural_error(self):
        """Evaluator-only, after the rollout: error of the RAW ManiCast forecast against the pose that really
        followed, per tick and horizon ([T, 10] active-wrist metres, NaN where the future is outside the episode).
        Lets natural forecast errors be found in the data (59/QĐ-DL N5: natural errors first)."""
        n = len(self.kp_log); err = np.full((n, len(HORIZON_TICKS)), np.nan, np.float32)
        wi = self.wrist_index()
        for i, raw in enumerate(self.raw_log):
            if raw is None:
                continue
            for j, k in enumerate(HORIZON_TICKS):
                if i + k < n:
                    err[i, j] = np.linalg.norm(raw[j, wi] - self.kp_log[i + k][wi])
        e1 = err[:, -1][np.isfinite(err[:, -1])]
        stats = {'wrist_1s_mean_m': None if not len(e1) else round(float(e1.mean()), 4),
                 'wrist_1s_p90_m': None if not len(e1) else round(float(np.percentile(e1, 90)), 4),
                 'ticks_over_target_radius': int((e1 > float(STATION['intent']['target_radius_m'])).sum())}
        return stats, err

    def sidecar_arrays(self):
        """Per-tick diagnostic arrays for the episode sidecar: raw / effective forecast, fault strength, natural
        error. Frame i matches LeRobot frame i (one frame per tick)."""
        n = len(self.trace)
        raw = np.zeros((n, len(HORIZON_TICKS), len(JOINTS), 3), np.float32); raw_valid = np.zeros(n, np.uint8)
        for i in range(min(n, len(self.raw_log))):
            if self.raw_log[i] is not None:
                raw[i] = self.raw_log[i]; raw_valid[i] = 1
        return {'raw_forecast': raw, 'raw_forecast_valid': raw_valid,
                'fault_strength': np.asarray(self.fault_log[:n], np.float32),
                'natural_error_wrist': self.natural_error()[1][:n],
                'human_keypoints': np.asarray(self.kp_log[:n], np.float32)}


# ---------------------------------------------------------------------- human motion primitives
def move(start, end, duration, lift=0., motion=None, motion_scale=1., **kw):
    """Minimum-jerk palm move with an optional vertical arc and HY-Motion curvature."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    n = max(1, int(round(duration / DT)))
    for i in range(1, n + 1):
        u = smooth(i / n)
        p = start + (end - start) * u + np.array([0., 0., lift * np.sin(np.pi * i / n)])
        if motion is not None:
            p = p + motion_scale * motion_deviation(motion, u)
        yield {'palm': p, 'progress': u, **kw}


def move_path(pts, duration, shape='grasp', shape_from=None, phase='', **kw):
    """One minimum-jerk easing over a whole polyline (no stop at the waypoints)."""
    pts = [np.asarray(p, float) for p in pts]
    dists = [float(np.linalg.norm(pts[k + 1] - pts[k])) for k in range(len(pts) - 1)]
    total, cum = sum(dists), np.cumsum([0] + dists)
    n = max(1, int(round(duration / DT)))
    for i in range(1, n + 1):
        u = smooth(i / n)
        d = u * total
        idx = max(0, min(np.searchsorted(cum, d) - 1, len(pts) - 2))
        frac = (d - cum[idx]) / max(dists[idx], 1e-9)
        yield {'palm': pts[idx] + frac * (pts[idx + 1] - pts[idx]), 'progress': u, 'shape': shape,
               'shape_from': shape_from, 'phase': phase, **kw}


def hold(palm, duration=None, until=None, **kw):
    n = 10**6 if duration is None else max(1, int(round(duration / DT)))
    for _ in range(n):
        if until is not None and until():
            return
        yield {'palm': np.asarray(palm, float), **kw}


HY_BANK = ROOT / 'outputs/human_motion/hymotion_ten_tasks_2026-10-30/hymotion_out'


def hy_motion(ep, clip, sample=0, hand='right', bank=HY_BANK):
    """Smoothed HY-Motion palm-path segments of one generated clip placed in this scene."""
    src = Path(bank) / f'{clip}.npz'
    with np.load(src, allow_pickle=False) as raw:
        k3d = raw['keypoints3d'][sample].copy(); text = str(raw['text'])
    world = to_scene(k3d, np.array([*ep.scene['human']['stand_xy'], 0.]), ep.scene['human']['facing_deg'], ep.scene['floor_z'])
    path = palm_path(world, hand)
    parts = segment_ranges(path, tails=True)
    return [smooth_motion_path(path[a:b]) for a, b in parts], {'clip': clip, 'sample': sample, 'prompt': text,
                                                              'source': str(src.relative_to(ROOT))}


def carry(ep, obj, dest, start, dz=.035, lift=.08, reach_s=.9, carry_s=1.0, lower_s=.5, shape='grasp', phase='carry',
          lift_first=0.):
    """Human pick-and-place of `obj` to `dest` (object centre), welded to the hand while carried (v2/v3 person).
    `lift_first`: raise the object straight up this much before carrying it (58/QĐ-DL W3: over other objects)."""
    start = np.asarray(start, float)
    top = ep.pos(obj) + np.array([0., 0., dz])
    yield from move(start, top + [0, 0, .05], reach_s, lift=.04, shape=shape, shape_from='relaxed', phase=phase + '_reach')
    yield from move(top + [0, 0, .05], top, .35, shape=shape, phase=phase + '_lower')
    # 68/QĐ-DL B.1: settle_hand NOT applied here yet (18/11): W1 cake0 lags 9-12 cm at the weld (same bug as C1), but
    # with settle_hand W2 block3 never settles (palm oscillates 3.7 cm/s in contact at 3.1 cm, 3/3 seeds
    # human_reach_failed) and W1 seed 300401 deadlocks (human yield_1 vs robot wait_place). Open, see
    # outputs/diag/grasp_gap_1118/TABLE.md
    ep.human_grasp(obj)
    # place with the REAL hand-to-object offset at the grasp (the smoothed palm lags its command; 55/QĐ-DL W3 fix)
    end = np.asarray(dest, float) + (np.asarray(ep.pose['palm'], float) - ep.pos(obj))
    if lift_first > 0:
        yield from move(top, top + [0, 0, lift_first], .3, shape=shape, phase=phase + '_lift')
        top = top + [0, 0, lift_first]
    yield from move(top, end + [0, 0, lift], carry_s, lift=.03, shape=shape, phase=phase + '_move')
    yield from move(end + [0, 0, lift], end, lower_s, shape=shape, phase=phase + '_place')
    ep.human_release(obj)
    yield from move(end, end + [0, 0, .06], .3, shape='relaxed', shape_from=shape, phase=phase + '_release')
    return end + np.array([0., 0., .06])


def select_pointed(index_base, index_tip, visible_objects, *, max_distance_m, margin_m):
    """Object nearest the index-finger ray (observed pointing), None if ambiguous. Moved from i3_observed.py."""
    base, tip = np.asarray(index_base, float), np.asarray(index_tip, float)
    vec = tip - base; norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return None
    direction = vec / norm
    scores = []
    for key, pos in visible_objects.items():
        rel = np.asarray(pos, float) - base
        along = float(rel @ direction)
        if along <= 0:
            continue
        scores.append((float(np.linalg.norm(rel - along * direction)), key))
    scores.sort()
    if not scores or scores[0][0] > max_distance_m:
        return None
    if len(scores) > 1 and scores[1][0] - scores[0][0] < margin_m:
        return None
    return scores[0][1]


def dirt_geoms(m):
    return [g for g in range(m.ngeom) if m.geom(g).name.startswith('dirt_preview_')]


def wipe_marks(ep, xy, radius=.03):
    """Hide dirt marks within `radius` of xy (visual wiping proxy)."""
    n = 0
    for g in dirt_geoms(ep.m):
        if ep.m.geom_rgba[g, 3] > 0 and np.linalg.norm(ep.m.geom_pos[g, :2] - np.asarray(xy)[:2]) < radius:
            ep.m.geom_rgba[g, 3] = 0.; n += 1
    return n


def marks_left(ep):
    return int(sum(ep.m.geom_rgba[g, 3] > 0 for g in dirt_geoms(ep.m)))
