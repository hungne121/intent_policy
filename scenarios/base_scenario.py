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
from benchmark.events import EventType as E
from benchmark.logger import EpisodeEventLogger
from env.base_env import ManipulationEnv, ROOT
from env.scene_builder import build_scene_xml, object_half_height, resolve_color, weld_name, robot_weld_name, BOARD_SHAPES
from human.scripted_human import make_human
from scenarios.config import ScenarioConfig, sample_variation

LIFT_HEIGHT = 0.03          # [m] above rest height => object counts as lifted by the robot
GRASP_DEBOUNCE = 2          # consecutive ticks required before grasp/release events
COMMIT_MARGIN = 0.04        # [m] the TCP approached one candidate this much more than every other one ...
COMMIT_WINDOW = 0.5         # [s] ... within this window => robot_target_commit (agent-independent)
COMMIT_APPROACH = 0.01      # [m] ... while actually getting closer to it
COMMIT_SWITCH_FACTOR = 2.0  # re-committing to another candidate needs this times the margin (servo drift of a
                            # few cm while descending must not count as a switch; a real switch moves ~0.2 m)


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
    """Subclasses define `robot_target`, `human_context_extras`, `update_task`, `success_predicate`."""

    def __init__(self, cfg: ScenarioConfig, robot_config=ROOT / 'configs/robot.yaml', gui: bool = False):
        self.cfg = cfg
        self.scene = cfg.scene
        self.env = ManipulationEnv(robot_config, gui=gui, model_xml=build_scene_xml(self.scene))
        m = self.env.model
        self.table_z = float(self.scene['table_top_z'])
        self.objects = list(self.scene.get('objects', {}))
        self.bowls = list(self.scene.get('bowls', {}))
        self.regions = list(self.scene.get('regions', {}))
        self.graspables = self.objects + self.bowls
        self.body_id = {k: m.body(k).id for k in self.graspables}
        self.qadr = {k: int(m.joint(f'{k}_free').qposadr[0]) for k in self.graspables}
        self.dadr = {k: int(m.joint(f'{k}_free').dofadr[0]) for k in self.graspables}
        self.weld_id = {k: m.equality(weld_name(k)).id for k in self.graspables}
        for k in self.graspables:
            self.env.register_graspable(self.body_id[k], m.equality(robot_weld_name(k)).id)
        self.mocap = {k: int(m.body(k).mocapid[0]) for k in ['human_hand', 'human_forearm', 'human_grasp_anchor'] + self.regions}
        root = m.body('robot_mount').id
        self.robot_geoms = []
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            while b > 0 and b != root:
                b = int(m.body_parentid[b])
            if b == root and m.geom_contype[g]:
                self.robot_geoms.append(g)
        self.human_geoms = [m.geom('human_palm').id, m.geom('human_forearm_geom').id]
        self.object_geom_ids = {k: [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.body_id[k]] for k in self.objects}
        if 'instruction_board' in self.scene:
            names = [m.geom(g).name for g in range(m.ngeom)]
            self.board_geom_ids = {s: [g for g, n in enumerate(names) if n.startswith(f'board_obj_{s}')] for s in BOARD_SHAPES}
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

    def region_pos(self, key: str) -> np.ndarray:
        return self.env.data.mocap_pos[self.mocap[key]].copy()

    def inside_region(self, p: np.ndarray, region: str, margin: float = 0.0) -> bool:
        half = self.scene['regions'][region]['half_size'] - margin
        return bool(np.all(np.abs(p[:2] - self.region_pos(region)[:2]) <= half))

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
    def reset(self, seed: int | None = None, episode_id: str | None = None) -> None:
        seed = self.cfg.seed if seed is None else int(seed)
        self.seed = seed
        self.episode_id = episode_id or f'{self.cfg.id}__seed{seed:05d}'
        self.variation = var = sample_variation(self.cfg, seed)
        env, m, d = self.env, self.env.model, self.env.data
        env.reset(np.asarray(env.cfg['home']) + var['robot_home_offset'])
        palette = self.scene['palette']
        for k in self.objects:
            obj = self.scene['objects'][k]
            q = self.qadr[k]
            yaw = var['object_yaw'][k]
            d.qpos[q:q + 3] = [*var['object_xy'][k], self.table_z + object_half_height(obj) + 0.001]
            d.qpos[q + 3:q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            for g in self.object_geom_ids[k]:
                m.geom_rgba[g] = resolve_color(var['object_color'][k], palette)
        for k in self.bowls:
            q = self.qadr[k]
            d.qpos[q:q + 3] = [*var['bowl_xy'][k], self.table_z + 0.0045]
            d.qpos[q + 3:q + 7] = [1, 0, 0, 0]
        for k in self.regions:
            d.mocap_pos[self.mocap[k]] = [*var['region_xy'][k], self.table_z]
            m.geom_rgba[m.geom(f'{k}_geom').id] = resolve_color(var['region_color'][k], palette)
        if 'instruction_board' in self.scene:
            for gid in self.board_geom_ids.values():
                for g in gid:
                    m.geom_rgba[g] = 0
            m.geom_rgba[m.geom('board_target').id] = 0
        for eid in self.weld_id.values():
            d.eq_active[eid] = False
        self.human = make_human(self.scene, self.cfg.human_behavior, var)
        self._apply_human_pose()
        mujoco.mj_forward(m, d)
        env.settle(10)
        self.rest_height = {k: float(self.pos(k)[2]) for k in self.graspables}
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
        self.commit_hist: deque = deque()
        self.commit_stage_name: str | None = None
        self.committed: str | None = None
        self.done = False
        self.logger.start(self.episode_id, self.time, seed=seed, variation=var)
        self.on_reset()

    def on_reset(self) -> None:
        """Subclass hook for task-specific trackers."""

    # ------------------------------------------------------------------ human
    def _apply_human_pose(self) -> None:
        d = self.env.data
        hand = self.human.pos
        shoulder = np.asarray(self.scene['human']['shoulder'], float)
        direction = shoulder - hand
        direction /= np.linalg.norm(direction) + 1e-12
        length = self.scene['human']['forearm_length']
        d.mocap_pos[self.mocap['human_hand']] = hand
        d.mocap_pos[self.mocap['human_forearm']] = hand + direction * (length / 2 + 0.02)
        d.mocap_quat[self.mocap['human_forearm']] = quat_z_to(direction)
        d.mocap_pos[self.mocap['human_grasp_anchor']] = hand

    def attach_to_hand(self, key: str) -> None:
        m, d = self.env.model, self.env.data
        eid = self.weld_id[key]
        m.eq_data[eid, 0:3] = 0
        m.eq_data[eid, 3:6] = self.pos(key) - d.mocap_pos[self.mocap['human_grasp_anchor']]
        m.eq_data[eid, 6:10] = self.quat(key)
        d.eq_active[eid] = True

    def detach_from_hand(self, key: str) -> None:
        self.env.data.eq_active[self.weld_id[key]] = False

    def human_context(self) -> dict:
        tcp = self.tcp()
        ctx = dict(t=self.time, table_top_z=self.table_z, tcp=tcp,
                   object_positions={k: self.pos(k) for k in self.objects},
                   region_positions={k: self.region_pos(k) for k in self.regions},
                   bowl_positions={k: self.pos(k) for k in self.bowls},
                   robot_grasping=dict(self.grasp_state), robot_lifted_object=self.robot_lifted,
                   human_robot_distance=self.human_distance, robot_speed=self.tcp_speed,
                   tcp_distance_to=lambda p: float(np.linalg.norm(np.asarray(p)[:2] - tcp[:2])))
        ctx.update(self.human_context_extras())
        return ctx

    def human_context_extras(self) -> dict:
        return {}

    def _show_instruction(self, obj_key: str) -> None:
        m = self.env.model
        palette = self.scene['palette']
        shape = self.scene['objects'][obj_key]['shape']
        for g in self.board_geom_ids[shape]:
            m.geom_rgba[g] = resolve_color(self.variation['object_color'][obj_key], palette)
        target = self.human.selected_target          # follows a mid-episode change of the instruction
        m.geom_rgba[m.geom('board_target').id] = resolve_color(self.variation['region_color'][target], palette)
        self.instruction_visible = True

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
            elif kind == 'show_instruction':
                self._show_instruction(key)
        for event_type, payload in self.human.events:
            if event_type == E.PROTOCOL_STEP_COMPLETE.value:
                self.protocol(payload.pop('step'), 'human', **payload)
            else:
                if event_type == E.HUMAN_INTENTION_CHANGE.value:
                    self.intention_changes.append(dict(t=t0, **payload))
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
        self.update_task()
        self._update_generic_failures()
        if not self.success and self.failure is None:
            self.stable = self.stable + 1 if self.success_predicate() else 0
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
        self.moving_now = moving              # instantaneous (safety); robot_moving has hysteresis (events)
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
        if self.robot_lifted is not None and self.robot_lifted != self.robot_target():
            self.fail('wrong_object_manipulated')

    def commit_stage(self) -> tuple[str, dict, str] | None:
        """(stage name, {candidate: position}, correct candidate) while a target choice is pending (subclasses)."""
        return None

    def _update_commitment(self) -> None:
        """Log robot_target_commit when the TCP clearly heads for one candidate (whatever the agent is)."""
        stage = self.commit_stage()
        if stage is None or len(stage[1]) < 2:
            self.commit_hist.clear()
            self.commit_stage_name = None
            return
        name, candidates, correct = stage
        if name != self.commit_stage_name:
            self.commit_stage_name, self.committed = name, None
            self.commit_hist.clear()
        tcp = self.tcp()[:2]
        d = {k: float(np.linalg.norm(tcp - np.asarray(p)[:2])) for k, p in candidates.items()}
        self.commit_hist.append((self.time, d))
        while self.time - self.commit_hist[0][0] > COMMIT_WINDOW + 1e-9:
            self.commit_hist.popleft()
        d_old = self.commit_hist[0][1]
        for k in d:
            gain = min((d[o] - d[k]) - (d_old[o] - d_old[k]) for o in d if o != k)
            approached = d_old[k] - d[k] >= COMMIT_APPROACH
            margin = COMMIT_MARGIN if self.committed is None else COMMIT_SWITCH_FACTOR * COMMIT_MARGIN
            if gain >= margin and approached and k != self.committed:
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
        return self.pos(key)

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
        violation = dist < self.safety['violation_distance_m'] and self.moving_now
        if violation and not self.violation_active:
            self.log(E.SAFETY_DISTANCE_VIOLATION, 'robot', distance=dist, tcp_speed=self.tcp_speed, kind='separation')
        self.violation_active = violation

    def _update_generic_failures(self) -> None:
        for k in self.graspables:
            if self.pos(k)[2] < self.table_z - 0.05:
                self.fail(f'{k}_fell_off_table')
        if not np.isfinite(self.env.data.qpos).all():
            self.fail('simulation_unstable')

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
        """World position of an oracle target key (graspable, region or scenario-specific point)."""
        if key in self.body_id:
            return self.pos(key)
        if key in self.regions:
            return self.region_pos(key)
        raise KeyError(key)

    def oracle_alternatives(self, key: str) -> list[str]:
        """Other candidates of the same slot (for the wrong-information condition)."""
        for group in (self.objects, self.bowls, self.regions):
            if key in group:
                return [k for k in group if k != key]
        return []

    # ------------------------------------------------------------------ public interface
    def get_observation(self, cameras=('scene', 'wrist'), width: int = 128, height: int = 96) -> dict:
        """Phase-1 policy observation: proprioception + RGB only. No privileged/intention fields."""
        obs = {'observation.state': self.env.observe().policy_vector()}
        for cam in cameras:
            obs[f'observation.images.{cam}'] = self.env.render(cam, width, height)
        return obs

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
                                     robot_grasped=self.grasp_state[k]) for k in self.graspables},
                    human_robot_distance=round(self.human_distance, 5), tcp_speed=round(self.tcp_speed, 5))

    def close(self) -> None:
        self.env.close()
