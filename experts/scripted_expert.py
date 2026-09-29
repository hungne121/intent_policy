"""Privileged waypoint expert that emits restricted actions (demonstration generator).

The expert reads simulator ground truth (object poses, the human's choice and stage). That is
acceptable for a *teacher*: the learned No-Intent policy only ever sees the recorded
observations (proprioception + images). One expert class handles all scenarios through a
per-scenario plan of generic steps; there is no per-skill model anywhere.

Scenario config `expert` block (teacher settings only):
  trigger: cue_complete  start once the human has finished the cue (Phase 1 default)
           cue_onset     clairvoyant: commit to the target as soon as the cue starts (upper-bound variant)
           evidence      information-timing contract (Phase 2): from the cue onset only target-agnostic
                         motion (open the gripper, go to a staging point between the candidates); commit
                         to a target only at `human_intention_evident` events, i.e. when the target has
                         become predictable from the observable cue. Commit times are recorded (L_demo).
  yield_prediction_horizon_s: intruder also yields on the predicted hand position this far ahead
                         (evidence mode: `predicted_hand_position`, cue_onset mode: the planned motion)
Guarded moves (cue_onset / evidence) hold while the human hand occupies the destination or is next to
the arm. The expert re-plans on intention changes (cue_onset: at the change, evidence: once the new
target is evident).
"""
from __future__ import annotations
import numpy as np
from benchmark.events import EventType as E
from controllers.restricted_action import RestrictedAction as A

TRAVEL_DZ = 0.13        # travel height above the table
GRASP_DZ = 0.012        # TCP height above object centre when grasping (fingertips reach ~3 cm below TCP)
SETTLE_TOL = 0.004      # hold until the measured TCP is this close to the set-point before gripping
YIELD_DISTANCE = 0.35   # intruder: hold while the human hand is closer than this to the TCP (the arm needs ~0.15 s to stop)
YIELD_GEOM_DISTANCE = 0.25  # ... or closer than this to any robot link (the hand may approach the wrist/forearm)
YIELD_ZONE_MARGIN = 0.05    # ... or inside the danger zone enlarged by this margin
# cue-onset expert, guarded moves (horizontal distances; only while the hand is within GUARD_Z in height):
GUARD_OCCUPIED_XY = 0.14    # the move's destination counts as occupied when the hand is this close to it ...
GUARD_APPROACH_XY = 0.20    # ... then the robot stops once it is this close to the hand
GUARD_CLOSE_XY = 0.10       # and it always stops when the hand is this close to the TCP / set-point
GUARD_Z = 0.15
TRIGGERS = ('cue_complete', 'cue_onset', 'evidence')


class Step:
    """One plan step. `kind`: wait | goto | grip | hold | settle. `guard`: hold while the hand is near (cue_onset)."""
    def __init__(self, kind, target=None, until=None, ticks=0, name='', guard=False):
        self.kind, self.target, self.until, self.ticks, self.name, self.guard = kind, target, until, ticks, name, guard


class ScriptedExpert:
    def __init__(self, scenario, mapper, rng_seed: int = 0):
        self.sc = scenario
        self.mapper = mapper
        self.rng = np.random.default_rng(rng_seed)
        self.tol = mapper.step / 2 + 1e-6

    # ------------------------------------------------------------------ plan construction
    def reset(self) -> None:
        sc = self.sc
        self.z0 = sc.table_z
        ecfg = sc.cfg.expert
        self.trigger = ecfg.get('trigger', 'cue_complete')
        if self.trigger not in TRIGGERS:
            raise ValueError(f'expert.trigger must be one of {TRIGGERS}, got {self.trigger!r}')
        self.anticipate = self.trigger in ('cue_onset', 'evidence')
        self.yield_horizon = float(ecfg.get('yield_prediction_horizon_s', 0.0))
        v = sc.variation
        self.obj = v['selected_object'] if 'selected_object' in v else v['requested_object']
        self.region = v.get('requested_target')
        self.n_changes = self.n_evidence = 0
        self.committed: str | None = None
        self.commit_log: list[dict] = []
        self._new_commit: str | None = None
        builder = {'instructor_object_to_target': self._plan_pick_place,
                   'intruder_pick_place_interruption': self._plan_pick_place,
                   'collaborator_object_handover': self._plan_handover,
                   'collaborator_bowl_assistance': self._plan_bowl}[sc.cfg.id]
        self.tcp_obj_offset = np.zeros(3)
        if self.trigger == 'evidence' and sc.cfg.role != 'intruder':
            self._set_plan(self._precommit_plan())
        else:
            self._set_plan(builder())

    def _set_plan(self, plan) -> None:
        self.plan = plan
        self.i = 0
        self.counter = 0
        self.axis = None

    def _trigger(self, protocol_step: str, name: str) -> Step:
        """Wait for the human cue: its completion (Phase 1) or its onset (anticipatory expert)."""
        sc = self.sc
        if self.anticipate:
            return Step('wait', until=lambda: sc.logger.count(E.HUMAN_CUE_ONSET) > 0, name='wait_cue_onset')
        return Step('wait', until=lambda: protocol_step in sc.protocol_done, name=name)

    def _staging_point(self) -> np.ndarray:
        """Target-agnostic waiting point: above the centroid of the candidate grasp points."""
        sc = self.sc
        if sc.cfg.id == 'collaborator_bowl_assistance':
            points = [self._bowl_handle(b)(sc.pos(b)) for b in sc.bowls]
        else:
            points = [sc.pos(k) for k in sc.objects]
        return np.r_[np.mean(np.asarray(points)[:, :2], axis=0), self.z0 + TRAVEL_DZ]

    def _precommit_plan(self):
        sc = self.sc
        return [Step('wait', until=lambda: sc.logger.count(E.HUMAN_CUE_ONSET) > 0, name='wait_cue_onset'),
                Step('grip', 'open', until=lambda: sc.env.data.qpos[sc.env.gid] > 0.075, ticks=25, name='open'),
                Step('goto', self._staging_point, name='staging', guard=True),
                Step('hold', ticks=10**6, name='await_evidence')]

    def _commit(self, target: str, plan) -> None:
        self.committed = self._new_commit = target
        self.commit_log.append(dict(t=round(self.sc.time, 4), target=target))
        self._set_plan(plan)

    def pop_commit(self) -> str | None:
        target, self._new_commit = self._new_commit, None
        return target

    def _lift_step(self, name: str) -> Step:
        """Rise to travel height (never down) before a re-planned lateral move."""
        return Step('goto', lambda: np.r_[self.mapper.setpoint[:2], max(self.mapper.setpoint[2], self.z0 + TRAVEL_DZ)],
                    name=name)

    def _grasp_steps(self, key, grasp_point):
        sc = self.sc
        return [
            Step('grip', 'open', until=lambda: sc.env.data.qpos[sc.env.gid] > 0.075, ticks=25, name='open'),
            Step('goto', lambda: np.r_[grasp_point()[:2], self.z0 + TRAVEL_DZ], name='above_grasp', guard=True),
            Step('goto', lambda: grasp_point(), name='descend', guard=True),
            Step('settle', ticks=20, name='settle_before_grasp'),
            Step('grip', 'close', until=lambda: sc.grasp_state[key], ticks=30, name='close'),
            Step('hold', ticks=3, name='settle_grasp'),
            Step('goto', lambda: np.r_[self.mapper.setpoint[:2], self.z0 + TRAVEL_DZ], name='lift'),
        ]

    def _object_grasp_point(self, key):
        return lambda: self.sc.pos(key) + [0, 0, GRASP_DZ]

    def _plan_pick_place(self):
        sc = self.sc
        obj = self.obj
        wait = [self._trigger('instruction_given', 'wait_instruction')] if sc.cfg.role == 'instructor' \
            else [Step('hold', ticks=int(self.rng.integers(3, 12)), name='start_delay')]
        return wait + self._grasp_steps(obj, self._object_grasp_point(obj)) + self._place_steps(obj)

    def _place_steps(self, obj):
        """Transport to / release in the *current* target region (it may be revised by the human)."""
        sc = self.sc
        region_xy = lambda: sc.region_pos(self.region)[:2]
        place = lambda: np.r_[region_xy(), sc.rest_height[obj] + 0.012 + GRASP_DZ]
        return [
            Step('goto', lambda: np.r_[region_xy(), self.z0 + TRAVEL_DZ], name='transport', guard=True),
            Step('goto', place, name='lower', guard=True),
            Step('settle', ticks=15, name='settle_before_release'),
            Step('grip', 'open', until=lambda: not sc.grasp_state[obj] and sc.env.data.qpos[sc.env.gid] > 0.06, ticks=25, name='release'),
            Step('goto', lambda: np.r_[self.mapper.setpoint[:2], self.z0 + TRAVEL_DZ], name='retreat'),
            Step('hold', ticks=10**6, name='done'),
        ]

    def _plan_handover(self):
        return [self._trigger('human_selection_shown', 'wait_selection')] + self._handover_steps(self.obj)

    def _handover_steps(self, obj):
        sc = self.sc
        receive = np.array(sc.variation['human']['receive_pose'])
        present = receive + np.asarray(sc.cfg.human_behavior['handover_offset'])
        staging = receive + np.asarray(sc.cfg.human_behavior['robot_wait_offset'])
        return self._grasp_steps(obj, self._object_grasp_point(obj)) + [
            Step('goto', lambda: staging + self.tcp_obj_offset, name='to_staging'),
            Step('wait', until=lambda: 'human_ready_to_receive' in sc.protocol_done, name='wait_human_ready'),
            Step('goto', lambda: present + self.tcp_obj_offset, name='to_handover'),
            Step('wait', until=lambda: sc.human.holding == obj, name='wait_human_grasp'),
            Step('grip', 'open', until=lambda: not sc.grasp_state[obj] and sc.env.data.qpos[sc.env.gid] > 0.06, ticks=25, name='release'),
            Step('wait', until=lambda: sc.human.stage in ('withdrawing', 'done'), name='wait_human_withdraw'),
            Step('goto', lambda: self.retreat_point, name='retreat'),
            Step('hold', ticks=10**6, name='done'),
        ]

    def _bowl_for(self, obj: str) -> str:
        return {v['pairs_with']: k for k, v in self.sc.scene['bowls'].items()}[obj]

    def _bowl_handle(self, bowl: str):
        r = self.sc.scene['bowls'][bowl].get('radius', 0.05)
        return lambda p: np.r_[p[0] - (r + 0.03), p[1], p[2] + 0.02 + GRASP_DZ]

    def _plan_bowl(self):
        return [self._trigger('human_picked_object', 'wait_pick')] + self._bowl_steps(self._bowl_for(self.obj))

    def _bowl_steps(self, bowl: str):
        sc = self.sc
        handle = self._bowl_handle(bowl)
        delivery = np.r_[sc.delivery, 0.0]
        return self._grasp_steps(bowl, lambda: handle(sc.pos(bowl))) + [
            Step('goto', lambda: handle(np.r_[delivery[:2], self.z0 + TRAVEL_DZ - 0.02]), name='carry'),
            Step('goto', lambda: handle(np.r_[delivery[:2], sc.rest_height[bowl] + 0.006]), name='lower'),
            Step('grip', 'open', until=lambda: not sc.grasp_state[bowl] and sc.env.data.qpos[sc.env.gid] > 0.06, ticks=25, name='release'),
            Step('goto', lambda: np.r_[self.mapper.setpoint[:2], self.z0 + TRAVEL_DZ], name='up'),
            Step('goto', lambda: np.r_[0.25, self.mapper.setpoint[1], self.z0 + TRAVEL_DZ], name='clear'),
            Step('hold', ticks=10**6, name='done'),
        ]

    # ------------------------------------------------------------------ intention changes
    def _follow_evidence(self) -> None:
        """Evidence mode: commit / re-commit when the human's target becomes predictable."""
        sc = self.sc
        new, self.n_evidence = sc.evidence_events[self.n_evidence:], len(sc.evidence_events)
        grasping = any(sc.grasp_state.values())
        for ev in new:
            kind = ev['kind']
            if kind == 'instruction' and not grasping:
                self.obj, self.region = ev['object'], ev['region']
                self._commit(self.obj, [self._lift_step('commit_lift')] + self._grasp_steps(self.obj, self._object_grasp_point(self.obj))
                             + self._place_steps(self.obj))
            elif kind == 'target_region':
                self.region = ev['region']
                if sc.grasp_state.get(self.obj, False):
                    self._commit(self.region, [self._lift_step('replan_lift')] + self._place_steps(self.obj))
            elif kind == 'target_object' and not grasping and (self.committed is None or ev['object'] != self.obj):
                self.obj = ev['object']
                if sc.cfg.id == 'collaborator_bowl_assistance':
                    bowl = self._bowl_for(self.obj)
                    self._commit(bowl, [self._lift_step('commit_lift')] + self._bowl_steps(bowl))
                else:
                    self._commit(self.obj, [self._lift_step('commit_lift')] + self._handover_steps(self.obj))

    def _follow_intention_change(self) -> None:
        sc = self.sc
        if len(sc.intention_changes) == self.n_changes:
            return
        self.n_changes = len(sc.intention_changes)
        change = sc.intention_changes[-1]
        if change['kind'] == 'target_object':
            if any(sc.grasp_state.values()):
                return
            self.obj = change['current']
            self._set_plan([self._lift_step('replan_lift')] + self._handover_steps(self.obj))
        elif change['kind'] == 'target_region':
            self.region = change['current']
            if sc.grasp_state.get(self.obj, False):
                self._set_plan([self._lift_step('replan_lift')] + self._place_steps(self.obj))

    # ------------------------------------------------------------------ acting
    def _toward(self, target) -> int | None:
        err = np.asarray(target, float) - self.mapper.setpoint
        rot = self.mapper.frame_rotation()
        err_frame = rot.T @ err
        over = np.abs(err_frame) > self.tol
        if not over.any():
            self.axis = None
            return None
        # Keep moving along the current axis until it is done: fewer direction switches
        # (a zig-zag staircase shakes the grasped object loose).
        if self.axis is None or not over[self.axis]:
            self.axis = int(np.argmax(np.abs(err_frame) - self.tol))
        axis = self.axis
        positive = err_frame[axis] > 0
        return [(A.MOVE_FORWARD, A.MOVE_BACKWARD), (A.MOVE_LEFT, A.MOVE_RIGHT), (A.MOVE_UP, A.MOVE_DOWN)][axis][0 if positive else 1]

    def _near_danger(self, hand: np.ndarray) -> bool:
        sc = self.sc
        in_zone = bool(np.all(hand >= sc.zone_lo - YIELD_ZONE_MARGIN) and np.all(hand <= sc.zone_hi + YIELD_ZONE_MARGIN))
        return in_zone or float(np.linalg.norm(hand - sc.tcp())) < YIELD_DISTANCE

    def _yield(self) -> bool:
        sc = self.sc
        if sc.cfg.role != 'intruder':
            return False
        if self._near_danger(sc.human.pos) or sc.human_distance < YIELD_GEOM_DISTANCE:
            return True
        if self.yield_horizon <= 0:
            return False
        future = sc.human.predicted_hand_position(self.yield_horizon) if self.trigger == 'evidence' \
            else sc.human.planned_hand_position(sc.time + self.yield_horizon)
        return self._near_danger(future)

    def _guard_hold(self, destination) -> bool:
        """Hold instead of moving towards a hand that occupies the destination or is next to the arm.

        Distances use both the measured TCP and the set-point (the arm lags the set-point by a few cm).
        """
        hand = self.sc.human.pos
        near = lambda p, r: float(np.linalg.norm(hand[:2] - np.asarray(p)[:2])) < r and abs(float(hand[2] - p[2])) < GUARD_Z
        arm = (self.sc.tcp(), self.mapper.setpoint)
        if any(near(p, GUARD_CLOSE_XY) for p in arm):
            return True
        return near(np.r_[np.asarray(destination)[:2], hand[2]], GUARD_OCCUPIED_XY) and any(near(p, GUARD_APPROACH_XY) for p in arm)

    def act(self) -> int:
        if self.trigger == 'evidence':
            self._follow_evidence()
        else:
            self._follow_intention_change()
        if self._yield():
            return int(A.HOLD)
        while self.i < len(self.plan):
            step = self.plan[self.i]
            if step.kind == 'wait':
                if step.until():
                    self._advance()
                    continue
                return int(A.HOLD)
            if step.kind == 'hold':
                if self.counter < step.ticks:
                    self.counter += 1
                    return int(A.HOLD)
                self._advance()
                continue
            if step.kind == 'settle':
                close = np.linalg.norm(self.sc.tcp() - self.mapper.setpoint) < SETTLE_TOL
                if close or self.counter >= step.ticks:
                    self._advance()
                    continue
                self.counter += 1
                return int(A.HOLD)
            if step.kind == 'grip':
                if (step.until is not None and step.until()) or self.counter >= step.ticks:
                    if step.name == 'release':
                        self.retreat_point = self.mapper.setpoint + [-0.10, 0.0, 0.05]
                    if step.name == 'close':
                        key = self._grasped_key()
                        if key is not None:
                            self.tcp_obj_offset = self.sc.tcp() - self.sc.pos(key)
                    self._advance()
                    continue
                self.counter += 1
                return int(A.OPEN_GRIPPER if step.target == 'open' else A.CLOSE_GRIPPER)
            if step.kind == 'goto':
                target = step.target()
                a = self._toward(target)
                if a is None:
                    self._advance()
                    continue
                if step.guard and self.anticipate and self._guard_hold(target):
                    return int(A.HOLD)
                return int(a)
        return int(A.HOLD)

    def _grasped_key(self):
        keys = [k for k, g in self.sc.grasp_state.items() if g]
        return keys[0] if keys else None

    def _advance(self):
        self.i += 1
        self.counter = 0

    @property
    def stage(self) -> str:
        return self.plan[self.i].name if self.i < len(self.plan) else 'finished'
