"""Privileged waypoint expert that emits restricted actions (demonstration generator) for the tasks T1-T5.

The expert reads simulator ground truth (object poses, the human's choice and stage). That is acceptable for a
*teacher*: the learned No-Intent policy only ever sees the recorded observations (proprioception + images). One
expert class handles all tasks through a per-task plan of generic steps; there is no per-skill model anywhere.

  T1       grasp the cube -> (place zone known) -> place it in P -> rest
  T4       no instruction: from start_s grasp the single cube -> place it in P1 -> rest; hold while a hand is in the
           robot zone (every trigger behaves the same)
  T2       grasp the object -> hold it at a staging point until a hand is out in a hand zone -> over the open palm,
           down into it, hold still -> release when the human has closed the hand on it and pulls -> rest
  T3       grasp the cup paired with the cube the human picks up, by the rim -> stand it in U_cup -> rest
  T5       re-plan to the new target (before any grasp)

Scenario config `expert` block (teacher settings only):
  trigger: cue_complete  act once the human's cue is complete (object_indicated, then place_indicated / target_indicated / block_picked;
                         after a change of mind: once the new target has been indicated)
           evidence      information-timing contract (Phase 2): from the cue onset only target-agnostic motion (open
                         the gripper, go to a staging point above the candidates); commit to a target only at
                         `human_intention_evident` events, i.e. when the target has become predictable from the
                         observable gesture, and touch it only CONFIRM_S after that. A withdrawn pointing hand (T5)
                         is evidence too: the robot rises and waits.
           cue_onset     clairvoyant: the target is known at the cue onset and after a change at the change itself
  travel_dz: travel height above the table
  T2 staging_offset: holding point (x, y) w.r.t. the centre of the hand zones, at handover height
  T4 yield_zone_margin_m / yield_prediction_horizon_s: hold while the (predicted) hand is in the robot zone
  T4 start_s: the robot starts by itself this long after the episode start (default 0.5)
  grasp_settle_ticks: HOLD decisions after the gripper closed, before lifting (default 3). Fixed waits like this and
                      start_s end on a timer the policy cannot observe (identical frames, different labels).
Guarded moves (cue_onset / evidence) hold while the human hand occupies the destination or is next to the arm.
"""
from __future__ import annotations
import numpy as np
from intent_policy.benchmark.events import EventType as E
from intent_policy.sim.restricted_action import RestrictedAction as A

SETTLE_TOL = 0.004      # hold until the measured TCP is this close to the set-point before gripping ...
SETTLE_SPEED = 0.02     # ... and has (nearly) stopped: a fast lateral move near the base overshoots while it descends
# guarded moves (horizontal distances; only while the hand is within GUARD_Z in height):
GUARD_OCCUPIED_XY = 0.14    # the move's destination counts as occupied when the hand is this close to it ...
GUARD_APPROACH_XY = 0.20    # ... then the robot stops once it is this close to the hand
GUARD_CLOSE_XY = 0.10       # and it always stops when the hand is this close to the TCP / set-point
GUARD_Z = 0.15
SEPARATION_M = 0.12     # T4: also hold while the palm is this close to the TCP ...
SEPARATION_GEOM_M = 0.04  # ... or any human proxy this close to a robot link
CONFIRM_S = 1.6         # evidence mode: touch the target only once its gesture has been held this long (a pointing
                        # that is withdrawn earlier, T5 early change, never leads to touching the old target)
R_SAFE = 0.22           # moves across y = 0 go around the robot base at least this far out (near-singular above it)
HANDOVER_DZ = 0.025     # T2: the object crosses the open fingers (up to 2.4 cm above the palm) this high above its
                        # place in the palm, then goes straight down into it (H2 is at the edge of the arm's reach)
TRIGGERS = ('cue_complete', 'cue_onset', 'evidence')
# cue_complete: each sub-intention is acted on once its deictic gesture is complete (the pointing hand arrived)
CUE_COMPLETE_STEP = {'T1': 'object_indicated', 'T2': 'target_indicated', 'T3': 'block_picked'}
FOREVER = 10 ** 6


class Step:
    """One plan step. kind: wait | goto | grip | hold | settle | call. `guard`: hold while the hand is near
    (anticipating triggers). `skip`: callable, the step ends as soon as it returns True."""
    def __init__(self, kind, target=None, until=None, ticks=0, name='', guard=False, skip=None):
        self.kind, self.target, self.until, self.ticks, self.name, self.guard, self.skip = kind, target, until, ticks, name, guard, skip


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
        self.code = sc.cfg.task_code
        self.travel = self.z0 + float(ecfg.get('travel_dz', 0.13))
        self.yield_margin = float(ecfg.get('yield_zone_margin_m', 0.0))
        self.yield_horizon = float(ecfg.get('yield_prediction_horizon_s', 0.0))
        self.grasp_settle = int(ecfg.get('grasp_settle_ticks', 3))
        self.obj: str | None = None
        self.zone: str | None = None
        self.n_changes = self.n_evidence = 0
        self.committed: str | None = None
        self.commit_t = 0.0
        self.commit_log: list[dict] = []
        self._new_commit: str | None = None
        self.tcp_obj_offset = np.zeros(3)
        self.retreat_point = None
        # change of mind with free timing (2026-10-03): the human signals 'wait', then points at the new target
        self.timed_change = 'change_delay_s' in sc.human.params
        self.stop_seen = self.change_seen = False
        if self.code == 'T4':                   # no instruction: the robot starts the pick-and-place by itself
            start = float(ecfg.get('start_s', 0.5))
            self._set_plan([Step('wait', until=lambda: sc.time >= start - 1e-9, name='wait_start'),
                            Step('call', target=self._start_known, name='start')])
        elif self.trigger == 'evidence':
            self._set_plan(self._precommit_plan())
        else:
            cue = (lambda: sc.logger.count(E.HUMAN_CUE_ONSET) > 0) if self.trigger == 'cue_onset' \
                else (lambda: CUE_COMPLETE_STEP[self.code] in sc.protocol_done)
            self._set_plan([Step('wait', until=cue, name='wait_cue'), Step('call', target=self._start_known, name='start')])

    def _start_known(self) -> None:
        """cue_complete / cue_onset (and T4): the current choice of the human is known."""
        self.obj, self.zone = self.sc.human.selected_object, self._known_zone()
        self._commit(self.obj, self._task_steps())

    def _known_zone(self) -> str | None:
        """Zone known without evidence: T3's cup spot U_cup and T4's fixed place zone P1, T1's place zone once
        instructed, T2's hand zone once the hand is out (cue_onset: from the start)."""
        sc = self.sc
        if self.code in ('T3', 'T4') or self.trigger == 'cue_onset':
            return sc.variation['target_zone']
        if self.code == 'T1':
            return sc.variation['target_zone'] if 'place_indicated' in sc.protocol_done else None
        return sc.variation['target_zone'] if 'hand_ready' in sc.protocol_done else None

    def _set_plan(self, plan) -> None:
        self.plan = plan
        self.i = 0
        self.counter = 0
        self.axis = None
        self.blocked: set[int] = set()
        self._moved_from = None
        self._band = False

    def _staging_point(self) -> np.ndarray:
        """Target-agnostic waiting point: above the centroid of the candidate grasp points."""
        sc = self.sc
        points = [sc.grasp_point(k) for k in sc.variation['spec']['layout']]
        return np.r_[np.mean(np.asarray(points)[:, :2], axis=0), self.travel]

    def _precommit_plan(self):
        sc = self.sc
        return [Step('wait', until=lambda: sc.logger.count(E.HUMAN_CUE_ONSET) > 0, name='wait_cue_onset'),
                Step('grip', 'open', until=lambda: sc.env.data.qpos[sc.env.gid] > 0.075, ticks=25, name='open'),
                Step('goto', self._staging_point, name='staging', guard=True),
                Step('hold', ticks=FOREVER, name='await_evidence')]

    def _commit(self, target: str, plan) -> None:
        self.commit_t = self.sc.time
        self.committed = self._new_commit = target
        self.commit_log.append(dict(t=round(self.sc.time, 4), target=target))
        self._set_plan(plan)

    def pop_commit(self) -> str | None:
        target, self._new_commit = self._new_commit, None
        return target

    def _lift_step(self, name: str) -> Step:
        """Rise to travel height (never down) before a re-planned lateral move."""
        return Step('goto', lambda: np.r_[self.mapper.setpoint[:2], max(self.mapper.setpoint[2], self.travel)], name=name)

    def _task_steps(self) -> list[Step]:
        obj = self.obj
        grasp = self._grasp_steps(obj)
        if self.code in ('T1', 'T4'):
            return grasp + [Step('wait', until=lambda: self.zone is not None, name='await_zone')] + self._place_steps(obj) + self._home_steps()
        if self.code == 'T2':
            return grasp + self._handover_steps(obj)
        # T3: the cup stands right in front of the human: first move away from them (+y, not into the base band)
        clear = Step('goto', lambda: np.r_[self.mapper.setpoint[0], max(self.mapper.setpoint[1], -R_SAFE - 0.005), self.travel],
                     name='clear_human')
        return grasp + self._place_steps(obj) + [clear] + self._home_steps()

    def _grasp_steps(self, key):
        sc = self.sc
        return [
            Step('grip', 'open', until=lambda: sc.env.data.qpos[sc.env.gid] > 0.075, ticks=25, name='open'),
            Step('goto', lambda: np.r_[sc.grasp_point(key)[:2], self.travel], name='above_grasp', guard=True),
            Step('settle', ticks=60, name='settle_above_grasp'),     # descend only once the TCP is above the grasp
            Step('wait', until=lambda: self.trigger != 'evidence' or self.code == 'T4' or sc.time - self.commit_t >= CONFIRM_S,
                 name='confirm_target'),
            Step('goto', lambda: sc.grasp_point(key), name='descend', guard=True),
            Step('settle', ticks=60, name='settle_before_grasp'),   # near the base the joints lag the set-point
            Step('grip', 'close', until=lambda: sc.grasp_state[key], ticks=30, name='close'),
            Step('hold', ticks=self.grasp_settle, name='settle_grasp'),
            Step('goto', lambda: np.r_[self.mapper.setpoint[:2], self.travel], name='lift'),
        ]

    def _place_steps(self, obj):
        """Transport to / release in the *current* zone (T1/T4: place zone P, T3: the cup spot U_cup)."""
        sc = self.sc
        cup = sc.scene['objects'][obj]['shape'] == 'cup'
        drop = 0.003 if cup else 0.012
        above = lambda: np.r_[sc.zone_pos(self.zone)[:2] + self.tcp_obj_offset[:2], self.travel]
        place = lambda: np.r_[sc.zone_pos(self.zone)[:2], sc.rest_height[obj] + drop] + self.tcp_obj_offset
        return [
            Step('goto', above, name='transport', guard=True),
            Step('goto', place, name='lower', guard=True),
            Step('settle', ticks=15, name='settle_before_release'),
            Step('grip', 'open', until=lambda: not sc.grasp_state[obj] and sc.env.data.qpos[sc.env.gid] > 0.06, ticks=25, name='release'),
            Step('goto', lambda: np.r_[self.mapper.setpoint[:2], self.travel], name='retreat'),
        ]

    def _home_steps(self):
        return [Step('goto', lambda: self.sc.rest_tcp, name='to_rest'), Step('hold', ticks=FOREVER, name='done')]

    def _handover_steps(self, obj):
        sc = self.sc
        hands = np.mean([sc.zone_pos(k) for k, z in sc.zones.items() if z['kind'] == 'hand'], axis=0)
        in_palm = np.asarray(sc.cfg.human_behavior['present_offset'], float) + [0.0, 0.0, sc.rest_height[obj] - sc.table_z]
        present = lambda: sc.zone_pos(self.zone) + in_palm + self.tcp_obj_offset     # resting in the upturned palm
        over_z = lambda: hands[2] + in_palm[2] + HANDOVER_DZ + self.tcp_obj_offset[2]  # the hand zones share one height
        staging = lambda: np.r_[hands[:2] + np.asarray(sc.cfg.expert['staging_offset'], float) + self.tcp_obj_offset[:2], over_z()]
        human = sc.human
        return [
            # always via the staging point at handover height: the arm rises where it can (not next to the base or at
            # the edge of its reach), then crosses over the open fingers, over the palm and straight down into it
            Step('goto', staging, name='to_staging', guard=True),
            Step('wait', until=lambda: self.zone is not None, name='await_hand'),
            Step('goto', lambda: np.r_[present()[:2], over_z()], name='over_hand'),
            Step('settle', ticks=15, name='settle_over_hand'),
            Step('goto', present, name='to_handover'),
            Step('wait', until=lambda: human.holding == obj and human.stage in ('pulling', 'wait_release'), name='wait_pull'),
            Step('grip', 'open', until=lambda: not sc.grasp_state[obj] and sc.env.data.qpos[sc.env.gid] > 0.06, ticks=25, name='release'),
            Step('wait', until=lambda: human.stage in ('withdraw_with_object', 'done'), name='wait_human_withdraw'),
            Step('goto', lambda: self.retreat_point, name='retreat'),
        ] + self._home_steps()

    # ------------------------------------------------------------------ following the human
    def _follow_evidence(self) -> None:
        """Evidence mode: commit / re-commit when the human's target becomes predictable."""
        sc = self.sc
        new, self.n_evidence = sc.evidence_events[self.n_evidence:], len(sc.evidence_events)
        grasping = any(sc.grasp_state.values())
        for ev in new:
            kind = ev['kind']
            if kind == 'target_zone':
                self.zone = ev['zone']              # pending steps read the zone when they start
                self.commit_log.append(dict(t=round(sc.time, 4), target=self.zone))
            elif kind == 'stop' and self.obj is not None:
                self._pause()                       # 'wait!' (evident from the hand moving out towards the robot)
            elif kind == 'target_withdrawn' and grasping and self.timed_change and self.obj is not None:
                self._pause()                       # T3: the cube goes back while the robot holds a cup
            elif kind == 'target_withdrawn' and not grasping and self.obj is not None:
                self.obj = self.committed = None    # the pointing hand is withdrawn: rise and wait
                self._set_plan([self._lift_step('withdrawn_lift'), Step('hold', ticks=FOREVER, name='await_evidence')])
            elif kind == 'target_object' and grasping and self.timed_change and ev['object'] != self.obj:
                self._redirect(ev['object'])        # changed while the old object is held: put it back first
            elif kind == 'target_object' and not grasping and ev['object'] != self.obj:
                self.obj = ev['object']
                if self.code == 'T3':
                    self.zone = sc.variation['target_zone']
                self._commit(self.obj, [self._lift_step('commit_lift')] + self._task_steps())

    def _follow_correction(self) -> None:
        """cue_complete, free-timing change of mind: pause once the 'wait' signal is complete, re-target once the new
        pointing has arrived (`change_indicated`), putting the old object back first if it is held."""
        done = self.sc.protocol_done
        if not self.stop_seen and 'stop_signaled' in done:
            self.stop_seen = True
            self._pause()
        if not self.change_seen and 'change_indicated' in done:
            self.change_seen = True
            self._redirect(self.sc.human.selected_object)

    def _pause(self) -> None:
        """'Wait!': stop where the arm is (holding whatever it holds) until the new target is known."""
        if self.obj is not None:
            self._set_plan([Step('hold', ticks=FOREVER, name='await_correction')])

    def _redirect(self, new: str) -> None:
        """Re-target to `new`: a held (old) object is put back where it stood first."""
        sc = self.sc
        held = next((k for k, g in sc.grasp_state.items() if g), None)
        if self.obj is None and held is None:
            return                                  # not started yet: the plan reads the choice when it starts
        old = sc.change_old
        self.obj = new
        if self.code == 'T3':
            self.zone = sc.variation['target_zone']
        if held is not None and held != new:
            steps = self._put_back_steps(held)
        elif old is not None and old != new and np.linalg.norm(sc.pos(old)[:2] - sc.start_pos[old][:2]) > 0.02:
            # the old object was already delivered (late: before the change was clear): fetch it back first
            steps = [self._lift_step('fetch_lift')] + self._grasp_steps(old) + self._put_back_steps(old)
        else:
            steps = []
        self._commit(new, steps + [self._lift_step('replan_lift')] + self._task_steps())

    def _put_back_steps(self, key: str) -> list[Step]:
        """Carry `key` back to where it stood and release it there."""
        sc = self.sc
        cup = sc.scene['objects'][key]['shape'] == 'cup'
        drop = 0.003 if cup else 0.012
        off = {}                                # TCP - object, measured once held (the 'wait' may cut the close short)
        spot = lambda: np.r_[sc.start_pos[key][:2], sc.rest_height[key] + drop] + off['v']
        return [Step('call', target=lambda: off.update(v=sc.tcp() - sc.pos(key)), name='put_back_hold'),
                self._lift_step('put_back_lift'),
                Step('goto', lambda: np.r_[spot()[:2], self.travel], name='put_back_transport', guard=True),
                Step('goto', spot, name='put_back_lower', guard=True),
                Step('settle', ticks=15, name='settle_put_back'),
                Step('grip', 'open', until=lambda: not sc.grasp_state[key] and sc.env.data.qpos[sc.env.gid] > 0.06,
                     ticks=25, name='put_back_release'),
                Step('goto', lambda: np.r_[self.mapper.setpoint[:2], self.travel], name='put_back_retreat')]

    def _follow_intention_change(self) -> None:
        """cue_onset: switch at the change; cue_complete: stop, and switch once the new target has been pointed at."""
        sc = self.sc
        if len(sc.intention_changes) == self.n_changes:
            return
        self.n_changes = len(sc.intention_changes)
        change = sc.intention_changes[-1]
        if self.obj is None or any(sc.grasp_state.values()):
            return                                  # not started yet (the plan reads the choice at its start) / too late
        self.obj = change['current']
        wait = [] if self.trigger == 'cue_onset' else \
            [Step('wait', until=lambda: 'change_indicated' in sc.protocol_done, name='wait_new_target')]
        self._commit(self.obj, [self._lift_step('replan_lift')] + wait + self._task_steps())

    # ------------------------------------------------------------------ acting
    def _toward(self, target) -> int | None:
        err = np.asarray(target, float) - self.mapper.setpoint
        rot = self.mapper.frame_rotation()
        err_frame = rot.T @ err
        over = np.abs(err_frame) > self.tol
        if not over.any():
            self.axis = None
            return None
        # A translation the controller refused (IK reach, e.g. passing over the robot base) blocks its axis until
        # the set-point moves again: the path then goes around along another axis.
        if self._moved_from is not None and np.array_equal(self._moved_from, self.mapper.setpoint) and self.axis is not None:
            self.blocked.add(self.axis)
            self.axis = None
        elif self._moved_from is not None:
            self.blocked.clear()
        free = over & ~np.isin(np.arange(3), list(self.blocked))
        # Moves along y through the band |y| < R_SAFE pass the robot base (the arm is near-singular above it and the
        # joints lag the set-point): first out to x >= R_SAFE along +x, then across, and only then inwards along x
        # (world frame = base_link frame here).
        sp, tg = self.mapper.setpoint, np.asarray(target, float)
        band = bool(over[1] and 0 not in self.blocked and min(sp[1], tg[1]) < R_SAFE and max(sp[1], tg[1]) > -R_SAFE)
        if self._band and not band:
            self.axis = None                    # out of the band: choose the axis again (e.g. inwards before going on
        self._band = band                       #   towards a human standing at the end of the y move)
        if band:
            if sp[0] < R_SAFE - float(np.max(self.tol)):
                self.axis = None
                self._moved_from = sp.copy()
                return A.MOVE_FORWARD
            if 1 not in self.blocked and err_frame[0] < 0:
                free[0] = False                 # no inward x move before the y move is done
        # Keep moving along the current axis until it is done: fewer direction switches
        # (a zig-zag staircase shakes the grasped object loose).
        if self.axis is None or not free[self.axis]:
            score = np.where(free, np.abs(err_frame) - self.tol, -np.inf) if free.any() else np.abs(err_frame) - self.tol
            self.axis = int(np.argmax(score))
        axis = self.axis
        positive = err_frame[axis] > 0
        self._moved_from = self.mapper.setpoint.copy()
        return [(A.MOVE_FORWARD, A.MOVE_BACKWARD), (A.MOVE_LEFT, A.MOVE_RIGHT), (A.MOVE_UP, A.MOVE_DOWN)][axis][0 if positive else 1]

    def _yield(self) -> bool:
        """T4: hold while the human hand is in the robot zone (optionally also the predicted hand), and while a
        hand is right next to the gripper anywhere (separation monitoring: never move into a hand)."""
        sc = self.sc
        if self.code != 'T4':
            return False
        if sc.in_robot_zone(sc.human.pos, self.yield_margin):
            return True
        if float(np.linalg.norm(sc.human.pos - sc.tcp())) < SEPARATION_M or sc.human_distance < SEPARATION_GEOM_M:
            return True
        if self.yield_horizon <= 0:
            return False
        future = sc.human.predicted_hand_position(self.yield_horizon) if self.trigger == 'evidence' \
            else sc.human.planned_hand_position(sc.time + self.yield_horizon)
        return sc.in_robot_zone(future, self.yield_margin)

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
        moved_from, self._moved_from = self._moved_from, None
        if self.trigger == 'evidence':
            self._follow_evidence()
        else:
            if self.trigger == 'cue_complete' and self.timed_change:
                self._follow_correction()
            else:
                self._follow_intention_change()
            if self.zone is None and self.obj is not None:
                self.zone = self._known_zone()
        if self._yield():
            return int(A.HOLD)
        while self.i < len(self.plan):
            step = self.plan[self.i]
            if step.skip is not None and step.skip():
                self._advance()
                continue
            if step.kind == 'call':
                self._advance()
                step.target()
                continue
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
                close = np.linalg.norm(self.sc.tcp() - self.mapper.setpoint) < SETTLE_TOL and self.sc.tcp_speed < SETTLE_SPEED
                if close or self.counter >= step.ticks:
                    self._advance()
                    continue
                self.counter += 1
                return int(A.HOLD)
            if step.kind == 'grip':
                if (step.until is not None and step.until()) or self.counter >= step.ticks:
                    if step.name == 'release':
                        away = [0.0, 0.06, 0.10] if self.code == 'T2' else [0.0, 0.0, 0.05]   # T2: up, then away from the hand (+Y)
                        self.retreat_point = self.mapper.setpoint + away
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
                self._moved_from = moved_from
                a = self._toward(target)
                if a is None:
                    self._advance()
                    continue
                if step.guard and self.anticipate and self._guard_hold(target):
                    self._moved_from = None
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
