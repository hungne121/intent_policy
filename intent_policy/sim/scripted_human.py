"""Deterministic, scripted human behaviours (one per HRI scenario type).

The human is a mocap hand driven by minimum-jerk segments. Behaviours are small reactive
state machines: they read a per-tick context dict supplied by the scenario (object poses,
robot grasp status, ...), schedule hand motion, emit benchmark events and request
attach/detach of objects to the hand. Given the same variation and the same robot
behaviour, the human trajectory is bit-identical.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from intent_policy.sim.human_state import HumanState


def min_jerk(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u ** 2)


@dataclass
class Segment:
    start_t: float
    duration: float
    p0: np.ndarray
    p1: np.ndarray
    label: str
    clearance: float | None = None   # cautious motion: stop if the robot is closer than this [m]

    @property
    def end_t(self) -> float:
        return self.start_t + self.duration

    def position(self, t: float) -> np.ndarray:
        return self.p0 + (self.p1 - self.p0) * min_jerk((t - self.start_t) / self.duration)


class ScriptedHuman:
    """Base class: segment playback, velocity estimate, event/request queues.

    Information-timing contract (Phase 2): the destination of a hand motion counts as predictable
    once `evidence_fraction` of the motion's duration has elapsed (about where reach-target predictors
    become reliable). Motions listed in EVIDENCE_LABELS reveal the human's intention; their evidence
    time is scheduled when the motion starts and a `human_intention_evident` event is emitted when it
    is reached (cancelled if the motion is replaced or stopped before). `predicted_hand_position`
    follows the same rule: the planned motion after the evidence point, constant-velocity
    extrapolation of the observed hand before it.
    """
    EVIDENCE_LABELS: dict[str, str] = {}     # motion label -> evidence kind

    def __init__(self, scene: dict, behavior: dict, variation: dict):
        self.scene_human = scene['human']
        self.behavior = behavior
        self.var = variation
        self.params = variation['human']
        self.speed = float(self.params['speed_scale'])
        self.rest = np.array(self.scene_human['hand_rest'], float)
        self.torso = np.array(self.scene_human['torso_pos'], float)
        self.evidence_fraction = float(behavior.get('evidence_fraction', 0.4))
        self.selected_object: str | None = None
        self.selected_target: str | None = None
        self.reset()

    def reset(self) -> None:
        self.pos = self.rest.copy()
        self.vel = np.zeros(3)
        self.segment: Segment | None = None
        self.stage = 'idle'
        self.stage_t = 0.0
        self.t_last = 0.0
        self.holding: str | None = None
        self.events: list[tuple[str, dict]] = []
        self.requests: list[tuple[str, str]] = []
        self.extras: dict = {}
        self.evidence: list[dict] = []       # scheduled evidence: t, kind, payload, segment, emitted, cancelled

    # -- helpers ---------------------------------------------------------------------------
    def emit(self, event_type: str, **payload) -> None:
        self.events.append((event_type, payload))

    def evidence_payload(self, label: str) -> dict:
        """Target information revealed by a motion (subclasses)."""
        return {}

    def add_evidence(self, t: float, kind: str, segment: Segment | None = None, **payload) -> None:
        self.evidence.append(dict(t=float(t), kind=kind, payload=payload, segment=segment, emitted=False, cancelled=False))

    def _cancel_pending_evidence(self, segment: Segment | None) -> None:
        for ev in self.evidence:
            if ev['segment'] is segment and segment is not None and not ev['emitted']:
                ev['cancelled'] = True

    def known_evidence(self, t: float) -> list[dict]:
        """Evidence entries that are predictable at time t (scheduled, not cancelled, t_ev <= t)."""
        return [ev for ev in self.evidence if not ev['cancelled'] and ev['t'] <= t + 1e-9]

    def move(self, t: float, target, duration: float, label: str, clearance: float | None = None) -> None:
        """Schedule a minimum-jerk hand motion; `duration` is scaled by the speed variant.

        With `clearance`, the motion is cautious: the hand stops where it is as soon as the
        robot is closer than `clearance` (the scripted human never pushes into the robot; a robot
        that keeps moving into a stopped hand still causes contact).
        """
        self._cancel_pending_evidence(self.segment)
        self.segment = Segment(t, max(duration / self.speed, 0.05), self.pos.copy(), np.asarray(target, float), label,
                               clearance)
        self.emit('human_motion_start', label=label, target=np.round(target, 4).tolist(), duration=self.segment.duration)
        if label in self.EVIDENCE_LABELS:
            self.add_evidence(t + self.evidence_fraction * self.segment.duration, self.EVIDENCE_LABELS[label],
                              self.segment, **self.evidence_payload(label))

    @property
    def moving(self) -> bool:
        return self.segment is not None

    def goto(self, stage: str, t: float) -> None:
        self.stage, self.stage_t = stage, t

    def planned_hand_position(self, t: float) -> np.ndarray:
        """Hand position at time t under the currently scheduled segment (clairvoyant future-motion GT)."""
        if self.segment is None or t <= self.segment.start_t:
            return self.pos.copy()
        return self.segment.position(min(t, self.segment.end_t))

    def predicted_hand_position(self, horizon: float, lead: float = 0.0) -> np.ndarray:
        """Future hand position `horizon` s after the last update under the information-timing contract.

        The current motion's plan is used once `evidence_fraction` of it has elapsed (at t + lead);
        before that, and at rest, the observed hand is extrapolated at constant velocity.
        """
        t, seg = self.t_last, self.segment
        if seg is not None and t + lead - seg.start_t >= self.evidence_fraction * seg.duration - 1e-9:
            return seg.position(min(t + horizon, seg.end_t))
        return self.pos + self.vel * horizon

    # -- main tick -------------------------------------------------------------------------
    def update(self, t: float, dt: float, ctx: dict) -> None:
        """Advance the behaviour to time t. Consumes nothing; fills self.events/self.requests."""
        self.events, self.requests = [], []
        previous = self.pos.copy()
        self.behave(t, ctx)
        seg = self.segment
        if seg is not None and seg.clearance is not None and ctx.get('human_robot_distance', np.inf) < seg.clearance:
            self.emit('human_motion_end', label=seg.label, stopped_for_clearance=True)
            self._cancel_pending_evidence(seg)
            self.segment = None
        for ev in self.evidence:
            if not ev['emitted'] and not ev['cancelled'] and ev['t'] <= t + 1e-9:
                ev['emitted'] = True
                self.emit('human_intention_evident', kind=ev['kind'], **ev['payload'])
        if self.segment is not None:
            self.pos = self.segment.position(t)
            if t >= self.segment.end_t - 1e-9:
                self.pos = self.segment.p1.copy()
                self.emit('human_motion_end', label=self.segment.label)
                self.segment = None
        self.vel = (self.pos - previous) / dt
        self.t_last = t

    def behave(self, t: float, ctx: dict) -> None:
        raise NotImplementedError

    def state(self, t: float) -> HumanState:
        return HumanState(timestamp=t, torso_position=self.torso.copy(), torso_velocity=np.zeros(3),
                          hand_position=self.pos.copy(), hand_velocity=self.vel.copy(), interaction_state=self.stage,
                          selected_object=self.selected_object, selected_target=self.selected_target,
                          holding=self.holding, motion_label=self.segment.label if self.segment else None,
                          extras=dict(self.extras))


class InstructorPointingHuman(ScriptedHuman):
    """Shows the instruction on the board, points at the object, then at the target region.

    Optional intention change (`variation.intention_change`, type target_region): a fixed delay after
    the robot has lifted the requested object, the human revises the target region once — the board
    switches to the other region's colour and the human points at the new region.
    """

    def __init__(self, scene, behavior, variation):
        super().__init__(scene, behavior, variation)
        self.selected_object = variation['requested_object']
        self.selected_target = variation['requested_target']
        self.changed = False
        self.lift_t = None

    def hover(self, xy_or_xyz) -> np.ndarray:
        p = np.asarray(xy_or_xyz, float)
        return np.array([p[0] + 0.08, p[1], self.behavior_table_z + self.behavior['hover_height_m']])

    def maybe_change_target(self, t, ctx) -> bool:
        ic = self.var.get('intention_change')
        if not ic or not ic['enabled'] or self.changed or self.stage == 'idle':
            return False
        if self.lift_t is None and ctx['robot_lifted_object'] == self.selected_object:
            self.lift_t = t
        if self.lift_t is None or t - self.lift_t < ic['delay_after_lift_s']:
            return False
        self.changed = True
        previous = self.selected_target
        self.selected_target = [r for r in ctx['region_positions'] if r != previous][0]
        self.emit('human_intention_change', kind='target_region', previous=previous, current=self.selected_target)
        self.requests.append(('show_instruction', self.selected_object))
        self.add_evidence(t, 'target_region', region=self.selected_target)     # the board shows the revision at once
        self.move(t, self.hover(ctx['region_positions'][self.selected_target]), 0.9, 'point_new_target')
        self.goto('to_target', t)
        return True

    def behave(self, t, ctx):
        self.behavior_table_z = ctx['table_top_z']
        p = self.params
        if self.maybe_change_target(t, ctx):
            return
        if self.stage == 'idle' and t >= p['cue_onset_s']:
            self.emit('human_cue_onset', modality='instruction_board+pointing',
                      requested_object=self.selected_object, requested_target=self.selected_target)
            self.requests.append(('show_instruction', self.selected_object))
            self.add_evidence(t, 'instruction', object=self.selected_object, region=self.selected_target)
            self.move(t, self.hover(ctx['object_positions'][self.selected_object]), 0.9, 'point_object')
            self.goto('instructing', t)
        elif self.stage == 'instructing' and not self.moving:
            self.goto('pointing_object', t)
        elif self.stage == 'pointing_object' and t - self.stage_t >= p['point_object_s']:
            self.move(t, self.hover(ctx['region_positions'][self.selected_target]), 0.9, 'point_target')
            self.goto('to_target', t)
        elif self.stage == 'to_target' and not self.moving:
            self.goto('pointing_target', t)
        elif self.stage == 'pointing_target' and t - self.stage_t >= p['point_target_s']:
            self.move(t, self.rest, 1.0, 'return_rest')
            self.goto('returning', t)
        elif self.stage == 'returning' and not self.moving:
            self.emit('protocol_step_complete', step='instruction_given')
            self.goto('observing', t)


class HandoverReceiverHuman(ScriptedHuman):
    """Indicates the selected object from a distance, waits at a receive pose, and takes the
    object once the robot presents it and holds still; withdraws after the robot releases.

    Optional intention change (`variation.intention_change`, type target_object): a fixed delay after
    the cue the human switches to the other object and indicates it instead. The change only happens
    while the robot has not grasped anything; otherwise it is skipped (recorded in `extras`).
    """

    EVIDENCE_LABELS = {'indicate_object': 'target_object', 'reindicate_object': 'target_object'}

    def __init__(self, scene, behavior, variation):
        super().__init__(scene, behavior, variation)
        self.selected_object = variation['selected_object']
        self.receive_pose = np.array(self.params['receive_pose'], float)
        self.still_since = None
        self.changed = False

    def evidence_payload(self, label):
        return dict(object=self.selected_object)

    def maybe_change_object(self, t, ctx) -> bool:
        ic, p = self.var.get('intention_change'), self.params
        if not ic or not ic['enabled'] or self.changed or self.stage not in ('indicating', 'selection_shown') \
                or t < p['cue_onset_s'] + ic['delay_after_cue_s']:
            return False
        self.changed = True
        if any(ctx['robot_grasping'].values()) or ctx['robot_lifted_object'] is not None:
            self.extras['intention_change_skipped_t'] = round(t, 4)     # robot already committed to the object
            return False
        previous = self.selected_object
        self.selected_object = [k for k in ctx['object_positions'] if k != previous][0]
        self.emit('human_intention_change', kind='target_object', previous=previous, current=self.selected_object)
        target = np.asarray(ctx['object_positions'][self.selected_object]) + np.asarray(self.behavior['indicate_offset'])
        self.move(t, target, ic['reach_s'], 'reindicate_object', clearance=self.behavior['hand_clearance_m'])
        self.goto('indicating', t)
        return True

    def behave(self, t, ctx):
        p, b = self.params, self.behavior
        if self.maybe_change_object(t, ctx):
            return
        obj = self.selected_object
        if self.stage == 'idle' and t >= p['cue_onset_s']:
            self.emit('human_cue_onset', modality='hand_indication', selected_object=obj)
            target = np.asarray(ctx['object_positions'][obj]) + np.asarray(b['indicate_offset'])
            self.move(t, target, p['reach_s'], 'indicate_object')
            self.goto('indicating', t)
        elif self.stage == 'indicating' and not self.moving:
            self.emit('protocol_step_complete', step='human_selection_shown')
            self.goto('selection_shown', t)
        elif self.stage == 'selection_shown' and ctx['robot_lifted_object'] is not None:
            # Withdraw towards the body first, then present the palm: never sweep across the
            # robot's workspace while it is carrying the object.
            self.move(t, self.pos + np.asarray(b['withdraw_offset']), 0.5 * p['receive_move_s'], 'withdraw_before_receive',
                      clearance=b['hand_clearance_m'])
            self.goto('withdrawing_to_receive', t)
        elif self.stage == 'withdrawing_to_receive' and not self.moving:
            self.move(t, self.receive_pose, 0.5 * p['receive_move_s'], 'move_to_receive', clearance=b['hand_clearance_m'])
            self.goto('to_receive', t)
        elif self.stage == 'to_receive' and not self.moving:
            self.emit('interaction_window_start', kind='handover', actor='human')
            self.emit('protocol_step_complete', step='human_ready_to_receive')
            self.goto('receiving', t)
        elif self.stage == 'receiving':
            offered = ctx['robot_lifted_object'] == obj and \
                np.linalg.norm(np.asarray(ctx['object_positions'][obj]) - self.pos) < b['accept_radius_m']
            self.still_since = (self.still_since if self.still_since is not None else t) \
                if offered and ctx['robot_speed'] < 0.02 else None
            if self.still_since is not None and t - self.still_since >= b['accept_still_s']:
                grasp = np.asarray(ctx['object_positions'][obj]) + np.asarray(b['grasp_offset'])
                self.move(t, grasp, b['final_reach_s'], 'take_object', clearance=b['hand_clearance_m'])
                self.goto('taking', t)
        elif self.stage == 'taking' and not self.moving:
            self.requests.append(('attach', obj))
            self.holding = obj
            self.emit('object_grasp', object=obj, actor='human')
            self.goto('holding_wait_release', t)
        elif self.stage == 'holding_wait_release' and not ctx['robot_grasping'].get(obj, False):
            self.emit('protocol_step_complete', step='object_transferred')
            self.emit('interaction_window_end', kind='handover', actor='human')
            self.goto('released_by_robot', t)
        elif self.stage == 'released_by_robot' and t - self.stage_t >= 0.3:
            self.move(t, self.rest, 1.2, 'withdraw_with_object')
            self.goto('withdrawing', t)
        elif self.stage == 'withdrawing' and not self.moving:
            self.goto('done', t)


class BowlRequesterHuman(ScriptedHuman):
    """Picks one object, waits with it; drops it only into the matching delivered bowl."""

    EVIDENCE_LABELS = {'reach_object': 'target_object'}

    def __init__(self, scene, behavior, variation):
        super().__init__(scene, behavior, variation)
        self.selected_object = variation['selected_object']
        self.bowl_for = {v['pairs_with']: k for k, v in scene['bowls'].items()}
        self.selected_target = self.bowl_for[self.selected_object]
        self.delivery = np.array(scene['delivery_spot'], float)

    def evidence_payload(self, label):
        return dict(object=self.selected_object)

    def behave(self, t, ctx):
        p, b = self.params, self.behavior
        obj = self.selected_object
        z0 = ctx['table_top_z']
        if self.stage == 'idle' and t >= p['cue_onset_s']:
            self.emit('human_cue_onset', modality='human_pick', selected_object=obj)
            grasp = np.asarray(ctx['object_positions'][obj]) + [0.0, 0.0, 0.03]
            self.move(t, grasp, p['reach_s'], 'reach_object')
            self.goto('reaching', t)
        elif self.stage == 'reaching' and not self.moving:
            self.requests.append(('attach', obj))
            self.holding = obj
            self.emit('object_grasp', object=obj, actor='human')
            self.emit('protocol_step_complete', step='human_picked_object')
            hold = np.r_[self.delivery, z0] + np.asarray(b['hold_pose_offset'])
            self.move(t, hold, p['lift_s'], 'lift_and_wait')
            self.goto('lifting', t)
        elif self.stage == 'lifting' and not self.moving:
            self.emit('interaction_window_start', kind='bowl_request', actor='human')
            self.goto('waiting_for_bowl', t)
        elif self.stage == 'waiting_for_bowl':
            bowl = self.selected_target
            if ctx['bowl_ready'].get(bowl, False) and ctx['tcp_distance_to'](ctx['bowl_positions'][bowl]) > 0.12:
                drop = np.asarray(ctx['bowl_positions'][bowl])[:3].copy()
                drop[2] = z0 + b['drop_height_m']
                self.move(t, drop, p['drop_move_s'], 'move_over_bowl')
                self.goto('to_bowl', t)
        elif self.stage == 'to_bowl' and not self.moving:
            self.requests.append(('detach', obj))
            self.holding = None
            self.emit('object_release', object=obj, actor='human')
            self.goto('dropped', t)
        elif self.stage == 'dropped' and t - self.stage_t >= 0.6:
            self.emit('interaction_window_end', kind='bowl_request', actor='human')
            self.move(t, self.rest, 1.0, 'withdraw')
            self.goto('withdrawing', t)
        elif self.stage == 'withdrawing' and not self.moving:
            self.goto('done', t)


class IntruderHuman(ScriptedHuman):
    """Moves the hand into the robot's workspace during the task, dwells, then withdraws."""
    EVIDENCE_LABELS = {'intrude': 'intrusion'}

    def evidence_payload(self, label):
        return dict(point=np.round(self.segment.p1, 4).tolist())

    def behave(self, t, ctx):
        p = self.params
        if self.stage == 'idle':
            triggered = ctx['robot_lifted_object'] is not None if p['trigger'] == 'during_transport' \
                else any(ctx['robot_grasping'].values())
            if triggered:
                self.extras['trigger_time'] = t
                self.goto('scheduled', t)
        elif self.stage == 'scheduled' and t - self.stage_t >= p['trigger_delay_s']:
            point = np.asarray(ctx['tcp']) + np.asarray(p['intrusion_offset'])
            point[2] = max(point[2], ctx['table_top_z'] + 0.06)
            self.extras['intrusion_point'] = point.round(4).tolist()
            self.move(t, point, p['approach_s'], 'intrude', clearance=self.behavior['hand_clearance_m'])
            self.goto('approaching', t)
        elif self.stage == 'approaching' and not self.moving:
            self.goto('dwelling', t)
        elif self.stage == 'dwelling' and t - self.stage_t >= p['dwell_s']:
            self.move(t, self.rest, p['withdraw_s'], 'withdraw')
            self.goto('withdrawing', t)
        elif self.stage == 'withdrawing' and not self.moving:
            self.goto('done', t)


BEHAVIORS = {
    'instructor_pointing': InstructorPointingHuman,
    'handover_receiver': HandoverReceiverHuman,
    'bowl_requester': BowlRequesterHuman,
    'intruder': IntruderHuman,
}


def make_human(scene: dict, behavior: dict, variation: dict) -> ScriptedHuman:
    return BEHAVIORS[behavior['type']](scene, behavior, variation)
