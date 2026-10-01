"""Deterministic, scripted human behaviours for the tasks T1-T5 (docs/requirements/scence_construct.md §4.4).

The human is a palm target driven by minimum-jerk segments (the mannequin follows it, sim/human_body.py). A
behaviour is a queue of gestures (point at a location, move the hand, wait for a condition, pause, call) that
reads a per-tick context dict supplied by the scenario (object poses, robot grasp status, ...), emits benchmark
events and requests attach/detach of objects to the hand. Reactive parts (change of mind, intrusion, handover)
rewrite the queue. Given the same variation and the same robot behaviour, the human trajectory is bit-identical.
Gestures: `point` (index finger aimed at a table location), `reach_out` (open hand held in a hand zone H),
`intrude` (hand into the robot zone); a change of mind has no dedicated gesture: the hand is withdrawn to rest and
then points at the new target.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from intent_policy.sim.human_state import HumanState


def segment_distance(q, a, b) -> float:
    """Distance from point q to the segment a-b."""
    ab = b - a
    u = float(np.clip(np.dot(q - a, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(q - (a + u * ab)))


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
        self.torso = np.zeros(3)                  # chest position, set by the scenario from the mannequin pose
        self.evidence_fraction = float(behavior.get('evidence_fraction', 0.4))
        self.selected_object: str | None = None
        self.selected_target: str | None = None
        self.reach = None                        # optional palm-target clamp to the body's reach (set by the scenario)
        self.point_pose = None                   # table location -> pointing palm position (set by the scenario)
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
        self._prev_robot_distance = np.inf
        self.point_at: np.ndarray | None = None   # what the index finger points at while gesturing (rendering only)
        self.shape: str | None = None             # hand shape set by the current gesture (None: from point_at / holding)
        self.palm_from = self.palm_to = 0.0       # palm orientation (0 down, 1 up) at the start / end of the motion

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
        target = np.asarray(target, float) if self.reach is None else self.reach(target)
        self.segment = Segment(t, max(duration / self.speed, 0.05), self.pos.copy(), target, label, clearance)
        self.emit('human_motion_start', label=label, target=np.round(target, 4).tolist(), duration=self.segment.duration)
        if label in self.EVIDENCE_LABELS:
            self.add_evidence(t + self.evidence_fraction * self.segment.duration, self.EVIDENCE_LABELS[label],
                              self.segment, **self.evidence_payload(label))

    @property
    def hand_shape(self) -> str:
        """Rendered hand shape: the gesture's shape, else point while gesturing (also with a held cube), grasp while
        holding, relaxed."""
        if self.shape is not None:
            return self.shape
        return 'point' if self.point_at is not None else 'grasp' if self.holding else 'relaxed'

    @property
    def palm_up(self) -> float:
        """Palm orientation for rendering (0 down, 1 up), turning along the current motion."""
        seg = self.segment
        if seg is None or self.palm_from == self.palm_to:
            return self.palm_to
        return self.palm_from + (self.palm_to - self.palm_from) * min_jerk((self.t_last - seg.start_t) / seg.duration)

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
        # predictive: also keep the distance by which the gap to the robot closes in the next tick (closing speed of
        # hand and robot; a hand moving past the robot, not towards it, needs no extra margin)
        d = float(ctx.get('human_robot_distance', np.inf))
        closing = max(0.0, (self._prev_robot_distance - d) / dt) if np.isfinite(d) and np.isfinite(self._prev_robot_distance) else 0.0
        self._prev_robot_distance = d
        if seg is not None and seg.clearance is not None and d < seg.clearance + 1.5 * closing * dt:
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




class Gesture:
    """One queued human action.

    kind: point (aim the index finger at `target(ctx)` from the pointing pose) | move (palm to `target(ctx)`) |
    wait (until `until(t, ctx)`) | sleep (`duration` s) | call (`fn(t, ctx)`, then continue at once).
    point / move finish `dwell` s after the hand arrived (or stopped for clearance).
    """
    def __init__(self, kind: str, label: str = '', target=None, duration: float = 0.0, dwell: float = 0.0,
                 until=None, fn=None, clearance: float | None = None, shape: str | None = None, palm_up: float = 0.0):
        self.kind, self.label, self.target, self.duration, self.dwell = kind, label, target, duration, dwell
        self.until, self.fn, self.clearance = until, fn, clearance
        self.shape, self.palm_up = shape, palm_up          # hand shape (None: derived) / palm up at the end of the motion
        self.t0 = self.arrived = None


class GestureHuman(ScriptedHuman):
    """Behaviour = queue of gestures; `stage` is the current gesture's label ('idle' before the cue)."""

    EVIDENCE_LABELS = {'point_object': 'target_object', 'point_place': 'target_zone', 'reach_out': 'target_zone',
                       'intrude': 'intrusion', 'withdraw_change': 'target_withdrawn'}

    def __init__(self, scene, behavior, variation):
        super().__init__(scene, behavior, variation)
        spec = variation['spec']
        self.spec = spec
        self.target = spec['target']
        self.change = variation.get('change')
        self.selected_object = self.change['old'] if self.change else self.target
        self.selected_target = variation.get('target_zone')
        self.changed = False
        self.started = False
        self._old_d = None

    def reset(self) -> None:
        super().reset()
        self.queue: list[Gesture] = []
        self.current: Gesture | None = None

    # -- gestures ---------------------------------------------------------------------------
    def G_point_object(self, key_fn, dwell) -> Gesture:
        """Point at the (current) selected object; key_fn() -> object key at the time the gesture starts."""
        return Gesture('point', 'point_object', target=lambda ctx: np.asarray(ctx['object_positions'][key_fn()], float),
                       duration=self.params['point_s'], dwell=dwell)

    def G_point_zone(self, zone, dwell) -> Gesture:
        return Gesture('point', 'point_place', target=lambda ctx: np.r_[np.asarray(ctx['zone_positions'][zone])[:2], ctx['table_top_z']],
                       duration=self.params['point_s'], dwell=dwell)

    def G_rest(self, label='return_rest', duration=None, pose=None, shape=None, palm_up=0.0) -> Gesture:
        pose = self.rest if pose is None else np.asarray(pose, float)
        return Gesture('move', label, target=lambda ctx: pose, duration=self.params['rest_s'] if duration is None else duration,
                       shape=shape, palm_up=palm_up)

    def G_call(self, fn) -> Gesture:
        return Gesture('call', fn=fn)

    def G_protocol(self, step, **payload) -> Gesture:
        return self.G_call(lambda t, ctx: self.emit('protocol_step_complete', step=step, **payload))

    # -- queue ------------------------------------------------------------------------------
    def evidence_payload(self, label):
        if label == 'point_object':
            return dict(object=self.selected_object)
        if label in ('point_place', 'reach_out'):
            return dict(zone=self.selected_target)
        if label == 'intrude':
            return dict(point=np.round(self.segment.p1, 4).tolist())
        return {}

    def _start(self, g: Gesture, t: float, ctx: dict) -> None:
        g.t0 = t
        if g.kind == 'call':
            g.fn(t, ctx)
            return
        self.goto(g.label, t)
        if g.kind in ('point', 'move'):
            self.palm_from, self.palm_to = self.palm_up, g.palm_up
        if g.kind == 'point':
            aim = np.asarray(g.target(ctx), float)
            self.point_at, self.shape = aim, 'point'
            self.move(t, self.point_pose(aim), g.duration, g.label, g.clearance)
        elif g.kind == 'move':
            self.point_at, self.shape = None, g.shape
            self.move(t, g.target(ctx), g.duration, g.label, g.clearance)

    def _finished(self, g: Gesture, t: float, ctx: dict) -> bool:
        if g.kind == 'call':
            return True
        if g.kind == 'wait':
            return bool(g.until(t, ctx))
        if g.kind == 'sleep':
            return t - g.t0 >= g.duration - 1e-9
        if self.moving:
            return False
        g.arrived = t if g.arrived is None else g.arrived
        return t - g.arrived >= g.dwell - 1e-9

    def run_queue(self, t: float, ctx: dict) -> None:
        while True:
            if self.current is None:
                if not self.queue:
                    return
                self.current = self.queue.pop(0)
                self._start(self.current, t, ctx)
            if not self._finished(self.current, t, ctx):
                return
            self.current = None

    def replace_queue(self, gestures: list[Gesture]) -> None:
        """Drop the current gesture and the queue; the next tick starts `gestures` from the current hand pose."""
        self.current = None
        self.queue = list(gestures)

    # -- change of mind (T5) ------------------------------------------------------------------
    def _switch_target(self, t: float) -> None:
        previous = self.selected_object
        self.selected_object = self.target
        self.changed = True
        self.emit('human_intention_change', kind='target_object', previous=previous, current=self.target,
                  timing=self.change['timing'])

    def G_change(self, rest_pose=None) -> list[Gesture]:
        """Withdraw the pointing hand all the way to rest (no dedicated cancel gesture; the hand keeps its pointing
        shape), pause, then point at the new target."""
        return [self.G_call(lambda t, ctx: self._switch_target(t)),
                self.G_rest('withdraw_change', self.params['withdraw_change_s'], rest_pose, shape='point'),
                Gesture('sleep', 'change_pause', duration=self.params.get('change_pause_s', 0.0)),
                self.G_point_object(lambda: self.selected_object, self.params['point_dwell_s'][2]),
                self.G_protocol('change_indicated', object=self.target)]

    def first_dwell(self) -> float:
        """Hold of the first pointing gesture (short before an early change of mind)."""
        early = self.change and self.change['timing'] == 'early'
        return self.params['early_dwell_s'] if early else self.params['point_dwell_s'][0]

    def late_change_due(self, ctx: dict) -> bool:
        """Late change: the robot has left its rest pose and is approaching the old target (within late_trigger_m),
        and has not grasped anything yet."""
        ch = self.change
        if not ch or ch['timing'] != 'late' or self.changed or not self.started:
            return False
        if any(ctx['robot_grasping'].values()) or ctx['robot_lifted_object'] is not None:
            return False
        if not any(ev['kind'] == 'target_object' and ev['emitted'] and ev['payload'].get('object') == ch['old']
                   for ev in self.evidence):
            return False                        # the old target must have been indicated first
        d = ctx['tcp_xy_distance'](ctx['object_positions'][ch['old']])
        approaching = self._old_d is not None and d < self._old_d - 1e-4
        self._old_d = d
        return approaching and ctx['robot_moving'] and not ctx['robot_at_rest'] and d <= float(ch['late_trigger_m'])


class PickPlaceInstructor(GestureHuman):
    """T1: point at the target cube, then at the place zone P, back to rest. T5: see GestureHuman.G_change."""

    def __init__(self, scene, behavior, variation):
        super().__init__(scene, behavior, variation)
        self.place_indicated = False

    def _mark_place(self, t, ctx):
        self.place_indicated = True

    def place_gestures(self) -> list[Gesture]:
        p = self.params
        return [self.G_point_zone(self.selected_target, p['point_dwell_s'][1]), self.G_call(self._mark_place),
                self.G_rest(), self.G_protocol('instruction_given'), self.G_call(lambda t, ctx: self.goto('observing', t))]

    def instruction(self) -> list[Gesture]:
        g = [self.G_point_object(lambda: self.selected_object, self.first_dwell()),
             self.G_call(self.after_object_indicated)]
        if self.change and self.change['timing'] == 'early':
            g += self.G_change()
        return g + self.place_gestures()

    def after_object_indicated(self, t, ctx) -> None:
        """Hook (T4 arms the intrusion here)."""

    def cue(self, t, ctx) -> None:
        self.started = True
        self.emit('human_cue_onset', modality='pointing', selected_object=self.selected_object, selected_zone=self.selected_target)
        self.queue = self.instruction()

    def behave(self, t, ctx):
        if not self.started and t >= self.params['cue_onset_s']:
            self.cue(t, ctx)
        if self.late_change_due(ctx):
            rest = self.place_gestures() if not self.place_indicated else \
                [self.G_rest(), self.G_protocol('instruction_given'), self.G_call(lambda t, ctx: self.goto('observing', t))]
            self.replace_queue(self.G_change() + rest)
        self.run_queue(t, ctx)


class HandoverReceiver(GestureHuman):
    """T2: point at the target object, then hold the open hand out palm up in the hand zone H (early / on time /
    late); once the robot has lowered the object into the palm and holds still, raise the palm under it and close
    the fingers around it, pull it (the robot must release on the pull), withdraw with it."""

    def __init__(self, scene, behavior, variation):
        super().__init__(scene, behavior, variation)
        self.timing = self.spec['timing']
        self.lift_t = None
        self.still_since = None

    def palm_at_zone(self, ctx) -> np.ndarray:
        return np.asarray(ctx['zone_positions'][self.selected_target], float)

    def expected_object_pos(self, ctx) -> np.ndarray:
        """Where the human expects the object: resting in the upturned palm (config present_offset + half height)."""
        h = ctx['object_half_height'][self.selected_object]
        return self.palm_at_zone(ctx) + np.asarray(self.behavior['present_offset'], float) + [0.0, 0.0, h]

    def indication(self) -> list[Gesture]:
        g = [self.G_point_object(lambda: self.selected_object, self.first_dwell())]
        if self.change and self.change['timing'] == 'early':
            g += self.G_change()
        return g + [self.G_protocol('target_indicated', object_by='pointing')] + self.receive_gestures()

    def _lifted(self, t, ctx) -> bool:
        if self.lift_t is None and ctx['robot_lifted_object'] == self.selected_object:
            self.lift_t = t
        return self.lift_t is not None

    def receive_gestures(self) -> list[Gesture]:
        p, b = self.params, self.behavior
        reach = Gesture('move', 'reach_out', target=self.palm_at_zone, duration=p['reach_out_s'], shape='offer', palm_up=1.0)
        ready = self.G_call(lambda t, ctx: (self.emit('interaction_window_start', kind='handover', actor='human'),
                                            self.emit('protocol_step_complete', step='hand_ready')))
        if self.timing == 'early':
            pre = []
        elif self.timing == 'on_time':
            pre = [self.G_rest(), Gesture('wait', 'wait_robot_grasp', until=lambda t, ctx: ctx['robot_grasping'].get(self.selected_object, False))]
        else:                                    # late: the robot holds the object and waits late_wait_s
            arrive = lambda t: self.lift_t + p['late_wait_s'] - p['reach_out_s'] / self.speed
            pre = [self.G_rest(), Gesture('wait', 'wait_robot_lift', until=self._lifted),
                   Gesture('wait', 'wait_late', until=lambda t, ctx: t >= arrive(t) - 1e-9)]
        take = Gesture('move', 'take_object', target=self._under_object, duration=b['take_s'], clearance=b['hand_clearance_m'],
                       shape='grasp', palm_up=1.0)
        pull = Gesture('move', 'pulling', duration=b['pull_s'], shape='grasp', palm_up=1.0,
                       target=lambda ctx: self.pos + np.asarray(b['pull_offset'], float))
        return pre + [reach, ready, Gesture('wait', 'waiting_offer', until=self._offered), take,
                      self.G_call(self._hold_object), pull,
                      Gesture('wait', 'wait_release', until=lambda t, ctx: not ctx['robot_grasping'].get(self.selected_object, False)),
                      self.G_call(lambda t, ctx: (self.emit('protocol_step_complete', step='object_transferred'),
                                                  self.emit('interaction_window_end', kind='handover', actor='human'))),
                      Gesture('sleep', 'released_by_robot', duration=0.3),
                      self.G_rest('withdraw_with_object', 1.2, shape='grasp', palm_up=1.0), self.G_call(lambda t, ctx: self.goto('done', t))]

    def _under_object(self, ctx) -> np.ndarray:
        """The palm rises straight up under the object held out in it: its bottom take_gap_m above the palm surface."""
        obj, b = self.selected_object, self.behavior
        bottom = ctx['object_positions'][obj][2] - ctx['object_half_height'][obj]
        return np.r_[self.pos[:2], bottom - b['take_gap_m'] - b['palm_surface_m']]

    def _offered(self, t, ctx) -> bool:
        obj = self.selected_object
        near = ctx['robot_lifted_object'] == obj and \
            np.linalg.norm(np.asarray(ctx['object_positions'][obj]) - self.expected_object_pos(ctx)) < self.behavior['accept_radius_m']
        self.still_since = (self.still_since if self.still_since is not None else t) if near and ctx['robot_speed'] < 0.02 else None
        return self.still_since is not None and t - self.still_since >= self.behavior['accept_still_s'] - 1e-9

    def _hold_object(self, t, ctx) -> None:
        self.requests.append(('attach', self.selected_object))
        self.holding = self.selected_object
        self.emit('object_grasp', object=self.selected_object, actor='human')

    def behave(self, t, ctx):
        if not self.started and t >= self.params['cue_onset_s']:
            self.started = True
            self.emit('human_cue_onset', modality='pointing', selected_object=self.selected_object)
            self.queue = self.indication()
        if self.late_change_due(ctx):
            self.replace_queue(self.G_change() + [self.G_protocol('target_indicated', object_by='pointing')] + self.receive_gestures())
        self._lifted(t, ctx)
        self.run_queue(t, ctx)


class AssistRequester(GestureHuman):
    """T3 (learned workflow, no pointing): reach for one of the two cubes in U (the intention cue: the reach reveals
    the cube and so its paired cup), take it, hold it above U until the robot has stood that cup in U_cup and moved
    clear, drop the cube into the cup and withdraw. `selected_object` is the cup; `block` the cube paired with it.
    T5: switch cubes before the robot grasps a cup. Early: hover above the first cube, pull back, reach for the
    other. Late (the robot is close to the first cube's cup): put the first cube back on its spot, take the other."""

    EVIDENCE_LABELS = {**GestureHuman.EVIDENCE_LABELS, 'reach_block': 'target_object'}

    def __init__(self, scene, behavior, variation):
        super().__init__(scene, behavior, variation)
        self.block_of = {cup: block for block, cup in variation['pairs'].items()}
        self.hold = np.asarray(behavior['hold_pose'], float)
        self.take_pose: dict[str, np.ndarray] = {}

    @property
    def block(self) -> str:
        """The cube the human (currently) wants to put into its cup."""
        return self.block_of[self.selected_object]

    def evidence_payload(self, label):
        if label == 'reach_block':
            return dict(object=self.selected_object, block=self.block)
        return super().evidence_payload(label)

    def G_reach(self, dwell) -> Gesture:
        return Gesture('move', 'reach_block', duration=self.params['u_reach_s'], dwell=dwell,
                       target=lambda ctx: np.asarray(ctx['object_positions'][self.block], float) + [0.0, 0.0, 0.03])

    def take(self) -> list[Gesture]:
        return [self.G_call(self._take_block), self.G_rest('lift_block', self.params['hold_s'], self.hold)]

    def switch_while_reaching(self) -> list[Gesture]:
        """Pull the hand back (nothing held yet) and reach for the other cube."""
        p = self.params
        return [self.G_call(lambda t, ctx: self._switch_target(t)),
                self.G_rest('withdraw_change', p['withdraw_change_s'], self.hold),
                Gesture('sleep', 'change_pause', duration=p.get('change_pause_s', 0.0)),
                self.G_reach(p['point_dwell_s'][0])] + self.take() + [self.G_protocol('change_indicated', object=self.target)]

    def switch_holding(self) -> list[Gesture]:
        """Put the held cube back where it was taken, then reach for the other cube."""
        p, old = self.params, self.block
        return [self.G_call(lambda t, ctx: self._switch_target(t)),
                Gesture('move', 'withdraw_change', target=lambda ctx: self.take_pose[old], duration=p['put_back_s'], shape='grasp'),
                self.G_call(lambda t, ctx: self._release_block(old)),
                Gesture('sleep', 'change_pause', duration=p.get('change_pause_s', 0.0)),
                self.G_reach(p['point_dwell_s'][0])] + self.take() + [self.G_protocol('change_indicated', object=self.target)]

    def request(self) -> list[Gesture]:
        g = [self.G_reach(self.first_dwell())]
        if self.change and self.change['timing'] == 'early':
            return g + self.switch_while_reaching() + self.after_pick()
        return g + self.take() + self.after_pick()

    def _robot_clear(self, t, ctx) -> bool:
        """The cup stands in U_cup, released, and the gripper has moved away from it."""
        cup = ctx['object_positions'][self.selected_object]
        return bool(ctx['cup_ready'] and not any(ctx['robot_grasping'].values())
                    and (ctx['robot_at_rest'] or ctx['tcp_xy_distance'](cup) >= self.behavior['robot_clear_m']))

    def after_pick(self) -> list[Gesture]:
        p, b = self.params, self.behavior
        over = lambda ctx: np.r_[np.asarray(ctx['object_positions'][self.selected_object])[:2], ctx['table_top_z'] + b['drop_height_m']]
        return [self.G_call(lambda t, ctx: self.emit('interaction_window_start', kind='assist', actor='human')),
                Gesture('wait', 'waiting_cup', until=self._robot_clear),
                Gesture('move', 'move_over_cup', target=over, duration=p['drop_move_s']),
                self.G_call(self._drop_block), Gesture('sleep', 'dropped', duration=0.6),
                self.G_call(lambda t, ctx: self.emit('interaction_window_end', kind='assist', actor='human')),
                self.G_rest('withdraw'), self.G_call(lambda t, ctx: self.goto('done', t))]

    def _take_block(self, t, ctx):
        self.take_pose[self.block] = self.pos.copy()
        self.requests.append(('attach', self.block))
        self.holding = self.block
        self.emit('object_grasp', object=self.block, actor='human')
        self.emit('protocol_step_complete', step='block_picked', object=self.block)

    def _release_block(self, key):
        self.requests.append(('detach', key))
        self.holding = None
        self.emit('object_release', object=key, actor='human')

    def _drop_block(self, t, ctx):
        self._release_block(self.block)

    def behave(self, t, ctx):
        if not self.started and t >= self.params['cue_onset_s']:
            self.started = True
            self.emit('human_cue_onset', modality='human_pick', block=self.block)
            self.queue = self.request()
        if self.late_change_due(ctx):
            switch = self.switch_holding() if self.holding else self.switch_while_reaching()
            self.replace_queue(switch + self.after_pick())
        self.run_queue(t, ctx)


class Interrupter(GestureHuman):
    """T4: no instruction (the robot starts the pick-and-place of the single cube by itself; the human stands at
    rest). When the robot is in the episode's phase (approaching the cube / carrying it / about to place it) the hand
    goes into the robot zone, is held there hold_s and withdrawn. The hand moves in slowly (approach_s) and stops
    short of the robot if it comes too close. T4-neg: the hand goes near but outside the robot zone."""

    def __init__(self, scene, behavior, variation):
        super().__init__(scene, behavior, variation)
        self.phase, self.hold_s = self.spec['phase'], float(self.spec['hold_s'])
        self.negative = bool(self.spec.get('negative'))
        self.intruded = False
        self.lift_t = None

    def triggered(self, t, ctx) -> bool:
        b, obj = self.behavior, self.target
        if self.phase == 'approach':
            return (not any(ctx['robot_grasping'].values()) and ctx['robot_moving']
                    and ctx['tcp_xy_distance'](ctx['object_positions'][obj]) <= b['approach_trigger_m'])
        if ctx['robot_lifted_object'] != obj:
            return False
        if self.phase == 'carry':
            self.lift_t = t if self.lift_t is None else self.lift_t
            return t - self.lift_t >= self.params['trigger_delay_s'] - 1e-9
        return ctx['tcp_xy_distance'](ctx['zone_positions'][self.selected_target]) <= b['place_trigger_m']

    def intrusion_path(self, ctx) -> list[np.ndarray]:
        """[optional waypoint, point]: the candidate path (config `intrusion_paths`) that keeps farthest from the
        robot (its links, and the way to where it is heading); negative targets go straight."""
        b, top = self.behavior, ctx['table_top_z']
        xyz = lambda q: np.array([q[0], q[1], top + q[2]], float)
        if self.negative:
            return [xyz(b['negative_points'][self.spec['neg_target']])]
        tcp = np.asarray(ctx['tcp'], float)
        dest = np.asarray(ctx['object_positions'][self.target] if self.phase == 'approach'
                          else ctx['zone_positions'][self.selected_target], float)
        dest = np.r_[dest[:2], tcp[2]]           # where the robot is heading (the cube / the place zone)
        chain = ctx['robot_chain']               # base, shoulder, elbow, wrists, gripper link origins
        robot = list(zip(chain, chain[1:])) + [(tcp, chain[-1]), (tcp, dest), (dest, dest + [0.0, 0.0, 0.3])]

        def clearance(path):
            pts = [self.pos, *path]
            samples = [a + (b_ - a) * u for a, b_ in zip(pts, pts[1:]) for u in np.linspace(0.0, 1.0, 15)]
            return min(segment_distance(q, a, b_) for q in samples for a, b_ in robot)
        paths = [[xyz(w) for w in c.get('via', [])] + [xyz(c['point'])] for c in b['intrusion_paths']]
        return max(paths, key=clearance)

    def intrusion(self, t, ctx) -> list[Gesture]:
        p, b = self.params, self.behavior
        path = self.intrusion_path(ctx)
        point = path[-1]
        self.extras.update(intrusion_point=point.round(4).tolist(), trigger_time=round(t, 4), phase=self.phase,
                           negative=self.negative, intrusion_via=[w.round(4).tolist() for w in path[:-1]])
        label = 'reach_near' if self.negative else 'intrude'
        share = p['approach_s'] / len(path)
        g = [Gesture('move', 'intrude_approach', target=lambda ctx, w=w: w, duration=share, clearance=b['hand_clearance_m'])
             for w in path[:-1]]
        g += [Gesture('move', label, target=lambda ctx: point, duration=share, clearance=b['hand_clearance_m']),
              Gesture('sleep', 'dwelling', duration=self.hold_s)]
        g += [Gesture('move', 'withdraw', target=lambda ctx, w=w: w, duration=p['withdraw_s'] / len(path)) for w in path[-2::-1]]
        return g + [self.G_rest('withdraw', p['withdraw_s'] / len(path)),
                    self.G_call(lambda t, ctx: self.extras.update(withdrawn_time=round(t, 4)))]

    def behave(self, t, ctx):
        self.started = True
        if not self.intruded and self.triggered(t, ctx):
            self.intruded = True
            self.replace_queue(self.intrusion(t, ctx) + [self.G_call(lambda t, ctx: self.goto('done', t))])
        self.run_queue(t, ctx)


BEHAVIORS = {
    't1_instructor': PickPlaceInstructor,
    't2_receiver': HandoverReceiver,
    't3_requester': AssistRequester,
    't4_intruder': Interrupter,
}


def make_human(scene: dict, behavior: dict, variation: dict) -> ScriptedHuman:
    return BEHAVIORS[behavior['type']](scene, behavior, variation)
