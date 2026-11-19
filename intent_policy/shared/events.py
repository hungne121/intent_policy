"""Human-facing event detector (09/QĐ-KT P0 items 1-5, 18/11/2026).

The scripted human (intent_policy/scenarios/*.py) reacts ONLY to the events below. Each event is a predicate on the
CURRENT world state of the episode:
  * robot state: TCP position (site), TCP linear velocity (MuJoCo site velocity of the current physics state),
    gripper opening (finger joint), the object the gripper holds (`env.attached`);
  * object state: body positions, dust-mark visibility (C2), the person's own palm pose (the person knows it);
  * scene constants of the task (spot / socket / handover coordinates).
No expert phase, expert flag, plan, controller set-point or future trajectory is read, so the SAME detector drives the
person for the expert (collection) and for every learned policy (intent_policy/envs/hrc_env.py).

Parameters: configs/hrc/events.yaml, each with its source (a measurement of the expert at the moment the old flag was
raised, outputs/diag/event_detector_1118, or an unchanged scenario constant). Ground-truth annotations used only for
scoring (success / grade) are NOT here: they stay in the experts' `success()` / scenario `check()` and the evaluator
(hrc_env `_expert_success`, named *_evaluator_only), never read by the human.

Every wait of the person that ends on a detector event logs `human_cue` {cue, t, ...} (ep.events), so the moment the
person reacts can be compared between expert and policy runs.
"""
from __future__ import annotations

import mujoco
import numpy as np
import yaml

from intent_policy.shared.sim_env import ROOT, TRAVEL_Z

CFG = yaml.safe_load((ROOT / 'configs/hrc/events.yaml').read_text())
STATION = yaml.safe_load((ROOT / 'configs/hrc/station.yaml').read_text())

# 11/QĐ-KT integration item 1: locked event schema. Bump DETECTOR_VERSION on ANY change of a definition / parameter.
DETECTOR_VERSION = 'hrc-events-1.2'      # 1.2 (19/11, 15/QĐ-KT A): C4 part_on_palm also once the robot let go of the
#                                          part and it rests on the supported palm (changelog: configs/hrc/events.yaml)
#                                          1.1 (18/11 22:5x, 12/QĐ-KT): geometric tolerances, I2/I3 cues, verified distances
if CFG.get('version') != DETECTOR_VERSION:          # 15/QĐ-KT A8: code and parameters are one locked version
    raise RuntimeError(f"configs/hrc/events.yaml version {CFG.get('version')} != {DETECTOR_VERSION}")
METRIC_SCHEMA_VERSION = 'hrc-metric-record-1.1'    # docs/reports/METRIC_SPEC_2026-11-18.md §5 (agent I)
SCHEMA = {
    'version': METRIC_SCHEMA_VERSION, 'detector_version': DETECTOR_VERSION,
    'time_unit': 's (episode time, tick * 0.05; t_start = 0)',
    'per_episode': ['detector_version', 'schema_version', 't_start', 't_success', 'success', 'rounds', 'anticipation',
                    'steps_required', 'step_predecessors', 'steps_ordered', 'steps_done', 'disrupted', 't_disruption',
                    't_recovery', 'recovered', 'human_robot_contact', 'human_robot_contact_classes',
                    'safety_distance_violation', 'min_robot_person_m', 'safety_audit'],
    'rounds': ['round_id', 'step', 't_R_ready', 't_H_ready', 't_cue_onset', 't_trigger', 't_response', 'response_missing'],
    'anticipation': ['round_id', 't_useful_onset', 't_trigger', 'AM'],
    'safety_audit': ['min_distance_m', 'closest_pair', 'interaction_phase', 'exemption_reason', 'contact_force_n', 'distance_check',
                     'penetration_m', 'violations', 'min_nonexempt_m', 'closest_nonexempt_pair', 'min_exempt_m',
                     'closest_exempt_pair', 'interaction_phase_ticks', 'static_threshold_violation_ticks', 'authorized_handover_contacts',
                     'ticks_columns', 'ticks'],
    'violation': ['kind', 'cause', 't', 't_end', 'pair', 'phase', 'distance_m', 'distance_lower_bound_m', 'threshold_m',
                  'robot_speed_m_s', 'hand_speed_m_s', 'force_n', 'penetration_m', 'verified', 'exempt', 'exemption_reason'],
    'n/a': 'None in a list / field = event not observed or not defined (10/QĐ-KT rule 4: report n/a, never 0)',
}


def detector(ep):
    """The episode's detector (one per episode, created on first use; the scenario creates it before the first
    tick). Creating it also starts the scoring-only MetricTracker (10/QĐ-KT), updated after every physics tick."""
    d = getattr(ep, '_event_detector', None)
    if d is None:
        d = ep._event_detector = EventDetector(ep)
        d.attach()
        MetricTracker.attach(ep, d)
    return d


class EventDetector:
    def __init__(self, ep, cfg=None):
        self.ep = ep
        self.cfg = (cfg or CFG)
        self.p = self.cfg.get(ep.task, {})
        self._vel = np.zeros(6)
        self._rest = {}                  # C4 (1.2): part -> consecutive physics ticks resting on the palm after release

    def attach(self):
        """Per-tick state of the human-facing predicates (1.2): after every physics tick (wraps ep.step, like
        MetricTracker) count how long each C4 part has rested on the supported palm with the robot's gripper empty.
        Reading a predicate never changes this state (read-only calls, e.g. traces, are safe)."""
        step = getattr(self.ep, 'step', None)
        if step is None or self.ep.task != 'C4' or getattr(self.ep, 'm', None) is None:   # no physics: unit-test fakes
            return

        def stepped(command):
            step(command)
            self.tick_update()
        self.ep.step = stepped

    def tick_update(self):
        from intent_policy.scenarios.c4 import SOCKETS
        for i in SOCKETS:
            part = f'part{i}'
            self._rest[part] = self._rest.get(part, 0) + 1 if self._c4_resting(part) else 0

    # ------------------------------------------------------------------ current robot / object state
    def tcp(self):
        return self.ep.tcp()

    def tcp_velocity(self):
        """Linear TCP velocity [m/s] of the current physics state (site velocity, world frame)."""
        ep = self.ep
        mujoco.mj_objectVelocity(ep.m, ep.d, mujoco.mjtObj.mjOBJ_SITE, ep.env.tcp, self._vel, 0)
        return self._vel[3:].copy()

    def tcp_speed(self):
        return float(np.linalg.norm(self.tcp_velocity()))

    def body_speed(self, name):
        """Linear speed [m/s] of a body's centre of mass, current physics state (mj_objectVelocity, world frame)."""
        ep = self.ep
        mujoco.mj_objectVelocity(ep.m, ep.d, mujoco.mjtObj.mjOBJ_BODY, ep.m.body(name).id, self._vel, 0)
        return float(np.linalg.norm(self._vel[3:]))

    def gripper_opening(self):
        return float(self.ep.d.qpos[self.ep.env.gid])

    def holding(self):
        return self.ep.attached_name()

    def footprint(self, name):
        """Half width of an object's collision geometry in the table plane (cylinder radius / smallest box half size),
        read from the scene model itself."""
        m = self.ep.m
        b = m.body(name).id
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == b and (m.geom_contype[g] or m.geom_conaffinity[g]):
                t = int(m.geom_type[g])
                return float(m.geom_size[g][0]) if t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                                         mujoco.mjtGeom.mjGEOM_SPHERE) else float(min(m.geom_size[g][:2]))
        return 0.

    def until(self, cue, pred, detail=None, **key):
        """`hold(..., until=...)` adapter: logs `human_cue` the first time `pred()` is true for this wait.
        `detail` (optional, 1.2): callable returning extra fields of that state for the log (e.g. C4 path)."""
        done = [False]

        def f():
            ok = bool(pred())
            if ok and not done[0]:
                done[0] = True
                self.ep.event('human_cue', cue=cue, **key, **(detail() if detail is not None else {}))
            return ok
        return f

    # ------------------------------------------------------------------ I1
    def i1_robot_prepared(self):
        """The robot hovers over a cup still on the plate with an empty, open gripper, aligned so that a vertical
        descent takes the cup, and slow enough to stop aligned (12/QĐ-KT A1-A3; old flag: i1_expert `robot_prepared`).
          xy:    |TCP - cup| <= (opening - cup diameter) / 2   (the cup fits between the open fingers)
          z:     grasp height (cup centre + GRASP_DZ) <= TCP z < rest height (i1_expert home z)
          speed: v <= sqrt(2 a xy_tol)   (stopping distance at the configured deceleration within the alignment)"""
        from intent_policy.scenarios.i1 import CUPS, PLATE_XY, PLATE_HALF
        ep, p = self.ep, self.p['robot_prepared']
        opening = self.gripper_opening()
        if self.holding() is not None:
            return False
        tcp = self.tcp()
        for c in CUPS:
            tol = (opening - 2 * self.footprint(c)) / 2
            if tol <= 0 or self.tcp_speed() > np.sqrt(2 * p['decel_m_s2'] * tol):
                continue
            pc = ep.pos(c)
            on_plate = np.all(np.abs(pc[:2] - PLATE_XY) <= PLATE_HALF)
            if (on_plate and np.linalg.norm(tcp[:2] - pc[:2]) <= tol
                    and pc[2] + p['grasp_dz_m'] <= tcp[2] < p['rest_z_m']):
                return True
        return False

    # ------------------------------------------------------------------ I2 / I3 (unchanged scenario predicates)
    def i2_cake_moves_to(self, plate_xy, prev_tcp):
        """The held cake is carried towards the plate: TCP xy distance to it shrinks since the previous tick."""
        p = self.p['cake_moves_to_plate']
        tcp = self.tcp()
        return (prev_tcp is not None and self.holding() == 'cake'
                and np.linalg.norm(prev_tcp[:2] - plate_xy[:2]) - np.linalg.norm(tcp[:2] - plate_xy[:2]) > p['approach_step_m']
                and tcp[2] > p['tcp_z_min_m'])

    def i3_robot_at_neutral(self):
        from intent_policy.scenarios.i3 import MIDPOINT
        return float(np.linalg.norm(self.tcp() - MIDPOINT)) < self.p['robot_at_neutral']['tcp_dist_m']

    def i3_robot_heads_to(self, cake_xy, d0):
        """The empty gripper is visibly heading for the pointed cake (closer than at the start of the point)."""
        p = self.p['robot_heads_to_cake']
        dist = float(np.linalg.norm(self.tcp()[:2] - cake_xy[:2]))
        return dist < max(p['near_m'], d0 - p['approach_m']) and self.holding() is None

    def tcp_xy_dist(self, xy):
        return float(np.linalg.norm(self.tcp()[:2] - np.asarray(xy)[:2]))

    # ------------------------------------------------------------------ C1
    def c1_robot_at_neutral(self):
        from intent_policy.scenarios.c1 import NEUTRAL
        return float(np.linalg.norm(self.tcp() - NEUTRAL)) < self.p['robot_at_neutral']['tcp_dist_m']

    def c1_cup_lifted_clear(self, cup):
        return self.ep.pos(cup)[2] > self.p['cup_lifted_clear']['cup_z_min_m']

    def c1_cup_returned(self, cup, target):
        p = self.p['cup_returned']
        c = self.ep.pos(cup)
        return (np.linalg.norm(c[:2] - target[:2]) < p['cup_xy_m'] and c[2] < p['cup_z_max_m'] and self.holding() is None
                and np.linalg.norm(self.tcp() - c) > p['tcp_cup_min_m'])

    # ------------------------------------------------------------------ C2
    def c2_robot_holds_brush(self):
        return self.holding() == 'cloth'

    def c2_robot_released(self):
        return self.holding() is None

    def c2_area_clean(self, obj):
        """No dust mark left in the grid under `obj` (a mark belongs to the object of its nearest dust spot)."""
        from intent_policy.scenarios.c2 import SPOTS
        from intent_policy.shared.episode import dirt_geoms
        m = self.ep.m
        spots = [(o, np.asarray(s)) for o, ss in SPOTS.items() for s in ss]
        for g in dirt_geoms(m):
            if m.geom_rgba[g, 3] <= 0:
                continue
            xy = m.geom_pos[g, :2]
            owner = min(spots, key=lambda os: np.linalg.norm(xy - os[1]))[0]
            if owner == obj:
                return False
        return True

    def c2_robot_left_shelf(self):
        p = self.p['robot_left_shelf']
        tcp = self.tcp()
        return tcp[1] > p['tcp_y_min_m'] and tcp[2] > p['tcp_z_min_m']

    def c2_brush_presented(self):
        """The robot holds the brush with the handover point inside the bristle block and keeps it there while the
        person reaches for it (12/QĐ-KT A1-A3; old: expert phase 'present_cloth' + set-point speed < 0.005).
          position: |brush - HANDOVER| <= smallest bristle half-extent
          speed:    v <= that half-extent / the person's reach-back duration (the brush stays in the zone meanwhile)
        The scenario requires it for `still_ticks` consecutive checks."""
        from intent_policy.scenarios.c2 import HANDOVER
        p = self.p['brush_presented']
        return (self.holding() == 'cloth'
                and np.linalg.norm(self.ep.pos('cloth') - HANDOVER) <= p['zone_m']
                and self.tcp_speed() <= p['zone_m'] / p['reach_s'])

    # ------------------------------------------------------------------ C3
    def c3_robot_holds_tray(self):
        return self.holding() == 'tray'

    # ------------------------------------------------------------------ C4
    def c4_part_on_palm(self, i, palm=None):
        """Part i is on the person's open palm where the hand can close on it, in one of two physical states
        (15/QĐ-KT A2, hrc-events-1.2; old: c4_expert `delivered` + expert-side grasp):
        (i)  receiver grasps before the robot releases (1.1 logic, unchanged; 12/QĐ-KT A1-A4): the robot holds part i
             over the palm and does not move it out of the hand's capture while the hand closes
               grasp:   hand-part distance (palm / finger capsules) <= HUMAN_GRASP_GAP_M (the person can close on it)
               support: part centre over the palm support box footprint (scene palm_support half size)
               secure:  TCP speed <= HUMAN_GRASP_GAP_M / the person's closing time (scenarios/c4.py grip move)
        (ii) the robot has let go and the part rests on the supported palm: gripper empty, the same grasp and support
             conditions on the part, the part's OWN speed <= the same bound, and all of it true for `rest_ticks`
             consecutive physics ticks (tick_update). No single distance fires it (15/QĐ-KT A4)."""
        p = self.p['part_on_palm']
        ep = self.ep
        part = f'part{i}'
        if ep.pose is None:
            return False
        held = self.holding()
        if held is None:                                                       # (ii)
            return self._rest.get(part, 0) >= p['rest_ticks']
        if held != part:
            return False
        if ep.hand_object_distance(part) > p['grasp_gap_m']:                 # (i)
            return False
        if self.tcp_speed() > p['grasp_gap_m'] / p['close_s']:
            return False
        return self._over_support(ep.pos(part), np.asarray(ep.pose['palm'], float), p)

    def c4_path(self, i):
        """Which state of c4_part_on_palm holds now (for the cue log): 'held' (i) or 'released' (ii)."""
        return {'path': 'held' if self.holding() == f'part{i}' else 'released'}

    def _c4_resting(self, part):
        """Instantaneous condition of path (ii): gripper empty, the person does not hold the part yet, the part is
        over the palm support, within the grasp gap of the hand and (almost) at rest."""
        ep, p = self.ep, self.p['part_on_palm']
        if ep.pose is None or self.holding() is not None or part in ep.held_by_human:
            return False
        if not self._over_support(ep.pos(part), np.asarray(ep.pose['palm'], float), p):
            return False
        if ep.hand_object_distance(part) > p['grasp_gap_m']:
            return False
        return self.body_speed(part) <= p['grasp_gap_m'] / p['close_s']

    def _over_support(self, xyz, palm, p):
        try:
            m, d = self.ep.m, self.ep.d
            g = m.geom('human_palm_support').id
            q = d.geom_xmat[g].reshape(3, 3).T @ (np.asarray(xyz) - d.geom_xpos[g])
            return bool(np.all(np.abs(q[:2]) <= m.geom_size[g][:2]))
        except (KeyError, AttributeError):     # no support geom in this model: its half size around the palm
            return bool(np.all(np.abs(np.asarray(xyz)[:2] - palm[:2]) <= min(p['support_half_xy_m'])))

    def c4_robot_clear(self, recv):
        return self.holding() is None and self.tcp()[2] > recv[2] + self.p['robot_clear']['tcp_above_palm_m']

    # ------------------------------------------------------------------ W1 / W2 / W3
    def w_robot_in_tray(self, tray_xy):
        """The robot works in the tray / at the destination the person wants (the person yields while true)."""
        p = self.p['robot_in_tray']
        tcp = self.tcp()
        d = float(np.linalg.norm(tcp[:2] - np.asarray(tray_xy, float)[:2]))
        if p.get('inclusive', False):
            return d <= p['tcp_xy_m'] and tcp[2] <= p['tcp_z_max_m']
        return d < p['tcp_xy_m'] and tcp[2] < p['tcp_z_max_m']

    def w_robot_claims(self, name, candidates=None):
        """The robot visibly takes `name`: it holds it, or its empty, open gripper is over the object's footprint below
        the rest height, slow enough to stop there (12/QĐ-KT A1-A3; old: avoider `ep.claimed`, set at the decision).
          xy:    |TCP - object| <= object half size (TCP over the footprint)
          z:     TCP z < rest height (w2 / w3 expert home z)
          speed: v <= sqrt(2 a half_size)"""
        p = self.p['robot_claims']
        held = self.holding()
        if held is not None:
            return held == name
        tcp, half = self.tcp(), self.footprint(name)
        return (self.gripper_opening() >= 2 * half
                and np.linalg.norm(tcp[:2] - self.ep.pos(name)[:2]) <= half
                and tcp[2] < p['rest_z_m']
                and self.tcp_speed() <= np.sqrt(2 * p['decel_m_s2'] * half))



# ====================================================================================================== SCORING ONLY
# 10/QĐ-KT rule 7, 11/QĐ-KT, 12/QĐ-KT: metric events for the benchmark (agent I). Separate from the human-driving
# predicates above: the person never reads these. They may use the scenario's evaluator-only plan (`info`) to know WHICH
# step is due, never to drive anyone. Per-tick predicates read the current state only; parameters: events.yaml `metric`.

class MetricTracker:
    def __init__(self, ep, det):
        self.ep, self.det = ep, det
        self.p = CFG['metric']
        self.info = None                       # evaluator-only plan (task_active event, or set by hrc_env)
        self.steps_done = {}                   # step name -> first t its completion predicate held
        self.rounds = {}                       # round index -> first t of each event
        self.prev_tcp = None
        self.frame = None                      # the person's last applied frame (visible hand shape / pointing)
        self.error = None
        self.safety = SafetyAudit(ep, det)

    @classmethod
    def attach(cls, ep, det):
        tr = ep._metric_tracker = cls(ep, det)
        step = getattr(ep, 'step', None)
        if step is None:                       # unit-test fakes without stepping
            return tr

        def stepped(command):                  # after every tick: the state just produced by physics
            step(command)
            try:
                tr.update()
            except Exception as exc:           # scoring must never break a rollout; the record says so
                tr.error = repr(exc)
        ep.step = stepped
        apply = getattr(ep, 'apply_human', None)
        if apply is not None:
            def applied(frame):
                tr.frame = frame
                return apply(frame)
            ep.apply_human = applied
        return tr

    # ------------------------------------------------------------------ plan / steps
    def plan(self):
        if self.info is None:
            ev = next((e for e in self.ep.events if e.get('event') == 'task_active'), {})
            self.info = {k[:-len('_evaluator_only')]: v for k, v in ev.items() if k.endswith('_evaluator_only')}
        return self.info

    def steps(self):
        """Required task steps in their required order: [(name, predicate)], ordered (False: independent, W tasks)."""
        ep, t, info = self.ep, self.ep.task, self.plan()
        pos, held = ep.pos, ep.attached_name
        if t == 'I1':
            from intent_policy.scenarios.i1 import CUPS, COASTER_T
            def on(co):
                c = pos(co)
                return any(np.linalg.norm(pos(k)[:2] - c[:2]) < .02 and abs(pos(k)[2] - (c[2] + COASTER_T / 2 + .037 + .008)) < .01
                           and held() != k for k in CUPS)                  # i1_expert.cups_on constants
            return [(f'cup_on_coaster{2 - r}', (lambda co: lambda: on(co))(f'coaster{2 - r}')) for r in range(3)], True
        if t == 'I2':
            from intent_policy.scenarios.i2 import PLATES
            pl = PLATES[info['plate']]
            return [('cake_on_plate', lambda: bool(np.all(np.abs(pos('cake')[:2] - pl[:2]) < [.035, .03])) and held() is None
                     and .63 < pos('cake')[2] < .71)], True                # i2_expert.success constants
        if t == 'I3':
            from intent_policy.scenarios.i3 import PLATE_XY, RECEIVE_TOL
            tg = info.get('B') or info['A']
            return [('cake_in_receive_tray', lambda: bool(np.all(np.abs(pos(tg)[:2] - PLATE_XY) < RECEIVE_TOL)) and .63 < pos(tg)[2] < .66
                     and held() is None)], True                            # scenarios/i3.check constants
        if t == 'C1':
            mk = lambda sd: (lambda: np.linalg.norm(pos('cloth' + sd)[:2] - ep.initial['cup' + sd][:2]) < .035
                             and np.linalg.norm(pos('cup' + sd)[:2] - ep.initial['cup' + sd][:2]) < .03 and pos('cup' + sd)[2] < .71
                             and held() is None)                           # c1_expert.success (+ c1 cup_returned z)
            return [(f'cloth_under_cup{sd}', mk(sd)) for sd in info['order']], True
        if t == 'C2':
            back = lambda o: np.linalg.norm(pos(o)[:2] - ep.initial[o][:2]) < .02 and abs(pos(o)[2] - ep.initial[o][2]) < .01
            st = [('brush_taken', lambda: held() == 'cloth')]
            st += [(f'clean_{o}', (lambda o: lambda: self.det.c2_area_clean(o))(o)) for o in info['order']]
            st += [('objects_back', lambda: back('cup') and back('clock')),
                   ('brush_on_stand', lambda: np.linalg.norm(pos('cloth')[:2] - ep.initial['cloth'][:2]) < .03
                    and abs(pos('cloth')[2] - ep.initial['cloth'][2]) < .02 and held() is None)]   # c2_expert.success
            return st, True
        if t == 'C3':
            from intent_policy.scenarios.c3 import TRAY_XY
            from intent_policy.shared.episode import marks_left
            return [('marks_clean', lambda: marks_left(ep) == 0),
                    ('tray_back', lambda: np.linalg.norm(pos('tray')[:2] - TRAY_XY) < .03 and abs(pos('tray')[2] - ep.initial['tray'][2]) < .01
                     and held() is None)], True                            # c3_expert.success constants
        if t == 'C4':
            from intent_policy.scenarios.c4 import SOCKETS
            mk = lambda i: (lambda: np.linalg.norm(pos(f'part{i}')[:2] - SOCKETS[i]) < .03 and pos(f'part{i}')[2] < .67
                            and held() != f'part{i}')                      # c4_expert.success constants
            return [(f'part{i}_installed', mk(i)) for i in info['order']], True
        if t == 'W1':
            from intent_policy.scenarios.w1 import in_tray
            st = [(f'cake{i}_in_tray', (lambda i: lambda: any(in_tray(ep, f'cake{i}', s) for s in 'LR'))(i)) for i in range(4)]
            st += [('cans_split', lambda: sorted(s for s in 'LR' if sum(in_tray(ep, f'can{k}', s) for k in range(2)) == 1) == ['L', 'R'])]
            return st, False                                               # w1_expert.success
        if t == 'W2':
            from intent_policy.scenarios.w2 import sorted_block
            return [(f'block{i}_sorted', (lambda i: lambda: sorted_block(ep, i))(i)) for i in range(6)], False
        if t == 'W3':
            from intent_policy.scenarios.w3 import packed
            return [(f'item{i}_packed', (lambda i: lambda: packed(ep, i))(i)) for i in range(6)], False
        return [], False

    # ------------------------------------------------------------------ per-round readiness / cue / response
    def _approach(self, target):
        """Useful robot motion towards `target`: TCP distance to it shrank by more than the step since the previous
        tick (the I2 scenario's 'carried towards' rule, events.yaml I2 approach_step_m; 3-D from 1.1)."""
        if self.prev_tcp is None or target is None:
            return False
        tcp, tg = self.ep.tcp(), np.asarray(target, float)
        return float(np.linalg.norm(self.prev_tcp - tg) - np.linalg.norm(tcp - tg)) > CFG['I2']['cake_moves_to_plate']['approach_step_m']

    def _pointing(self):
        """The person's visible pointing gesture: (target point or None, gesture completion 0-1) of the last frame."""
        f = self.frame or {}
        return (None if f.get('point_at') is None else np.asarray(f['point_at'], float)), float(f.get('progress') or 0.)

    def round_events(self, r):
        """(R_ready, H_ready, cue_onset, trigger, response_target) predicates of round r, or None if not defined."""
        ep, t, info, m = self.ep, self.ep.task, self.plan(), self.p
        held, pos = ep.attached_name, ep.pos
        still = lambda: ep.hand_speed <= m['hand_still_max_m_s']
        never = lambda: False
        if t == 'I1':
            co = f'coaster{2 - r}'
            placed = lambda: np.linalg.norm(pos(co) - ep.initial[co]) > .01 and co not in ep.held_by_human   # i1_expert: off the stack
            return (lambda: (held() or '').startswith('cup'), placed, never, placed,
                    lambda: pos(co) if (held() or '').startswith('cup') else None)
        if t in ('I2', 'I3'):
            # 12/QĐ-KT C17: cue = the pointing gesture; onset = the gesture starts, completion = fully formed (the
            # scenarios' own strength > 0.95) with the hand still; response target = what is pointed at
            onset = lambda: self._pointing()[0] is not None and self._pointing()[1] > m['point_onset_min']
            done = lambda: self._pointing()[0] is not None and self._pointing()[1] >= m['point_complete_min'] and still()
            if t == 'I2':
                ready = lambda: held() is None and self.det.tcp_xy_dist(pos('cake')) <= self.det.footprint('cake')
                tgt = lambda: pos('cake') if held() is None else self._pointing()[0]
            else:
                ready = self.det.i3_robot_at_neutral
                tgt = lambda: self._pointing()[0] if held() is None else None
            return ready, done, onset, done, tgt
        if t == 'C1':
            sd = info['order'][r]
            lifted = lambda: ('cloth' + sd) in ep.held_by_human and still()
            return self.det.c1_robot_at_neutral, lifted, never, lifted, lambda: pos('cup' + sd) if held() is None else None
        if t == 'C2' and 1 <= r <= 2:
            o = info['order'][r - 1]
            lifted = lambda: pos(o)[2] > ep.initial[o][2] + .07                 # c2_expert lifted()
            from intent_policy.scenarios.c2 import SPOTS, SHELF_FLOOR
            spot = np.r_[SPOTS[o][0], SHELF_FLOOR]
            return never, lifted, never, lifted, lambda: spot if held() == 'cloth' else None
        if t == 'C4':
            from intent_policy.scenarios.c4 import SOCKETS
            i = info['order'][r]
            offer = lambda: (ep.pose is not None and still()
                             and np.linalg.norm(np.asarray(ep.pose['palm'])[:2] - [SOCKETS[i][0], SOCKETS[i][1] - .02]) <= m['c4_palm_receive_xy_m'])
            tgt = lambda: (np.asarray(ep.pose['palm']) if held() == f'part{i}' else pos(f'part{i}')) if held() in (None, f'part{i}') else None
            return (lambda: held() == f'part{i}'), offer, never, offer, tgt
        return None

    def update(self):
        ep = self.ep
        t = round(ep.t, 3)
        steps, _ = self.steps()
        for name, pred in steps:
            if name not in self.steps_done and pred():
                self.steps_done[name] = t
        r = next((k for k, (n, _) in enumerate(steps) if n not in self.steps_done), None)   # current round
        if r is not None:
            ev = self.round_events(r)
            if ev is not None:
                R, H, O, T, target = ev
                rec = self.rounds.setdefault(r, dict.fromkeys(('R_ready', 'H_ready', 'cue_onset', 'trigger', 'response', 'onset')))
                tg = target()
                for k, f in (('R_ready', R), ('H_ready', H), ('cue_onset', O), ('trigger', T)):
                    if rec[k] is None and f():
                        rec[k] = t
                moving = self._approach(tg)
                if rec['onset'] is None and moving:
                    rec['onset'] = t                  # anticipation: useful motion in the round, before or after the cue
                if rec['trigger'] is not None and rec['response'] is None and moving:
                    rec['response'] = t
        self.prev_tcp = ep.tcp()
        self.safety.update(t)

    # ------------------------------------------------------------------ record
    def record(self, success, t_success=None, summary=None):
        ep = self.ep
        steps, ordered = self.steps()
        names = [s for s, _ in steps]
        n = max(self.rounds) + 1 if self.rounds else 0
        g = lambda r, k: self.rounds.get(r, {}).get(k)
        rounds = [{'round_id': r, 'step': names[r] if r < len(names) else None, 't_R_ready': g(r, 'R_ready'),
                   't_H_ready': g(r, 'H_ready'), 't_cue_onset': g(r, 'cue_onset'), 't_trigger': g(r, 'trigger'),
                   't_response': g(r, 'response'), 'response_missing': g(r, 'trigger') is not None and g(r, 'response') is None}
                  for r in range(n)]
        # anticipation (11/QĐ-KT C4, separate from Rsp): useful-action onset = first useful motion towards the round's
        # target in the round (may precede the cue); AM = t_trigger - t_useful_onset (> 0: the robot moved before the cue)
        antic = [{'round_id': r, 't_useful_onset': g(r, 'onset'), 't_trigger': g(r, 'trigger'),
                  'AM': None if g(r, 'onset') is None or g(r, 'trigger') is None else round(g(r, 'trigger') - g(r, 'onset'), 3)}
                 for r in range(n)]
        preds = {nm: (names[:k] if ordered else []) for k, nm in enumerate(names)}   # dependency graph (11/QĐ-KT C3)
        fault = (summary or {}).get('fault') or {}
        t_dis = fault.get('fault_start') if fault.get('fault_applied') else None
        ch = next((e['t'] for e in ep.events if e.get('event') == 'human_intention_change'), None)
        if ch is not None:
            t_dis = ch if t_dis is None else min(t_dis, ch)
        disrupted = t_dis is not None
        classes = (summary or {}).get('contact_classes') or {}
        contact = any(v.get('category') == 'invalid_contact' for v in classes.values())
        near = [r['robot_person_m'] for r in ep.trace if r.get('robot_person_m') is not None]
        done = sorted(self.steps_done.items(), key=lambda kv: kv[1])
        saf = self.safety.summary(classes)
        return {'detector_version': DETECTOR_VERSION, 'schema_version': METRIC_SCHEMA_VERSION,
                't_start': 0.0, 't_success': t_success if success else None, 'success': bool(success),
                'rounds': rounds, 'anticipation': antic,
                't_R_ready': [x['t_R_ready'] for x in rounds], 't_H_ready': [x['t_H_ready'] for x in rounds],
                't_trigger': [x['t_trigger'] for x in rounds], 't_response': [x['t_response'] for x in rounds],
                'steps_required': names, 'step_predecessors': preds, 'steps_ordered': ordered,
                'steps_done': [[k, v] for k, v in done],
                'disrupted': disrupted, 't_disruption': t_dis,
                't_recovery': next((v for _, v in done if disrupted and v >= t_dis), None),
                'recovered': bool(disrupted and success and not contact),
                'human_robot_contact': contact,
                'human_robot_contact_classes': sorted(k for k, v in classes.items() if v.get('category') == 'invalid_contact'),
                'min_robot_person_m': None if not near else round(float(min(near)), 4),
                'safety_distance_violation': any(v['kind'] == 'clearance' and v['verified'] and not v['exempt'] for v in saf['violations']),
                'safety_audit': saf, 'metric_events_version': CFG['metric']['version'], 'tracker_error': self.error}


def metric_record(ep, success, t_success=None, summary=None):
    """Per-episode scoring record (keys fixed for agent I's metric code, 10/11/12/QĐ-KT). `summary`: Episode.finish()
    output (contact classes, fault). Scoring only."""
    tr = getattr(ep, '_metric_tracker', None)
    if tr is None:
        return None
    return tr.record(success, t_success, summary)


def _point_box(points, centre, xmat, half):
    """Distance from each point to a box (0 inside)."""
    q = (np.asarray(points) - centre) @ xmat.reshape(3, 3)
    return np.linalg.norm(np.maximum(np.abs(q) - half, 0.), axis=1)


class SafetyAudit:
    """Per-tick robot-person clearance over ALL robot collision geoms (+ the object the robot alone holds) vs ALL
    collidable human segments (+ objects the person alone holds); mj_geomDistance with distmax 10 m (no 0.25 m cut-off,
    11/QĐ-KT C5). Inside a declared close-interaction phase the declared pairs (gripper fingers / exchanged object vs
    the person's hand / that object) are exempt from the clearance rule, never from contact classification.

    Distance check (12/QĐ-KT B13-B14): MuJoCo 3.3.7 returns 0 for some mesh-box pairs that are apart. When it returns
    <= 0 and mj_collision has no contact for the pair, the value is invalid: the audit keeps a LOWER bound (bounding-
    sphere gap or contact margin; conservative variant only) and, for mesh-box pairs, an independent estimate from the
    mesh vertices (exact point-to-box distance of every vertex: an UPPER bound of the true distance, equal to it for a
    vertex-face closest feature). A violation is `verified` when the distance used (MuJoCo where valid, else the vertex
    estimate) is below the threshold; a violation only the lower bound shows is `verified: false` (never evidence).

    Non-contact clearance threshold: protective separation distance of speed and separation monitoring (ISO/TS
    15066:2016 §5.5.4; Marvel and Norcross 2017) with the station's safety configuration (station.yaml
    predictive_avoidance.ssm_reaction_s, controller.work_accel_m_s2):
        S_p(t) = v_h (T_r + T_s) + v_r T_r + v_r^2 / (2 a),  T_s = v_r / a,  C = Z_d = Z_r = 0
    v_h = observed hand speed now, v_r = TCP speed now. 0.08 m (56/QĐ-DL trial) and 0.04 m are sensitivity values only."""

    def __init__(self, ep, det):
        self.ep, self.det = ep, det
        self.p = CFG['metric']['safety']
        self.ci = self.p['close_interaction'].get(ep.task)
        self.tr = float(STATION['predictive_avoidance']['ssm_reaction_s'])
        self.a = float(STATION['controller']['work_accel_m_s2'])
        self.ticks = []
        self.best = {'all': (np.inf, None), 'non': (np.inf, None), 'ex': (np.inf, None)}
        self.viol = []
        self.static = {str(x): 0 for x in self.p['static_thresholds_m']}
        self.phase_ticks, self.exemptions = 0, {}
        self._ft = np.zeros(6)
        self._geoms = None
        self.phase_now, self.phase_at = None, {}
        self.check = {'invalid_mujoco_pairs': 0, 'verified_by_vertices': 0, 'unverifiable': 0,
                      'lower_bound_below_threshold_but_verified_clear': 0, 'lower_bound_below_threshold': 0}

    def _sets(self):
        if self._geoms is None:
            v, m = self.ep.validator, self.ep.m
            hand = [g for g in v.human_geoms if any(w in m.geom(g).name for w in self.p['hand_geom_words'])]
            fingers = [g for g in v.robot_collision if any(w in m.body(m.geom_bodyid[g]).name for w in self.p['robot_finger_bodies'])]
            self._geoms = (list(v.robot_collision), list(v.human_geoms), set(hand), set(fingers))
        return self._geoms

    def _phase(self):
        """Declared close-interaction phase: the exchange object is within the robot's final approach of its grasp point
        AND at the person's hand. Returns the object or None."""
        if self.ci is None:
            return None
        ep = self.ep
        for o in self.ci['objects']:
            if o not in ep.positions:
                continue
            po = ep.pos(o)
            gg = self.ci.get('grasp_geom')
            base = ep.d.geom_xpos[ep.m.geom(gg).id].copy() if gg else po
            grasp = base + [0., 0., self.ci['robot_grip_dz_m']]
            robot_at = ep.attached_name() == o or np.linalg.norm(ep.tcp() - grasp) <= self.ci['robot_approach_m']
            palm = None if ep.pose is None else np.asarray(ep.pose['palm'], float)
            human_at = o in ep.held_by_human or (palm is not None and np.linalg.norm(palm - po) <= self.ci['hand_object_m'])
            if robot_at and human_at:
                return o
        return None

    def _vertex_distance(self, a, b):
        """Independent mesh-box distance from the mesh vertices (upper bound), or None for other geometry types."""
        m, d = self.ep.m, self.ep.d
        MESH, BOX = mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_BOX
        ta, tb = int(m.geom_type[a]), int(m.geom_type[b])
        if {ta, tb} != {int(MESH), int(BOX)}:
            return None
        mesh, box = (a, b) if ta == int(MESH) else (b, a)
        k = int(m.geom_dataid[mesh])
        v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
        world = d.geom_xpos[mesh] + v @ d.geom_xmat[mesh].reshape(3, 3).T
        return float(_point_box(world, d.geom_xpos[box], d.geom_xmat[box], m.geom_size[box]).min())

    def update(self, t):
        ep = self.ep
        if getattr(ep, 'validator', None) is None:       # unit-test fakes without physics
            return
        m, d = ep.m, ep.d
        robot, human, hand, fingers = self._sets()
        mine, theirs = ep.attached_name(), set(ep.held_by_human)
        shared = {mine} & theirs
        hg = ep.validator.held_geoms
        r_extra = [] if mine is None or mine in shared else hg(mine)
        h_extra = [g for n in sorted(theirs - shared) for g in hg(n)]
        ex_obj = self._phase()
        ex_r = set(fingers) | (set(hg(ex_obj)) if ex_obj else set())
        ex_h = set(hand) | (set(hg(ex_obj)) if ex_obj else set())
        touching = set()
        for i in range(d.ncon):
            c = d.contact[i]
            touching.add((int(c.geom1), int(c.geom2))); touching.add((int(c.geom2), int(c.geom1)))
        vr = self.det.tcp_speed(); vh = float(ep.hand_speed)
        sp = vh * (self.tr + vr / self.a) + vr * self.tr + vr * vr / (2 * self.a)
        dmax = float(self.p['distmax_m'])
        best = {'non': [np.inf, np.inf, None, True], 'ex': [np.inf, np.inf, None, True], 'all': [np.inf, np.inf, None, True]}
        for a in robot + r_extra:
            for b in human + h_extra:
                if a == b:
                    continue
                dist = float(mujoco.mj_geomDistance(m, d, a, b, dmax, self._ft)); lb, ok = dist, True
                if dist <= 0. and (a, b) not in touching:
                    self.check['invalid_mujoco_pairs'] += 1
                    lb = max(float(np.linalg.norm(d.geom_xpos[a] - d.geom_xpos[b]) - m.geom_rbound[a] - m.geom_rbound[b]),
                             float(max(m.geom_margin[a], m.geom_margin[b])), 0.)
                    ub = self._vertex_distance(a, b)
                    if ub is None:
                        dist, ok = lb, False; self.check['unverifiable'] += 1
                    else:
                        dist = ub; self.check['verified_by_vertices'] += 1
                        if lb < sp:
                            self.check['lower_bound_below_threshold'] += 1
                            if ub >= sp:
                                self.check['lower_bound_below_threshold_but_verified_clear'] += 1
                pair = (m.geom(a).name or m.body(m.geom_bodyid[a]).name, m.geom(b).name or m.body(m.geom_bodyid[b]).name)
                keys = ['all'] + (['ex'] if ex_obj is not None and a in ex_r and b in ex_h else ['non'])
                for k in keys:
                    if dist < best[k][0]:
                        best[k][0], best[k][2], best[k][3] = dist, pair, ok
                    best[k][1] = min(best[k][1], lb)
        if ex_obj is not None:
            self.phase_ticks += 1
            self.exemptions.setdefault(ex_obj, {'reason': self.ci['reason'], 'first_t': t, 'last_t': t})['last_t'] = t
        self.phase_now = None if ex_obj is None else f'close_interaction:{ex_obj}'
        self.phase_at[t] = self.phase_now
        for k in ('all', 'non', 'ex'):
            if best[k][0] < self.best[k][0]:
                self.best[k] = (best[k][0], best[k][2])
        for k, ex in (('non', False), ('ex', True)):
            dist, lb, pair, ok = best[k]
            if lb < sp:                                    # conservative check; `verified` says whether it holds
                self.viol.append(dict(t=t, d=dist, lb=lb, sp=sp, pair=pair, verified=bool(ok and dist < sp), exempt=ex,
                                      vr=vr, vh=vh))
        for x in self.p['static_thresholds_m']:
            if best['non'][0] < float(x):
                self.static[str(x)] += 1
        r4 = lambda x: None if not np.isfinite(x) else round(float(x), 4)
        self.ticks.append([t, r4(best['non'][0]), r4(best['non'][1]), round(sp, 4), int(ex_obj is not None), r4(best['all'][0]),
                           round(vr, 3), round(vh, 3)])

    def _grouped(self):
        """Clearance violations merged over consecutive ticks with the same pair / exemption / verification."""
        out = []
        for v in self.viol:
            last = out[-1] if out else None
            key = (v['exempt'], list(v['pair'] or []), v['verified'])
            if last is not None and (last['exempt'], last['pair'], last['verified']) == key and round(v['t'] - last['t_end'], 3) <= .051:
                last['t_end'] = v['t']
                if v['d'] - v['sp'] < last['_m']:
                    last.update(distance_m=round(v['d'], 4), distance_lower_bound_m=round(v['lb'], 4), threshold_m=round(v['sp'], 4),
                                robot_speed_m_s=round(v['vr'], 3), hand_speed_m_s=round(v['vh'], 3), _m=v['d'] - v['sp'])
                continue
            out.append({'kind': 'clearance', 'cause': 'proximity_without_contact', 't': v['t'], 't_end': v['t'],
                        'pair': key[1], 'phase': self.phase_at.get(v['t']) or 'non_contact',
                        'distance_m': round(v['d'], 4), 'distance_lower_bound_m': round(v['lb'], 4), 'threshold_m': round(v['sp'], 4),
                        'robot_speed_m_s': round(v['vr'], 3), 'hand_speed_m_s': round(v['vh'], 3), 'force_n': None,
                        'penetration_m': None, 'verified': v['verified'], 'exempt': v['exempt'],
                        'exemption_reason': self.ci['reason'] if v['exempt'] else None, '_m': v['d'] - v['sp']})
        for v in out:
            v.pop('_m')
        return out

    def summary(self, classes):
        classes = classes or {}
        bad = {k: v for k, v in classes.items() if v.get('category') == 'invalid_contact'}
        cause = {'robot_touches_person': 'robot_caused', 'person_touches_robot_held_object': 'via_held_object'}
        contacts = [{'kind': 'contact', 'cause': cause.get(v.get('rule'), v.get('rule')), 't': v.get('first_t_s'),
                     't_end': v.get('t_max_s'), 'pair': (v.get('pair') or '').split('|'), 'phase': v.get('robot_phase'),
                     'distance_m': -float(v.get('max_depth_m') or 0.), 'distance_lower_bound_m': None, 'threshold_m': 0.,
                     'robot_speed_m_s': None, 'hand_speed_m_s': None, 'force_n': v.get('max_force_n'),
                     'penetration_m': v.get('max_depth_m'), 'verified': True, 'exempt': False, 'exemption_reason': None,
                     'class': k} for k, v in sorted(bad.items())]             # contacts are never exempt (11/QĐ-KT C2)
        handover = sorted(k for k, v in classes.items() if v.get('rule') == 'handover')
        f = lambda k: None if not np.isfinite(self.best[k][0]) else round(self.best[k][0], 4)
        pr = lambda k: None if self.best[k][1] is None else list(self.best[k][1])
        cl = self._grouped()
        return {'min_distance_m': f('all'), 'closest_pair': pr('all'),
                'interaction_phase': sorted({p for p in self.phase_at.values() if p}),
                'exemption_reason': sorted({e['reason'] for e in self.exemptions.values()}),
                'contact_force_n': max([v.get('max_force_n') or 0. for v in bad.values()], default=None),
                'penetration_m': max([v.get('max_depth_m') or 0. for v in bad.values()], default=None),
                'violations': contacts + cl, 'authorized_handover_contacts': handover,
                'min_nonexempt_m': f('non'), 'closest_nonexempt_pair': pr('non'),
                'min_exempt_m': f('ex'), 'closest_exempt_pair': pr('ex'),
                'interaction_phase_ticks': self.phase_ticks, 'exemptions': self.exemptions,
                'clearance_violation_ticks': sum(1 for v in self.viol if v['verified'] and not v['exempt']),
                'unverified_clearance_ticks': sum(1 for v in self.viol if not v['verified'] and not v['exempt']),
                'exempt_clearance_ticks': sum(1 for v in self.viol if v['exempt']),
                'unsafe_contacts': sorted(bad), 'distance_check': dict(self.check),
                'static_threshold_violation_ticks': dict(self.static),
                'clearance_rule': 'd_nonexempt < S_p = v_h (T_r + v_r/a) + v_r T_r + v_r^2/(2a)',
                'ticks_columns': ['t', 'd_nonexempt_m', 'd_nonexempt_lower_bound_m', 'S_p_m', 'close_interaction', 'd_all_m',
                                  'robot_speed_m_s', 'hand_speed_m_s'],
                'ticks': self.ticks}
