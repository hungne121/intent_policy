"""Unit tests of the human-facing event detector (09/QĐ-KT P0 item 5): every predicate on a fake current state."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from intent_policy.shared.events import CFG, EventDetector, detector
from intent_policy.shared.sim_env import TRAVEL_Z


TEXT_KEYS = {'version', 'reason', 'objects', 'hand_geom_words', 'robot_finger_bodies', 'grasp_geom'}


def _walk(x, key=None):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _walk(v, k)
    elif isinstance(x, list) and key not in TEXT_KEYS:
        for v in x:
            yield from _walk(v, key)
    else:
        yield key, x


def test_config_complete():
    """No placeholder left; every parameter is a number / flag (names and reasons are text)."""
    for k, v in _walk(CFG):
        assert k in TEXT_KEYS or isinstance(v, (int, float, bool)), (k, v)


class Fake:
    """Minimal episode: TCP, held object, body positions; velocity / opening are patched on the detector."""

    def __init__(self, task, tcp, objects, held=None, m=None):
        self.task, self._tcp, self.objects, self.held = task, np.asarray(tcp, float), objects, held
        self.m = m
        self.events = []
        self.t = 0.

    def tcp(self):
        return self._tcp.copy()

    def attached_name(self):
        return self.held

    def pos(self, name):
        return np.asarray(self.objects[name], float)

    def event(self, name, **kw):
        self.events.append({'event': name, **kw})

    def hand_object_distance(self, name):
        return getattr(self, 'hand_gap', 0.)


FOOT = {'cup': .034, 'block': .02, 'item': .02, 'cake': .03}


def det_for(ep, speed=0., opening=.085, vel=None):
    d = EventDetector(ep)
    d.footprint = lambda name: next(v for k, v in FOOT.items() if name.startswith(k))
    v = np.zeros(3) if vel is None else np.asarray(vel, float)
    if vel is None:
        v[0] = speed
    d.tcp_velocity = lambda: v.copy()
    d.gripper_opening = lambda: opening
    return d


# ---------------------------------------------------------------------------------------------------------- I1
def i1_ep(tcp, held=None, cup0=(.24, .08, .665)):
    from intent_policy.scenarios.i1 import CUPS, CUP_Z
    objs = {k: (xy[0], xy[1], CUP_Z) for k, xy in CUPS.items()}
    objs['cup0'] = cup0
    return Fake('I1', tcp, objs, held)


def test_i1_prepared_over_plate_cup():
    """Valid offset placement (12/QĐ-KT A4): aligned within the finger gap (8.5 mm with the gripper open), anywhere
    between grasp height and rest height, slow enough to stop aligned."""
    gap = (.085 - 2 * .034) / 2
    assert det_for(i1_ep([.24 + .8 * gap, .08, TRAVEL_Z])).i1_robot_prepared()          # 6.8 mm off: still fits
    assert det_for(i1_ep([.24, .08 - .8 * gap, .70])).i1_robot_prepared()                # low, just above grasp height
    assert det_for(i1_ep([.24, .08, TRAVEL_Z]), speed=.9 * np.sqrt(2 * gap)).i1_robot_prepared()


@pytest.mark.parametrize('case', ['xy', 'below_grasp', 'resting_high', 'speed', 'closed', 'held', 'cup_served', 'nowhere'])
def test_i1_not_prepared(case):
    """Wrong placement / not yet stable: the person must not start."""
    gap = (.085 - 2 * .034) / 2
    tcp, kw, held, cup0 = [.24, .08, TRAVEL_Z], {}, None, (.24, .08, .665)
    if case == 'xy': tcp[0] += 1.2 * gap                          # a finger would land on the cup
    if case == 'below_grasp': tcp[2] = .665 + .02                 # lower than the grasp height: into the cup
    if case == 'resting_high': tcp[2] = .825                      # at / above the rest height
    if case == 'speed': kw['speed'] = 1.5 * np.sqrt(2 * gap)      # passing over: cannot stop aligned
    if case == 'closed': kw['opening'] = .06                      # fingers narrower than the cup
    if case == 'held': held = 'cup1'
    if case == 'cup_served':              # cup0 already on a coaster spot (off the plate), robot retreated above it
        cup0 = (.155, -.08, .70); tcp = [.155, -.08, TRAVEL_Z]
    if case == 'nowhere': tcp = [.28, .05, TRAVEL_Z]
    assert not det_for(i1_ep(tcp, held, cup0), **kw).i1_robot_prepared()


def test_until_logs_cue_once():
    ep = i1_ep([.24, .08, TRAVEL_Z])
    d = det_for(ep)
    f = d.until('robot_prepared', d.i1_robot_prepared, round=0)
    assert f() and f()
    assert [e for e in ep.events if e['event'] == 'human_cue'] == [{'event': 'human_cue', 'cue': 'robot_prepared', 'round': 0}]


def test_detector_cached_per_episode():
    ep = i1_ep([0, 0, 0])
    assert detector(ep) is detector(ep)


# ---------------------------------------------------------------------------------------------------------- I2 / I3 / C1
def test_i2_cake_moves_to_plate():
    plate = np.array([.30, -.11, .64])
    ep = Fake('I2', [.30, .0, .80], {}, held='cake')
    d = det_for(ep)
    assert d.i2_cake_moves_to(plate, np.array([.30, .01, .80]))
    assert not d.i2_cake_moves_to(plate, np.array([.30, -.0005, .80]))      # step below 1 mm
    assert not d.i2_cake_moves_to(plate, None)
    ep.held = None
    assert not d.i2_cake_moves_to(plate, np.array([.30, .01, .80]))


def test_i3_neutral_and_heading():
    from intent_policy.scenarios.i3 import MIDPOINT
    ep = Fake('I3', MIDPOINT + [.02, 0, 0], {})
    d = det_for(ep)
    assert d.i3_robot_at_neutral()
    ep._tcp = MIDPOINT + [.03, 0, 0]
    assert not d.i3_robot_at_neutral()
    cake = np.array([.10, -.10, .65])
    ep._tcp = np.array([.20, -.10, .80])                 # 10 cm from the cake, started at 15 cm
    assert d.i3_robot_heads_to(cake, .15)
    assert not d.i3_robot_heads_to(cake, .12)            # only 2 cm closer than at the start of the point
    ep.held = 'cake0'
    assert not d.i3_robot_heads_to(cake, .15)


def test_c1_predicates():
    from intent_policy.scenarios.c1 import NEUTRAL
    target = np.array([.12, -.10, .69])
    ep = Fake('C1', NEUTRAL + [0, .02, 0], {'cupL': [.12, -.10, .69]})
    d = det_for(ep)
    assert d.c1_robot_at_neutral()
    assert not d.c1_cup_lifted_clear('cupL')
    assert d.c1_cup_returned('cupL', target)
    ep._tcp = np.array([.12, -.10, .80])                  # gripper still 11 cm over the cup
    assert not d.c1_cup_returned('cupL', target)
    ep.objects['cupL'] = [.12, -.10, .78]
    assert d.c1_cup_lifted_clear('cupL')


# ---------------------------------------------------------------------------------------------------------- C2
def dust_model():
    from intent_policy.scenarios.c2 import SPOTS
    g = ''.join(f'<geom name="dirt_preview_{i}" type="box" size=".004 .004 .0001" pos="{x} {y} .62" contype="0" conaffinity="0"/>'
                for i, (x, y) in enumerate(s for ss in SPOTS.values() for s in ss))
    return mujoco.MjModel.from_xml_string(f'<mujoco><worldbody>{g}</worldbody></mujoco>')


def test_c2_area_clean():
    m = dust_model()
    d = det_for(Fake('C2', [.28, .1, .8], {}, m=m))
    assert not d.c2_area_clean('cup') and not d.c2_area_clean('clock')
    m.geom_rgba[:3, 3] = 0                                # the three cup spots wiped, clock spots dusty
    assert d.c2_area_clean('cup') and not d.c2_area_clean('clock')
    m.geom_rgba[1, 3] = 1
    assert not d.c2_area_clean('cup')


def test_c2_left_shelf_and_brush_presented():
    from intent_policy.scenarios.c2 import HANDOVER
    p = CFG['C2']['brush_presented']
    ep = Fake('C2', [.28, .0, .80], {'cloth': HANDOVER + [.8 * p['zone_m'], 0, 0]}, held='cloth')   # offset but in zone
    d = det_for(ep)
    assert d.c2_robot_left_shelf() and d.c2_robot_holds_brush() and not d.c2_robot_released()
    assert d.c2_brush_presented()
    ep._tcp = np.array([.28, -.05, .80])
    assert not d.c2_robot_left_shelf()
    assert not det_for(ep, speed=1.5 * p['zone_m'] / p['reach_s']).c2_brush_presented()     # still moving
    ep.objects['cloth'] = HANDOVER + [0, 0, 1.5 * p['zone_m']]
    assert not d.c2_brush_presented()                                                      # wrong place
    ep.objects['cloth'] = HANDOVER; ep.held = None
    assert not d.c2_brush_presented() and d.c2_robot_released()


def test_c3_robot_holds_tray():
    ep = Fake('C3', [0, 0, 0], {}, held='tray')
    assert det_for(ep).c3_robot_holds_tray()
    ep.held = None
    assert not det_for(ep).c3_robot_holds_tray()


# ---------------------------------------------------------------------------------------------------------- C4
def test_c4_part_on_palm():
    """Valid offset (within the hand's grasp gap), wrong part, not-yet-secure (moving), out of reach, off the palm."""
    p = CFG['C4']['part_on_palm']
    palm = np.array([.28, -.08, .71])
    ep = Fake('C4', [.28, -.08, .80], {'part1': palm + [.01, 0, .06]}, held='part1')
    ep.pose = {'palm': palm}
    ep.hand_gap = .8 * p['grasp_gap_m']
    d = det_for(ep)
    assert d.c4_part_on_palm(1)
    assert not d.c4_part_on_palm(0)                                                      # wrong part
    assert not det_for(ep, speed=1.5 * p['grasp_gap_m'] / p['close_s']).c4_part_on_palm(1)   # not yet secure
    ep.hand_gap = 1.5 * p['grasp_gap_m']
    assert not d.c4_part_on_palm(1)                                                      # hand cannot close on it
    ep.hand_gap = 0.; ep.objects['part1'] = palm + [.05, 0, .06]
    assert not d.c4_part_on_palm(1)                                                      # beyond the palm support
    ep.objects['part1'] = palm + [0, 0, .06]; ep.held = None
    assert not d.c4_part_on_palm(1)                                                      # the robot no longer holds it


def test_c4_robot_clear():
    recv = np.array([.28, -.08, .71])
    ep = Fake('C4', recv + [0, 0, .13], {})
    assert det_for(ep).c4_robot_clear(recv)
    ep._tcp = recv + [0, 0, .11]
    assert not det_for(ep).c4_robot_clear(recv)


# ---------------------------------------------------------------------------------------------------------- W
def test_w_robot_in_tray_boundaries():
    ep = Fake('W1', [.10, .0, .75], {})
    assert det_for(ep).w_robot_in_tray([.0, .0])                            # W1: xy 0.10 < 0.11, z < 0.76
    ep2 = Fake('W2', [.10, .0, .78], {})
    assert det_for(ep2).w_robot_in_tray([.0, .0])                           # W2: inclusive (old: yield until > 0.10 or > 0.78)
    ep2._tcp = np.array([.1001, .0, .78])
    assert not det_for(ep2).w_robot_in_tray([.0, .0])


def test_w_robot_claims():
    objs = {'block0': [.10, .0, .64], 'block1': [.40, .0, .64]}
    ep = Fake('W2', [.25, .0, TRAVEL_Z], objs, held='block0')
    cands = list(objs)
    assert det_for(ep).w_robot_claims('block0', cands) and not det_for(ep).w_robot_claims('block1', cands)
    ep.held = None
    assert not det_for(ep).w_robot_claims('block0', cands)                   # empty gripper at rest between both
    assert not det_for(ep, vel=[-.3, 0, 0]).w_robot_claims('block0', cands)  # moving towards it is not a claim
    ep._tcp = np.array([.40 + .015, 0, TRAVEL_Z])                           # over block1's footprint (half 0.02)
    assert det_for(ep).w_robot_claims('block1', cands) and not det_for(ep).w_robot_claims('block0', cands)
    assert not det_for(ep, speed=.3).w_robot_claims('block1', cands)         # sweeping over it
    ep._tcp = np.array([.40 + .025, 0, TRAVEL_Z])
    assert not det_for(ep).w_robot_claims('block1', cands)                   # beside the footprint
    ep._tcp = np.array([.40, 0, .83])
    assert not det_for(ep).w_robot_claims('block1', cands)                   # resting high


# ---------------------------------------------------------------------------------------------------------- scoring-only metric events
class StepEp(Fake):
    """Fake episode that advances time on step() (the tracker wraps ep.step)."""
    def __init__(self, task, info, objects, initial, tcp=(.28, .05, .82)):
        super().__init__(task, tcp, objects)
        self.events = [{'t': 0., 'event': 'task_active', **{k + '_evaluator_only': v for k, v in info.items()}}]
        self.initial = {k: np.asarray(v, float) for k, v in initial.items()}
        self.held_by_human, self.hand_speed, self.pose, self.trace, self.tick = set(), 0., {'palm': np.zeros(3)}, [], 0

    @property
    def t(self):
        return self.tick * .05

    @t.setter
    def t(self, v):
        pass

    def step(self, cmd=None):
        self.tick += 1
        self.trace.append({'robot_person_m': .3})


METRIC_KEYS = {'t_start', 't_success', 'success', 't_R_ready', 't_H_ready', 't_trigger', 't_response', 'steps_required',
               'steps_done', 'disrupted', 'recovered', 'human_robot_contact', 'min_robot_person_m'}


def test_metric_record_i1_round():
    from intent_policy.scenarios.i1 import CUPS, CUP_Z, COASTER_T
    from intent_policy.shared.events import metric_record
    objs = {k: [xy[0], xy[1], CUP_Z] for k, xy in CUPS.items()}
    objs.update({f'coaster{i}': [.52, -.28, .623 + i * COASTER_T] for i in range(3)})
    ep = StepEp('I1', {'spot_order': [0, 1, 2]}, objs, dict(objs))
    d = detector(ep)
    ep.step(None)                                          # nothing happened
    ep.held = 'cup0'; ep.step(None)                        # t 0.10 robot holds a cup: R ready
    ep.objects['coaster2'] = [.155, -.08, .623]; ep.step(None)      # t 0.15 coaster down at a spot, not held: H ready + trigger
    ep._tcp = np.array([.25, .0, .80]); ep.step(None)      # t 0.20 TCP moves towards the coaster: response
    ep.objects['cup0'] = [.155, -.08, .623 + COASTER_T / 2 + .037 + .008]; ep.held = None; ep.step(None)   # t 0.25 step done
    rec = metric_record(ep, True, 0.25, {'contact_classes': {}, 'fault': None})
    assert METRIC_KEYS <= set(rec)
    assert rec['t_R_ready'][0] == .1 and rec['t_H_ready'][0] == .15 and rec['t_trigger'][0] == .15 and rec['t_response'][0] == .2
    assert rec['steps_required'] == ['cup_on_coaster2', 'cup_on_coaster1', 'cup_on_coaster0']
    assert rec['steps_done'] == [['cup_on_coaster2', .25]] and not rec['disrupted'] and not rec['human_robot_contact']
    assert rec['min_robot_person_m'] == .3 and rec['tracker_error'] is None


def test_metric_record_contact_and_disruption():
    from intent_policy.scenarios.c4 import SOCKETS
    from intent_policy.shared.events import metric_record
    objs = {f'part{i}': [SOCKETS[i][0], .12, .65] for i in range(3)}
    ep = StepEp('C4', {'order': [1, 0, 2]}, objs, dict(objs))
    detector(ep)
    ep.events.append({'t': .05, 'event': 'human_intention_change'})
    ep.pose['palm'] = np.array([SOCKETS[1][0], SOCKETS[1][1] - .02, .71]); ep.step(None)   # palm still at socket 1: H ready
    ep.objects['part1'] = [SOCKETS[1][0], SOCKETS[1][1], .65]; ep.step(None)             # part 1 installed
    cc = {'invalid_contact|robot_touches_person|x': {'category': 'invalid_contact'}}
    rec = metric_record(ep, True, .1, {'contact_classes': cc, 'fault': None})
    assert rec['t_H_ready'][0] == .05 and rec['steps_done'] == [['part1_installed', .1]]
    assert rec['disrupted'] and rec['t_disruption'] == .05 and rec['t_recovery'] == .1
    assert rec['human_robot_contact'] and not rec['recovered']
