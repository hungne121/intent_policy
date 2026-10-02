"""IntentTracker and HoldingDetector (INTENT_ACT_GUIDE_v2.md §3.4, §9.5)."""
import numpy as np
import pytest

from intent_policy.intent.contract import one_hot
from intent_policy.intent.labels import load_schema
from intent_policy.intent.places import target_positions
from intent_policy.intent.tracker import HoldingDetector, IntentTracker, RobotState

S = load_schema()
K, W, G = len(S['targets']), len(S['who']), len(S['gestures'])
POS = target_positions(S)
REST_TCP = np.array([0.15, 0.0, 0.85])


class Sim:
    """Drives a tracker with one-hot gestures / targets and a scripted robot."""

    def __init__(self, **kw):
        self.tr = IntentTracker(S, **{**S['tracker'], **kw})
        self.tcp, self.holding, self.hands = REST_TCP.copy(), False, np.zeros((6, 3))
        self.out = None

    def frames(self, n, gesture='rest', target='none'):
        for _ in range(n):
            self.out = self.tr.step(one_hot(S['targets'].index(target), K), one_hot(0, W),
                                    one_hot(S['gestures'].index(gesture), G), RobotState(self.tcp, 0.04, self.holding), self.hands)
        return self

    @property
    def c(self):
        return S['targets'][int(np.argmax(self.out['c_target']))], S['who'][int(np.argmax(self.out['c_who']))]

    def grasp_and_lift(self, slot, gesture='rest', target='none'):
        self.tcp = POS[slot].copy()
        self.holding = True
        self.frames(2, gesture, target)
        self.tcp = self.tcp + [0, 0, 0.05]
        return self.frames(2, gesture, target)

    def release_at(self, xyz):
        self.tcp = np.asarray(xyz, float).copy()
        self.frames(1)
        self.holding = False
        return self.frames(2)


def test_t1_pick_then_place():
    s = Sim().frames(3)
    assert s.c == ('none', 'none')
    s.frames(4, 'point', 'S3')
    assert s.c == ('none', 'none')                                  # not stable for hold_frames yet
    s.frames(1, 'point', 'S3')
    assert s.c == ('S3', 'robot')                                   # committed
    s.frames(20, 'point', 'P1')
    assert s.c == ('S3', 'robot') and s.out['queue'] == [('pick', 'S3'), ('place', 'P1')]
    s.frames(30)                                                    # retract: the commitment stays (memory)
    assert s.c == ('S3', 'robot')
    s.grasp_and_lift('S3')
    assert s.c == ('P1', 'robot')                                   # pick done -> place is next
    s.release_at(POS['P2'])
    assert s.c == ('P1', 'robot')                                   # released in the wrong zone: not done
    s.holding = True
    s.frames(2)
    s.release_at(POS['P1'] + [0.02, 0.0, 0.1])
    assert s.c == ('none', 'none') and s.out['queue'] == []


def test_change_of_mind_replaces_until_the_grasp():
    s = Sim().frames(6, 'point', 'S3').frames(10, 'rest')
    s.tcp = POS['S3'] + [0, 0, 0.1]                                 # robot approaching the old target (late change)
    s.frames(6, 'point', 'S5')
    assert s.c == ('S5', 'robot') and s.out['queue'] == [('pick', 'S5')]
    s.grasp_and_lift('S5')
    assert s.c == ('none', 'none')
    s2 = Sim().frames(6, 'point', 'S3')
    s2.tcp, s2.holding = POS['S3'].copy(), True                     # grasped: a new pointing no longer replaces it
    s2.frames(2).frames(10, 'rest').frames(6, 'point', 'S5')
    assert s2.out['queue'][0] == ('pick', 'S3')


def test_t2_handover_and_no_recommit_while_the_palm_stays_out():
    s = Sim().frames(6, 'point', 'S4').frames(10, 'rest').frames(6, 'palm_up', 'H2')
    assert s.out['queue'] == [('pick', 'S4'), ('handover', 'H2')]
    s.grasp_and_lift('S4', 'palm_up', 'H2')                           # the palm stays out the whole time
    assert s.c == ('H2', 'joint')
    s.hands[:] = POS['H2'] + [0.0, 0.0, 0.02]
    s.tcp = POS['H2'] + [0, 0, 0.05]
    s.frames(1, 'palm_up', 'H2')
    s.holding = False
    s.frames(10, 'palm_up', 'H2')
    assert s.c == ('none', 'none')                                  # completed and not committed again


def test_cancel_is_off_by_default():
    s = Sim().frames(6, 'point', 'S3').frames(10, 'palm_forward', 'S3')
    assert s.c == ('S3', 'robot')
    s = Sim(cancel=True).frames(6, 'point', 'S3').frames(10, 'palm_forward', 'S3')
    assert s.c == ('none', 'none')


def test_reach_never_commits_and_place_needs_a_pick():
    s = Sim().frames(20, 'reach', 'U1').frames(20, 'point', 'P1')
    assert s.c == ('none', 'none') and s.out['queue'] == []


def test_slip_keeps_the_pick_for_a_retry():
    s = Sim().frames(6, 'point', 'S3')
    s.tcp, s.holding = POS['S3'].copy(), True
    s.frames(2)
    s.holding = False                                               # slipped before the lift
    s.frames(3)
    assert s.c == ('S3', 'robot')


def test_holding_detector():
    det = HoldingDetector(-0.005, 0.06, steady_m=0.002, steady_frames=1)
    seq = [0.085, 0.085, 0.07, 0.055, 0.045, 0.041, 0.040, 0.040, 0.040, 0.05, 0.07, 0.085]
    out = [det.step(x) for x in seq]
    assert out == [False, False, False, False, False, False, True, True, True, False, False, False]
    det.reset()
    assert [det.step(x) for x in (0.08, 0.04, 0.001, 0.0005, 0.0005)][-1]   # a cup rim: nearly closed but held
