"""Unit tests of the I3 grasp-sequence rule and of scenarios.i3.check with retry tolerance (31/QĐ-HL)."""
import numpy as np
import pytest

from intent_policy.benchmark.grasp_sequence import collapse_retries, sequence_reason
from intent_policy.scenarios import i3

A, B, W = 'Rx', 'Lx', 'Ly'


@pytest.mark.parametrize('variant,grasps,v1_ok,v2_ok', [
    ('nominal', [B], True, True),
    ('nominal', [B, B, B], False, True),          # retry of the target
    ('nominal', [W], False, False),
    ('nominal', [W, B], False, False),            # another object is not a retry
    ('nominal', [B, W], False, False),
    ('nominal', [], False, False),
    ('switch_before_commit', [B, B], False, True),
    ('switch_before_commit', [A, B], False, False),   # A must not be grasped
    ('switch_before_commit', [A, A, B], False, False),
    ('switch_after_commit', [A, B], True, True),
    ('switch_after_commit', [B], True, True),
    ('switch_after_commit', [A, A, B, B], False, True),
    ('switch_after_commit', [B, B], False, True),
    ('switch_after_commit', [B, A], False, False),
    ('switch_after_commit', [A], False, False),
    ('switch_after_commit', [A, B, A], False, False),  # A again after B is not a retry
])
def test_sequence_rule(variant, grasps, v1_ok, v2_ok):
    tgt = B if variant != 'nominal' else B
    assert (sequence_reason(variant, grasps, A, tgt, retry_ok=False) is None) == v1_ok
    assert (sequence_reason(variant, grasps, A, tgt, retry_ok=True) is None) == v2_ok


def test_collapse():
    assert collapse_retries(['a', 'a', 'b', 'a', 'a']) == ['a', 'b', 'a']
    assert collapse_retries([]) == []


def test_v1_wording_unchanged():
    assert sequence_reason('switch_before_commit', [A, B], A, B) == "grasp sequence ['Rx', 'Lx'] (A must not be grasped)"
    assert sequence_reason('nominal', [W], A, B) == "grasp sequence ['Ly']"
    assert sequence_reason('switch_after_commit', [W], A, B) == "grasp sequence ['Ly'] (A then B, or only B, expected)"


class FakeEp:
    def __init__(self, final, initial, grasps):
        self._f, self.initial = final, initial
        self.events = [{'event': 'robot_grasp', 'object': g} for g in grasps]

    def pos(self, k):
        return np.array(self._f[k])


def _scene(target_xy=(0.28, 0.08), target_z=0.641, moved_mm=0.0, target_in_tray=True):
    init = {k: np.array([x, -0.10, 0.641]) for k, x in zip(i3.CAKES, (0.11, 0.21, 0.35, 0.45))}
    fin = {k: v.copy() for k, v in init.items()}
    fin[B] = np.array([*target_xy, target_z]) if target_in_tray else np.array([0.5, 0.0, 0.641])
    if moved_mm:
        fin[A] = init[A] + np.array([moved_mm / 1000, 0, 0])
    return fin, init


INFO = dict(variant='switch_before_commit', A=A, B=B)


def test_check_retry_still_requires_target_in_tray():
    fin, init = _scene(target_in_tray=False)
    ok, why = i3.check(FakeEp(fin, init, [B, B]), INFO, retry_ok=True)
    assert not ok and 'not in receive tray' in why


def test_check_retry_still_requires_unmoved_cakes():
    fin, init = _scene(moved_mm=12.0)
    ok, why = i3.check(FakeEp(fin, init, [B, B]), INFO, retry_ok=True)
    assert not ok and f'{A} moved' in why
    fin, init = _scene(moved_mm=5.0)
    assert i3.check(FakeEp(fin, init, [B, B]), INFO, retry_ok=True)[0]


def test_check_default_is_old_evaluator():
    fin, init = _scene()
    assert not i3.check(FakeEp(fin, init, [B, B]), INFO)[0]               # configs default false
    assert i3.check(FakeEp(fin, init, [B]), INFO)[0]
    assert i3.check(FakeEp(fin, init, [B, B]), INFO, retry_ok=True)[0]
