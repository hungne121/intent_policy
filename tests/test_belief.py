"""Belief over the human's target (intent_policy/intent/belief.py, docs/requirements/INTENT_ACT_v3.2.md §3-4)."""
import numpy as np
import pytest

from intent_policy.intent.belief import (GestureRamp, HandKinematics, OnlineDTW, PointingBelief, ReachBelief,
                                         ReachLibrary, load_library, pointing_likelihood, save_libraries)
from intent_policy.intent.labels import load_schema

S = load_schema()
B = S['belief']


def test_pointing_likelihood_peaks_on_the_pointed_candidate_and_is_flat_when_not_pointing():
    base, tip = np.array([0.0, 0.0, 1.0]), np.array([0.1, 0.0, 1.0])            # ray along +x
    cands = np.array([[1.0, 0.0, 1.0], [1.0, 0.3, 1.0], [0.0, 1.0, 1.0]])
    lik = pointing_likelihood(base, tip, cands, 0.9, 4.61)
    assert np.argmax(lik) == 0 and lik[0] == pytest.approx(1.0)
    assert lik[2] == pytest.approx(0.1, abs=1e-6)                               # far off the ray: 1 - p_point
    assert np.allclose(pointing_likelihood(base, tip, cands, 0.0, 4.61), 1.0)


def test_pointing_belief_follows_a_change_of_mind():
    b = PointingBelief(['A', 'B'], lam=B['lambda'])
    for _ in range(20):
        b.update(np.array([1.0, 0.2]))
    assert b.b['A'] > 0.99
    for _ in range(20):                                                         # the human now points at B
        b.update(np.array([0.2, 1.0]))
    assert b.b['B'] > 0.99


def test_pointing_belief_forgets_towards_uniform_without_evidence():
    b = PointingBelief(['A', 'B'], lam=0.9)
    for _ in range(20):
        b.update(np.array([1.0, 0.2]))
    for _ in range(60):
        b.update(np.ones(2))
    assert abs(b.b['A'] - 0.5) < 0.05


def test_reach_belief_mixes_the_posterior_by_the_gesture_probability_without_recounting():
    r = ReachBelief(['H1', 'H2'], lam=0.9)
    for _ in range(30):                                  # the same posterior every frame: no accumulation
        r.update(np.array([0.9, 0.1]), p_g=0.5)
    assert r.b['H1'] == pytest.approx(0.5 * 0.9 + 0.5 * 0.5)
    for _ in range(80):
        r.update(None, 0.0)
    assert abs(r.b['H1'] - 0.5) < 0.02


def _line(end, T=20):
    u = np.linspace(0.0, 1.0, T)[:, None]
    return np.concatenate([u * end, u * end], 1)                                 # wrist and tip move together


def test_online_dtw_classifies_a_partial_reach():
    a, b = np.array([0.3, 0.0, 0.0]), np.array([0.0, 0.3, 0.0])
    lib = ReachLibrary.fit({'A': [_line(a, 18), _line(a, 22)], 'B': [_line(b, 18), _line(b, 22)]}, 20, 0.01)
    dtw = OnlineDTW(lib, ['A', 'B'])
    post = [dtw.step(x) for x in _line(b, 25)[:15]]                              # 60 % of a slower reach to B
    assert post[0][0] == pytest.approx(0.5, abs=1e-6)                            # nothing moved yet
    assert post[-1][1] > 0.9


def test_start_clusters_pick_the_matching_library_and_survive_save_load(tmp_path):
    a, b = np.array([0.3, 0.0, 0.0]), np.array([0.0, 0.3, 0.0])
    off = np.r_[0.2, 0.2, 0.0, 0.2, 0.2, 0.0]
    demos = {'A': [_line(a), _line(a), _line(a) + off, _line(a) + off, _line(a) + off],
             'B': [_line(b), _line(b), _line(b)]}
    lib = ReachLibrary.fit(demos, 20, 0.01, families=[['A', 'B']], radius=0.06, min_demos=2)
    assert lib.keys_for(['A', 'B'], np.zeros(6)) == ['A@0', 'B@0']
    assert lib.keys_for(['A', 'B'], off) == ['A', 'B']                            # cluster 1 has no B: all starts
    save_libraries(tmp_path / 'lib.npz', {0: lib, 1: lib}, {})
    back = load_library(tmp_path / 'lib.npz', 1)
    assert sorted(back.mean) == sorted(lib.mean) and back.radius == pytest.approx(0.06)
    assert np.allclose(back.start['A@0'], lib.start['A@0'])


def test_gesture_ramp_rises_after_the_detected_onset():
    kin = HandKinematics(S, B['onset_speed_m_s'], B['still_speed_m_s'])
    ramp = GestureRamp(S['gestures'], B['ramp_s'], B['p_max'], S['fps'])
    kp = np.zeros((6, 3))
    i = S['gestures'].index('point')
    p = None
    for t in range(40):
        moved = kp.copy()
        if t >= 5:                                                               # the hand moves 2 cm per frame
            moved[:3] += [0.02 * (t - 4), 0.0, 0.0]
        kin.step(t, moved)
        p = ramp.step(t, 'point' if t >= 5 else 'rest', kin)
        if t == 4:
            assert p[S['gestures'].index('rest')] == pytest.approx(B['p_max'])
        if t == 5:
            assert kin.onset_t == 5 and p[i] == pytest.approx(1.0 / len(S['gestures']))
    assert p[i] == pytest.approx(B['p_max'])                                      # ramp_s (23 frames) is over
