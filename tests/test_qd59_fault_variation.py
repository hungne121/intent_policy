"""59/QĐ-DL: N5 fault injector and N1 variation (no simulation)."""
import numpy as np
import pytest
import yaml

from intent_policy.shared.sim_env import ROOT, HORIZONS, DT
from intent_policy.shared.forecast_fault import FAULT_TYPES, FaultInjector, sample_fault, solve_elbow, ARMS
from intent_policy.shared.variation import sample_variation, timed_human

CFG = yaml.safe_load((ROOT / 'configs/hrc/collection.yaml').read_text())


def pose(t=0.):
    """Plausible 6-joint pose (JOINTS order); the person moves along +x (bone lengths constant)."""
    p = np.array([[.10, -.60, 1.35], [.45, -.60, 1.35], [.05, -.55, 1.10], [.50, -.45, 1.12],
                  [.05, -.50, .90], [.55, -.25, .95]])
    p[:, 0] += .3 * t
    return p


def raw_forecast(t):
    return np.stack([pose(t + h) for h in HORIZONS])


def run(fault_type, seed=3):
    spec = sample_fault(fault_type, seed, CFG['n5'])
    spec.onset_index, spec.delay_s = 0, 0.
    inj = FaultInjector(spec, CFG['n5'])
    outs = []
    for i in range(80):
        t = i * DT
        used = i >= 10                                     # the expert starts reading at 0.5 s
        eff, k = inj.apply(raw_forecast(t), pose(t), t, used, 'r')
        outs.append((t, k, eff, raw_forecast(t)))
    return inj, outs


@pytest.mark.parametrize('fault_type', FAULT_TYPES)
def test_fault_window_continuity_geometry(fault_type):
    inj, outs = run(fault_type)
    s = inj.summary()
    assert s['fault_applied'] and not s['violations'], s
    assert abs(s['fault_start'] - .5) < 1e-9
    for t, k, eff, raw in outs:
        if k == 0:
            assert np.array_equal(eff, raw)                  # true forecast outside the window
        else:
            assert np.allclose(eff[:, [0, 2, 4]], raw[:, [0, 2, 4]])   # idle (left) arm untouched
    assert s['max_wrist_shift_1s_m'] > .05


def test_spatial_bone_lengths_kept():
    inj, outs = run('spatial_fault')
    s_i, e_i, w_i = ARMS['r']; cur = pose(0.)  # lengths are time-invariant
    l1, l2 = np.linalg.norm(cur[e_i] - cur[s_i]), np.linalg.norm(cur[w_i] - cur[e_i])
    for t, k, eff, raw in outs:
        if k > 0:
            assert np.allclose(np.linalg.norm(eff[:, e_i] - eff[:, s_i], axis=1), l1, atol=1e-6)
            assert np.allclose(np.linalg.norm(eff[:, w_i] - eff[:, e_i], axis=1), l2, atol=1e-6)


def test_solve_elbow_clamps_unreachable():
    e, w, clamped = solve_elbow([0, 0, 0], [1., 0, 0], .3, .3, [0, .1, 0])
    assert clamped and np.isclose(np.linalg.norm(e), .3) and np.isclose(np.linalg.norm(w - e), .3)


def test_fault_never_triggers_without_motion():
    spec = sample_fault('spatial_fault', 1, CFG['n5'])
    inj = FaultInjector(spec, CFG['n5'])
    for i in range(60):
        eff, k = inj.apply(raw_forecast(0.), pose(0.), i * DT, False, 'r')
        assert k == 0
    assert not inj.summary()['fault_applied']


def test_samples_are_deterministic():
    assert sample_fault('direction_fault', 7, CFG['n5']).to_dict() == sample_fault('direction_fault', 7, CFG['n5']).to_dict()
    assert sample_variation(7, CFG['n1']) == sample_variation(7, CFG['n1'])
    lo, hi = CFG['n1']['speed_scale']
    assert all(lo <= sample_variation(s, CFG['n1'])['speed_scale'] <= hi for s in range(50))


def _script(log):
    """Toy human: 10 stationary frames, then a 20-frame move with a side effect at frame 15."""
    for _ in range(10):
        yield {'palm': np.zeros(3)}
    for i in range(20):
        if i == 15:
            log.append('grasp')
        yield {'palm': np.array([.01 * (i + 1), 0, 0])}


@pytest.mark.parametrize('speed', [.8, 1., 1.2])
def test_timed_human_speed_and_side_effects(speed):
    side, log = [], {}
    frames = list(timed_human(_script(side), {'speed_scale': speed, 'start_shift_s': 0., 'pause': None}, log))
    assert side == ['grasp']
    assert np.allclose(frames[-1]['palm'], [.2, 0, 0])
    assert abs(len(frames) - 30 / speed) <= 2


def test_timed_human_shift_and_pause():
    log = {}
    early = list(timed_human(_script([]), {'speed_scale': 1., 'start_shift_s': -.3, 'pause': None}, log))
    assert log['dropped_start_frames'] == 6 and len(early) == 24
    log = {}
    late = list(timed_human(_script([]), {'speed_scale': 1., 'start_shift_s': .2,
                                          'pause': {'after_s': .6, 'duration_s': .5}}, log))
    assert log['held_start_frames'] == 4 and log['pause_at_s'] is not None
    assert len(late) == 30 + 4 + 10
