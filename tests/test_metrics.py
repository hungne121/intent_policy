"""HRIBench-style metrics on synthetic event streams (no simulator, no policy)."""
import ast
from pathlib import Path

import pytest

from intent_policy.benchmark import metrics as M
from intent_policy.benchmark.metrics import compute_episode_metrics, aggregate, ALL_METRICS
from conftest import ev

CTX = M.EpisodeContext(['a', 'b', 'c'])


def stream(*events, end=20.0, success=True):
    base = [ev(0.0, 'episode_start')] + list(events)
    if success:
        base.append(ev(end, 'task_success'))
    base.append(ev(end, 'episode_end'))
    return sorted(base, key=lambda e: e['timestamp'])


def test_metric_interface_reset_update_compute():
    m = M.CSR()
    m.update(stream(success=True), CTX)
    m.update(stream(success=False), CTX)
    out = m.compute()
    assert out['value'] == pytest.approx(0.5) and out['n'] == 2
    m.reset()
    assert m.compute()['value'] is None


def test_csr_ct():
    s = stream(end=12.5)
    assert M.CSR().episode_value(s, CTX) == 1.0
    assert M.CT().episode_value(s, CTX) == pytest.approx(12.5)
    f = stream(success=False)
    assert M.CSR().episode_value(f, CTX) == 0.0
    assert M.CT().episode_value(f, CTX) is None


def test_idle_ratio_uses_motion_intervals():
    s = stream(ev(2, 'human_cue_onset'), ev(4, 'robot_motion_start'), ev(10, 'robot_motion_end', stopped_at=9.0), end=12)
    # task window [2, 12] = 10 s, moving [4, 9] = 5 s -> IR = 0.5
    assert M.IR().episode_value(s, CTX) == pytest.approx(0.5)


def test_rsp_motion_start_and_stop():
    s = stream(ev(3, 'human_cue_onset'), ev(4.25, 'robot_motion_start'), ev(8, 'robot_motion_end', stopped_at=7.9))
    assert M.Rsp().episode_value(s, CTX) == pytest.approx(1.25)
    moving = stream(ev(1, 'robot_motion_start'), ev(3, 'human_cue_onset'), ev(9, 'robot_motion_end', stopped_at=8.5))
    assert M.Rsp().episode_value(moving, CTX) == 0.0
    ctx = M.EpisodeContext([], {'rsp_trigger': {'event_type': 'disruption_start'}, 'rsp_response': 'robot_motion_end'})
    stop = stream(ev(1, 'robot_motion_start'), ev(5, 'disruption_start'), ev(5.6, 'robot_motion_end', stopped_at=5.4))
    assert M.Rsp().episode_value(stop, ctx) == pytest.approx(0.4)
    assert M.Rsp().episode_value(stream(), CTX) is None       # no trigger -> not observable


def test_tsync():
    ctx = M.EpisodeContext([], {'tsync_human': {'event_type': 'interaction_window_start'},
                                'tsync_robot': {'event_type': 'protocol_step_complete', 'step': 'robot_ready'}})
    s = stream(ev(5, 'protocol_step_complete', step='robot_ready'), ev(6.5, 'interaction_window_start'))
    assert M.TSync().episode_value(s, ctx) == pytest.approx(1.5)
    assert M.TSync().episode_value(stream(), ctx) is None


def test_order_compliance():
    ok = stream(*[ev(i + 1, 'protocol_step_complete', step=s) for i, s in enumerate('abc')])
    assert M.OC().episode_value(ok, CTX) == 1.0
    partial = stream(ev(1, 'protocol_step_complete', step='a'), ev(2, 'protocol_step_complete', step='b'))
    assert M.OC().episode_value(partial, CTX) == pytest.approx(2 / 3)
    wrong = stream(ev(1, 'protocol_step_complete', step='b'), ev(2, 'protocol_step_complete', step='a'))
    assert M.OC().episode_value(wrong, CTX) == 0.0
    extra = stream(ev(1, 'protocol_step_complete', step='a'), ev(2, 'protocol_step_complete', step='other'),
                   ev(3, 'protocol_step_complete', step='b'), ev(4, 'protocol_step_complete', step='c'))
    assert M.OC().episode_value(extra, CTX) == 1.0            # steps outside the protocol are ignored


def test_collision_free_and_contact_safety():
    clean = stream()
    assert M.CFR().episode_value(clean, CTX) == 1.0 and M.HCS().episode_value(clean, CTX) == 1.0
    near = stream(ev(3, 'safety_distance_violation', distance=0.03))
    assert M.CFR().episode_value(near, CTX) == 1.0 and M.HCS().episode_value(near, CTX) == 0.0
    hit = stream(ev(3, 'human_robot_contact'), success=False)
    assert M.CFR().episode_value(hit, CTX) == 0.0 and M.HCS().episode_value(hit, CTX) == 0.0


def test_cir():
    assert M.CIR().episode_value(stream(), CTX) is None
    s = stream(ev(2, 'human_intention_change', contradictory=True), ev(3, 'protocol_step_complete', step='contradiction_recognized'),
               ev(5, 'human_intention_change', contradictory=True))
    assert M.CIR().episode_value(s, CTX) == pytest.approx(0.5)


def test_dsr():
    assert M.DSR().episode_value(stream(), CTX) is None
    good = stream(ev(2, 'disruption_start'), ev(4, 'disruption_end'), ev(5, 'recovery_start'), ev(8, 'recovery_complete'))
    assert M.DSR().episode_value(good, CTX) == 1.0
    no_recovery = stream(ev(2, 'disruption_start'), ev(4, 'disruption_end'), success=False)
    assert M.DSR().episode_value(no_recovery, CTX) == 0.0


def test_applicability_and_aggregation():
    s = stream(ev(1, 'human_cue_onset'), ev(2, 'robot_motion_start'), ev(9, 'robot_motion_end', stopped_at=9))
    m = compute_episode_metrics(s, ['CSR', 'CT', 'IR'], ['a'])
    assert set(m) == set(ALL_METRICS)
    assert m['CSR'] == 1.0 and m['TSync'] == 'n/a' and m['DSR'] == 'n/a'
    agg = aggregate([m, compute_episode_metrics(stream(success=False), ['CSR', 'CT', 'IR'], ['a'])])
    assert agg['CSR']['value'] == pytest.approx(0.5) and agg['CSR']['n'] == 2
    assert agg['CT']['n'] == 1                                  # failed episode has no completion time
    assert agg['OC'] == 'n/a'


def test_metrics_are_independent_of_policy_code():
    src = Path(M.__file__).read_text()
    imported = {n.names[0].name if isinstance(n, ast.Import) else n.module
                for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.Import, ast.ImportFrom))}
    assert not any(m and (m.startswith('policies') or m.startswith('lerobot') or m.startswith('torch')) for m in imported)
