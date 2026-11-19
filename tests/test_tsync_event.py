"""26/QĐ-HL Phase 4 item 4: unit tests of the event-based TSync (tools/hribench_table.tsync_event, contract 18/QĐ-KT):
early / on time / late / missing events, N/A. s = exp(-max(t_R_ready - t_H_ready - u, 0)/tau)."""
import math
from intent_policy.benchmark.tsync import tsync_event


def rec(rounds, req=('s',), done=()):
    return {'metrics': {'steps_required': list(req), 'steps_done': [list(d) for d in done], 'rounds': rounds}}


def rnd(tr, th, step='s'):
    return {'step': step, 't_R_ready': tr, 't_H_ready': th}


def test_robot_early_is_not_penalised():
    s, n = tsync_event(rec([rnd(0.5, 5.0)]), 0.5, 1.0)
    assert s == 1.0 and n == 1


def test_on_time_within_u():
    s, _ = tsync_event(rec([rnd(5.4, 5.0)]), 0.5, 1.0)
    assert s == 1.0


def test_late_decays_exponentially():
    s, _ = tsync_event(rec([rnd(7.0, 5.0)]), 0.5, 1.0)
    assert math.isclose(s, math.exp(-1.5))


def test_missing_time_scores_zero_but_counts_as_required():
    s, n = tsync_event(rec([rnd(None, 5.0)]), 0.5, 1.0)
    assert s == 0.0 and n == 0
    s, n = tsync_event(rec([]), 0.5, 1.0)
    assert s == 0.0 and n == 0


def test_step_already_satisfied_at_start_is_excluded_and_all_excluded_is_na():
    s, n = tsync_event(rec([], req=('s',), done=[('s', 0.0)]), 0.5, 1.0)
    assert s is None and n == 0


def test_mean_over_required_steps():
    r = rec([rnd(5.0, 5.0, 'a'), rnd(8.0, 5.0, 'b')], req=('a', 'b'))
    s, n = tsync_event(r, 0.5, 1.0)
    assert math.isclose(s, (1.0 + math.exp(-2.5)) / 2) and n == 2
