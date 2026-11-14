"""Forecast correction is not a human change-of-mind event."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from i3_collect import events_from


def clip():
    return {'target_A_evaluator_only': 'A', 'target_B_evaluator_only': 'B', 'trace': []}


def result(phases):
    return {'phases': phases, 'attach_log': [], 'last_phase': 'observe'}


def test_initial_forecast_correction_is_not_human_switch():
    phases = [{'phase': 'wait_hand_clear', 't': 1., 'item': 'A', 'switched_from': 'B'},
              {'phase': 'pointing_confirmed', 't': 1., 'item': 'A'},
              {'phase': 'switch_pending', 't': 4., 'item': 'A'},
              {'phase': 'pointing_confirmed', 't': 5., 'item': 'B'}]
    assert events_from(result(phases), clip())['switch_observed'] == 4.


def test_no_human_switch_without_second_cue():
    phases = [{'phase': 'wait_hand_clear', 't': 1., 'item': 'A', 'switched_from': 'B'},
              {'phase': 'pointing_confirmed', 't': 1.1, 'item': 'A'}]
    assert 'switch_observed' not in events_from(result(phases), clip())


def test_confirmation_of_b_is_sufficient_after_a():
    phases = [{'phase': 'pointing_confirmed', 't': 1., 'item': 'A'},
              {'phase': 'pointing_confirmed', 't': 2., 'item': 'B'}]
    assert events_from(result(phases), clip())['switch_observed'] == 2.
