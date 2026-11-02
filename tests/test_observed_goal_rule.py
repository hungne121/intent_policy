from types import SimpleNamespace
from unittest.mock import patch
from intent_policy.intent.goal_belief import GoalBelief


def test_observed_never_constructs_forecaster():
    with patch('intent_policy.intent.goal_belief.ManiCastForecaster', side_effect=AssertionError('forecast leak')):
        bel = GoalBelief(SimpleNamespace(dt=.05), 'observed', .3, .1)
    assert bel.fc is None
    assert bel.approach_star == 0


def test_observed_commit_does_not_wait_for_motion_end():
    bel = GoalBelief.__new__(GoalBelief)
    bel.rule, bel.obj_keys, bel.zone_keys = 'observed', ['a', 'b'], ['x', 'y']
    bel.after_motion, bel.evidence = False, {}
    bel.b_star_obj, bel.b_star_zone = .8, .75
    assert bel.commit_star('object') == .8
    assert bel.commit_star('zone') == .75
    bel.obj_keys = ['common_tool']
    assert bel.commit_star('object') == 0
    bel.rule = 'late'
    bel.complete = lambda _: False
    assert bel.commit_star('object') > 1
