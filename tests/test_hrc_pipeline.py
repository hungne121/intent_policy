"""Shared intent rule and schema of the HRC pipeline (53/QĐ-DL, 54/QĐ-DL), without MuJoCo."""
import numpy as np

from intent_policy.shared.forecast_planner import ForecastPlanner, AVOID_RADIUS, TARGET_RADIUS
from intent_policy.data.dataset_schema import features
from intent_policy.data.episode_validator import validate


class FakeEpisode:
    def __init__(self, mode, forecast, wrist=(0., 0., .8), speed=.2):
        self.mode, self._f, self.hand_speed = mode, forecast, speed
        self.kp = np.zeros((6, 3)); self.kp[5] = wrist

    def forecast_available(self):
        return self._f if self.mode == 'forecast' else None

    def forecast_for_expert(self):
        return self._f if self.mode == 'forecast' else None

    def wrist_index(self):
        return 5

    def wrist(self):
        return self.kp[5]


def straight_forecast(end=(.3, 0., .8)):
    f = np.zeros((10, 6, 3)); f[:, 5] = np.linspace([0., 0., .8], end, 11)[1:]
    return f


def test_observed_mode_never_reads_the_forecast():
    plan = ForecastPlanner(FakeEpisode('observed', straight_forecast()))
    assert plan.follow_point() is None and not plan.active()
    assert not plan.tube_hits((.3, 0.))            # only the observed wrist counts


def test_follow_reads_the_1s_end_point_plus_offset():
    plan = ForecastPlanner(FakeEpisode('forecast', straight_forecast()))
    assert np.allclose(plan.follow_point((0., .1, 0.)), [.3, .1, .8])
    assert plan.heads_to((.3 + TARGET_RADIUS / 2, 0.))
    assert plan.nearest({'a': (.3, 0.), 'b': (-.3, 0.)}) == 'a'


def test_follow_needs_a_moving_hand():
    plan = ForecastPlanner(FakeEpisode('forecast', straight_forecast(), speed=0.))
    assert plan.follow_point() is None


def test_avoid_uses_the_whole_trajectory():
    plan = ForecastPlanner(FakeEpisode('forecast', straight_forecast()))
    assert plan.tube_hits((.15, AVOID_RADIUS / 2))
    assert not plan.tube_hits((.15, AVOID_RADIUS * 2))


def test_schema_is_fixed():
    f = features(['high', 'top', 'wrist'], (256, 256))
    assert f['action']['shape'] == (7,) and f['observation.human_forecast']['shape'] == (180,)
    assert f['observation.images.top']['shape'] == (256, 256, 3)


def test_validator_rejects_forecast_given_to_observed():
    s = {'success': True, 'physics_valid': True, 'mode': 'observed', 'contact_violations': {},
         'human_self_violations': {}, 'robot_limits_ok': True, 'robot_limits': {},
         'forecast_validity_summary': {'frames': 3, 'valid': 3, 'used_by_expert': 1}}
    ok, reasons = validate(s, 3, 3)
    assert not ok and 'observed' in reasons[0]
