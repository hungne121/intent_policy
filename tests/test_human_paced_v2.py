import numpy as np
from intent_policy.experts.human_paced import HumanPace, CartesianFollower


def test_forecast_cannot_raise_pace_above_observed_motion():
    pace = HumanPace()
    for _ in range(100):
        pace.update(.20, 2.0)
    assert pace.speed(following=True) <= .20 * .8 + 1e-8


def test_work_continues_after_pointing_stops():
    pace = HumanPace()
    for _ in range(40): pace.update(.30, .25)
    for _ in range(40): pace.update(0., 0.)
    assert .14 <= pace.speed() <= .24


def test_cartesian_follower_limits_speed_and_acceleration():
    follower = CartesianFollower(dt=.05, acceleration=.65)
    current = np.zeros(3)
    velocity = np.zeros(3)
    for _ in range(80):
        nxt = follower.step(current, [2., .2, 0.], .18)
        assert np.linalg.norm(follower.velocity) <= .18 + 1e-9
        assert np.linalg.norm(follower.velocity - velocity) <= .65 * .05 + 1e-9
        current, velocity = nxt, follower.velocity.copy()


def test_new_target_turns_without_position_jump():
    follower = CartesianFollower()
    current = np.zeros(3)
    for _ in range(20): current = follower.step(current, [1., 0., 0.], .20)
    previous = current.copy()
    current = follower.step(current, [0., 1., 0.], .20)
    assert np.linalg.norm(current - previous) <= .20 * .05 + 1e-9
