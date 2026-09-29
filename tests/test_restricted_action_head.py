"""Restricted 9-action interface: enum, head output, deterministic controller mapping."""
import numpy as np
import pytest
import torch

from controllers.restricted_action import (RestrictedAction as A, RestrictedActionConfig, RestrictedActionMapper,
                                           NUM_RESTRICTED_ACTIONS, validate_action)
from policies.restricted_action_head import RestrictedActionHead

EXPECTED = ['HOLD', 'MOVE_FORWARD', 'MOVE_BACKWARD', 'MOVE_LEFT', 'MOVE_RIGHT', 'MOVE_UP', 'MOVE_DOWN',
            'OPEN_GRIPPER', 'CLOSE_GRIPPER']
# Documented convention in the default control frame (base_link == world axes): +X towards the human.
DIRECTION = {A.MOVE_FORWARD: (0, 1), A.MOVE_BACKWARD: (0, -1), A.MOVE_LEFT: (1, 1), A.MOVE_RIGHT: (1, -1),
             A.MOVE_UP: (2, 1), A.MOVE_DOWN: (2, -1)}


def test_exactly_nine_generic_actions_in_order():
    assert NUM_RESTRICTED_ACTIONS == 9
    assert [a.name for a in A] == EXPECTED
    assert [int(a) for a in A] == list(range(9))
    assert not {'PICK', 'PLACE', 'HANDOVER', 'RETRACT'} & {a.name for a in A}


@pytest.mark.parametrize('hidden', [None, 64])
def test_head_outputs_nine_logits_per_chunk_step(hidden):
    head = RestrictedActionHead(32, 9, hidden)
    out = head(torch.randn(4, 20, 32))
    assert out.shape == (4, 20, 9)
    with pytest.raises(ValueError):
        RestrictedActionHead(32, 5)


@pytest.mark.parametrize('bad', [-1, 9, 42, 1.0, True, '1', None])
def test_invalid_action_ids_fail(bad):
    with pytest.raises((ValueError, TypeError)):
        validate_action(bad)


@pytest.fixture
def mapper(scenarios):
    sc = scenarios['intruder_pick_place_interruption']
    sc.reset(0)
    m = RestrictedActionMapper(sc.env, RestrictedActionConfig.load())
    return m


def test_every_action_maps_to_the_documented_command(mapper):
    step = mapper.cfg.step_vector()
    for action in A:
        before = mapper.setpoint.copy()
        grip_before = mapper.gripper_target
        cmd = mapper.map(int(action))
        assert cmd.action_id == int(action) and cmd.action_name == action.name
        assert cmd.joint_target.shape == (7,) and np.isfinite(cmd.joint_target).all()
        if action in DIRECTION:
            axis, sign = DIRECTION[action]
            expected = np.zeros(3)
            expected[axis] = sign * step[axis]
            np.testing.assert_allclose(cmd.delta_frame, expected)
            np.testing.assert_allclose(cmd.delta_world, expected, atol=1e-9)
            np.testing.assert_allclose(cmd.cartesian_target, before + expected, atol=1e-12)
            assert cmd.gripper_target == grip_before
        else:
            assert not np.any(cmd.delta_frame) and not np.any(cmd.delta_world)
            np.testing.assert_array_equal(cmd.cartesian_target, before)
        if action is A.OPEN_GRIPPER:
            assert cmd.gripper_target == mapper.cfg.gripper['open'] and cmd.joint_target[6] == mapper.cfg.gripper['open']
        if action is A.CLOSE_GRIPPER:
            assert cmd.gripper_target == mapper.cfg.gripper['closed'] and cmd.joint_target[6] == mapper.cfg.gripper['closed']


def test_hold_generates_no_translation_and_identical_joint_target(mapper):
    first = mapper.map(int(A.HOLD))
    second = mapper.map(int(A.HOLD))
    assert not np.any(first.delta_world)
    np.testing.assert_array_equal(first.joint_target, second.joint_target)
    np.testing.assert_array_equal(first.cartesian_target, second.cartesian_target)


def test_mapping_is_deterministic(scenarios):
    sc = scenarios['intruder_pick_place_interruption']
    seq = [1, 3, 5, 0, 8, 2, 4, 6, 7, 0]
    runs = []
    for _ in range(2):
        sc.reset(3)
        m = RestrictedActionMapper(sc.env, RestrictedActionConfig.load())
        runs.append([m.map(a).joint_target.copy() for a in seq])
    for a, b in zip(*runs):
        np.testing.assert_array_equal(a, b)


def test_translation_magnitude_and_frame_are_configurable(scenarios):
    sc = scenarios['intruder_pick_place_interruption']
    sc.reset(0)
    cfg = RestrictedActionConfig.load()
    cfg.translation_step = {'x': 0.02, 'y': 0.005, 'z': 0.03}
    m = RestrictedActionMapper(sc.env, cfg)
    np.testing.assert_allclose(m.map(int(A.MOVE_FORWARD)).delta_world, [0.02, 0, 0], atol=1e-12)
    np.testing.assert_allclose(m.map(int(A.MOVE_RIGHT)).delta_world, [0, -0.005, 0], atol=1e-12)
    np.testing.assert_allclose(m.map(int(A.MOVE_DOWN)).delta_world, [0, 0, -0.03], atol=1e-12)
    # UR 'base' frame is rotated 180 deg about Z w.r.t. base_link: forward/left flip in the world.
    cfg_base = RestrictedActionConfig.load()
    cfg_base.control_frame = 'base'
    mb = RestrictedActionMapper(sc.env, cfg_base)
    np.testing.assert_allclose(mb.map(int(A.MOVE_FORWARD)).delta_world, [-0.01, 0, 0], atol=1e-9)
    np.testing.assert_allclose(mb.map(int(A.MOVE_LEFT)).delta_world, [0, -0.01, 0], atol=1e-9)


def test_controller_limits_refuse_out_of_workspace_setpoints(scenarios):
    sc = scenarios['intruder_pick_place_interruption']
    sc.reset(0)
    m = RestrictedActionMapper(sc.env, RestrictedActionConfig.load())
    zmax = m.cfg.limits['z'][1]
    for _ in range(100):
        cmd = m.map(int(A.MOVE_UP))
    assert m.setpoint[2] <= zmax + 1e-9 and cmd.rejected


@pytest.mark.parametrize('action,axis,sign', [(A.MOVE_FORWARD, 0, 1), (A.MOVE_BACKWARD, 0, -1), (A.MOVE_LEFT, 1, 1),
                                              (A.MOVE_RIGHT, 1, -1), (A.MOVE_UP, 2, 1), (A.MOVE_DOWN, 2, -1)])
def test_executed_motion_moves_tcp_in_the_commanded_direction(scenarios, action, axis, sign):
    sc = scenarios['intruder_pick_place_interruption']
    sc.reset(1)
    m = RestrictedActionMapper(sc.env, RestrictedActionConfig.load())
    start = sc.tcp()
    for _ in range(5):
        sc.step(m.map(int(action)).joint_target)
    for _ in range(10):
        sc.step(m.map(int(A.HOLD)).joint_target)
    moved = sc.tcp() - start
    assert sign * moved[axis] == pytest.approx(0.05, abs=0.006)
    assert np.abs(np.delete(moved, axis)).max() < 0.006


def test_gripper_commands_use_the_gripper_controller(scenarios):
    sc = scenarios['intruder_pick_place_interruption']
    sc.reset(1)
    m = RestrictedActionMapper(sc.env, RestrictedActionConfig.load())
    for _ in range(25):
        sc.step(m.map(int(A.CLOSE_GRIPPER)).joint_target)
    assert sc.env.observe().gripper < 0.01
    for _ in range(25):
        sc.step(m.map(int(A.OPEN_GRIPPER)).joint_target)
    assert sc.env.observe().gripper > 0.08
