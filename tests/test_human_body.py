"""Human mannequin (HY-Motion wooden model): reach, pose continuity, rendering hooks."""
import numpy as np
import pytest

from intent_policy.scenarios.config import ScenarioConfig
from intent_policy.sim.human_body import HumanBody


@pytest.fixture(scope='module')
def body():
    scene = ScenarioConfig.load('t2_handover').scene
    return HumanBody({**scene['human'], 'floor_z': scene['floor_z']})


def test_asset_segments_and_proportions(body):
    assert set(body.segments) == {'pelvis', 'torso', 'head', 'l_upperarm', 'l_forearm', 'l_hand',
                                  'r_upperarm', 'r_forearm', 'r_hand'}
    assert 1.6 < body.skeleton['height_m'] < 1.9
    assert 0.2 < body.L1['r'] < 0.35 and 0.2 < body.L2['r'] < 0.35
    variants = set(body.skeleton['segments']['r_hand']['variants'])
    assert {'relaxed', 'point', 'grasp', 'offer'} <= variants and 'relaxed-point-50' in variants   # + blended shapes


def test_pointing_index_finger_ray_hits_every_zone(body):
    """The index finger ray (knuckle -> tip of the HY-Motion hand) passes within 2 cm of every S / P target."""
    from intent_policy.utils import load_yaml
    zones = load_yaml('configs/layout.yaml')['zones']
    for k, z in zones.items():
        if z['kind'] not in ('slot', 'place'):
            continue
        tgt = np.r_[z['xy'], 0.62]
        w, R = body.pose(body.pointing_palm(tgt, 0.50), tgt)['segments']['r_hand']
        knuckle, tip = (w + R @ body.R_yaw.T @ (q - body.J['R_Wrist']) for q in body.index_ray['r'])
        d = (tip - knuckle) / np.linalg.norm(tip - knuckle)
        v = tgt - tip
        assert np.linalg.norm(v - np.dot(v, d) * d) < 0.02, k


def test_palm_turns_up_for_receiving(body):
    palm = [0.36, -0.15, 0.77]
    normal = lambda f: body.pose(palm, palm_up=f)['segments']['r_hand'][1] @ np.array([0.0, 0.0, -1.0])
    assert normal(0.0)[2] < -0.8 and normal(1.0)[2] > 0.8 and abs(normal(0.5)[2]) < 0.3
    assert normal(1.0)[2] > 0.99                     # held out flat to receive an object into the palm
    assert body.pose(palm, palm_up=1.0)['reach_error'] < 1e-6


# zones of configs/layout.yaml for the side-standing human (H1, H2 at hand height, U cube, U_cup) and near P
@pytest.mark.parametrize('palm', [[0.24, -0.15, 0.77], [0.36, -0.15, 0.77], [0.41, -0.26, 0.65], [0.205, -0.26, 0.74],
                                  [0.36, -0.03, 0.74]])
def test_palm_targets_on_the_table_are_reached(body, palm):
    pose = body.pose(palm)
    assert pose['reach_error'] < 1e-6
    assert 0.0 <= pose['lean'] <= body.max_lean + 1e-9


def test_clamp_returns_a_reachable_point_towards_the_target(body):
    far = np.array([0.0, 0.45, 0.66])             # beyond the far robot-zone slots, out of reach even leaning
    near = body.clamp(far)
    assert body.pose(near)['reach_error'] < 1e-6 and np.linalg.norm(near - far) > 0.01
    shoulder = body.J['R_Shoulder']
    assert np.linalg.norm(near - far) < np.linalg.norm(shoulder - far)


def test_no_lean_when_the_target_is_within_reach(body):
    assert body.pose(body.J['R_Shoulder'] + [-0.2, 0.05, -0.35])['lean'] == 0.0


def test_pose_is_continuous_in_the_palm_target(body):
    a, b = body.pose([0.55, 0.05, 0.80]), body.pose([0.551, 0.05, 0.80])
    for seg in body.segments:
        (pa, Ra), (pb, Rb) = a['segments'][seg], b['segments'][seg]
        assert np.linalg.norm(pa - pb) < 0.01 and np.abs(Ra - Rb).max() < 0.05


def test_segment_rotations_are_proper(body):
    pose = body.pose([0.45, -0.10, 0.75], point_at=[0.30, -0.12, 0.62])
    for _, R in pose['segments'].values():
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0)


def test_scenario_poses_mannequin_and_changes_hand_shape_gradually(scenarios):
    sc = scenarios['t2_handover']
    sc.reset(0)
    m = sc.env.model
    b = sc.body_model
    assert m.geom_dataid[b.hand_geom] == b.hand_meshes['relaxed']
    sc.human.point_at = sc.pos(sc.on_table[0])
    steps = []
    for _ in range(4):                               # 25 % per tick through the blended meshes
        sc._apply_human_pose()
        steps.append(int(m.geom_dataid[b.hand_geom]))
    assert steps[:3] == [b.hand_meshes[f'relaxed-point-{p}'] for p in (25, 50, 75)] and steps[3] == b.hand_meshes['point']
    hand = sc.env.data.mocap_pos[sc.mocap['human_hand']]
    assert np.allclose(hand, sc.human.pos) and sc.max_reach_error < 1e-6
