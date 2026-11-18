"""Shared intent rule and schema of the HRC pipeline (53/QĐ-DL, 54/QĐ-DL), without MuJoCo; plus one MuJoCo
regression check that the recorded cameras are identical in every task (68/QĐ-DL A)."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from intent_policy.shared.forecast_planner import ForecastPlanner, AVOID_RADIUS, TARGET_RADIUS
from intent_policy.data.dataset_schema import features
from intent_policy.data.episode_validator import validate

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = yaml.safe_load((ROOT / 'configs/hrc/collection.yaml').read_text())


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
    f = features(COLLECTION['cameras'], COLLECTION['image_hw'])
    assert f['action']['shape'] == (7,) and f['observation.human_forecast']['shape'] == (180,)
    images = sorted(k for k in f if k.startswith('observation.images.'))
    assert images == sorted(f'observation.images.{c}' for c in ('front_oblique', 'overhead_oblique', 'wrist'))   # 68/QĐ-DL
    assert all(f[k]['shape'] == (256, 256, 3) for k in images) and COLLECTION['fps'] == 20


def test_validator_rejects_forecast_given_to_observed():
    s = {'success': True, 'physics_valid': True, 'mode': 'observed', 'contact_violations': {},
         'human_self_violations': {}, 'robot_limits_ok': True, 'robot_limits': {},
         'forecast_validity_summary': {'frames': 3, 'valid': 3, 'used_by_expert': 1}}
    ok, reasons = validate(s, 3, 3)
    assert not ok and 'observed' in reasons[0]


def test_validator_never_accepts_false_physics_without_detailed_violations():
    s = {'success': True, 'physics_valid': False, 'mode': 'forecast',
         'contact_violations': {}, 'human_self_violations': {},
         'robot_limits_ok': True, 'robot_limits': {},
         'forecast_validity_summary': {'frames': 3, 'valid': 3, 'used_by_expert': 3}}
    ok, reasons = validate(s, 3, 3)
    assert not ok and 'physics_valid=false' in reasons


def test_quality_classification_never_promotes_physics_failure_to_a():
    from intent_policy.data.episode_quality import classify
    s = {'success': True, 'physics_valid': False, 'mode': 'forecast',
         'contact_violations': {'grasp': .001}, 'human_self_violations': {},
         'robot_limits_ok': True, 'robot_limits': {},
         'contact_classes': {'grasp': {'category': 'allowed_contact', 'violation': 'depth'}},
         'forecast_validity_summary': {'frames': 3, 'valid': 3, 'used_by_expert': 3}}
    assert classify(s, 3, 3)[0] == 'C'
    assert classify(s, 3, 3, True)[0] == 'B'
    s['contact_classes']['grasp']['category'] = 'invalid_contact'
    assert classify(s, 3, 3, True)[0] == 'C'
    s['physics_valid'] = True
    s['contact_classes'] = {}; s['contact_violations'] = {}
    assert classify(s, 3, 3)[0] == 'A'


# ---------------------------------------------------------------- 68/QĐ-DL A: recorded cameras identical in all tasks
def _camera(m, name):
    import mujoco
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
    assert i >= 0, f'camera {name} missing'
    return {'body': mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.cam_bodyid[i])), 'pos': m.cam_pos[i].copy(),
            'quat': m.cam_quat[i].copy(), 'mat0': m.cam_mat0[i].copy(), 'pos0': m.cam_pos0[i].copy(),
            'fovy': float(m.cam_fovy[i])}


@pytest.fixture(scope='module')
def task_cameras():
    """Every task and variant built as collection builds it (scenario build -> Episode), compiled model cameras."""
    from intent_policy.experts import TASKS
    from intent_policy.scenarios import scenario
    from intent_policy.shared.episode import Episode
    out = {}
    for task in TASKS:
        sc = scenario(task)
        for variant in getattr(sc, 'VARIANTS', (None,)):
            scene, positions, xml, info = sc.build(300000, variant) if variant else sc.build(300000)
            ep = Episode(task, 'observed', 300000, scene, positions, xml, None, out=None, max_ticks=1)
            out[(task, variant or 'nominal')] = {c: _camera(ep.m, c) for c in COLLECTION['cameras']}
            del ep
    return out


def test_recorded_cameras_identical_in_all_tasks(task_cameras):
    from intent_policy.shared.sim_env import RECORDED_CAMERAS, WRIST_CAMERA
    from intent_policy.sim.scene_builder_v2 import look_at_xyaxes
    assert len({t for t, _ in task_cameras}) == 10
    assert set(COLLECTION['cameras']) == set(RECORDED_CAMERAS) | {'wrist'}
    ref_key = next(iter(task_cameras)); ref = task_cameras[ref_key]
    for key, cams in task_cameras.items():
        for c, v in cams.items():
            for f in ('pos', 'quat', 'mat0', 'pos0'):
                assert np.allclose(v[f], ref[c][f], rtol=0, atol=1e-9), (key, c, f, v[f], ref_key, ref[c][f])
            assert v['fovy'] == ref[c]['fovy'] and v['body'] == ref[c]['body'], (key, c)
    for c, cfg in RECORDED_CAMERAS.items():                           # equal to configs/hrc/station.yaml
        v = ref[c]
        assert v['body'] == 'world' and abs(v['fovy'] - float(cfg['fovy'])) < 1e-9
        assert np.allclose(v['pos'], cfg['pos'], rtol=0, atol=1e-9)
        ax = np.array(look_at_xyaxes(cfg['pos'], cfg['target']).split(), float)
        x = ax[:3] / np.linalg.norm(ax[:3]); y = ax[3:] - x * (x @ ax[3:]); y /= np.linalg.norm(y)
        assert np.allclose(v['mat0'].reshape(3, 3), np.column_stack([x, y, np.cross(x, y)]), rtol=0, atol=1e-9)
        look = np.asarray(cfg['target'], float) - np.asarray(cfg['pos'], float)
        assert np.allclose(-v['mat0'].reshape(3, 3)[:, 2], look / np.linalg.norm(look), atol=1e-5)
    w = ref['wrist']                                                  # equal to configs/scene/common.yaml cameras.wrist
    assert w['body'] == WRIST_CAMERA['body'] and abs(w['fovy'] - float(WRIST_CAMERA['fovy'])) < 1e-9
    assert np.allclose(w['pos'], WRIST_CAMERA['pos'], rtol=0, atol=1e-9)
