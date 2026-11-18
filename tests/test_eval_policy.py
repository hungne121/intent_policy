"""Closed-loop eval (12/QĐ-HL, 08/QĐ-KT): the env observation equals the recorded features, key mapping, seed guard."""
import numpy as np
import pytest

from intent_policy.data.lerobot_recorder import LeRobotRecorder
from intent_policy.envs import HRCEnv, policy_observation
from intent_policy.envs.hrc_env import CFG
from intent_policy.eval.runner import to_policy_batch


class _Capture(LeRobotRecorder):
    """LeRobotRecorder.add_frame unchanged; its LeRobot dataset replaced by a list, plus the env observation at the
    same moment (observation at t, before the physics step)."""

    def __init__(self):
        self.cameras, self.image_hw, self.frames, self.task_string = list(CFG['cameras']), tuple(CFG['image_hw']), 0, 'I2'
        self.pairs = []
        self.ds = self

    def add_frame(self, ep, action=None):          # called by Episode.step as recorder.add_frame(ep, command)
        if action is None:                         # LeRobot dataset stub
            self.pairs[-1].insert(0, ep); return
        self.pairs.append([policy_observation(ep, self.cameras, self.image_hw)])
        LeRobotRecorder.add_frame(self, ep, action)


def test_observation_equals_recorded_frame():
    from intent_policy.shared.episode import Episode
    from intent_policy.shared.manicast_adapter import load_predictor
    from intent_policy.experts import expert_class
    from intent_policy.scenarios import scenario
    sc = scenario('I2')
    scene, positions, xml, info = sc.build(300800)
    rec = _Capture()
    ep = Episode('I2', 'forecast', 300800, scene, positions, xml, load_predictor(), max_ticks=14, recorder=rec)
    expert = expert_class('I2')(ep)
    ep.run(sc.human(ep, info), expert.run(), done=lambda: sc.human_done(ep))
    ep.env.close()
    assert len(rec.pairs) == 14
    assert any(f['observation.forecast_valid'][0] == 1 for f, _ in rec.pairs), 'no valid forecast in 14 ticks'
    for frame, obs in rec.pairs:
        for c in CFG['cameras']:
            np.testing.assert_array_equal(obs['pixels'][c], frame[f'observation.images.{c}'])
        np.testing.assert_array_equal(obs['agent_pos'], frame['observation.state'])
        np.testing.assert_array_equal(obs['human_forecast'], frame['observation.human_forecast'])
        np.testing.assert_array_equal(obs['forecast_valid'], frame['observation.forecast_valid'])


def test_policy_batch_keys():
    h, w = CFG['image_hw']
    obs = {'pixels': {c: np.full((h, w, 3), 255, np.uint8) for c in CFG['cameras']},
           'agent_pos': np.arange(13, dtype=np.float32), 'human_forecast': np.zeros(180, np.float32),
           'forecast_valid': np.ones(1, np.float32)}
    b = to_policy_batch(obs, 'I2')
    for c in CFG['cameras']:
        assert b[f'observation.images.{c}'].shape == (1, 3, h, w) and float(b[f'observation.images.{c}'].max()) == 1.
    assert b['observation.state'].shape == (1, 13) and b['observation.human_forecast'].shape == (1, 180)
    assert b['observation.forecast_valid'].shape == (1, 1) and b['task'] == ['I2']


def test_test_split_refused():
    with pytest.raises(ValueError, match='split'):
        HRCEnv('I2').reset(seed=CFG['splits']['test'][0])
