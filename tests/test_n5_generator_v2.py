"""16/QĐ-HL 7: generator v2 passes the locked validator (finite / bone / start / jump) on 100 % of emitted fault ticks,
both val seeds x the 3 fault types; v1 stays selectable; v2 never emits an invalid forecast (raises instead)."""
import numpy as np
import pytest

from intent_policy.envs.hrc_env import HRCEnv, HRCEnvConfig, CFG
from intent_policy.shared import forecast_fault as ff

GEN_V1, GEN_V2 = 'v1_collection_hrc-ds-0.5', 'v2_continuous_eval_2026-11-19'


def test_config_names():
    assert CFG['n5']['generator_version'] == GEN_V1 and CFG['n5']['policy_generator_version'] == GEN_V2
    assert set(ff.GENERATORS) == {GEN_V1, GEN_V2}
    with pytest.raises(ValueError):
        ff.FaultInjector(ff.sample_fault('spatial_fault', 300800, CFG['n5']), dict(CFG['n5'], generator_version='v3'))


@pytest.mark.parametrize('fault', ff.FAULT_TYPES)
@pytest.mark.parametrize('seed', [300800, 300801])
def test_v2_all_ticks_pass(fault, seed):
    env = HRCEnv('C4', 'nominal', HRCEnvConfig(group='N5', fault_type=fault))
    obs, _ = env.reset(seed=seed)
    a = np.r_[obs['agent_pos'][:6], obs['agent_pos'][12]].astype(np.float32)
    for _ in range(60):
        env.step(a)                                  # FaultConfigError would propagate here (never emit invalid)
    s = env.injector.summary()
    env.close()
    assert s['generator_version'] == GEN_V2
    assert s['active_ticks'] > 0 and s['violations'] == []
    assert s['max_bone_dev_m'] <= 0.01 and s['max_start_err_m'] <= 0.04 and s['max_jump_m'] <= 0.05
    if fault == 'spatial_fault':                     # severity kept: wrist shift at 1 s = amplitude x strength
        # any loss is reach-limited (clamped: the human arm cannot extend further, bones kept) or slew-limited (>= 0.9)
        assert s['min_wrist_shift_ratio'] >= (0.9 if s['clamped_ticks'] == 0 else s['min_wrist_shift_ratio_target'] - 1e-9)


def test_v2_raises_instead_of_invalid():
    cfg = dict(CFG['n5'], generator_version=GEN_V2)
    inj = ff.FaultInjector(ff.sample_fault('spatial_fault', 300800, cfg), cfg)
    raw = np.random.default_rng(1).normal(size=(10, 6, 3)); cur = raw[0].copy()     # incoherent forecast: bones inconsistent
    inj.start, inj.end = 0., 5.
    inj.bone_tol = -1.                                # a validator that cannot be met
    with pytest.raises(ff.FaultConfigError):
        inj.apply(raw, cur, 1., False, 'r')
