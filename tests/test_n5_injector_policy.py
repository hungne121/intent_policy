"""14/QĐ-HL P3: N5 forecast injector in the policy rollout (HRCEnv). Short sim rollouts, val seed 300800 (never test seeds)."""
import numpy as np
import pytest

from intent_policy.envs.hrc_env import HRCEnv, HRCEnvConfig, CFG, POLICY_TRIGGERS
from intent_policy.shared.forecast_fault import FaultInjector, sample_fault
from tools.collect import task_cfg

SEED, TASK, FAULT, TICKS = 300800, 'C4', 'spatial_fault', 40


def hold(env, obs):
    """Fixed action: hold the current joint pose and gripper (same sequence for every env under comparison)."""
    return np.r_[obs['agent_pos'][:6], obs['agent_pos'][12]].astype(np.float32)


def rollout(group, fault, ticks=TICKS):
    env = HRCEnv(TASK, 'nominal', HRCEnvConfig(group=group, fault_type=fault))
    obs, _ = env.reset(seed=SEED)
    a0 = hold(env, obs); rows = []
    for _ in range(ticks):
        obs, *_ = env.step(a0)
        ep = env.ep
        rows.append(dict(kp=ep.kp_log[-1].copy(), q=ep.d.qpos[ep.env.qids].copy(), obs_f=obs['human_forecast'].copy(),
                         raw=None if ep.manicast.raw_forecast is None else ep.manicast.raw_forecast.copy(),
                         mc=None if ep.manicast.forecast is None else ep.manicast.forecast.copy(),
                         strength=ep.manicast.fault_strength))
    return env, rows


def test_config_trigger_declared():
    assert CFG['n5']['policy_trigger'] in POLICY_TRIGGERS


def test_injector_on_for_n5_off_for_n1_n6():
    env = HRCEnv(TASK, 'nominal', HRCEnvConfig(group='N5', fault_type=FAULT)); env.reset(seed=SEED)
    spec = sample_fault(FAULT, SEED, task_cfg('n5', TASK))          # same draw as tools/collect.py
    assert isinstance(env.ep.injector, FaultInjector) and env.ep.injector.spec == spec
    assert env.fault_case['seed'] == SEED and env.fault_case['fault_type'] == FAULT
    env.close()
    for g in ('N1', 'N6'):
        e = HRCEnv(TASK, 'nominal', HRCEnvConfig(group=g)); e.reset(seed=SEED)
        assert e.ep.injector is None and e.injector is None
        e.close()
    with pytest.raises(ValueError):
        HRCEnv(TASK, 'nominal', HRCEnvConfig(group='N5')).reset(seed=SEED)       # no hidden default fault type
    with pytest.raises(ValueError):
        HRCEnv(TASK, 'nominal', HRCEnvConfig(group='N1', fault_type=FAULT)).reset(seed=SEED)
    with pytest.raises(ValueError):
        HRCEnv(TASK, 'nominal', HRCEnvConfig(group='N5', fault_type=FAULT)).reset(seed=300900)   # test seed refused


def test_trigger_uses_only_past():
    cfg = task_cfg('n5', TASK)
    spec = sample_fault(FAULT, SEED, cfg)
    raw = np.random.default_rng(0).normal(size=(10, 6, 3)); cur = raw[0] * 0 + 0.1
    used_a = [False] * 3 + [True] * 40
    used_b = used_a[:10] + [False] * 33                           # differs only after tick 10
    outs = []
    for used in (used_a, used_b):
        inj = FaultInjector(spec, cfg); outs.append([inj.apply(raw, cur, 0.05 * i, u, 'r')[0] for i, u in enumerate(used)])
    for i in range(11):                                           # tick i sees used[i-1] at most: identical up to 10
        assert np.array_equal(outs[0][i], outs[1][i])
    assert inj.start is not None and spec.onset_index == 0


def test_only_forecast_changes_physics_and_human_identical():
    env_f, f = rollout('N5', FAULT)


    env_1, n = rollout('N1', None)                                # N1: same variation as N5
    inwin = [i for i, r in enumerate(f) if r['strength'] > 0]
    assert inwin, 'fault never active'
    for rf, rn in zip(f, n):
        assert np.array_equal(rf['kp'], rn['kp'])                 # human ground truth unchanged
        assert np.array_equal(rf['q'], rn['q'])                   # robot physics unchanged for equal actions
        assert np.array_equal(rf['raw'], rn['raw'])               # raw ManiCast unchanged
    for i, (rf, rn) in enumerate(zip(f, n)):
        dev = np.abs(rf['obs_f'] - rn['obs_f']).max()
        if rf['strength'] > 0:
            assert dev > 1e-3, (i, dev)                           # changed inside the window
        t_ = 0.05 * (i + 1)
        if not (env_f.injector.start - 0.06 <= t_ <= env_f.injector.end + 0.5):
            assert dev <= 1e-3, (i, dev)                          # untouched outside (v2 decays for a few ticks after)
        if rf['mc'] is not None and np.any(rf['obs_f']):
            assert np.allclose(rf['obs_f'], np.asarray(rf['mc'], np.float32).reshape(-1))  # policy tensor = effective forecast
    rec = env_f._finish(False)
    f_ = rec['fault']
    for k in ('task', 'fault_type', 'seed', 'group', 'trigger', 'fault_start', 'fault_end', 'fault_applied', 'active_ticks',
              'max_wrist_shift_1s_m', 'max_bone_dev_m', 'max_start_err_m', 'max_jump_m', 'violations', 'fault_parameters'):
        assert k in f_, k
    assert f_['active_ticks'] >= len(inwin) > 0
    assert f_['generator_version'] == 'v2_continuous_eval_2026-11-19'
    row = next(r for r in rec['fault_trace'] if r['strength'] > 0)
    for k in ('tick', 't', 'forecast_valid', 'used', 'strength', 'raw_wrist_1s', 'eff_wrist_1s', 'wrist_dev_1s_m'):
        assert k in row, k
    bound = env_f.hand_motion_bound
    assert bound == 0.08                                          # events.yaml metric.hand_still_max_m_s, not station 0.03
    assert all(r['used'] == (r['hand_speed'] > bound) for r in rec['fault_trace'])     # no forecast-valid condition
    assert all(r['forecast_valid'] for r in rec['fault_trace'] if r['strength'] > 0)   # never forced invalid
    assert all(r['checks_pass'] is not None for r in rec['fault_trace'] if r['strength'] > 0)
    assert abs(rec['fault']['use_onsets_s'][0] - (next(r['t'] for r in rec['fault_trace'] if r['used']) + 0.05)) < 1e-6   # injector reads the flag one tick later
