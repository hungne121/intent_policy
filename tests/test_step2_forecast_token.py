"""Step 2 (07/QĐ-HL D, F, G, H): UR3e FK, forecast preprocessing (shapes, TCP-relative, velocity, mask), the M1 token
in ACT, M1-noH vs M1 configs, per-sample L1 and task-balanced weights."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from lerobot.configs.types import FeatureType, PolicyFeature

from intent_policy.policies.forecast_token import (ForecastEncoder, TcpFK, forecast_features, forecast_position_stats,
                                                   ur3e_tcp_chain)
from intent_policy.policies.hri_act import HRIACTConfig, HRIACTPolicy

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / 'outputs/diag/preview_d1_1118/ds/hrc_forecast'
CAMS = ['observation.images.front_oblique', 'observation.images.overhead_oblique', 'observation.images.wrist']
SMALL = dict(chunk_size=6, n_action_steps=1, dim_model=32, n_heads=4, dim_feedforward=64, n_encoder_layers=1,
             n_decoder_layers=1, latent_dim=8, pretrained_backbone_weights=None, use_vae=True, action_mode='continuous')
B = 3


def test_fk_matches_mujoco():
    mujoco = pytest.importorskip('mujoco')
    st = yaml.safe_load((ROOT / 'configs/hrc/station.yaml').read_text())
    root = ET.parse(ROOT / 'assets/scene.xml').getroot()
    mount = root.find("./worldbody/body[@name='robot_mount']")             # = sim_env.apply_station
    mount.set('pos', f'{st["robot_base_xy"][0]} {st["robot_base_xy"][1]} 0.62')
    half = np.radians(float(st['robot_base_yaw_deg'])) / 2
    mount.set('quat', f'{np.cos(half):.7f} 0 0 {np.sin(half):.7f}')
    root.find('compiler').set('meshdir', str(ROOT / 'assets/meshes'))
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    d = mujoco.MjData(m)
    chain = ur3e_tcp_chain()
    fk = TcpFK(chain)
    adr = [m.joint(n).qposadr[0] for n in chain['joints']]
    rng = np.random.default_rng(0)
    q = rng.uniform(-np.pi, np.pi, (20, 6))
    ref = []
    for qi in q:
        d.qpos[adr] = qi
        mujoco.mj_kinematics(m, d)
        ref.append(d.site_xpos[m.site('tcp').id].copy())
    err = np.linalg.norm(fk(torch.as_tensor(q)).numpy() - np.array(ref), axis=1)
    assert err.max() < 1e-6


@pytest.mark.skipif(not PREVIEW.exists(), reason='preview dataset not on this machine')
def test_fk_matches_recorded_tcp():
    pq = pytest.importorskip('pyarrow.parquet')
    t = pq.read_table(sorted(PREVIEW.glob('data/*/*.parquet'))[0], columns=['observation.state', 'observation.tcp_pose'])
    st = np.stack(t.column('observation.state').to_numpy(zero_copy_only=False))[::50]
    tcp = np.stack(t.column('observation.tcp_pose').to_numpy(zero_copy_only=False))[::50, :3]
    err = np.linalg.norm(TcpFK(ur3e_tcp_chain())(torch.as_tensor(st[:, :6], dtype=torch.float64)).numpy() - tcp, axis=1)
    assert err.max() < 1e-3                                                # < 1 mm (float32 storage)


def test_forecast_features_shapes_rel_velocity_mask():
    g = torch.Generator().manual_seed(0)
    p0, v = torch.randn(B, 1, 6, 3, generator=g), torch.randn(B, 1, 6, 3, generator=g)
    k = torch.arange(10, dtype=torch.float32)[None, :, None, None]
    fc = p0 + (k + 1) * 0.1 * v                                            # linear forecast, horizons 0.1 ... 1.0 s
    tcp = torch.randn(B, 3, generator=g)
    valid = torch.tensor([1., 1., 0.])
    x = forecast_features(fc, valid, tcp, torch.zeros(3), torch.ones(3), 0.1)
    assert x.shape == (B, 10, 6, 9)
    assert torch.allclose(x[:2, ..., :3], fc[:2])                          # Ĥ
    assert torch.allclose(x[:2, ..., 3:6], fc[:2] - tcp[:2, None, None])   # Ĥ - p_TCP
    assert torch.allclose(x[:2, ..., 6:], v[:2].expand(-1, 10, -1, -1), atol=1e-5)   # dĤ/dt, incl. the one-sided v_0
    assert (x[2] == 0).all()                                               # invalid -> zeros
    mean, std = torch.tensor([0.3, -0.2, 0.9]), torch.tensor([0.2, 0.3, 0.1])
    xn = forecast_features(fc, valid, tcp, mean, std, 0.1)
    assert torch.allclose(xn[:2, ..., 3:6], (fc[:2] - tcp[:2, None, None]) / std, atol=1e-5)   # normalise, then relative
    assert torch.allclose(xn[:2, ..., 6:], v[:2].expand(-1, 10, -1, -1) / std, atol=1e-4)


def enc_cfg():
    return dict(key='observation.human_forecast', valid_key='observation.forecast_valid', horizons=10, joints=6, dt_s=0.1,
                hidden_dim=16, in_cvae=True, fk_chain=ur3e_tcp_chain())


def test_encoder_mask_flag_and_qpos():
    enc = ForecastEncoder(8, enc_cfg())
    with pytest.raises(RuntimeError):
        enc({'observation.state': torch.zeros(1, 13), 'observation.human_forecast': torch.zeros(1, 180),
             'observation.forecast_valid': torch.ones(1, 1)})
    qm, qs = np.linspace(-1, 1, 6), np.linspace(0.1, 0.6, 6)
    enc.set_stats([0.3, -0.2, 0.9], [0.2, 0.3, 0.1], qm, qs)
    q = torch.randn(4, 6)
    state = torch.cat([(q - torch.as_tensor(qm, dtype=torch.float32)) / torch.as_tensor(qs, dtype=torch.float32), torch.zeros(4, 7)], 1)
    assert torch.allclose(enc.qpos(state), q, atol=1e-5)
    b = {'observation.state': state[:2], 'observation.human_forecast': torch.randn(2, 180),
         'observation.forecast_valid': torch.zeros(2, 1)}
    z = enc(b)
    assert z.shape == (2, 1, 8) and torch.allclose(z[0], z[1])            # invalid: content ignored, same token
    b['observation.forecast_valid'] = torch.ones(2, 1)
    assert not torch.allclose(enc(b)[0], z[0])
    mu, sd = forecast_position_stats(np.r_[np.ones((1, 180)), np.zeros((1, 180))], np.array([1, 0]))
    assert np.allclose(mu, 1) and np.allclose(sd, 0)


def config(forecast: bool) -> HRIACTConfig:
    inputs = {k: PolicyFeature(FeatureType.VISUAL, (3, 32, 32)) for k in CAMS}
    inputs['observation.state'] = PolicyFeature(FeatureType.STATE, (13,))
    kw = dict(SMALL, device='cpu', push_to_hub=False)
    if forecast:
        kw.update(use_forecast=True, forecast_fk_chain=ur3e_tcp_chain(), forecast_hidden_dim=16)
    return HRIACTConfig(input_features=inputs, output_features={'action': PolicyFeature(FeatureType.ACTION, (7,))}, **kw)


def batch(seed=0, pad=0):
    g = torch.Generator().manual_seed(seed)
    b = {k: torch.rand(B, 3, 32, 32, generator=g) for k in CAMS}
    b.update({'observation.state': torch.randn(B, 13, generator=g), 'action': torch.randn(B, 6, 7, generator=g),
              'action_is_pad': torch.zeros(B, 6, dtype=torch.bool),
              'observation.human_forecast': torch.randn(B, 180, generator=g),
              'observation.forecast_valid': torch.tensor([[1.], [0.], [1.]])})
    if pad:
        b['action_is_pad'][0, -pad:] = True
    return b


def make(forecast):
    torch.manual_seed(0)
    p = HRIACTPolicy(config(forecast))
    if forecast:
        p.set_forecast_stats([0.3, -0.2, 0.9], [0.2, 0.3, 0.1], np.zeros(6), np.ones(6))
    return p


def test_m1_token_params_forward_and_select_action():
    noh, m1 = make(False), make(True)
    n = lambda p: sum(x.numel() for x in p.parameters())
    n_fc = sum(x.numel() for x in m1.forecast_encoder.parameters())
    assert n(m1) - n(noh) == n_fc + SMALL['dim_model']                    # + MLP + one 1-D position embedding row
    assert m1.model.n_extra == 1 and m1.model.intent_in_cvae
    m1.train()
    loss, info = m1.forward(batch())
    loss.backward()
    assert m1.forecast_encoder.mlp[0].weight.grad.abs().sum() > 0
    assert set(info) == {'l1_loss', 'kld_loss'}
    m1.eval()
    m1.reset()
    a = m1.select_action({k: v for k, v in batch().items() if not k.startswith('action')})
    assert a.shape == (B, 7)
    b1, b2 = batch(1), batch(1)
    b2['observation.human_forecast'] = b2['observation.human_forecast'] + 1.0
    with torch.no_grad():
        c1, c2 = m1.predict_action_chunk(b1), m1.predict_action_chunk(b2)
    assert not torch.allclose(c1[0], c2[0]) and torch.allclose(c1[1], c2[1])   # sample 1 invalid: forecast ignored


def test_per_sample_l1_is_lerobot_l1():
    p = make(False)
    p.eval()
    b = batch(2)
    loss, info = p.forward(b)
    assert abs(float(p.per_sample_l1(b).mean()) - info['l1_loss']) < 1e-5
    b = batch(2, pad=3)
    p.forward(b)
    per = p.per_sample_l1(b)
    err = (b['action'][0, :3] - p._actions_hat[0, :3]).abs().mean()
    assert torch.allclose(per[0], err, atol=1e-6)                          # padded steps masked out


def test_configs_differ_only_where_allowed():
    y = {n: yaml.safe_load((ROOT / f'configs/policy/{n}.yaml').read_text()) for n in ('m0', 'm1_noh', 'm1', 'act_continuous')}
    assert {k for k in set(y['m0']) | set(y['m1_noh']) if y['m0'].get(k) != y['m1_noh'].get(k)} == {'train_data'}
    assert y['m0']['train_data']['dataset'] == 'D_observed' and y['m1_noh']['train_data']['dataset'] == 'D_forecast'
    diff = {k for k in set(y['m1_noh']) | set(y['m1']) if y['m1_noh'].get(k) != y['m1'].get(k)}
    assert diff and all(k.startswith('forecast_') or k == 'use_forecast' for k in diff)
    base = {k: v for k, v in y['act_continuous'].items() if k != 'use_intent'}
    assert all(y['m0'][k] == v for k, v in base.items())                  # the repo's ACT baseline, unchanged


def test_task_balanced_weights():
    from tools.train_policy import frame_weights
    meta = {0: dict(task_name='I1', group='N1'), 1: dict(task_name='I1', group='N6'), 2: dict(task_name='C1', group='N1')}
    ep = np.r_[np.zeros(100), np.ones(300), np.full(50, 2)].astype(int)
    w = frame_weights(ep, meta, 'task_balanced', None)
    assert np.isclose(w[ep <= 1].sum(), 0.5) and np.isclose(w[ep == 2].sum(), 0.5)
    assert np.isclose(w[0], w[150])                                        # uniform within a task
    w2 = frame_weights(ep, meta, 'task_balanced', {'N6': 2.0})
    assert w2[150] > w2[0]


def test_grade_fault_checks_never_a(tmp_path):
    from tools.build_train_set import episode_grade
    eid = 'N5-C3-nominal-direction_fault-300000-forecast'
    d = tmp_path / 'rejected_episodes' / eid
    d.mkdir(parents=True)
    summary = dict(success=True, robot_limits_ok=True, contact_classes={},
                   reject_reasons=['physics_valid=false', "fault_checks ['jump 0.063'] at t=1.05"])
    (d / 'summary.json').write_text(json.dumps(summary))
    assert episode_grade(tmp_path, dict(episode_id=eid, quality_grade='C')) == 'C'
    summary['reject_reasons'] = ['physics_valid=false']                   # unobservable-only -> re-derived A
    (d / 'summary.json').write_text(json.dumps(summary))
    assert episode_grade(tmp_path, dict(episode_id=eid, quality_grade='C')) == 'A'
    assert episode_grade(tmp_path, dict(episode_id=eid, quality_grade='C',
                                        reject_reasons=["fault_checks ['jump 0.155'] at t=0.85"])) == 'C'
