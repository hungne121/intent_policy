"""M* (19/QĐ-HL): structured forecast encoder, KL isolation, decision-aware loss, decision windows, diagnostics."""
import numpy as np
import torch
import yaml

from intent_policy.policies.decision_windows import decision_mask, episode_windows
from intent_policy.policies.hri_act import HRIACTConfig, HRIACTPolicy
from intent_policy.policies.forecast_token import ur3e_tcp_chain
from test_step2_forecast_token import B, CAMS, SMALL, batch
from lerobot.configs.types import FeatureType, PolicyFeature
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'configs/policy/m_star.yaml').read_text())


def mstar(**over):
    inputs = {k: PolicyFeature(FeatureType.VISUAL, (3, 32, 32)) for k in CAMS}
    inputs['observation.state'] = PolicyFeature(FeatureType.STATE, (13,))
    kw = dict(SMALL, device='cpu', push_to_hub=False, use_forecast=True, forecast_fk_chain=ur3e_tcp_chain(), forecast_hidden_dim=16,
              forecast_arch='structured_transformer', forecast_in_cvae=False, forecast_n_layers=1,
              decision_loss=dict(enabled=True, alpha=2.0, event_windows={}))
    kw.update(over)
    torch.manual_seed(0)
    p = HRIACTPolicy(HRIACTConfig(input_features=inputs, output_features={'action': PolicyFeature(FeatureType.ACTION, (7,))}, **kw))
    p.set_forecast_stats([0.3, -0.2, 0.9], [0.2, 0.3, 0.1], np.zeros(6), np.ones(6))
    return p


def test_yaml_matches_config_fields():
    for k in CFG:
        assert k in ('type', 'train_data', 'diagnostics') or k in HRIACTConfig.__dataclass_fields__, k
    m1 = yaml.safe_load((ROOT / 'configs/policy/m1.yaml').read_text())
    same = ['dim_model', 'n_heads', 'dim_feedforward', 'n_encoder_layers', 'n_decoder_layers', 'kl_weight', 'chunk_size', 'optimizer_lr', 'train_data']
    assert all(CFG[k] == m1[k] for k in same)                         # M* vs M1: same ACT, budget, data


def test_structured_tokens_and_kl_isolation():
    p = mstar()
    fe = p.forecast_encoder
    assert fe.n_tokens == 60 and p.model.n_extra == 60 and not p.model.intent_in_cvae
    p.train()
    loss, info = p.forward(dict(batch(), decision_mask=torch.tensor([1, 0, 1])))
    assert torch.isfinite(loss)
    # KL alone must not reach the forecast encoder (forecast_in_cvae false)
    p.zero_grad()
    b = dict(batch())
    b['observation.forecast_valid'] = torch.ones(B, 1)
    p.model.training = True
    from lerobot.utils.constants import OBS_IMAGES
    b[OBS_IMAGES] = [b[k] for k in p.config.image_features]
    _, (mu, ls) = p.model(b)
    (-0.5 * (1 + ls - mu.pow(2) - ls.exp())).sum(-1).mean().backward()
    assert all(q.grad is None or q.grad.abs().max() == 0 for q in fe.parameters())
    # M1 setting: the KL does reach it
    p1 = mstar(forecast_arch='mlp_flatten', forecast_in_cvae=True)
    p1.train()
    b1 = dict(batch(), **{'observation.forecast_valid': torch.ones(B, 1)})
    b1[OBS_IMAGES] = [b1[k] for k in p1.config.image_features]
    _, (mu, ls) = p1.model(b1)
    (-0.5 * (1 + ls - mu.pow(2) - ls.exp())).sum(-1).mean().backward()
    assert any(q.grad is not None and q.grad.abs().max() > 0 for q in p1.forecast_encoder.parameters())


def test_decision_loss_alpha0_equals_lerobot_and_weights():
    p = mstar()
    p.train()
    b = batch(pad=2)
    torch.manual_seed(1)
    base, _ = p._forward_decision(dict(b, decision_mask=torch.tensor([1, 0, 1])), 0.0)
    torch.manual_seed(1)
    ref, _ = super(HRIACTPolicy, p).forward(dict(b))
    assert torch.allclose(base, ref, atol=1e-6)
    torch.manual_seed(1)
    up, info = p._forward_decision(dict(b, decision_mask=torch.tensor([1, 0, 1])), 2.0)
    assert info['decision_frac'] > 0 and not torch.allclose(up, base)


def test_gradient_reaches_forecast_encoder_and_diagnostics():
    p = mstar()
    p.train()
    b = dict(batch(), decision_mask=torch.tensor([1, 0, 1]))
    b['observation.forecast_valid'] = torch.ones(B, 1)
    loss, _ = p.forward(b)
    loss.backward()
    g = sum(float(q.grad.abs().sum()) for q in p.forecast_encoder.parameters() if q.grad is not None)
    assert g > 0
    d = p.forecast_diagnostics(b, torch.roll(torch.arange(B), 1))
    assert d['zh_var'] > 0 and 0 <= d['dead_unit_ratio'] <= 1 and d['attn_mass_forecast'] > 0
    assert d['attn_uniform_share'] < 1 and d['sens_shuffled'] > 0 and d['sens_masked'] > 0
    assert all(np.isfinite(v) for v in d.values())


def test_decision_windows():
    w = CFG['decision_loss']['event_windows']
    meta = {0: dict(task_name='I3', events=[dict(event='human_motion_onset', t=1.0), dict(event='switch_observed', t=4.0),
                                            dict(event='pointing_confirmed', t=6.0)], fault_start=None),
            1: dict(task_name='C4', events=[], fault_start=2.0, fault_end=3.0)}
    t = np.arange(0, 8, 0.5)
    ep = np.r_[np.zeros(len(t)), np.ones(len(t))].astype(int)
    m, share = decision_mask(ep, np.r_[t, t], meta, w)
    m0, m1 = m[:len(t)], m[len(t):]
    assert m0[t == 1.0][0] and m0[t == 2.0][0] and not m0[t == 2.5][0]            # onset [1, 2]
    assert m0[t == 3.0][0] and m0[t == 5.0][0]             # switch [3, 5]
    assert m0[t == 6.0][0] and not m0[t == 6.5][0] and not m0[t == 2.5][0]        # pre-commit [5, 6]
    assert m1[(t >= 2) & (t <= 4)].all() and not m1[t == 4.5][0]                  # N5 fault [2, 3 + 1]
    assert set(share) == {'I3', 'C4', 'all'}
