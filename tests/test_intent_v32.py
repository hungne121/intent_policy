"""Intent contract v3.2 (docs/requirements/INTENT_ACT_v3.2.md §5): source, encoder, masked IntentACT."""
import numpy as np
import pytest
import torch

from intent_policy.benchmark.runner import ExpertAgent, run_episode
from intent_policy.intent.contract import field_shapes_v32, validate_intent_v32
from intent_policy.intent.labels import load_schema
from intent_policy.intent.source_v32 import label_episode
from intent_policy.policies.intent_encoder_v32 import TOKENS_V32, IntentEncoderV32
from intent_policy.scenarios.scenario_registry import make_scenario
from intent_policy.utils import resolve

S = load_schema()


def random_intent(B, seed=0, present=1.0):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, shape in field_shapes_v32(S).items():
        v = torch.rand((B,) + tuple(shape), generator=g)
        if k in ('p_who', 'p_target', 'c_who', 'c_target', 'phase', 'p_gesture'):
            v = v / v.sum(-1, keepdim=True)
        out[k] = v
    out['human_present'] = torch.full((B, 1), present)
    return out


def encoder(d=32):
    return IntentEncoderV32(d, len(S['targets']), len(S['who']), len(S['phases']), len(S['gestures']) + 1,
                            S['n_waypoints'], len(S['keypoints']), 10)


@pytest.fixture(scope='module')
def t1_belief_episode():
    """A T1 episode of the belief expert: per frame the keypoints, the human stage, the state and the expert's belief."""
    sc = make_scenario('t1_pick_place', {'expert': {'trigger': 'belief', 'grasp_settle_ticks': 0}})
    frames = []

    def on_frame(obs, decision, target, s):
        frames.append(dict(kp=s.human_keypoints(S['keypoints']), stage=s.human.stage,
                           state=np.asarray(obs['observation.state']), belief=decision.extras['expert_belief']))
    rec = run_episode(sc, ExpertAgent(), 3, keep_trace=False, on_frame=on_frame)
    stages, runs = [f['stage'] for f in frames], []
    for i, n in enumerate(stages):
        if runs and runs[-1]['name'] == n:
            runs[-1]['end'] = i
        else:
            runs.append(dict(name=n, start=i, end=i))
    meta = dict(variation=sc.variation, seed=sc.seed, task_code=sc.cfg.task_code, human_stages=runs)
    sc.close()
    return rec, frames, meta


def test_labels_replay_the_belief_the_expert_acted_on(t1_belief_episode):
    rec, frames, meta = t1_belief_episode
    assert rec['success']
    kp = np.stack([f['kp'] for f in frames])
    state = np.stack([f['state'] for f in frames])
    out = label_episode(meta, kp, state, S, S['belief'], resolve(S['belief']['reach']['library']))
    validate_intent_v32(out, S, batched=True)
    slot = meta['variation']['spec']['layout']
    T = S['targets']
    worst = 0.0
    for t, f in enumerate(frames):
        b = {**f['belief']['objects'], **f['belief']['zones']}
        for k, v in b.items():                                   # T1: p_target = the joint pointing belief
            worst = max(worst, abs(out['p_target'][t, T.index(slot.get(k, k))] - v))
    assert worst < 0.01


def test_encoder_masks_absent_human_and_dropped_tokens():
    enc = encoder()
    toks, mask = enc(random_intent(4))
    assert toks.shape == (4, 6, 32) and not mask.any()
    _, mask = enc(random_intent(4, present=0.0))
    assert mask[:, TOKENS_V32.index('human')].all() and mask[:, TOKENS_V32.index('motion')].all()
    assert not mask[:, TOKENS_V32.index('target')].any()
    keep = {t: torch.tensor([True, False, True, True]) for t in TOKENS_V32}
    _, mask = enc(random_intent(4), keep)
    assert mask[1].all() and not mask[0].any()


def test_masked_decoder_equals_lerobot_decoder_without_mask():
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTDecoder
    from intent_policy.policies.intent_act import decode_masked
    torch.manual_seed(0)
    dec = ACTDecoder(ACTConfig(dim_model=32, n_heads=4, dim_feedforward=64, n_decoder_layers=1)).eval()
    x, mem = torch.randn(5, 2, 32), torch.randn(7, 2, 32)
    dpos, epos = torch.randn(5, 1, 32), torch.randn(7, 1, 32)
    ref = dec(x, mem, decoder_pos_embed=dpos, encoder_pos_embed=epos)
    got = decode_masked(dec, x, mem, dpos, epos, torch.zeros(2, 7, dtype=torch.bool))
    assert torch.allclose(ref, got, atol=1e-6)


def test_fully_masked_intent_does_not_change_the_actions():
    """IntentACT v3.2: with every intent token masked, the actions do not depend on the intent values."""
    from intent_policy.policies.hri_act import HRIACTConfig
    from intent_policy.policies.intent_act import IntentACT
    from lerobot.configs.types import FeatureType, PolicyFeature
    cfg = HRIACTConfig(input_features={'observation.state': PolicyFeature(FeatureType.STATE, (10,)),
                                       'observation.environment_state': PolicyFeature(FeatureType.ENV, (4,))},
                       output_features={'action': PolicyFeature(FeatureType.ACTION, (7,))}, action_mode='continuous',
                       dim_model=32, n_heads=4, dim_feedforward=64, chunk_size=4, n_action_steps=1, latent_dim=8,
                       use_intent=True, intent_arch='tokens', intent_schema=S, intent_source='belief', intent_version='v32',
                       use_task_token=True, device='cpu', push_to_hub=False)
    torch.manual_seed(0)
    model = IntentACT(cfg, cfg.intent_token_cfg, cfg.n_tasks).eval()
    base = {'observation.state': torch.randn(2, 10), 'observation.environment_state': torch.randn(2, 4),
            'task_id': torch.tensor([0, 1])}
    none = {t: torch.zeros(2, dtype=torch.bool) for t in TOKENS_V32}
    a1, _ = model({**base, 'intent': random_intent(2, 1), 'intent_keep': none})
    a2, _ = model({**base, 'intent': random_intent(2, 2), 'intent_keep': none})
    b1, _ = model({**base, 'intent': random_intent(2, 1)})
    b2, _ = model({**base, 'intent': random_intent(2, 2)})
    assert torch.allclose(a1, a2, atol=1e-6)
    assert not torch.allclose(b1, b2, atol=1e-4)
