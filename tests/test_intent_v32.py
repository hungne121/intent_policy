"""Intent contract v3.2 (docs/requirements/INTENT_ACT_v3.2.md §5): device source, encoder, masked IntentACT."""
import json

import numpy as np
import pytest
import torch

from intent_policy.benchmark.runner import ExpertAgent, run_episode
from intent_policy.intent.contract import field_shapes_v32, validate_intent_v32
from intent_policy.intent.labels import load_schema
from intent_policy.policies.intent_encoder_v32 import TOKENS_V32, IntentEncoderV32
from intent_policy.scenarios.scenario_registry import make_scenario

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


class FakeDeePoint:
    """Stands in for the DeePoint server (intent/device.DeePointClient): fixed class probabilities and direction; logs
    the model selected per episode and the frames it saw."""
    n_classes = 6

    def __init__(self, probs=(0.05, 0.9, 0.01, 0.01, 0.02, 0.01)):
        self.probs, self.models, self.frames = list(probs), [], 0

    def reset(self, model=0):
        self.models.append(model)

    def step(self, rgb):
        self.frames += 1
        kp = [[0.0, 0.0, 0.0]] * 17
        kp[8], kp[10] = [150.0, 90.0, 1.0], [170.0, 80.0, 1.0]           # right elbow, wrist (COCO-17)
        return dict(probs=self.probs, direction=[0.3, 0.9, -0.3], kp=kp)


@pytest.fixture
def fake_device(tmp_path, monkeypatch):
    from intent_policy.intent import device
    folds = {str(k): dict(checkpoint=f'fold{k}.ckpt', sigma_deg={'point': 3.0, 'reach': 8.0, 'palm_up': 8.0},
                          hold_frames={'late': {'palm_forward': 2, 'withdraw': 7}, 'early': {'palm_forward': 2, 'withdraw': 8}},
                          pause_timeout_s=1.3) for k in range(5)}
    f = tmp_path / 'device_folds.json'
    f.write_text(json.dumps(dict(classes=S['device']['classes'], folds=folds)))
    client = FakeDeePoint()
    monkeypatch.setattr(device.DeePointClient, 'shared', classmethod(lambda cls, ckpts: client))
    return client, str(f)


def device_scenario(folds_file):
    return make_scenario('t1_pick_place', {
        'expert': {'trigger': 'device', 'grasp_settle_ticks': 0,
                   'device': {'folds_file': folds_file, 'width': 160, 'height': 90}},
        'scene': {'cameras': {'sensor_side': {'pos': [2.30, -0.48, 1.25], 'target': [0.30, -0.48, 0.85], 'fovy': 50}}}})


class Stop(Exception):
    pass


def test_recorded_labels_are_the_belief_the_expert_acted_on(fake_device):
    """Collection (scripts/collect_demos.py): the contract frames come from the expert's own DeviceBelief, so p_target
    equals the belief the expert decided on; the episode's fold model is selected and the pre-roll is sent."""
    from intent_policy.intent.folds import fold_of
    from intent_policy.intent.source_v32 import IntentV32
    client, folds_file = fake_device
    sc = device_scenario(folds_file)
    agent, src, out, beliefs = ExpertAgent(), [], [], []

    def on_frame(obs, decision, target, s):
        if not src:
            src.append(IntentV32(s, S, belief=agent.expert.belief))
            src[0].reset()
        out.append(src[0].step(np.asarray(obs['observation.state'], np.float64)))
        beliefs.append(decision.extras['expert_belief'])
        if len(out) == 40:
            raise Stop
    with pytest.raises(Stop):
        run_episode(sc, agent, 3, keep_trace=False, on_frame=on_frame)
    assert client.models[-1] == fold_of(sc.seed) and client.frames >= S['device']['preroll_frames']
    frames = {k: np.stack([f[k] for f in out]) for k in out[0]}
    validate_intent_v32(frames, S, batched=True)
    slot, T = sc.variation['spec']['layout'], S['targets']
    worst = max(abs(frames['p_target'][t, T.index(slot.get(k, k))] - v)
                for t, b in enumerate(beliefs) for k, v in {**b['objects'], **b['zones']}.items())
    assert worst < 0.01
    assert np.allclose(frames['p_gesture'][-1], np.asarray(client.probs) / sum(client.probs), atol=1e-5)
    sc.close()


def test_lock_opens_at_a_new_gesture(fake_device):
    """The decision lock holds for the rest of a gesture and opens when the device's decision turns from none to the
    gesture class again (a new gesture: the late T5 idx 484 timeout, 2026-10-12)."""
    from intent_policy.intent.device import DeviceBelief
    client, folds_file = fake_device
    sc = device_scenario(folds_file)
    sc.reset(seed=3)
    b = DeviceBelief(sc, S, {**S['device'], 'folds_file': folds_file, 'width': 160, 'height': 90})
    b.reset()
    key = ('point', tuple(sorted(b._pointing_candidates())))
    held = []
    for n, c in enumerate(('point', 'point', 'none', 'point')):
        client.probs = [0.9 if i == S['device']['classes'].index(c) else 0.02 for i in range(6)]
        for _ in range(2):                                      # 2 control frames: at least one device frame
            b.step()
        if n == 0:                                              # the hand stopped during the first pointing
            b.frozen[key] = {k: 1.0 / len(key[1]) for k in key[1]}
        held.append(key in b.frozen)
    assert held == [True, True, True, False]
    sc.close()


def test_gesture_complete_needs_the_gesture_and_a_still_hand(fake_device):
    """The late rule acts only once the directed gesture is complete: detected by the device while the hand is
    still at its pose (or its decision locked when the hand stopped)."""
    from intent_policy.intent.device import DeviceBelief
    client, folds_file = fake_device
    sc = device_scenario(folds_file)
    sc.reset(seed=3)
    b = DeviceBelief(sc, S, {**S['device'], 'folds_file': folds_file, 'width': 160, 'height': 90})
    b.reset()
    seen = []
    for c in ('none', 'point'):
        client.probs = [0.9 if i == S['device']['classes'].index(c) else 0.02 for i in range(6)]
        for _ in range(2):
            b.step()
        seen.append(b.complete('object'))
    assert seen == [False, True]                     # the human stands still: complete once pointing is detected
    b.moving = True
    assert not b.complete('object')                  # the hand still moving: not complete
    sc.close()


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
                       use_intent=True, intent_arch='tokens', intent_schema=S, intent_source='device', intent_version='v32',
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
