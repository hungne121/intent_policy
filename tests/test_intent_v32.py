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


def image_model(**over):
    """A small v3.2 IntentACT with one camera (no pretrained backbone download)."""
    from intent_policy.policies.hri_act import HRIACTConfig
    from intent_policy.policies.intent_act import IntentACT
    from lerobot.configs.types import FeatureType, PolicyFeature
    cfg = HRIACTConfig(input_features={'observation.state': PolicyFeature(FeatureType.STATE, (10,)),
                                       'observation.images.top': PolicyFeature(FeatureType.VISUAL, (3, 64, 64))},
                       output_features={'action': PolicyFeature(FeatureType.ACTION, (7,))}, action_mode='continuous',
                       dim_model=32, n_heads=4, dim_feedforward=64, chunk_size=4, n_action_steps=1, latent_dim=8,
                       pretrained_backbone_weights=None, use_vae=False, use_intent=True, intent_arch='tokens',
                       intent_schema=S, intent_source='device', intent_version='v32', use_task_token=True,
                       device='cpu', push_to_hub=False, **over)
    return IntentACT(cfg, cfg.intent_token_cfg, cfg.n_tasks)


def image_batch(seed=0, intent_seed=1):
    g = torch.Generator().manual_seed(seed)
    return {'observation.state': torch.randn(2, 10, generator=g), 'observation.images': [torch.rand(2, 3, 64, 64, generator=g)],
            'task_id': torch.tensor([0, 1]), 'intent': random_intent(2, intent_seed)}


def test_v32_film_starts_as_identity_and_then_conditions_every_stage():
    torch.manual_seed(0)
    plain = image_model().eval()
    torch.manual_seed(0)
    film = image_model(intent_film=True).eval()
    assert set(film.intent_film_v32.heads) == {'layer1', 'layer2', 'layer3', 'layer4'}
    film.load_state_dict({**film.state_dict(), **plain.state_dict()})
    b = image_batch()
    assert torch.allclose(plain(b)[0], film(b)[0], atol=1e-6)
    for h in film.intent_film_v32.heads.values():                  # a trained FiLM: the intent changes the image path
        torch.nn.init.normal_(h.weight, std=0.1)
    film.encoder_1d_feature_pos_embed.weight.data.zero_()
    none = {t: torch.zeros(2, dtype=torch.bool) for t in TOKENS_V32}
    a1, a2 = film({**image_batch(intent_seed=1), 'intent_keep': none})[0], film({**image_batch(intent_seed=2), 'intent_keep': none})[0]
    assert torch.allclose(a1, a2, atol=1e-6)                       # masked intent: FiLM sees zeros either way
    assert not torch.allclose(film(image_batch(intent_seed=1))[0], film(image_batch(intent_seed=2))[0], atol=1e-4)


def test_image_dropout_masks_the_cameras_in_training_only():
    torch.manual_seed(0)
    model = image_model(image_p_drop=0.999999)
    for m in model.modules():                                      # no random dropout: compare two training passes
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
        elif isinstance(m, torch.nn.MultiheadAttention):
            m.dropout = 0.0
    model.train()
    b1, b2 = image_batch(), image_batch()
    b2['observation.images'] = [torch.rand(2, 3, 64, 64)]
    assert torch.allclose(model(b1)[0], model(b2)[0], atol=1e-6)   # every image token masked: images do not matter
    model.eval()
    assert not torch.allclose(model(b1)[0], model(b2)[0], atol=1e-4)
    with pytest.raises(ValueError):
        image_model(image_p_drop=1.0)


def test_swap_test_partners_and_classes():
    """§5.3 swap test: the alternative is the twin (T1 / T2) or the other cup (T3); p_target mass is conserved; the
    first commit per decision is classified against the human's target and the alternative."""
    from intent_policy.benchmark.counterfactual import alternative_partners, permute_targets, slot_partners
    from scripts.intent_swap_test import decisions
    t1 = {'task': 'T1', 'target': 'B1', 'layout': {'B1': 'S2', 'B1p': 'S3', 'B3': 'S1'}, 'place': 'P1'}
    assert alternative_partners(t1)['B1'] == 'B1p' and slot_partners(t1)['S2'] == 'S3'
    t3 = {'task': 'T3', 'target': 'C1', 'layout': {'C1': 'S1', 'C3': 'S5'}, 'u_blocks': ['B1', 'B2']}
    assert slot_partners(t3) ['S1'] == 'S5' and alternative_partners({'task': 'T4'}) == {}
    T = S['targets']
    p = np.zeros(len(T), np.float32)
    p[T.index('S2')], p[T.index('P1')] = 0.7, 0.3
    q = permute_targets(p, slot_partners(t1), T)
    assert q[T.index('S3')] == np.float32(0.7) and q[T.index('P2')] == np.float32(0.3) and np.isclose(q.sum(), 1.0)
    ev = lambda *ts: [dict(event_type='robot_target_commit', payload=dict(target=t)) for t in ts]
    rec = dict(spec=t1, list_task='T1', events=ev('B1p', 'B1', 'P1'))
    assert decisions(rec) == {'object': 'alt', 'place': 'human'}
    assert decisions(dict(rec, events=ev('B3'))) == {'object': 'other', 'place': 'none'}
    assert decisions(dict(spec=t3, list_task='T3', events=ev('C3'))) == {'object': 'alt'}


def test_f1_target_init_only_rescales_the_target_embeddings():
    """§5.3 F1: 'type' draws the same random numbers as D0 and only rescales E_target / committed to N(0,1)."""
    torch.manual_seed(0)
    small = image_model().state_dict()
    torch.manual_seed(0)
    unit = image_model(intent_target_init='type').state_dict()
    for k in small:
        if k.endswith(('E_target', 'committed')):
            assert torch.allclose(unit[k], small[k] / 0.02, atol=1e-5)
        else:
            assert torch.equal(unit[k], small[k]), k
    with pytest.raises(ValueError):
        image_model(intent_target_init='big')


def test_unified_token_reads_only_the_device_output():
    """Trial 2026-10-18: intent_repr 'unified' = ONE token u_t = [(1 - hold) p_target, hold] (hold = p(palm_forward) +
    p(withdraw)); the other contract fields (simulator keypoints, forecasts) do not change the actions; masked, the
    intent does not matter; FiLM + image dropout work with it."""
    torch.manual_seed(0)
    model = image_model(intent_repr='unified', intent_film=True, image_p_drop=0.1).eval()
    assert model.intent_encoder.n_tokens == 1
    for h in model.intent_film_v32.heads.values():
        torch.nn.init.normal_(h.weight, std=0.1)
    b1, b2 = image_batch(intent_seed=1), image_batch(intent_seed=1)
    for k in b2['intent']:                                         # change every field except p_target / p_gesture
        if k not in ('p_target', 'p_gesture'):
            b2['intent'][k] = b2['intent'][k] + 1.0
    assert torch.allclose(model(b1)[0], model(b2)[0], atol=1e-6)
    b3 = image_batch(intent_seed=2)
    assert not torch.allclose(model(b1)[0], model(b3)[0], atol=1e-4)
    none = {t: torch.zeros(2, dtype=torch.bool) for t in TOKENS_V32}
    assert torch.allclose(model({**b1, 'intent_keep': none})[0], model({**b3, 'intent_keep': none})[0], atol=1e-6)
    u = model.intent_encoder.unified(b1['intent'])
    assert u.shape[-1] == len(S['targets']) + 1 and torch.allclose(u.sum(-1), torch.ones(2), atol=1e-5)
