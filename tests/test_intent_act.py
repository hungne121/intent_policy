"""Intent tokens in LeRobot ACT (INTENT_ACT_GUIDE.md M2): baseline identity, token layout, dropout tokens, soft
labels, gradients, checkpoint round trip and the training-script path."""
import copy
import itertools

import numpy as np
import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.modeling_act import ACT

from intent_policy.intent.labels import load_schema
from intent_policy.policies.hri_act import HRIACTConfig, HRIACTPolicy
from intent_policy.policies.intent_act import IntentACT
from intent_policy.policies.intent_encoder import ALL_COMPONENTS, TYPE_ID, IntentEncoder

SCHEMA = load_schema()
M, J = SCHEMA['n_waypoints'], len(SCHEMA['keypoints'])
N_OBJ, N_ACT = len(SCHEMA['obj_vocab']), len(SCHEMA['act_vocab'])
SMALL = dict(chunk_size=6, n_action_steps=1, dim_model=32, n_heads=4, dim_feedforward=64, n_encoder_layers=1,
             n_decoder_layers=1, latent_dim=8, pretrained_backbone_weights=None)
B = 3


def config(mode='continuous', intent=True, **over) -> HRIACTConfig:
    kw = dict(SMALL, action_mode=mode, use_vae=(mode == 'continuous'), device='cpu', push_to_hub=False)
    if intent:
        kw.update(use_intent=True, intent_arch='tokens', intent_schema=SCHEMA)
    kw.update(over)
    return HRIACTConfig(input_features={'observation.state': PolicyFeature(FeatureType.STATE, (10,)),
                                        'observation.images.high': PolicyFeature(FeatureType.VISUAL, (3, 32, 32))},
                        output_features={'action': PolicyFeature(FeatureType.ACTION, (7,))}, **kw)


def batch(seed=0, intent=True) -> dict:
    g = torch.Generator().manual_seed(seed)
    b = {'observation.state': torch.randn(B, 10, generator=g), 'observation.images.high': torch.rand(B, 3, 32, 32, generator=g),
         'action': torch.randn(B, SMALL['chunk_size'], 7, generator=g),
         'action_is_pad': torch.zeros(B, SMALL['chunk_size'], dtype=torch.bool),
         'restricted_action': torch.randint(0, 9, (B, SMALL['chunk_size']), generator=g),
         'restricted_action_is_pad': torch.zeros(B, SMALL['chunk_size'], dtype=torch.bool)}
    if intent:
        b.update({'intent.obj': torch.randint(0, N_OBJ, (B,), generator=g), 'intent.act': torch.randint(0, N_ACT, (B,), generator=g),
                  'intent.phase': torch.rand(B, generator=g), 'intent.tte': 3 * torch.rand(B, generator=g),
                  'intent.xi': 0.1 * torch.randn(B, M, J, 3, generator=g)})
    return b


def intent_dict(seed=0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return dict(obj=torch.randint(0, N_OBJ, (B,), generator=g), act=torch.randint(0, N_ACT, (B,), generator=g),
                tau=torch.stack([torch.rand(B, generator=g), 3 * torch.rand(B, generator=g)], -1),
                xi=torch.randn(B, M, J, 3, generator=g))


def model_batch(cfg, seed=0) -> dict:
    b = batch(seed, intent=False)
    b['observation.images'] = [b['observation.images.high']]
    return b


def test_without_intent_the_model_is_lerobot_act():
    """intent_cfg=None: same state_dict keys / shapes and, with the same seed, the same parameters and outputs."""
    cfg = config(intent=False)
    torch.manual_seed(0)
    ref = ACT(cfg)
    torch.manual_seed(0)
    ours = IntentACT(cfg, None)
    a, b = ref.state_dict(), ours.state_dict()
    assert list(a) == list(b) and all(a[k].shape == b[k].shape and torch.equal(a[k], b[k]) for k in a)
    for train in (False, True):
        ref.train(train), ours.train(train)
        torch.manual_seed(1)
        out_ref = ref(model_batch(cfg))
        torch.manual_seed(1)
        out = ours(model_batch(cfg))
        assert torch.equal(out_ref[0], out[0])
    policy = HRIACTPolicy(config(intent=False))
    assert type(policy.model) is ACT and policy.intent_encoder is None          # the baseline policy is untouched


@pytest.mark.parametrize('mode', ['continuous', 'restricted'])
def test_every_component_subset_trains_and_infers(mode):
    for r in range(1, 5):
        for comps in itertools.combinations(ALL_COMPONENTS, r):
            policy = HRIACTPolicy(config(mode, intent_components=list(comps)))
            enc = policy.intent_encoder
            assert enc.components == comps and enc.n_tokens == sum(M if c == 'xi' else 1 for c in comps)
            n_1d = 2                                                            # latent + robot state
            assert policy.model.encoder_1d_feature_pos_embed.num_embeddings == n_1d + enc.n_tokens
            if mode == 'continuous':                                           # cls, state, intent, actions
                assert policy.model.vae_encoder_pos_enc.shape[1] == 2 + enc.n_tokens + SMALL['chunk_size']
            policy.train()
            loss, info = policy.forward(batch())
            assert torch.isfinite(loss)
            policy.eval()
            with torch.no_grad():
                out = policy.predict_action_chunk(batch()) if mode == 'continuous' else policy.predict_restricted_chunk(batch())
            assert out.shape == (B, SMALL['chunk_size'], 7 if mode == 'continuous' else 9) and torch.isfinite(out).all()


def test_cvae_without_intent_keeps_the_original_table():
    policy = HRIACTPolicy(config('continuous', intent_in_cvae=False))
    assert policy.model.vae_encoder_pos_enc.shape[1] == 2 + SMALL['chunk_size']
    policy.train()
    assert torch.isfinite(policy.forward(batch())[0])


def test_dropped_component_gives_its_unknown_token():
    torch.manual_seed(0)
    enc = IntentEncoder(16, N_OBJ, N_ACT, M, J)
    it = intent_dict()
    keep = {c: torch.ones(B, dtype=torch.bool) for c in ALL_COMPONENTS}
    full = enc(it, keep)
    for c in ALL_COMPONENTS:
        k = dict(keep, **{c: torch.tensor([False, True, False])})
        tok = enc(it, k)
        sl = slice(TYPE_ID[c], TYPE_ID[c] + 1) if c != 'xi' else slice(3, 3 + M)
        unknown = {'obj': enc.obj_emb.weight[N_OBJ][None], 'act': enc.act_emb.weight[N_ACT][None],
                   'tau': enc.null_tau, 'xi': enc.null_xi}[c] + enc.type_emb.weight[TYPE_ID[c]]
        for i in (0, 2):
            assert torch.allclose(tok[i, sl], unknown)
        assert torch.allclose(tok[1], full[1])                                    # kept rows unchanged
        others = [j for j in range(full.shape[1]) if j not in range(sl.start, sl.stop)]
        assert torch.allclose(tok[:, others], full[:, others])
    drop_all = enc.sample_keep(1000, 'cpu', 0.0, 1.0)
    assert not any(v.any() for v in drop_all.values())


def test_changing_the_human_action_changes_the_output():
    torch.manual_seed(0)
    policy = HRIACTPolicy(config('continuous')).eval()
    b = batch()
    with torch.no_grad():
        a = policy.predict_action_chunk(b)
        b2 = dict(b, **{'intent.act': (b['intent.act'] + 1) % N_ACT})
        a2 = policy.predict_action_chunk(b2)
    assert (a - a2).abs().amax(dim=(1, 2)).min() > 1e-6


def test_one_hot_float_equals_the_hard_label():
    torch.manual_seed(0)
    enc = IntentEncoder(16, N_OBJ, N_ACT, M, J)
    it = intent_dict()
    soft = dict(it, obj=torch.nn.functional.one_hot(it['obj'], N_OBJ).float(),
                act=torch.nn.functional.one_hot(it['act'], N_ACT).float())
    assert torch.allclose(enc(it), enc(soft), atol=1e-6)


@pytest.mark.parametrize('mode', ['continuous', 'restricted'])
def test_loss_reaches_the_intent_encoder(mode):
    torch.manual_seed(0)
    policy = HRIACTPolicy(config(mode, intent_p_drop=0.0, intent_p_drop_all=0.0)).train()
    loss, _ = policy.forward(batch())
    loss.backward()
    grads = {n: p.grad for n, p in policy.named_parameters() if n.startswith('model.intent_encoder.')}
    for name in ('obj_emb.weight', 'act_emb.weight', 'tau_mlp.0.weight', 'xi_proj.weight', 'xi_time', 'type_emb.weight'):
        g = grads[f'model.intent_encoder.{name}']
        assert g is not None and g.abs().sum() > 0, name


def test_component_dropout_only_in_training():
    policy = HRIACTPolicy(config('restricted', intent_p_drop=1.0))
    policy.train()
    assert not any(v.any() for v in policy._with_intent(batch())['intent_keep'].values())
    policy.eval()
    assert policy._with_intent(batch()).get('intent_keep') is None


def test_xi_is_normalised_with_the_training_statistics():
    policy = HRIACTPolicy(config('restricted'))
    mean, std = np.full((M, J, 3), 0.5), np.full((M, J, 3), 1e-4)
    policy.intent_encoder.set_xi_stats(mean, std)
    assert torch.allclose(policy.intent_encoder.xi_std, torch.full((M, J, 3), 1e-2))     # std clipped at 1e-2
    b = batch()
    xi = policy._with_intent(b)['intent']['xi']
    assert torch.allclose(xi, (b['intent.xi'] - 0.5) / 1e-2)


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    policy = HRIACTPolicy(config('continuous', intent_components=['obj', 'tau'], intent_film=True)).eval()
    policy.intent_encoder.set_xi_stats(np.ones((M, J, 3)), 2 * np.ones((M, J, 3)))
    policy.save_pretrained(tmp_path)
    loaded = HRIACTPolicy.from_pretrained(tmp_path).eval()
    assert loaded.config.intent_arch == 'tokens' and loaded.config.intent_components == ['obj', 'tau']
    assert loaded.config.intent_schema['obj_vocab'] == SCHEMA['obj_vocab'] and loaded.config.intent_film
    assert torch.equal(loaded.intent_encoder.xi_std, policy.intent_encoder.xi_std)
    with torch.no_grad():
        assert torch.allclose(policy.predict_action_chunk(batch()), loaded.predict_action_chunk(batch()))


def test_film_starts_as_identity():
    torch.manual_seed(0)
    a = HRIACTPolicy(config('restricted', intent_film=False)).eval()
    torch.manual_seed(0)
    b = HRIACTPolicy(config('restricted', intent_film=True)).eval()
    b.load_state_dict({**b.state_dict(), **a.state_dict()})
    with torch.no_grad():
        assert torch.allclose(a.predict_restricted_chunk(batch()), b.predict_restricted_chunk(batch()), atol=1e-6)


def test_config_validation():
    with pytest.raises(ValueError):
        config(intent_schema={})
    with pytest.raises(ValueError):
        config(intent_components=['obj', 'gaze'])
    with pytest.raises(ValueError):
        config(intent_mask=True)
    with pytest.raises(ValueError):
        config(intent_arch='late')


def test_training_script_path(labelled_dataset):
    """scripts/train_policy.py pieces on a labelled dataset: the same episodes train the no-intent and the intent
    policy; intent.* bypass the LeRobot preprocessor; xi statistics come from the training episodes."""
    import argparse
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from intent_policy.policies.hri_act import make_hri_act_pre_post_processors
    from scripts.train_policy import build_config, guide_overrides, run_batch, xi_stats

    root = labelled_dataset[0]
    meta = LeRobotDatasetMetadata('local/intent', root=root)
    base = dict(type='hri_act', action_mode='restricted', use_vae=False, **SMALL)
    args = argparse.Namespace(use_intent=True, intent_schema='configs/intent_schema.yaml', intent_components='obj,act,tau',
                              intent_in_cvae=None, p_drop=0.3, p_drop_all=None, action_mode='discrete', intent_film=None)
    over = guide_overrides(args)
    assert over['intent_components'] == ['obj', 'act', 'tau'] and over['intent_p_drop'] == 0.3
    assert over['action_mode'] == 'restricted' and 'intent_p_drop_all' not in over
    for policy_yaml in (base, dict(base, **over)):
        cfg = build_config(policy_yaml, meta, 'cpu', ['high', 'wrist'])
        assert not any(k.startswith(('intent.', 'human.')) for k in cfg.input_features)
        policy = HRIACTPolicy(cfg)
        pre, _ = make_hri_act_pre_post_processors(cfg, dataset_stats=meta.stats)
        delta = {k: [i / 20 for i in range(cfg.chunk_size)] for k in ('action', cfg.restricted_label_key)}
        ds = LeRobotDataset('local/intent', root=root, episodes=[0], delta_timestamps=delta)
        if policy.intent_encoder is not None:
            mean, std = xi_stats(ds)
            assert mean.shape == (M, J, 3) and (std > 0).any()
            policy.intent_encoder.set_xi_stats(mean, std)
        b = next(iter(torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)))
        policy.train()
        loss, info = run_batch(policy, pre, b, 'cpu', cfg.restricted_label_key)
        assert torch.isfinite(loss)
    bad = dict(base, **dict(over, intent_schema=dict(SCHEMA, obj_vocab=SCHEMA['obj_vocab'][:-1])))
    with pytest.raises(ValueError):
        build_config(bad, meta, 'cpu', ['high', 'wrist'])                     # labels built with another vocabulary
