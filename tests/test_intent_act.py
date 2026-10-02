"""Task and intent tokens in LeRobot ACT (INTENT_ACT_GUIDE_v2.md §4.6): baseline identity, task token, contract
inputs and gradients, group dropout tokens, target swap, ensembling reset; plus checkpoints, evaluation-time group
keep, xi normalisation and the training-script path."""
import argparse

import numpy as np
import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.modeling_act import ACT

from intent_policy.intent.contract import field_shapes, one_hot, validate_intent
from intent_policy.intent.labels import load_schema
from intent_policy.policies.hri_act import HRIACTConfig, HRIACTPolicy
from intent_policy.policies.intent_act import IntentACT
from intent_policy.policies.intent_encoder import GROUP_OF, GROUPS, TOKENS, IntentEncoder

S = load_schema()
K, W, P, M, J = len(S['targets']), len(S['who']), len(S['phases']), S['n_waypoints'], len(S['keypoints'])
SMALL = dict(chunk_size=6, n_action_steps=1, dim_model=32, n_heads=4, dim_feedforward=64, n_encoder_layers=1,
             n_decoder_layers=1, latent_dim=8, pretrained_backbone_weights=None)
B = 3


def config(mode='restricted', intent=True, task=True, **over) -> HRIACTConfig:
    kw = dict(SMALL, action_mode=mode, use_vae=(mode == 'continuous'), device='cpu', push_to_hub=False)
    if intent or task:
        kw['intent_schema'] = S
    if intent:
        kw.update(use_intent=True, intent_arch='tokens')
    kw['use_task_token'] = task
    kw.update(over)
    return HRIACTConfig(input_features={'observation.state': PolicyFeature(FeatureType.STATE, (10,)),
                                        'observation.images.high': PolicyFeature(FeatureType.VISUAL, (3, 32, 32))},
                        output_features={'action': PolicyFeature(FeatureType.ACTION, (7,))}, **kw)


def dist(g, n, rows=B):
    return torch.softmax(torch.randn(rows, n, generator=g), -1)


def contract(seed=0) -> dict:
    """A random batch of valid contract fields (dataset layout: intent.<field>)."""
    g = torch.Generator().manual_seed(seed)
    d = dict(p_who=dist(g, W), c_who=dist(g, W), p_target=dist(g, K), c_target=dist(g, K),
             occupancy=torch.cat([torch.randn(B, 3, generator=g), torch.rand(B, 4, generator=g)], -1),
             tte=3 * torch.rand(B, 1, generator=g), tte_std=torch.rand(B, 1, generator=g), phase=dist(g, P),
             confidence=torch.rand(B, 2, generator=g), xi=0.1 * torch.randn(B, M, J, 3, generator=g))
    return {f'intent.{k}': v for k, v in d.items()}


def batch(seed=0, intent=True, task=True) -> dict:
    g = torch.Generator().manual_seed(seed)
    b = {'observation.state': torch.randn(B, 10, generator=g), 'observation.images.high': torch.rand(B, 3, 32, 32, generator=g),
         'action': torch.randn(B, SMALL['chunk_size'], 7, generator=g),
         'action_is_pad': torch.zeros(B, SMALL['chunk_size'], dtype=torch.bool),
         'restricted_action': torch.randint(0, 9, (B, SMALL['chunk_size']), generator=g),
         'restricted_action_is_pad': torch.zeros(B, SMALL['chunk_size'], dtype=torch.bool)}
    if task:
        b['hri.task_id'] = torch.randint(0, 4, (B,), generator=g)
    if intent:
        b.update(contract(seed))
    return b


def chunk(policy, b):
    with torch.no_grad():
        return policy.predict_restricted_chunk(b) if policy.config.action_mode == 'restricted' else policy.predict_action_chunk(b)


def test_1_without_task_and_intent_the_model_is_lerobot_act():
    cfg = config('continuous', intent=False, task=False)
    torch.manual_seed(0)
    ref = ACT(cfg)
    torch.manual_seed(0)
    ours = IntentACT(cfg, None, 0)
    a, b = ref.state_dict(), ours.state_dict()
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    mb = batch(intent=False, task=False)
    mb['observation.images'] = [mb['observation.images.high']]
    for train in (False, True):
        ref.train(train), ours.train(train)
        torch.manual_seed(1)
        x = ref(mb)[0]
        torch.manual_seed(1)
        assert torch.equal(x, ours(mb)[0])
    policy = HRIACTPolicy(config(intent=False, task=False))
    assert type(policy.model) is ACT and policy.intent_encoder is None


@pytest.mark.parametrize('mode', ['restricted', 'continuous'])
def test_2_task_token_only(mode):
    torch.manual_seed(0)
    policy = HRIACTPolicy(config(mode, intent=False, task=True))
    assert policy.model.use_task and not policy.model.use_intent and policy.model.n_extra == 1
    policy.train()
    assert torch.isfinite(policy.forward(batch(intent=False))[0])
    policy.eval()
    b = batch(intent=False)
    b2 = dict(b, **{'hri.task_id': (b['hri.task_id'] + 1) % 4})
    assert (chunk(policy, b) - chunk(policy, b2)).abs().amax(dim=(1, 2)).min() > 1e-6


@pytest.mark.parametrize('mode', ['restricted', 'continuous'])
def test_3_intent_contract_loss_and_gradients(mode):
    torch.manual_seed(0)
    policy = HRIACTPolicy(config(mode, intent_p_drop_group=0.0, intent_p_drop_all=0.0))
    enc = policy.intent_encoder
    assert policy.model.n_extra == 1 + enc.n_tokens == 1 + 5 + M
    assert policy.model.encoder_1d_feature_pos_embed.num_embeddings == 2 + 1 + 5 + M
    if mode == 'continuous':
        assert policy.model.vae_encoder_pos_enc.shape[1] == 2 + 1 + 5 + M + SMALL['chunk_size']
    b = batch()
    validate_intent({k.removeprefix('intent.'): v.numpy().reshape((B,) + field_shapes(S)[k.removeprefix('intent.')])
                     for k, v in b.items() if k.startswith('intent.')}, S, batched=True)
    policy.train()
    loss, _ = policy.forward(b)
    assert torch.isfinite(loss)
    loss.backward()
    for name in ('who_proj.weight', 'E_target', 'committed', 'occ_proj.weight', 'time_mlp.0.weight', 'xi_proj.weight', 'xi_time',
                 'type_emb.weight'):
        g = dict(policy.named_parameters())[f'model.intent_encoder.{name}'].grad
        assert g is not None and g.abs().sum() > 0, name
    assert dict(policy.named_parameters())['model.task_emb.weight'].grad.abs().sum() > 0
    policy.eval()
    out = chunk(policy, b)
    assert out.shape == (B, SMALL['chunk_size'], 9 if mode == 'restricted' else 7) and torch.isfinite(out).all()


def test_4_dropped_group_gives_null_tokens():
    torch.manual_seed(0)
    enc = IntentEncoder(16, K, W, P, M, J)
    it = {k.removeprefix('intent.'): v for k, v in contract().items()}
    full = enc(it)
    pos = {name: [i] for i, name in enumerate(TOKENS)} | {'xi': list(range(len(TOKENS), len(TOKENS) + M))}
    for g in GROUPS:
        keep = {gr: torch.ones(B, dtype=torch.bool) for gr in GROUPS}
        keep[g] = torch.tensor([False, True, False])
        tok = enc(it, keep)
        dropped = [n for n, gr in GROUP_OF.items() if gr == g]
        type_e = enc.type_emb.weight[GROUPS.index(g)]
        for n in dropped:
            null = enc.null_xi if n == 'xi' else enc.null[n]
            for i in (0, 2):
                assert torch.allclose(tok[i, pos[n]], null + type_e)
        other = [j for n, idx in pos.items() if n not in dropped for j in idx]
        assert torch.allclose(tok[:, other], full[:, other]) and torch.allclose(tok[1], full[1])
    assert not any(v.any() for v in enc.sample_keep(500, 'cpu', 0.0, 1.0).values())


def test_5_swapping_the_target_changes_the_output():
    torch.manual_seed(0)
    policy = HRIACTPolicy(config('restricted')).eval()
    b = batch()
    s3, s1 = one_hot(S['targets'].index('S3'), K), one_hot(S['targets'].index('S1'), K)
    a = chunk(policy, dict(b, **{'intent.p_target': torch.as_tensor(s3).expand(B, -1)}))
    a2 = chunk(policy, dict(b, **{'intent.p_target': torch.as_tensor(s1).expand(B, -1)}))
    assert (a - a2).abs().amax(dim=(1, 2)).min() > 1e-6
    soft = dict(b, **{'intent.p_target': torch.as_tensor(s3).expand(B, -1).clone()})     # one-hot = soft encoding
    assert torch.allclose(policy.intent_encoder(policy._with_intent(soft)['intent'])[:, 1],
                          policy.intent_encoder(policy._with_intent(dict(b, **{'intent.p_target': torch.as_tensor(s3).expand(B, -1)}))['intent'])[:, 1])


@pytest.mark.parametrize('mode', ['restricted', 'continuous'])
def test_6_committed_target_change_resets_ensembling(mode):
    torch.manual_seed(0)
    policy = HRIACTPolicy(config(mode))
    policy.set_temporal_ensemble(-0.1)
    one = lambda b: {k: (v[:1] if torch.is_tensor(v) else v) for k, v in b.items()}
    b = one(batch())
    b['intent.confidence'] = torch.tensor([[0.5, 0.5]])
    select = policy.select_restricted_action if mode == 'restricted' else policy.select_action
    ens = lambda: policy._restricted_ensembler if mode == 'restricted' else policy.temporal_ensembler
    for _ in range(4):
        select(b)
    assert int(ens().ensembled_actions_count.max()) > 1 and policy.ensembler_resets == 0
    b['intent.c_target'] = torch.as_tensor(one_hot(S['targets'].index('P1'), K))[None]
    select(b)
    assert policy.ensembler_resets == 1 and int(ens().ensembled_actions_count.max()) == 1
    select(b)
    b['intent.confidence'] = torch.tensor([[0.9, 0.5]])                                  # confidence crosses 0.8
    select(b)
    assert policy.ensembler_resets == 2
    select(b)
    assert policy.ensembler_resets == 2                                                  # only the first crossing


def test_group_keep_at_evaluation_and_training_dropout():
    policy = HRIACTPolicy(config(intent_keep=['spatial', 'memory'], intent_p_drop_group=1.0))
    policy.eval()
    keep = policy._with_intent(batch())['intent_keep']
    assert {g for g, k in keep.items() if k.all()} == {'spatial', 'memory'} and not keep['time'].any()
    policy.train()
    assert not any(v.any() for v in policy._with_intent(batch())['intent_keep'].values())
    full = HRIACTPolicy(config()).eval()
    assert full._with_intent(batch()).get('intent_keep') is None


def test_xi_is_normalised_with_the_training_statistics():
    policy = HRIACTPolicy(config())
    policy.intent_encoder.set_xi_stats(np.full((M, J, 3), 0.5), np.full((M, J, 3), 1e-4))
    assert torch.allclose(policy.intent_encoder.xi_std, torch.full((M, J, 3), 1e-2))
    b = batch()
    assert torch.allclose(policy._with_intent(b)['intent']['xi'], (b['intent.xi'] - 0.5) / 1e-2)


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    policy = HRIACTPolicy(config('continuous', intent_film=True, intent_source='perfect')).eval()
    policy.intent_encoder.set_xi_stats(np.ones((M, J, 3)), 2 * np.ones((M, J, 3)))
    policy.save_pretrained(tmp_path)
    loaded = HRIACTPolicy.from_pretrained(tmp_path).eval()
    c = loaded.config
    assert c.intent_arch == 'tokens' and c.use_task_token and c.intent_film and c.intent_source == 'perfect'
    assert c.intent_schema['targets'] == S['targets'] and torch.equal(loaded.intent_encoder.xi_std, policy.intent_encoder.xi_std)
    assert torch.allclose(chunk(policy, batch()), chunk(loaded, batch()))


def test_film_starts_as_identity():
    torch.manual_seed(0)
    a = HRIACTPolicy(config(intent_film=False)).eval()
    torch.manual_seed(0)
    b = HRIACTPolicy(config(intent_film=True)).eval()
    b.load_state_dict({**b.state_dict(), **a.state_dict()})
    assert torch.allclose(chunk(a, batch()), chunk(b, batch()), atol=1e-6)


def test_config_validation():
    with pytest.raises(ValueError):
        config(intent_schema={})
    with pytest.raises(ValueError):
        config(intent_keep=['gaze'])
    with pytest.raises(ValueError):
        config(intent_source='oracle')
    with pytest.raises(ValueError):
        config(intent_mask=True)


def test_training_script_path(labelled_dataset):
    """train_policy pieces on a labelled dataset: the same episodes train the no-intent and the intent policy, the
    intent columns bypass the preprocessor, a source mix picks per sample, xi statistics come from the training split."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from intent_policy.policies.hri_act import make_hri_act_pre_post_processors
    from scripts.train_policy import IntentSelector, build_config, guide_overrides, parse_mix, run_batch, source_prefixes, xi_stats

    root = labelled_dataset[0]
    meta = LeRobotDatasetMetadata('local/intent', root=root)
    base = dict(type='hri_act', action_mode='restricted', use_vae=False, **SMALL)
    args = argparse.Namespace(use_intent=True, intent_schema='configs/intent_schema.yaml', use_task_token=True, intent_keep=None,
                              intent_source='perfect', intent_source_mix='perfect:0.5,predicted:0.5', intent_in_cvae=None,
                              p_drop_group=0.2, p_drop_all=None, intent_film=None, n_action_steps=None, action_mode='discrete')
    over = guide_overrides(args)
    assert over['intent_source'] == 'perfect' and over['intent_p_drop_group'] == 0.2 and over['action_mode'] == 'restricted'
    assert source_prefixes('predicted', meta) == ['intent_pr0', 'intent_pr1', 'intent_pr2']
    for policy_yaml in (dict(base, use_task_token=True, intent_schema=S), dict(base, **over)):
        mix = policy_yaml.get('intent_source_mix')
        cfg = build_config(dict(policy_yaml), meta, 'cpu', ['high', 'wrist'])
        assert not any(k.startswith(('intent', 'human', 'hri')) for k in cfg.input_features)
        policy = HRIACTPolicy(cfg)
        pre, _ = make_hri_act_pre_post_processors(cfg, dataset_stats=meta.stats)
        delta = {k: [i / 20 for i in range(cfg.chunk_size)] for k in ('action', cfg.restricted_label_key)}
        ds = LeRobotDataset('local/intent', root=root, episodes=[0], delta_timestamps=delta)
        selector = None
        if policy.intent_encoder is not None:
            w = parse_mix(cfg.intent_source, mix)
            selector = IntentSelector({s: source_prefixes(s, meta) for s in w}, w, 0)
            assert selector.primary == 'intent_pp'
            mean, std = xi_stats(ds, selector.primary)
            policy.intent_encoder.set_xi_stats(mean, std)
        b = next(iter(torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)))
        policy.train()
        loss, _ = run_batch(policy, pre, b, 'cpu', cfg.restricted_label_key, selector=selector)
        assert torch.isfinite(loss)
    bad = dict(base, **dict(over, intent_schema=dict(S, targets=S['targets'][:-1])))
    with pytest.raises(ValueError):
        build_config(bad, meta, 'cpu', ['high', 'wrist'])
