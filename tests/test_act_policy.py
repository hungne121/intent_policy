"""LeRobot ACT integration: construction, shared backbone, feature interface, both action paths."""
import ast
import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACT, ACTPolicy

from intent_policy.benchmark.runner import ExpertAgent, ObservationConfig, run_episode
from intent_policy.sim.base_env import ROOT
from intent_policy.policies.hri_act import HRIACTConfig, HRIACTPolicy, make_hri_act_pre_post_processors
from intent_policy.policies.policy_agent import PolicyAgent
from intent_policy.scenarios.scenario_registry import make_scenario
from scripts.collect_demos import features
from scripts.train_policy import build_config, run_batch

SMALL = dict(chunk_size=8, n_action_steps=1, dim_model=64, n_heads=4, dim_feedforward=128, n_encoder_layers=1,
             n_decoder_layers=1, pretrained_backbone_weights=None)
OBS = ObservationConfig(('high', 'wrist'), 48, 36)


@pytest.fixture(scope='module')
def tiny_dataset(tmp_path_factory, scenarios):
    """Two expert episodes recorded exactly like scripts/collect_demos.py (small images)."""
    root = tmp_path_factory.mktemp('ds') / 'tiny'
    ds = LeRobotDataset.create(repo_id='local/tiny', fps=20, features=features(OBS), root=root, use_videos=True)
    for sid, seed in (('t1_pick_place', 2), ('t4_interrupt', 3)):
        frames = []

        def on_frame(obs, decision, target, scenario):
            frames.append({**obs, 'action': np.asarray(target, np.float32),
                           'restricted_action': np.array([decision.action_id], np.int64)})
        rec = run_episode(scenarios[sid], ExpertAgent(), seed, obs_cfg=OBS, on_frame=on_frame, keep_trace=False)
        assert rec['success']
        for fr in frames:
            ds.add_frame({**fr, 'task': scenarios[sid].cfg.task})
        ds.save_episode()
    ds.finalize()
    return root


def make_policy(meta, mode, **over):
    yaml_cfg = dict(type='hri_act', action_mode=mode, use_vae=(mode == 'continuous'), **SMALL)
    yaml_cfg.update(over)
    cfg = build_config(yaml_cfg, meta, 'cpu', ['high', 'wrist'])
    return HRIACTPolicy(cfg), cfg


def loaded(root, cfg):
    delta = {k: [i / 20 for i in range(cfg.chunk_size)] for k in ('action', cfg.restricted_label_key)}
    return LeRobotDataset('local/tiny', root=root, delta_timestamps=delta)


def test_policy_is_lerobot_act_with_one_shared_backbone(tiny_dataset):
    meta = LeRobotDatasetMetadata('local/tiny', root=tiny_dataset)
    policy, cfg = make_policy(meta, 'restricted')
    assert isinstance(policy, ACTPolicy) and isinstance(cfg, HRIACTConfig) and cfg.type == 'hri_act'
    assert sum(isinstance(m, ACT) for m in policy.modules()) == 1          # exactly one ACT network
    assert hasattr(policy.model, 'backbone') and hasattr(policy.model, 'action_head')   # continuous head kept
    assert policy.restricted_head.net.out_features == 9


def test_no_skill_specific_models_or_routing():
    forbidden = ('ACT_PICK', 'ACT_PLACE', 'ACT_HANDOVER', 'ACT_RETRACT', 'SkillSelector', 'skill_selector')
    for pkg in ('intent_policy', 'scripts'):
        for f in (ROOT / pkg).rglob('*.py'):
            text = f.read_text()
            assert not any(word in text for word in forbidden), f'{f} mentions a skill-specific model'
            tree = ast.parse(text)
            n_policies = sum(isinstance(n, ast.Call) and getattr(n.func, 'id', getattr(n.func, 'attr', '')) == 'HRIACTPolicy'
                             for n in ast.walk(tree))
            assert n_policies <= 1, f'{f} instantiates several policies'


def test_dataset_features_map_to_act_features(tiny_dataset):
    meta = LeRobotDatasetMetadata('local/tiny', root=tiny_dataset)
    _, cfg = make_policy(meta, 'restricted')
    assert set(cfg.input_features) == {'observation.state', 'observation.images.high', 'observation.images.wrist'}
    assert set(cfg.output_features) == {'action'} and cfg.action_feature.shape == (7,)
    assert cfg.robot_state_feature.shape == (10,)
    assert cfg.image_features['observation.images.high'].shape == (3, 36, 48)
    assert cfg.restricted_label_key not in cfg.output_features          # never normalised as an action
    item = loaded(tiny_dataset, cfg)[5]
    assert item['action'].shape == (cfg.chunk_size, 7) and item['restricted_action'].shape == (cfg.chunk_size,)
    assert item['action_is_pad'].shape == (cfg.chunk_size,)


@pytest.mark.parametrize('mode', ['restricted', 'continuous'])
def test_forward_shapes_training_step_and_checkpoint(tiny_dataset, tmp_path, mode):
    torch.manual_seed(0)
    meta = LeRobotDatasetMetadata('local/tiny', root=tiny_dataset)
    policy, cfg = make_policy(meta, mode)
    pre, post = make_hri_act_pre_post_processors(cfg, meta.stats)
    batch = next(iter(torch.utils.data.DataLoader(loaded(tiny_dataset, cfg), batch_size=4, shuffle=True)))
    opt = torch.optim.AdamW(policy.get_optim_params(), lr=1e-3)
    losses = []
    for _ in range(15):                                   # overfit one batch: loss must go down
        loss, info = run_batch(policy, pre, {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()},
                               'cpu', cfg.restricted_label_key)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    policy.eval()
    obs = pre({k: v[:1] for k, v in batch.items() if k.startswith('observation')})
    if mode == 'restricted':
        logits = policy.predict_restricted_chunk(obs)
        assert logits.shape == (1, cfg.chunk_size, 9)
        # the restricted head reads exactly the latent the continuous head reads
        assert policy._latent.shape == (1, cfg.chunk_size, cfg.dim_model)
        out = policy.select_restricted_action(obs)
        assert out['action_logits'].shape == (1, 9) and 0 <= int(out['selected_action']) < 9
        torch.testing.assert_close(out['action_probabilities'].sum(-1), torch.ones(1))
    chunk = policy.predict_action_chunk(obs)               # continuous ACT path always available
    assert chunk.shape == (1, cfg.chunk_size, 7)
    ckpt = tmp_path / 'ckpt'
    policy.save_pretrained(ckpt); pre.save_pretrained(ckpt); post.save_pretrained(ckpt)
    again = HRIACTPolicy.from_pretrained(ckpt)
    again.eval()
    torch.testing.assert_close(again.predict_action_chunk(obs), chunk)
    if mode == 'restricted':
        torch.testing.assert_close(again.predict_restricted_chunk(obs), policy.predict_restricted_chunk(obs))


def test_config_guards():
    with pytest.raises(ValueError):
        HRIACTConfig(action_mode='skills', device='cpu', **SMALL)
    with pytest.raises(ValueError):                         # intention needs explicit numeric oracle branches
        HRIACTConfig(use_intent=True, device='cpu', **SMALL)


@pytest.mark.parametrize('mode', ['restricted', 'continuous'])
def test_policy_runs_on_the_simulator_without_intention_input(tiny_dataset, tmp_path, scenarios, mode):
    meta = LeRobotDatasetMetadata('local/tiny', root=tiny_dataset)
    policy, cfg = make_policy(meta, mode)
    pre, post = make_hri_act_pre_post_processors(cfg, meta.stats)
    ckpt = tmp_path / mode
    policy.save_pretrained(ckpt); pre.save_pretrained(ckpt); post.save_pretrained(ckpt)
    agent = PolicyAgent(ckpt, device='cpu')
    assert agent.action_mode == mode and agent.describe()['use_intent'] is False
    raw = copy.deepcopy(scenarios['t3_assist'].cfg.to_dict())
    raw['timing']['timeout_s'] = 1.0                       # short smoke episode
    sc = make_scenario(raw)
    try:
        rec = run_episode(sc, agent, 0, obs_cfg=OBS)
    finally:
        sc.close()
    assert rec['n_decisions'] >= 19 and rec['failure'] == 'timeout'
    step = rec['trace'][3]
    if mode == 'restricted':
        assert len(step['action_logits']) == 9 and len(step['action_probabilities']) == 9
        assert 0 <= step['selected_action'] < 9 and len(step['mapped_controller_command']['joint_target']) == 7
    else:
        assert step['selected_action_name'] == 'continuous' and len(step['mapped_controller_command']['joint_target']) == 7


def test_training_resume_is_exact(tiny_dataset, tmp_path, monkeypatch):
    """12 uninterrupted steps == 6 steps + resume to 12 (optimizer, RNG and epoch position restored)."""
    import sys
    import yaml
    from safetensors.torch import load_file
    from scripts import train_policy
    policy_cfg = tmp_path / 'policy.yaml'
    policy_cfg.write_text(yaml.safe_dump(dict(type='hri_act', action_mode='restricted', use_vae=False, dropout=0.1, **SMALL)))
    exp = dict(experiment='resume_test', seed=3, policy=dict(config=str(policy_cfg)), intent=dict(provider='none'),
               observation=dict(cameras=['high', 'wrist'], width=48, height=36),
               data=dict(repo_id='local/tiny', root=str(tiny_dataset)),
               training=dict(steps=12, batch_size=4, num_workers=0, log_every=3, save_every=6, output_dir=str(tmp_path / 'x')))
    exp_cfg = tmp_path / 'exp.yaml'
    exp_cfg.write_text(yaml.safe_dump(exp))

    def train(out, *extra):
        monkeypatch.setattr(sys, 'argv', ['train_policy', '--config', str(exp_cfg), '--output-dir', str(out),
                                          '--device', 'cpu', '--val-every', '0', *extra])
        train_policy.main()
        return load_file(out / 'checkpoints/last/pretrained_model/model.safetensors')

    full = train(tmp_path / 'full')
    train(tmp_path / 'split', '--steps', '6')
    resumed = train(tmp_path / 'split', '--resume')
    assert full.keys() == resumed.keys()
    for k in full:
        torch.testing.assert_close(full[k], resumed[k], msg=k)
