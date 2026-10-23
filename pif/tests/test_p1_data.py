"""P1 tests (handoff §3 "Test P1"): token shapes / dtypes, padding mask with different K, causality, backward velocity,
object permutation; eval labels stay out of the training samples."""
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from pif.data.dataset import PIFWindowDataset, collate
from pif.data.tokens import HUMAN_DIM, OBJECT_DIM, backward_velocity, build_tokens, norm_stats, permute_objects

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / 'configs/pif/base.yaml').read_text())
J = len(CFG['tokens']['keypoints'])


def episode(T=80, K=4, seed=0):
    rng = np.random.default_rng(seed)
    holding = np.full(T, -1)
    holding[T // 3:T // 2] = 1
    return dict(fps=10.0, human_pos=rng.normal(size=(T, J, 3)), human_vis=rng.random((T, J)) > 0.1,
                obj_pos=rng.normal(size=(T, K, 3)), obj_vis=np.ones((T, K), bool), obj_type=rng.integers(0, 3, K),
                obj_has_item=rng.random((T, K)) > 0.5, obj_done=rng.random((T, K)) > 0.5, holding=holding,
                obj_names=[f'o{k}' for k in range(K)], meta={})


def test_token_shapes_and_dtypes():
    tok = build_tokens(episode(T=30, K=5))
    assert tok['human'].shape == (30, J, HUMAN_DIM) and tok['human'].dtype == np.float32
    assert tok['objects'].shape == (30, 5, OBJECT_DIM) and tok['objects'].dtype == np.float32
    assert tok['obj_type'].shape == (5,) and tok['obj_type'].dtype == np.int64
    assert tok['holding'].shape == (30,) and tok['holding'].dtype == np.int64


def test_hidden_keypoint_is_zero_with_flag():
    ep = episode(T=10)
    tok = build_tokens(ep, norm_stats([ep]))
    hidden = ~ep['human_vis']
    assert hidden.any()
    assert np.all(tok['human'][hidden] == 0.0)
    assert np.all(tok['human'][~hidden][:, -1] == 1.0)


def test_velocity_is_backward_difference():
    pos = np.cumsum(np.arange(12, dtype=float))[:, None, None] * np.ones((1, 2, 3))
    vis = np.ones((12, 2), bool)
    vel = backward_velocity(pos, vis, fps=10.0)
    assert np.allclose(vel[0], 0.0)
    assert np.allclose(vel[1:], (pos[1:] - pos[:-1]) * 10.0)
    vis[5, 0] = False                         # a hidden frame breaks the difference on both sides of it
    vel = backward_velocity(pos, vis, fps=10.0)
    assert np.allclose(vel[5, 0], 0.0) and np.allclose(vel[6, 0], 0.0) and not np.allclose(vel[7, 0], 0.0)


def test_tokens_are_causal():
    ep = episode(T=60)
    tok = build_tokens(ep)
    t = 25
    ep2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ep.items()}
    for key in ('human_pos', 'obj_pos'):
        ep2[key][t + 1:] += 5.0
    ep2['holding'][t + 1:] = 2
    tok2 = build_tokens(ep2)
    for key in ('human', 'objects', 'holding'):
        assert np.array_equal(tok[key][:t + 1], tok2[key][:t + 1])
        assert not np.array_equal(tok[key][t + 1:], tok2[key][t + 1:])


def test_dataset_inputs_are_causal():
    """Changing frames after the input window does not change the input frames of a sample."""
    ep = episode(T=120)
    ds = PIFWindowDataset([ep], CFG, train=False)
    s = 30
    a = ds[ds.index.index((0, s))]
    ep2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ep.items()}
    ep2['human_pos'][s + ds.L:] += 3.0
    b = PIFWindowDataset([ep2], CFG, train=False)[ds.index.index((0, s))]
    assert torch.equal(a['human'][:ds.L], b['human'][:ds.L])
    assert not torch.equal(a['human'][ds.L:], b['human'][ds.L:])


def test_permutation_moves_tokens_and_holding():
    ep = episode(T=40, K=5)
    perm = np.array([3, 0, 4, 1, 2])
    tok, tokp = build_tokens(ep), build_tokens(permute_objects(ep, perm))
    assert np.array_equal(tokp['objects'], tok['objects'][:, perm])
    assert np.array_equal(tokp['obj_type'], tok['obj_type'][perm])
    held = tok['holding'] >= 0
    assert np.array_equal(perm[tokp['holding'][held]], tok['holding'][held])
    assert np.all(tokp['holding'][~held] == -1)


def test_collate_pads_objects_with_mask():
    ds = PIFWindowDataset([episode(T=90, K=3, seed=1), episode(T=90, K=6, seed=2)], CFG, train=True)
    a, b = ds[0], ds[len(ds) - 1]
    batch = collate([a, b])
    assert batch['objects'].shape[2] == 6
    assert batch['obj_mask'].tolist() == [[True] * 3 + [False] * 3, [True] * 6]
    assert torch.all(batch['objects'][0, :, 3:] == 0)
    assert batch['human'].shape == (2, ds.L + ds.H, J, HUMAN_DIM)


def test_target_validity_past_episode_end():
    ep = episode(T=70)
    ds = PIFWindowDataset([ep], CFG, train=False)
    s = len(ep['holding']) - ds.L                          # last window: no future frames at all
    x = ds[ds.index.index((0, s))]
    assert x['valid'][:ds.L].all() and not x['valid'][ds.L:].any()
    assert not x['target_valid'][-1].any()
    h0 = int(x['horizons'][0])
    assert x['target_valid'][ds.L - h0 - 1, 0] and not x['target_valid'][ds.L - h0, 0]
    assert torch.all(x['holding'][~x['valid']] == -1)


def test_no_labels_in_training_samples():
    ep = episode(T=60)
    x = PIFWindowDataset([ep], CFG, train=True, labels=[dict(target=1)])[0]
    assert 'labels' not in x
    y = PIFWindowDataset([ep], dict(CFG, supervised_warmstart=True), train=True, labels=[dict(target=1)])[0]
    assert y['labels'] == dict(target=1)


REPLAY = ROOT / 'outputs/pif/sim_replay/intent_act_late_v5'


@pytest.mark.skipif(not (REPLAY / 'episode_000000.npz').exists(), reason='sim replay not available')
def test_sim_episode_reader():
    from pif.data.episode_reader import load_meta, read_sim_dataset
    from pif.data.labels import TARGET_NONE, sim_labels
    ds_root = ROOT / 'outputs/datasets/intent_act_late_v5'
    ep = read_sim_dataset(ds_root, REPLAY, CFG, episodes=[0])[0]
    meta = load_meta(ds_root)[0]                              # T1: B1 (twin B1p) -> P1, 250 frames at 20 fps
    assert len(ep['holding']) == 125 and ep['meta']['stride'] == 2
    assert ep['obj_names'][-3:] == ['P1', 'P2', 'U'] and ep['human_pos'].shape == (125, J, 3)
    assert abs(np.median(ep['obj_pos'][0, :, 2])) < 0.05     # table frame: objects near z = 0
    held = ep['holding'][ep['holding'] >= 0]
    assert len(held) and np.all(np.array(ep['obj_names'])[held] == 'B1')
    assert ep['obj_done'][-1, ep['obj_names'].index('B1')] and ep['obj_has_item'][-1, ep['obj_names'].index('P1')]
    lab = sim_labels(meta, ep, CFG)
    targets = [(g['label'], g['target']) for g in lab['gestures'] if g['target'] != TARGET_NONE]
    assert targets == [('point_object', ep['obj_names'].index('B1')), ('point_place', ep['obj_names'].index('P1'))]
    assert set(lab['intent_dest']) == {ep['obj_names'].index('P1')}
