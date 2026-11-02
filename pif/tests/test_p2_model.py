"""P2 tests (docs/PIF.md): forward with different K, belief sums to 1 without NaN, causality, object
permutation, parameter count; one loss step runs and is finite."""
from pathlib import Path

import numpy as np
import pytest
import torch

from pif.config import load_config
from pif.data.dataset import PIFWindowDataset, collate
from pif.losses import SIGReg, pif_losses
from pif.model import PIF
from test_p1_data import episode

ROOT = Path(__file__).resolve().parents[2]
CFG = load_config('configs/pif/pif.yaml')


def batch(seed=0):
    ds = PIFWindowDataset([episode(T=90, K=3, seed=seed), episode(T=90, K=5, seed=seed + 1)], CFG, train=False)
    return collate([ds[0], ds[len(ds) - 1]])


def test_forward_belief_normalised():
    torch.manual_seed(0)
    m = PIF(CFG).eval()
    b = batch()
    log_b, rho = m.infer(b)
    assert log_b.shape == (2, b['human'].shape[1], CFG['model']['num_relations'], 6)
    p = log_b.exp()
    assert torch.isfinite(log_b[p > 0]).all()
    assert torch.allclose(p.flatten(2).sum(-1), torch.ones(p.shape[:2]), atol=1e-5)
    assert p[0, :, :, 4:].max() < 1e-6                     # padded objects of the K = 3 episode get no mass


def test_belief_is_causal():
    torch.manual_seed(0)
    m = PIF(CFG).eval()
    b = batch()
    t = 30
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    b2['human'][:, t + 1:] += 1.0
    lb, _ = m.infer(b)
    lb2, _ = m.infer(b2)
    assert torch.equal(lb[:, :t + 1], lb2[:, :t + 1]) and not torch.equal(lb[:, t + 1:], lb2[:, t + 1:])


def test_object_permutation_moves_belief():
    torch.manual_seed(0)
    m = PIF(CFG).eval()
    b = batch()
    perm = torch.tensor([2, 0, 4, 1, 3])
    b2 = dict(b, objects=b['objects'][:, :, perm], obj_type=b['obj_type'][:, perm], obj_mask=b['obj_mask'][:, perm])
    inv = torch.argsort(perm)
    b2['holding'] = torch.where(b['holding'] >= 0, inv[b['holding'].clamp(min=0)], b['holding'])
    lb, _ = m.infer(b)
    lb2, _ = m.infer(b2)
    assert torch.allclose(lb[1, :, :, 1:][..., perm], lb2[1, :, :, 1:], atol=1e-4)
    assert torch.allclose(lb[1, :, :, 0], lb2[1, :, :, 0], atol=1e-4)


def test_parameter_count():
    n = sum(p.numel() for p in PIF(CFG).parameters())
    assert 0.5e6 < n < 3e6, n                             # handoff B4: 1-3 M parameters


def test_one_loss_step():
    pytest.importorskip('lejepa')
    torch.manual_seed(0)
    m = PIF(CFG)
    out = pif_losses(m, batch(), CFG, 0, SIGReg(64), 2, torch.Generator().manual_seed(0))
    assert torch.isfinite(out['total'])
    out['total'].backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
