"""PIF v3 core (pif/filter.py): exact filter against brute-force enumeration, normalised densities, a synthetic reach."""
import itertools
import math

import numpy as np
import torch

from pif.filter import Params, batch, belief, fit, forward, log_vmf3

INIT = dict(sigma_m=0.02, beta=0.0, kappa=5.0, a=0.9, hazard=0.05, w_none=0.3, act_prob=0.5)


def test_forward_matches_enumeration():
    """b_t and log p(o_1:T) of the forward recursion = sums over every state sequence (G = 3, T = 4)."""
    p = Params(INIT)
    rng = np.random.default_rng(0)
    G, T = 3, 4
    log_e = torch.as_tensor(rng.normal(size=(1, T, G)))
    cmask = torch.ones(1, G - 1, dtype=torch.bool)
    log_b, ll = forward(log_e, cmask, p)
    h, w0 = torch.sigmoid(p.logit_h).item(), torch.sigmoid(p.logit_w0).item()
    pi = np.array([w0] + [(1 - w0) / (G - 1)] * (G - 1))
    A = (1 - h) * np.eye(G) + h * pi[None]
    e = np.exp(log_e[0].numpy())
    marg, total = np.zeros((T, G)), 0.0
    for seq in itertools.product(range(G), repeat=T):
        w = pi[seq[0]] * e[0, seq[0]]
        for t in range(1, T):
            w *= A[seq[t - 1], seq[t]] * e[t, seq[t]]
        total += w
    assert math.isclose(ll.item(), math.log(total), rel_tol=1e-9)
    for t in range(T):                                    # filtered belief at t = sums over prefixes of length t + 1
        f = np.zeros(G)
        for seq in itertools.product(range(G), repeat=t + 1):
            w = pi[seq[0]] * e[0, seq[0]]
            for s in range(1, t + 1):
                w *= A[seq[s - 1], seq[s]] * e[s, seq[s]]
            f[seq[-1]] += w
        np.testing.assert_allclose(log_b[0, t].detach().exp().numpy(), f / f.sum(), rtol=1e-9)


def test_vmf_is_normalised():
    rng = np.random.default_rng(1)
    d = rng.normal(size=(400000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    for kappa in (0.5, 5.0, 40.0):
        dens = log_vmf3(torch.as_tensor(d[:, 2]), torch.tensor(kappa, dtype=torch.float64)).exp().numpy()
        assert abs(dens.mean() * 4 * math.pi - 1) < 0.02     # E_uniform[f] x area of the sphere


def _reach_episode(target: int, T: int = 40, seed: int = 0) -> dict:
    """Right index finger moves from (0.6, 0, 0.2) towards container `target`, pointing at it; left hand still."""
    rng = np.random.default_rng(seed)
    cand = np.array([[0.3, 0.3, 0.0], [0.3, 0.1, 0.0], [0.3, -0.1, 0.0], [0.3, -0.3, 0.0]])
    start = np.array([0.6, 0.0, 0.2])
    s = np.clip((np.arange(T) - 10) / 20, 0, 1)[:, None]          # still for 1 s, then 2 s of reaching
    tip = start + s * 0.6 * (cand[target] - start) + rng.normal(0, 0.003, (T, 3))
    base = tip + 0.05 * (start - cand[target]) / np.linalg.norm(start - cand[target])
    kp = ['r_index_base', 'r_index_tip', 'l_index_base', 'l_index_tip']
    human = np.zeros((T, 4, 3))
    human[:, 0], human[:, 1] = base, tip
    human[:, 2], human[:, 3] = [0.6, 0.4, 0.1], [0.6, 0.45, 0.1]
    return dict(human_pos=human, human_vis=np.ones((T, 4), bool), obj_pos=np.repeat(cand[None], T, 0),
                holding=-np.ones(T, int)), kp


def test_synthetic_reach_is_recovered():
    eps, kp = zip(*[_reach_episode(k % 4, seed=k) for k in range(8)])
    x = batch(list(eps), [[0, 1, 2, 3]] * 8, kp[0])
    p, hist = fit(x, INIT, steps=150, lr=0.05, seed=0)
    assert hist[-1] < hist[0]
    b = belief(x, p)
    for i in range(8):
        assert b[i, 9, 1:].max() < 0.5                                # nothing before the motion starts (P5)
        assert int(np.argmax(b[i, -1, 1:])) == i % 4                  # the reached container wins
