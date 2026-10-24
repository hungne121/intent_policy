"""PIF v3 core (docs/PIF.md §4): normalised Boltzmann-rational emission + exact Bayes filter over g in {none, 1..K}.

  g = (robot-acts-on, k) or none. Per frame t >= 1 and hand h (index tip p, pointing direction d = norm(tip - base)):
    acting hand, target k:  dp ~ N(beta sigma^2 u_k, sigma^2 I),  d_t ~ vMF(norm(a d_{t-1} + (1 - a) u_k), kappa)
    idle hand / none:       dp ~ N(0, sigma^2 I),                  d_t ~ vMF(d_{t-1}, kappa)
  u_k = norm(c_k - p_{t-1}) (object position at t - 1: causal). Every term is a normalised density whose constant does
  not depend on g, so -log p(o_1:T) from the forward algorithm is a proper likelihood (docs/PIF.md §4.3, fix 1).
  p(o_t | k) = rho * 1/2 sum_h [act_h(k) prod_{h' != h} idle_h'] + (1 - rho) prod_h idle_h,  p(o_t | none) = prod_h idle_h:
  under an intent the person acts towards k on a fraction rho of the frames and is still / retracting on the others,
  so a retracted hand does not cancel a held intent (stage A found hazard -> 0.985 without this term: the filter had
  no memory). A hand that is not seen at t or t - 1 contributes 1 (missing data).
  Transition: b^-_t = (1 - h) b_{t-1} + h pi,  pi = [w_none, (1 - w_none) / K, ...]; b_0 = pi (docs/PIF.md §4.4).
Emission uses only frames t and t - 1; all other history goes through b_{t-1} (no double counting).
Episodes are batched: time is padded with log e = 0 (the predicted belief is normalised, so padding adds 0 to the
likelihood) and candidates with a mask.
"""
from __future__ import annotations
import math

import numpy as np
import torch

HANDS = (('r_index_base', 'r_index_tip'), ('l_index_base', 'l_index_tip'))
NEG = -1e9                                  # log of a masked candidate (finite: keeps logaddexp gradients defined)


def _unit(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def log_vmf3(cos: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """log density of a 3D von Mises-Fisher at cos(angle to the mean): kappa cos + log kappa - log 4pi - log sinh kappa."""
    log_sinh = kappa + torch.log1p(-torch.exp(-2 * kappa)) - math.log(2.0)
    return kappa * cos + torch.log(kappa) - math.log(4 * math.pi) - log_sinh


class Params(torch.nn.Module):
    """The learned scalars of the core (configs/pif/pif.yaml v3.init), kept in unconstrained form."""

    def __init__(self, init: dict):
        super().__init__()
        logit = lambda p: math.log(p / (1 - p))
        f = lambda v: torch.nn.Parameter(torch.tensor(v, dtype=torch.float64))
        self.log_sigma, self.beta = f(math.log(init['sigma_m'])), f(float(init['beta']))
        self.log_kappa, self.logit_a = f(math.log(init['kappa'])), f(logit(init['a']))
        self.logit_h, self.logit_w0 = f(logit(init['hazard'])), f(logit(init['w_none']))
        self.logit_rho = f(logit(init['act_prob']))

    def values(self) -> dict:
        return dict(sigma_m=self.log_sigma.exp().item(), beta=self.beta.item(), kappa=self.log_kappa.exp().item(),
                    a=torch.sigmoid(self.logit_a).item(), hazard=torch.sigmoid(self.logit_h).item(),
                    w_none=torch.sigmoid(self.logit_w0).item(), act_prob=torch.sigmoid(self.logit_rho).item())


def batch(eps: list[dict], cands: list[list[int]], keypoints: list[str]) -> dict:
    """Pad episodes to [B, T, ...]: index tip / base per hand, hand visibility, candidate positions + masks."""
    J = {n: i for i, n in enumerate(keypoints)}
    B, T, K = len(eps), max(len(e['holding']) for e in eps), max(len(c) for c in cands)
    tip, base = np.zeros((B, T, 2, 3)), np.zeros((B, T, 2, 3))
    vis, cand = np.zeros((B, T, 2), bool), np.zeros((B, T, K, 3))
    cmask, tlen = np.zeros((B, K), bool), np.array([len(e['holding']) for e in eps])
    for i, (e, c) in enumerate(zip(eps, cands)):
        n = tlen[i]
        for h, (b, t) in enumerate(HANDS):
            tip[i, :n, h], base[i, :n, h] = e['human_pos'][:, J[t]], e['human_pos'][:, J[b]]
            vis[i, :n, h] = e['human_vis'][:, J[b]] & e['human_vis'][:, J[t]]
        cand[i, :n, :len(c)] = e['obj_pos'][:, c]
        cmask[i, :len(c)] = True
    d = lambda a: torch.as_tensor(a, dtype=torch.float64)
    return dict(tip=d(tip), base=d(base), vis=torch.as_tensor(vis), cand=d(cand), cmask=torch.as_tensor(cmask),
                tlen=torch.as_tensor(tlen))


def log_emission(x: dict, p: Params) -> torch.Tensor:
    """[B, T, 1 + K] log p(o_t | g) for g = none, 1..K; frame 0 and padding are 0, masked candidates NEG."""
    tip, base, vis, cand = x['tip'], x['base'], x['vis'], x['cand']
    B, T, K = cand.shape[:3]
    sigma, kappa, a = p.log_sigma.exp(), p.log_kappa.exp(), torch.sigmoid(p.logit_a)
    dp = tip[:, 1:] - tip[:, :-1]                                              # [B, T-1, H, 3]
    ok = (vis[:, 1:] & vis[:, :-1]).double()                                   # [B, T-1, H]
    d_now, d_prev = _unit(tip[:, 1:] - base[:, 1:]), _unit(tip[:, :-1] - base[:, :-1])
    u = _unit(cand[:, :-1, None] - tip[:, :-1, :, None])                       # [B, T-1, H, K, 3]
    gauss = lambda r2: -r2 / (2 * sigma ** 2) - 3 * torch.log(sigma) - 1.5 * math.log(2 * math.pi)
    idle = gauss((dp ** 2).sum(-1)) + log_vmf3((d_now * d_prev).sum(-1), kappa)
    mean_d = _unit(a * d_prev[..., None, :] + (1 - a) * u)
    act = gauss(((dp[..., None, :] - p.beta * sigma ** 2 * u) ** 2).sum(-1)) + log_vmf3((d_now[..., None, :] * mean_d).sum(-1), kappa)
    idle, act = idle * ok, act * ok[..., None]                                 # unseen hand -> factor 1
    comp = act + (idle.sum(2, keepdim=True) - idle)[..., None]                 # hand h acts, the other is idle
    tgt = torch.logsumexp(comp, dim=2) - math.log(comp.shape[2])               # [B, T-1, K]
    none = idle.sum(2)[..., None]
    rho = torch.sigmoid(p.logit_rho)
    tgt = torch.logaddexp(torch.log(rho) + tgt, torch.log1p(-rho) + none)     # acting or still under intent k
    out = torch.cat([none, tgt], -1)
    out = torch.cat([out.new_zeros(B, 1, 1 + K), out], 1)
    live = (torch.arange(T)[None] < x['tlen'][:, None]).double()[..., None]
    out = out * live
    return torch.where(torch.cat([torch.ones(B, 1, dtype=torch.bool), x['cmask']], 1)[:, None], out, out.new_full((), NEG))


def log_prior(cmask: torch.Tensor, p: Params) -> torch.Tensor:
    """[B, 1 + K] log pi: w_none on none, the rest uniform over each episode's candidates."""
    w0, K = torch.sigmoid(p.logit_w0), cmask.sum(1, keepdim=True).double()
    rest = torch.where(cmask, torch.log1p(-w0) - torch.log(K), torch.full_like(K, NEG).expand_as(cmask))
    return torch.cat([torch.log(w0).expand(len(cmask), 1), rest], 1)


def forward(log_e: torch.Tensor, cmask: torch.Tensor, p: Params) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact filter in log space: (log b_t [B, T, 1 + K], log p(o_1:T) per episode [B])."""
    h = torch.sigmoid(p.logit_h)
    lp = log_prior(cmask, p)
    out, ll, prev = [], log_e.new_zeros(len(log_e)), lp
    for t in range(log_e.shape[1]):
        joint = log_e[:, t] + torch.logaddexp(torch.log1p(-h) + prev, torch.log(h) + lp)
        z = torch.logsumexp(joint, 1, keepdim=True)
        ll, prev = ll + z[:, 0], joint - z
        out.append(prev)
    return torch.stack(out, 1), ll


def fit(x: dict, init: dict, steps: int, lr: float, seed: int) -> tuple[Params, list[float]]:
    """Maximise sum log p(o_1:T) over the scalars (Baum-Welch by gradient, docs/PIF.md §4.5). Uses no labels."""
    torch.manual_seed(seed)
    p = Params(init)
    opt = torch.optim.Adam(p.parameters(), lr=lr)
    n = x['tlen'].sum().double()
    hist = []
    for _ in range(steps):
        opt.zero_grad()
        nll = -forward(log_emission(x, p), x['cmask'], p)[1].sum() / n
        nll.backward()
        opt.step()
        hist.append(nll.item())
    return p, hist


@torch.no_grad()
def belief(x: dict, p: Params) -> np.ndarray:
    """[B, T, 1 + K] filtered belief (none first)."""
    return forward(log_emission(x, p), x['cmask'], p)[0].exp().numpy()
