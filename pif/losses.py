"""PIF losses (docs/PIF.md). Intent labels are never used here (docs/PIF.md).

For S sampled input frames t of each window and every valid code Z = (r, k):
  L_pred(Z)   = sum_h tv_h ( mean_j |ô^j_{t+h}(Z) - sg(o^j_{t+h})| + |û_{t+h}(Z) - sg(u_{t+h})| )   (L1, mean over dims)
  w(Z)        = softmax(-L_pred(Z) / T) over valid Z (hindsight posterior, detached)
  L_predict   = sum_Z w(Z) L_pred(Z)                                    (soft EM)
  L_filter    = KL(w || b_t)                                            (distil hindsight into the causal belief)
  L_sigreg    = SIGReg (LeJEPA, Balestriero & LeCun 2025: sliced Epps-Pulley test, lejepa package) on e_t and o_t
  L_state     = l_occ KL(mean_batch p(r) || uniform) + l_ent mean H(p(r)),  p(r) = sum_k w(r, k)
  L_kl        = KL(b_t || uniform)                                      (compression)
Note: the code-usage terms need a gradient, so they use the non-detached softmax(-L_pred / T) (w itself is detached
for L_predict / L_filter as the handoff says).
"""
from __future__ import annotations
import math

import torch


def temperature(step: int, cfg: dict) -> float:
    t = cfg['loss']['temperature']
    a = min(step / max(t['anneal_steps'], 1), 1.0)
    return float(t['t0'] * (t['t_min'] / t['t0']) ** a)                # geometric anneal t0 -> t_min


class SIGReg:
    def __init__(self, num_slices: int = 1024, n_points: int = 17):
        import lejepa
        self.loss = lejepa.multivariate.SlicingUnivariateTest(univariate_test=lejepa.univariate.EppsPulley(n_points=n_points),
                                                              num_slices=num_slices)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.loss.to(x.device)                      # the lejepa module keeps its integration points as a CPU buffer
        return self.loss(x.float())


def pif_losses(model, batch, cfg: dict, step: int, sigreg: SIGReg, n_sample: int, rng: torch.Generator) -> dict:
    L, H = batch['n_input'], batch['horizons'][0]
    e, o, u = model.encode(batch)                                     # [B, N, ...] input + future frames
    obj_mask, valid = batch['obj_mask'], batch['valid']
    B, N, K, d = o.shape
    log_b, rho, m, ok_all = model.belief(e[:, :L], o[:, :L], obj_mask)
    # sampled input frames with at least one valid horizon
    tv = batch['target_valid']                                        # [B, L, nH]
    score = tv.any(-1).float() + torch.rand(B, L, generator=rng, device=tv.device) * 0.5
    t_idx = score.topk(n_sample, dim=1).indices                       # [B, S]
    gather = lambda x, idx: torch.gather(x, 1, idx.reshape(B, -1, *[1] * (x.dim() - 2)).expand(-1, -1, *x.shape[2:]))
    e_s, o_s, ok_s = gather(e, t_idx), gather(o, t_idx), gather(ok_all, t_idx)
    m_s = gather(m, t_idx) if model.predictor.in_m is not None else None
    obj_hat, hum_hat = model.predictor(e_s, ok_s, o_s, m_s)           # [B,S,R,K+1,H,K,d], [B,S,R,K+1,H,d]
    fut = t_idx[:, :, None] + H[None, None]                           # [B, S, nH]
    fut_c = fut.clamp(max=N - 1)
    o_tgt = gather(o.detach(), fut_c.reshape(B, -1)).reshape(B, *fut.shape[1:], K, d)
    u_tgt = gather(u.detach(), fut_c.reshape(B, -1)).reshape(B, *fut.shape[1:], d)
    tv_s = torch.gather(tv, 1, t_idx[..., None].expand(-1, -1, tv.shape[-1])).float()          # [B, S, nH]
    om = obj_mask.float()[:, None, None, None, None, :]
    err_o = ((obj_hat - o_tgt[:, :, None, None]).abs().mean(-1) * om).sum(-1) / om.sum(-1).clamp(min=1)
    err_u = (hum_hat - u_tgt[:, :, None, None]).abs().mean(-1)
    l_pred = ((err_o + err_u) * tv_s[:, :, None, None]).sum(-1)       # [B, S, R, K+1]
    zvalid = model.filter.valid_z(obj_mask)[:, None].expand_as(l_pred)
    T = temperature(step, cfg)
    neg = torch.where(zvalid, -l_pred / T, torch.full_like(l_pred, -1e9))
    w_soft = torch.softmax(neg.flatten(2), -1).reshape(l_pred.shape)
    w = w_soft.detach()
    l_predict = (w * l_pred).flatten(2).sum(-1).mean()
    lb_s = gather(log_b, t_idx)
    l_filter = (w * (torch.log(w + 1e-12) - lb_s)).flatten(2).sum(-1).mean()
    p_r = w_soft.sum(-1)                                              # [B, S, R]
    R = p_r.shape[-1]
    mean_p = p_r.mean((0, 1))
    l_occ = (mean_p * (torch.log(mean_p + 1e-12) + math.log(R))).sum()
    l_ent = -(p_r * torch.log(p_r + 1e-12)).sum(-1).mean()
    b = log_b.exp()
    vz = model.filter.valid_z(obj_mask)
    n_valid = vz.flatten(1).sum(1).float()
    l_kl = (b * (log_b + torch.log(n_valid)[:, None, None, None])).masked_fill(~vz[:, None], 0).flatten(2).sum(-1)
    l_kl = (l_kl * valid[:, :L]).sum() / valid[:, :L].sum()
    ev = valid.reshape(-1)
    cap = cfg['loss']['sigreg_max_samples']
    sub = lambda x: x[torch.randperm(len(x), device=x.device, generator=rng)[:cap]] if len(x) > cap else x
    l_sig = sigreg(sub(e.reshape(B * N, d)[ev])) + sigreg(sub(o[obj_mask[:, None].expand(B, N, K) & valid[..., None]]))
    lw = cfg['loss']['weights']
    total = (l_predict + lw['filter'] * l_filter + lw['sigreg'] * l_sig + lw['occ'] * l_occ + lw['ent'] * l_ent
             + lw['kl'] * l_kl)
    with torch.no_grad():
        p_k_none = w[..., 0].sum(-1).mean()
        eff_codes = torch.exp(-(mean_p * torch.log(mean_p + 1e-12)).sum())
        h_b = -(b * log_b.clamp(min=-30)).flatten(2).sum(-1)
        h_b = (h_b * valid[:, :L]).sum() / valid[:, :L].sum()
    return dict(total=total, predict=l_predict, filter=l_filter, sigreg=l_sig, occ=l_occ, ent=l_ent, kl=l_kl,
                temperature=torch.tensor(T), eff_codes=eff_codes, belief_entropy=h_b, w_none=p_k_none,
                rho=(rho * valid[:, :L]).sum() / valid[:, :L].sum())
