"""Predictive Intent Filter (handoff §4; Huang 2026, "Your Probabilistic JEPA Is Secretly a Hidden Markov Model").

FrameEncoder f  : Transformer over one frame's token set -> e_t (CLS, pooled), o_t^k (per object), u_t (human).
BeliefFilter    : m_t = GRU(m_{t-1}, e_t); evidence log e_t(r, k) = MLP_r(m_t) + <W m_t, o_t^k> (k = none: learned
                  embedding); keep probability rho_t; prediction b-_t = rho b_{t-1} + (1 - rho) uniform; update
                  b_t ∝ e_t * b-_t, in log space over the valid Z = (r, k), k in {none} + real objects.
Predictor g     : sees e_t (the CURRENT frame only), the code embedding C_r + o_t^k and the horizon embedding; it never
                  gets m_t or any history (`predictor_sees_history` = ablation), so everything beyond the current frame
                  that helps predicting the future has to go through Z = the belief. Outputs the future embedding of
                  every object and of the human.
Index 0 of the k axis is `none`; objects are 1..K (padding masked).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

HUMAN_DIM, OBJECT_DIM = 7, 6


def mlp(din: int, dh: int, dout: int, layers: int = 2) -> nn.Sequential:
    mods, d = [], din
    for _ in range(layers - 1):
        mods += [nn.Linear(d, dh), nn.GELU()]
        d = dh
    return nn.Sequential(*mods, nn.Linear(d, dout))


class FrameEncoder(nn.Module):
    def __init__(self, d: int, layers: int, heads: int, n_joints: int, n_types: int):
        super().__init__()
        self.human_in = nn.Linear(HUMAN_DIM, d)
        self.joint_emb = nn.Parameter(torch.randn(n_joints, d) * 0.02)
        self.obj_in = nn.Linear(OBJECT_DIM + 1, d)                     # + held-by-robot flag
        self.type_emb = nn.Embedding(n_types, d)
        self.robot_emb = nn.Embedding(2, d)                            # robot holds nothing / something
        self.cls = nn.Parameter(torch.randn(d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, human, objects, obj_type, obj_mask, holding):
        """human [B, N, J, 7], objects [B, N, K, 6], obj_type [B, K], obj_mask [B, K], holding [B, N] (-1 = none).
        Returns e [B, N, d], o [B, N, K, d], u [B, N, d]."""
        B, N, J, _ = human.shape
        K = objects.shape[2]
        held = (holding[..., None] == torch.arange(K, device=holding.device)).float()          # [B, N, K]
        h = self.human_in(human) + self.joint_emb
        o = self.obj_in(torch.cat([objects, held[..., None]], -1)) + self.type_emb(obj_type)[:, None]
        r = self.robot_emb((holding >= 0).long())[:, :, None]
        c = self.cls.expand(B, N, 1, -1)
        x = torch.cat([c, r, h, o], 2).reshape(B * N, 2 + J + K, -1)
        pad = torch.cat([torch.zeros(B, 2 + J, dtype=torch.bool, device=obj_mask.device), ~obj_mask], 1)
        pad = pad[:, None].expand(B, N, -1).reshape(B * N, -1)
        y = self.norm(self.tf(x, src_key_padding_mask=pad)).reshape(B, N, 2 + J + K, -1)
        return y[:, :, 0], y[:, :, 2 + J:], y[:, :, 2:2 + J].mean(2)


class BeliefFilter(nn.Module):
    def __init__(self, d: int, R: int):
        super().__init__()
        self.R = R
        self.gru = nn.GRUCell(d, d)
        self.rel = mlp(d, d, R)
        self.W = nn.Linear(d, d, bias=False)
        self.none = nn.Parameter(torch.randn(d) * 0.02)
        self.keep = mlp(d, d // 2, 1)

    def valid_z(self, obj_mask):
        """[B, R, K + 1] valid codes (none always valid)."""
        B = obj_mask.shape[0]
        k = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=obj_mask.device), obj_mask], 1)
        return k[:, None].expand(B, self.R, -1)

    def forward(self, e, o, obj_mask, history: bool = True):
        """e [B, T, d], o [B, T, K, d] -> log_b [B, T, R, K + 1], rho [B, T], m [B, T, d], ok_all [B, T, K+1, d].
        history=False: no filter, belief = softmax of the current evidence (ablation `no_filter`)."""
        B, T, d = e.shape
        valid = self.valid_z(obj_mask)
        n_valid = valid.flatten(1).sum(1).float()
        log_u = torch.where(valid, -torch.log(n_valid)[:, None, None], torch.full_like(valid, -1e9, dtype=e.dtype))
        m = torch.zeros(B, d, device=e.device, dtype=e.dtype)
        log_b = log_u
        out_b, out_rho, out_m = [], [], []
        ok_all = torch.cat([self.none.expand(B, T, 1, d), o], 2)
        for t in range(T):
            m = self.gru(e[:, t], m) if history else self.gru(e[:, t], torch.zeros_like(m))
            ev = self.rel(m)[:, :, None] + torch.einsum('bd,bkd->bk', self.W(m), ok_all[:, t])[:, None] / d ** 0.5
            rho = torch.sigmoid(self.keep(m))[:, 0]
            if history:
                prior = torch.logaddexp(torch.log(rho + 1e-6)[:, None, None] + log_b,
                                        torch.log(1 - rho + 1e-6)[:, None, None] + log_u)
            else:
                prior = log_u
            logits = torch.where(valid, ev + prior, torch.full_like(ev, -1e9))
            log_b = logits - torch.logsumexp(logits.flatten(1), 1)[:, None, None]
            out_b.append(log_b)
            out_rho.append(rho)
            out_m.append(m)
        return torch.stack(out_b, 1), torch.stack(out_rho, 1), torch.stack(out_m, 1), ok_all


class Predictor(nn.Module):
    """g(e_t, Z, h) -> future embeddings. The first layer is split by input so the (Z x object x horizon) grid is a
    broadcast sum instead of a concatenation."""

    def __init__(self, d: int, R: int, n_h: int, hidden: int, history: bool = False):
        super().__init__()
        self.C = nn.Parameter(torch.randn(R, d) * 0.02)
        self.h_emb = nn.Embedding(n_h, hidden)
        self.in_e, self.in_z, self.in_o = nn.Linear(d, hidden), nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False)
        self.in_m = nn.Linear(d, hidden, bias=False) if history else None
        self.obj_out = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.hum_out = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, e, ok, o, m=None):
        """e [B, S, d], ok [B, S, K+1, d] (none + objects), o [B, S, K, d] -> obj [B, S, R, K+1, H, K, d],
        hum [B, S, R, K+1, H, d] for every code Z = (r, k) and horizon index."""
        z = self.C[None, None, :, None] + ok[:, :, None]                                   # [B, S, R, K+1, d]
        base = self.in_e(e)[:, :, None, None] + self.in_z(z)                               # [B, S, R, K+1, hid]
        if self.in_m is not None:
            base = base + self.in_m(m)[:, :, None, None]
        base = base[:, :, :, :, None] + self.h_emb.weight                                   # [B, S, R, K+1, H, hid]
        hum = self.hum_out(base)
        obj = self.obj_out(base[:, :, :, :, :, None] + self.in_o(o)[:, :, None, None, None])
        return obj, hum


class PIF(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d, R = cfg['d_model'], cfg['num_relations']
        ab = cfg.get('ablation') or {}
        self.history = not ab.get('no_filter', False)
        self.encoder = FrameEncoder(d, cfg['encoder_layers'], cfg['heads'], len(cfg['tokens']['keypoints']),
                                    len(cfg['tokens']['object_types']))
        self.filter = BeliefFilter(d, R)
        self.predictor = Predictor(d, R, len(cfg['horizons_s']), cfg.get('predictor_hidden', 2 * d),
                                   history=ab.get('predictor_sees_history', False))

    def encode(self, batch):
        return self.encoder(batch['human'], batch['objects'], batch['obj_type'], batch['obj_mask'], batch['holding'])

    def belief(self, e, o, obj_mask):
        return self.filter(e, o, obj_mask, history=self.history)

    @torch.no_grad()
    def infer(self, batch):
        """Causal belief over a whole sequence: log_b [B, T, R, K+1], rho [B, T]."""
        e, o, _ = self.encode(batch)
        log_b, rho, _, _ = self.belief(e, o, batch['obj_mask'])
        return log_b, rho
