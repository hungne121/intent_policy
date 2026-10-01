"""Intent tokens for the ACT transformer (docs/requirements/INTENT_ACT_GUIDE.md §2.1, §2.6).

    intent {obj, act, tau, xi} -> IntentEncoder -> tokens (B, N, D), N = [obj] + [act] + [tau] + M [xi]

Each component has a learned UNKNOWN / null token used when it is dropped (`keep[c] = False`): component dropout
during training, ablations at test time. `none` (a data label: no intention) is a normal class, UNKNOWN is the
extra embedding row at index n_cls. obj / act accept hard labels (LongTensor (B,)) or soft probabilities
(FloatTensor (B, n_cls)). `xi_mean` / `xi_std` (buffers, set from the training split) normalise raw xi.
"""
import math

import torch
import torch.nn as nn

ALL_COMPONENTS = ('obj', 'act', 'tau', 'xi')
TYPE_ID = {'obj': 0, 'act': 1, 'tau': 2, 'xi': 3}


class IntentEncoder(nn.Module):
    """intent dict -> tokens (B, N, d).
    keep[c] (BoolTensor B): False -> the learned UNKNOWN / null token of component c."""

    def __init__(self, d, n_obj, n_act, n_waypoints, n_joints, kp_dim=3, components=ALL_COMPONENTS, n_fourier=6,
                 tte_max=3.0):
        super().__init__()
        unknown = set(components) - set(ALL_COMPONENTS)
        if unknown or not components:
            raise ValueError(f'intent components must be a non-empty subset of {ALL_COMPONENTS}, got {components}')
        self.components = tuple(c for c in ALL_COMPONENTS if c in components)
        self.n_obj, self.n_act, self.M = n_obj, n_act, n_waypoints
        self.n_fourier, self.tte_max = n_fourier, tte_max
        self.obj_emb = nn.Embedding(n_obj + 1, d)          # index n_obj = UNKNOWN
        self.act_emb = nn.Embedding(n_act + 1, d)          # index n_act = UNKNOWN
        self.tau_mlp = nn.Sequential(nn.Linear(2 + 2 * n_fourier, d), nn.GELU(), nn.Linear(d, d))
        self.null_tau = nn.Parameter(torch.randn(1, d) * 0.02)
        self.xi_proj = nn.Linear(n_joints * kp_dim, d)
        self.xi_time = nn.Parameter(torch.randn(n_waypoints, d) * 0.02)
        self.null_xi = nn.Parameter(torch.randn(n_waypoints, d) * 0.02)
        self.type_emb = nn.Embedding(4, d)
        self.register_buffer('xi_mean', torch.zeros(n_waypoints, n_joints, kp_dim))
        self.register_buffer('xi_std', torch.ones(n_waypoints, n_joints, kp_dim))

    @property
    def n_tokens(self):
        return sum(self.M if c == 'xi' else 1 for c in self.components)

    def set_xi_stats(self, mean, std, min_std: float = 1e-2) -> None:
        self.xi_mean.copy_(torch.as_tensor(mean, dtype=self.xi_mean.dtype).reshape(self.xi_mean.shape))
        self.xi_std.copy_(torch.as_tensor(std, dtype=self.xi_std.dtype).reshape(self.xi_std.shape).clamp_min(min_std))

    def normalize_xi(self, xi):
        return (xi - self.xi_mean) / self.xi_std

    @staticmethod
    def _label_tok(x, emb, n_cls, keep):
        tok = emb(x.long()) if not x.is_floating_point() else x @ emb.weight[:n_cls]
        return torch.where(keep[:, None], tok, emb.weight[n_cls].expand_as(tok))

    def _tau_feat(self, tau):
        phase, tte = tau[:, :1], tau[:, 1:].clamp(0, self.tte_max)
        k = torch.arange(self.n_fourier, device=tau.device, dtype=tau.dtype)
        ang = (2.0 ** k) * math.pi * phase
        return torch.cat([phase, torch.log1p(tte), ang.sin(), ang.cos()], dim=-1)

    def sample_keep(self, B, device, p_drop, p_drop_all):
        keep = {c: torch.rand(B, device=device) >= p_drop for c in self.components}
        drop_all = torch.rand(B, device=device) < p_drop_all
        return {c: k & ~drop_all for c, k in keep.items()}

    def forward(self, intent, keep=None):
        B = intent[self.components[0] if self.components[0] in intent else 'obj'].shape[0]
        dev = self.type_emb.weight.device
        if keep is None:
            keep = {c: torch.ones(B, dtype=torch.bool, device=dev) for c in self.components}
        toks = []
        for c in self.components:
            if c == 'obj':
                t = self._label_tok(intent['obj'], self.obj_emb, self.n_obj, keep['obj'])[:, None]
            elif c == 'act':
                t = self._label_tok(intent['act'], self.act_emb, self.n_act, keep['act'])[:, None]
            elif c == 'tau':
                t = self.tau_mlp(self._tau_feat(intent['tau']))
                t = torch.where(keep['tau'][:, None], t, self.null_tau.expand(B, -1))[:, None]
            else:
                t = self.xi_proj(intent['xi'].flatten(2)) + self.xi_time
                t = torch.where(keep['xi'][:, None, None], t, self.null_xi.expand(B, -1, -1))
            toks.append(t + self.type_emb.weight[TYPE_ID[c]])
        return torch.cat(toks, dim=1)


class IntentFiLM(nn.Module):
    """FiLM of a backbone feature map by the mean intent token (identity at initialisation). Fallback, §2.6."""

    def __init__(self, d, C):
        super().__init__()
        self.to_gb = nn.Linear(d, 2 * C)
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)

    def forward(self, feat, intent_tok):          # feat (B, C, H, W)
        g, b = self.to_gb(intent_tok.mean(1)).chunk(2, dim=-1)
        return feat * (1 + g[..., None, None]) + b[..., None, None]
