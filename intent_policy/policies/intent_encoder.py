"""Intent tokens for the ACT transformer (docs/requirements/INTENT_ACT_GUIDE_v2.md §4.1-4.2, §4.5).

    contract fields -> IntentEncoder -> [WHO, TARGET, C_TARGET, OCC, TIME, XI_1..XI_M] (B, 5 + M, d)

  WHO       p_who ⊕ c_who               Linear(2W, d)                              group semantic
  TARGET    p_target                    p_target @ E_target  (E_target: K x d, shared)    group spatial
  C_TARGET  c_target                    c_target @ E_target + committed embedding  group memory
  OCC       occupancy                   Linear(7, d)                               group spatial
  TIME      tte, tte_std, phase, conf.  Fourier(tte) ⊕ the others -> MLP           group time
  XI_m      xi[m] (J * 3, normalised)   Linear(J * 3, d) + waypoint embedding m    group motion
Every token adds its group's type embedding. Group dropout (training) and `keep` (evaluation ablations) replace a
group's tokens by learned null tokens (one per token) + the type embedding. Soft and one-hot distributions are
encoded identically (a distribution times E_target), so hard labels are a special case.
"""
import math

import torch
import torch.nn as nn

GROUPS = ('semantic', 'spatial', 'memory', 'time', 'motion')
TOKENS = ('who', 'target', 'c_target', 'occ', 'time')           # + M xi tokens
GROUP_OF = {'who': 'semantic', 'target': 'spatial', 'c_target': 'memory', 'occ': 'spatial', 'time': 'time', 'xi': 'motion'}


class IntentEncoder(nn.Module):
    def __init__(self, d, n_targets, n_who, n_phases, n_waypoints, n_joints, kp_dim=3, n_fourier=6, tte_max=3.0):
        super().__init__()
        self.K, self.W, self.P, self.M = n_targets, n_who, n_phases, n_waypoints
        self.n_fourier, self.tte_max = n_fourier, tte_max
        self.who_proj = nn.Linear(2 * n_who, d)
        self.E_target = nn.Parameter(torch.randn(n_targets, d) * 0.02)
        self.committed = nn.Parameter(torch.randn(d) * 0.02)
        self.occ_proj = nn.Linear(7, d)
        self.time_mlp = nn.Sequential(nn.Linear(2 + 2 * n_fourier + 1 + n_phases + 2, d), nn.GELU(), nn.Linear(d, d))
        self.xi_proj = nn.Linear(n_joints * kp_dim, d)
        self.xi_time = nn.Parameter(torch.randn(n_waypoints, d) * 0.02)
        self.null = nn.ParameterDict({k: nn.Parameter(torch.randn(1, d) * 0.02) for k in TOKENS})
        self.null_xi = nn.Parameter(torch.randn(n_waypoints, d) * 0.02)
        self.type_emb = nn.Embedding(len(GROUPS), d)
        self.register_buffer('xi_mean', torch.zeros(n_waypoints, n_joints, kp_dim))
        self.register_buffer('xi_std', torch.ones(n_waypoints, n_joints, kp_dim))

    @property
    def n_tokens(self) -> int:
        return len(TOKENS) + self.M

    def set_xi_stats(self, mean, std, min_std: float = 1e-2) -> None:
        self.xi_mean.copy_(torch.as_tensor(mean, dtype=self.xi_mean.dtype).reshape(self.xi_mean.shape))
        self.xi_std.copy_(torch.as_tensor(std, dtype=self.xi_std.dtype).reshape(self.xi_std.shape).clamp_min(min_std))

    def normalize_xi(self, xi):
        return (xi - self.xi_mean) / self.xi_std

    def _time_feat(self, tte, tte_std, phase, conf):
        x = tte.clamp(0, self.tte_max) / self.tte_max
        k = torch.arange(self.n_fourier, device=x.device, dtype=x.dtype)
        ang = (2.0 ** k) * math.pi * x
        return torch.cat([x, torch.log1p(tte.clamp(0, self.tte_max)), ang.sin(), ang.cos(), tte_std / self.tte_max,
                          phase, conf], dim=-1)

    def sample_keep(self, B, device, p_drop_group, p_drop_all):
        keep = {g: torch.rand(B, device=device) >= p_drop_group for g in GROUPS}
        drop_all = torch.rand(B, device=device) < p_drop_all
        return {g: k & ~drop_all for g, k in keep.items()}

    def forward(self, intent: dict, keep: dict | None = None) -> torch.Tensor:
        """intent: p_who, c_who (B, W), p_target, c_target (B, K), occupancy (B, 7), tte, tte_std (B, 1), phase (B, P),
        confidence (B, 2), xi (B, M, J, D) normalised. keep: group -> BoolTensor (B,) (None: all kept)."""
        B = intent['p_target'].shape[0]
        toks = {'who': self.who_proj(torch.cat([intent['p_who'], intent['c_who']], -1)),
                'target': intent['p_target'] @ self.E_target,
                'c_target': intent['c_target'] @ self.E_target + self.committed,
                'occ': self.occ_proj(intent['occupancy']),
                'time': self.time_mlp(self._time_feat(intent['tte'], intent['tte_std'], intent['phase'], intent['confidence']))}
        out = []
        for name in TOKENS:
            g = GROUP_OF[name]
            t = toks[name]
            if keep is not None:
                t = torch.where(keep[g][:, None], t, self.null[name].expand(B, -1))
            out.append((t + self.type_emb.weight[GROUPS.index(g)])[:, None])
        xi = self.xi_proj(intent['xi'].flatten(2)) + self.xi_time
        if keep is not None:
            xi = torch.where(keep['motion'][:, None, None], xi, self.null_xi.expand(B, -1, -1))
        out.append(xi + self.type_emb.weight[GROUPS.index('motion')])
        return torch.cat(out, dim=1)


class IntentFiLM(nn.Module):
    """FiLM of a backbone feature map by the mean intent token (identity at initialisation). Fallback, §4.5."""

    def __init__(self, d, C):
        super().__init__()
        self.to_gb = nn.Linear(d, 2 * C)
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)

    def forward(self, feat, intent_tok):          # feat (B, C, H, W)
        g, b = self.to_gb(intent_tok.mean(1)).chunk(2, dim=-1)
        return feat * (1 + g[..., None, None]) + b[..., None, None]
