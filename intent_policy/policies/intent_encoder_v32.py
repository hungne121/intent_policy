"""Intent tokens v3.2 (docs/requirements/INTENT_ACT_v3.2.md §5.2): 6 tokens + a key padding mask.

    contract v3.2 fields -> IntentEncoderV32 -> [HUMAN, SEM, TARGET, C_TARGET, TIME, MOTION] (B, 6, d), mask (B, 6)

  HUMAN     hand_hist (H x J x 3, normalised) ⊕ occupancy ⊕ min_dist ⊕ point_dir   -> MLP
  SEM       p_who ⊕ c_who ⊕ phase ⊕ p_gesture                                    -> Linear
  TARGET    p_target @ E_target + Linear([normalised entropy, max] of p_target)   (E_target: K x d, shared)
  C_TARGET  c_target @ E_target + committed embedding
  TIME      Fourier(tte / tte_max) ⊕ tte_std / tte_max                            -> MLP
  MOTION    xi (M x J x 3, normalised), flattened                                 -> Linear (one token)
Every token adds its own type embedding. The mask (True = the transformer ignores the token, in the encoder's
self-attention and the decoder's cross-attention, policies/intent_act.py) replaces v2's learned null tokens:
training drops all intent tokens with p_drop_all and each token with p_drop_token; human_present = 0 masks HUMAN and
MOTION; `keep` (evaluation ablations) masks the tokens not kept.
"""
import math

import torch
import torch.nn as nn

TOKENS_V32 = ('human', 'sem', 'target', 'c_target', 'time', 'motion')


class IntentEncoderV32(nn.Module):
    def __init__(self, d, n_targets, n_who, n_phases, n_gestures, n_waypoints, n_joints, hand_hist, kp_dim=3,
                 n_fourier=6, tte_max=3.0, target_init_std=0.02):
        """`target_init_std`: initial std of E_target and `committed`. 0.02 (D0) left the target identity 20-40 times
        smaller than the type embedding (N(0,1)); 1.0 gives them the type embedding's scale (§5.3 F1)."""
        super().__init__()
        self.K, self.n_fourier, self.tte_max = n_targets, n_fourier, tte_max
        h_in = hand_hist * n_joints * kp_dim + 7 + 1 + 6
        self.human_mlp = nn.Sequential(nn.Linear(h_in, d), nn.GELU(), nn.Linear(d, d))
        self.sem_proj = nn.Linear(2 * n_who + n_phases + n_gestures, d)
        self.E_target = nn.Parameter(torch.randn(n_targets, d) * target_init_std)
        self.target_conf = nn.Linear(2, d)
        self.committed = nn.Parameter(torch.randn(d) * target_init_std)
        self.time_mlp = nn.Sequential(nn.Linear(1 + 2 * n_fourier + 1, d), nn.GELU(), nn.Linear(d, d))
        self.motion_proj = nn.Linear(n_waypoints * n_joints * kp_dim, d)
        self.type_emb = nn.Embedding(len(TOKENS_V32), d)
        self.register_buffer('xi_mean', torch.zeros(n_waypoints, n_joints, kp_dim))
        self.register_buffer('xi_std', torch.ones(n_waypoints, n_joints, kp_dim))
        self.register_buffer('hist_mean', torch.zeros(hand_hist, n_joints, kp_dim))
        self.register_buffer('hist_std', torch.ones(hand_hist, n_joints, kp_dim))

    @property
    def n_tokens(self) -> int:
        return len(TOKENS_V32)

    def set_stats(self, xi_mean, xi_std, hist_mean, hist_std, min_std: float = 1e-2) -> None:
        """Normalisation of xi and hand_hist (training-split mean / std, std floored at 1 cm)."""
        for buf, val in ((self.xi_mean, xi_mean), (self.hist_mean, hist_mean)):
            buf.copy_(torch.as_tensor(val, dtype=buf.dtype).reshape(buf.shape))
        for buf, val in ((self.xi_std, xi_std), (self.hist_std, hist_std)):
            buf.copy_(torch.as_tensor(val, dtype=buf.dtype).reshape(buf.shape).clamp_min(min_std))

    def _time_feat(self, tte, tte_std):
        x = tte.clamp(0, self.tte_max) / self.tte_max
        k = torch.arange(self.n_fourier, device=x.device, dtype=x.dtype)
        ang = (2.0 ** k) * math.pi * x
        return torch.cat([x, ang.sin(), ang.cos(), tte_std.clamp(0, self.tte_max) / self.tte_max], dim=-1)

    @staticmethod
    def sample_keep(B, device, p_drop_token, p_drop_all) -> dict:
        """Training dropout: every token kept with 1 - p_drop_token, all of them dropped together with p_drop_all."""
        drop_all = torch.rand(B, device=device) < p_drop_all
        return {t: (torch.rand(B, device=device) >= p_drop_token) & ~drop_all for t in TOKENS_V32}

    def forward(self, intent: dict, keep: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """intent: the contract v3.2 fields (raw xi / hand_hist, normalised here), batched. Returns the tokens
        (B, 6, d) and the key padding mask (B, 6), True = ignored."""
        B = intent['p_target'].shape[0]
        hist = ((intent['hand_hist'] - self.hist_mean) / self.hist_std).flatten(1)
        human = self.human_mlp(torch.cat([hist, intent['occupancy'], intent['min_dist'], intent['point_dir']], -1))
        sem = self.sem_proj(torch.cat([intent['p_who'], intent['c_who'], intent['phase'], intent['p_gesture']], -1))
        p = intent['p_target'].clamp_min(1e-12)
        conf = torch.stack([1.0 + (p * p.log()).sum(-1) / math.log(self.K), p.max(-1).values], -1)
        target = intent['p_target'] @ self.E_target + self.target_conf(conf)
        c_target = intent['c_target'] @ self.E_target + self.committed
        time = self.time_mlp(self._time_feat(intent['tte'], intent['tte_std']))
        motion = self.motion_proj(((intent['xi'] - self.xi_mean) / self.xi_std).flatten(1))
        toks = torch.stack([human, sem, target, c_target, time, motion], 1) + self.type_emb.weight[None]
        mask = torch.zeros(B, len(TOKENS_V32), dtype=torch.bool, device=toks.device)
        absent = intent['human_present'].reshape(B) < 0.5
        mask[:, TOKENS_V32.index('human')] |= absent
        mask[:, TOKENS_V32.index('motion')] |= absent
        if keep is not None:
            for i, t in enumerate(TOKENS_V32):
                mask[:, i] |= ~keep[t]
        return toks, mask


class IntentFiLMV32(nn.Module):
    """FiLM of every ResNet stage by the v3.2 intent tokens (INTENT_ACT_v3.2.md §5.3; FiLM in each backbone block as in
    BC-Z / RT-1 / MT-ACT). The tokens are concatenated with the masked ones zeroed, so the modulation sees which token
    carries what and a dropped / absent token contributes nothing. Zero-initialised heads: identity at initialisation
    (adaLN-Zero, DiT)."""

    def __init__(self, d, n_tokens, channels: dict[str, int]):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(n_tokens * d, d), nn.GELU())
        self.heads = nn.ModuleDict({k: nn.Linear(d, 2 * c) for k, c in channels.items()})
        for h in self.heads.values():
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> dict:
        """tokens (B, N, d), mask (B, N) True = masked -> {stage: (gamma, beta)} each (B, C)."""
        z = self.trunk((tokens * (~mask)[..., None].to(tokens.dtype)).flatten(1))
        return {k: h(z).chunk(2, dim=-1) for k, h in self.heads.items()}

    @staticmethod
    def modulate(feat: torch.Tensor, gb) -> torch.Tensor:
        g, b = gb
        return feat * (1 + g[..., None, None]) + b[..., None, None]


class IntentEncoderUnified(nn.Module):
    """One intent token from the measuring device's output only (trial of 2026-10-18; intent/unified.py, doc 'Hướng
    triển khai thesis hiện tại' §2.2 / §4): u_t over the K schema targets + `hold`

        hold = p_gesture(palm_forward)                            ('wait'; withdraw left out on 2026-10-18: under the
                                                                    continuous robot motion it had 12-48 % false-alarm
                                                                    episodes, no hold reached <= 5 %, report doc 2b)
        u_t  = [(1 - hold) p_target, hold]                         (p_target: the device's FETCH-POMDP belief)
        token = u_t @ E_u + type embedding

    E_u ~ N(0, 1), the scale of the type embedding (the F1 finding of 2026-10-14: a target embedding 20-40x smaller
    than the type embedding left D ignoring the intent). It reads the contract v3.2 fields p_target / p_gesture and
    nothing else (no simulator keypoints: HUMAN / MOTION tokens of v3.2 carried those). The mask / keep interface is
    the v3.2 one: the single token follows keep['target']."""

    def __init__(self, d, n_targets, n_gestures, hold_classes=(4,)):
        super().__init__()
        self.K = n_targets
        self.hold_idx = list(hold_classes)            # contract p_gesture order: rest, point, reach, palm_up, palm_forward, withdraw
        if max(self.hold_idx) >= n_gestures:
            raise ValueError(f'hold classes {self.hold_idx} outside the {n_gestures} gesture classes')
        self.E_u = nn.Parameter(torch.randn(n_targets + 1, d))
        self.type_emb = nn.Embedding(1, d)

    @property
    def n_tokens(self) -> int:
        return 1

    def set_stats(self, *args, **kwargs) -> None:     # nothing to normalise
        return None

    @staticmethod
    def sample_keep(B, device, p_drop_token, p_drop_all) -> dict:
        return IntentEncoderV32.sample_keep(B, device, p_drop_token, p_drop_all)

    def unified(self, intent: dict) -> torch.Tensor:
        p = intent['p_target'].clamp_min(0)
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
        hold = intent['p_gesture'][..., self.hold_idx].sum(-1, keepdim=True).clamp(0, 1)
        return torch.cat([(1 - hold) * p, hold], -1)

    def forward(self, intent: dict, keep: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.unified(intent)
        tok = (u @ self.E_u + self.type_emb.weight[0])[:, None]
        mask = torch.zeros(u.shape[0], 1, dtype=torch.bool, device=u.device)
        if keep is not None:
            mask[:, 0] |= ~keep['target']
        return tok, mask
