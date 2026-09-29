"""Intention encoders and fusion with the ACT policy latent h (agent.md section 2 architecture).

    h (B, S, D) ─────────────────────────────┐
    branch features (normalised) -> branch MLP -> z_b (B, E) ─┴─> h' = h + W · MLP([h, z_1 .. z_n])

Each branch (e.g. spatial, motion) has its own encoder with the same embedding size. The output
projection W is zero-initialised, so an untrained fusion leaves h unchanged. With `mask=True`
the branch inputs are zeroed (capacity-matched No-Information control: same parameters, no information).
"""
import torch
from torch import Tensor, nn


class BranchEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, embed: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, embed), nn.LayerNorm(embed))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class IntentFusion(nn.Module):
    def __init__(self, latent_dim: int, branch_dims: dict[str, int], hidden: int = 128, embed: int = 64):
        super().__init__()
        self.branch_names = list(branch_dims)
        self.encoders = nn.ModuleDict({b: BranchEncoder(d, hidden, embed) for b, d in branch_dims.items()})
        self.fuse = nn.Sequential(nn.Linear(latent_dim + embed * len(branch_dims), latent_dim), nn.ReLU())
        self.out = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h: Tensor, branches: dict[str, Tensor], mask: bool = False) -> Tensor:
        """h: (B, S, D); branches: name -> (B, F_b) normalised features."""
        z = [self.encoders[b](torch.zeros_like(branches[b]) if mask else branches[b]) for b in self.branch_names]
        z = torch.cat(z, dim=-1).unsqueeze(1).expand(-1, h.shape[1], -1)
        return h + self.out(self.fuse(torch.cat([h, z.to(h.dtype)], dim=-1)))
