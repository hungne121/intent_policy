"""Restricted diagnostic action head: 9 generic motor primitives on top of the ACT latent.

The head is intentionally tiny (a linear layer, optionally a one-hidden-layer MLP). It maps
each ACT decoder token (one per chunk step) to 9 logits, so the restricted head keeps ACT's
action chunking: logits have shape (B, chunk_size, 9).
"""
from torch import Tensor, nn

from controllers.restricted_action import NUM_RESTRICTED_ACTIONS


class RestrictedActionHead(nn.Module):
    def __init__(self, latent_dim: int, num_actions: int = NUM_RESTRICTED_ACTIONS, hidden_dim: int | None = None):
        super().__init__()
        if num_actions != NUM_RESTRICTED_ACTIONS:
            raise ValueError(f'the restricted action space has exactly {NUM_RESTRICTED_ACTIONS} actions')
        self.num_actions = num_actions
        if hidden_dim:
            self.net = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_actions))
        else:
            self.net = nn.Linear(latent_dim, num_actions)

    def forward(self, latent: Tensor) -> Tensor:
        """latent: (B, S, D) ACT decoder output -> logits (B, S, 9)."""
        return self.net(latent)
