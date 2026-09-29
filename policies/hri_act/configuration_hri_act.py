"""Configuration for the project's ACT policy: LeRobot `ACTConfig` + action-mode selection + intent fusion.

Everything ACT-related (backbone, transformer, chunking, VAE, normalisation, optimiser preset)
is inherited unchanged from LeRobot. Project-specific fields select the action head and, from
Phase 2, the intention-information branches fused with the ACT latent.
"""
from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig

ACTION_MODES = ('restricted', 'continuous')
INTENT_PREFIX = 'observation.oracle.'


@PreTrainedConfig.register_subclass('hri_act')
@dataclass
class HRIACTConfig(ACTConfig):
    # 'restricted': 9-action diagnostic head on the ACT decoder latent (Phases 1-3).
    # 'continuous': original LeRobot ACT regression head / action chunk (Phase 4).
    action_mode: str = 'restricted'
    num_restricted_actions: int = 9
    restricted_head_hidden_dim: int | None = None
    # Dataset key holding the restricted action id. Deliberately NOT prefixed with "action" so
    # LeRobot never treats it as a continuous ACTION feature (it would get normalised).
    restricted_label_key: str = 'restricted_action'
    # Optional auxiliary L1 weight on the continuous head while training in restricted mode.
    continuous_aux_weight: float = 0.0
    # Intention information (Phase 2+): numeric oracle features grouped into branches, each with its own
    # encoder; the branch embeddings are fused with the ACT latent h before the action heads.
    use_intent: bool = False
    intent_branches: dict = field(default_factory=dict)      # branch name -> list of observation.oracle.* keys
    intent_branch_dims: dict = field(default_factory=dict)   # branch name -> feature width (set from the dataset)
    intent_embed_dim: int = 64
    intent_hidden_dim: int = 128
    # Capacity-matched No-Information control: same encoders/fusion, intention inputs replaced by zeros.
    intent_mask: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.action_mode not in ACTION_MODES:
            raise ValueError(f'action_mode must be one of {ACTION_MODES}, got {self.action_mode!r}')
        if self.num_restricted_actions != 9:
            raise ValueError('the restricted action space has exactly 9 actions')
        if self.use_intent:
            if not self.intent_branches or not all(self.intent_branches.values()):
                raise ValueError('use_intent needs non-empty intent_branches')
            bad = [k for keys in self.intent_branches.values() for k in keys if not k.startswith(INTENT_PREFIX)]
            if bad:
                raise ValueError(f'intent features must be numeric oracle fields ({INTENT_PREFIX}*), got {bad}')
            if set(self.intent_branch_dims) != set(self.intent_branches):
                raise ValueError('intent_branch_dims must give the width of every intent branch')

    @property
    def intent_keys(self) -> list[str]:
        """Oracle features the policy reads (none for the masked No-Information control)."""
        if not self.use_intent or self.intent_mask:
            return []
        return [k for keys in self.intent_branches.values() for k in keys]
