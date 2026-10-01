"""Configuration for the project's ACT policy: LeRobot `ACTConfig` + action-mode selection + intention input.

Everything ACT-related (backbone, transformer, chunking, VAE, normalisation, optimiser preset)
is inherited unchanged from LeRobot. Project-specific fields select the action head and how intention
information enters the policy: `fusion` (Phase 2: oracle feature branches fused with the ACT latent) or
`tokens` (docs/requirements/INTENT_ACT_GUIDE.md: intent labels {obj, act, tau, xi} as transformer tokens).
"""
from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig

ACTION_MODES = ('restricted', 'continuous')
INTENT_PREFIX = 'observation.oracle.'
INTENT_ARCHS = ('fusion', 'tokens')
INTENT_COMPONENTS = ('obj', 'act', 'tau', 'xi')
SCHEMA_KEYS = ('obj_vocab', 'act_vocab', 'n_waypoints', 'keypoints', 'keypoint_dim', 'tte_max_s')


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
    # Intention information. intent_arch 'fusion' (Phase 2+): numeric oracle features grouped into branches, each
    # with its own encoder; the branch embeddings are fused with the ACT latent h before the action heads.
    # 'tokens' (INTENT_ACT_GUIDE.md): per-timestep intent labels (dataset keys intent.*) encoded as N extra tokens of
    # the ACT transformer encoder and of the CVAE encoder (policies/intent_act.py).
    use_intent: bool = False
    intent_arch: str = 'fusion'
    intent_branches: dict = field(default_factory=dict)      # branch name -> list of observation.oracle.* keys
    intent_branch_dims: dict = field(default_factory=dict)   # branch name -> feature width (set from the dataset)
    intent_embed_dim: int = 64
    intent_hidden_dim: int = 128
    # Capacity-matched No-Information control: same encoders/fusion, intention inputs replaced by zeros.
    intent_mask: bool = False
    # tokens: configs/intent_schema.yaml (vocabularies, waypoints, keypoints, tte_max_s), components fed to the
    # model, intent tokens also in the CVAE encoder, component dropout (training only), FiLM fallback (§2.6).
    intent_schema: dict = field(default_factory=dict)
    intent_components: list[str] = field(default_factory=lambda: list(INTENT_COMPONENTS))
    intent_in_cvae: bool = True
    intent_p_drop: float = 0.2
    intent_p_drop_all: float = 0.1
    intent_film: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.action_mode not in ACTION_MODES:
            raise ValueError(f'action_mode must be one of {ACTION_MODES}, got {self.action_mode!r}')
        if self.num_restricted_actions != 9:
            raise ValueError('the restricted action space has exactly 9 actions')
        if self.use_intent and self.intent_arch not in INTENT_ARCHS:
            raise ValueError(f'intent_arch must be one of {INTENT_ARCHS}, got {self.intent_arch!r}')
        if self.use_intent and self.intent_arch == 'tokens':
            missing = [k for k in SCHEMA_KEYS if k not in self.intent_schema]
            if missing:
                raise ValueError(f'intent tokens need intent_schema fields {missing} (configs/intent_schema.yaml)')
            if not self.intent_components or set(self.intent_components) - set(INTENT_COMPONENTS):
                raise ValueError(f'intent_components must be a non-empty subset of {INTENT_COMPONENTS}')
            if self.intent_mask:
                raise ValueError('intent_mask is a fusion control; not implemented for intent tokens')
        elif self.use_intent:
            if not self.intent_branches or not all(self.intent_branches.values()):
                raise ValueError('use_intent needs non-empty intent_branches')
            bad = [k for keys in self.intent_branches.values() for k in keys if not k.startswith(INTENT_PREFIX)]
            if bad:
                raise ValueError(f'intent features must be numeric oracle fields ({INTENT_PREFIX}*), got {bad}')
            if set(self.intent_branch_dims) != set(self.intent_branches):
                raise ValueError('intent_branch_dims must give the width of every intent branch')

    @property
    def intent_keys(self) -> list[str]:
        """Oracle features the fusion policy reads (none for the masked No-Information control or intent tokens)."""
        if not self.use_intent or self.intent_mask or self.intent_arch != 'fusion':
            return []
        return [k for keys in self.intent_branches.values() for k in keys]

    @property
    def intent_token_cfg(self) -> dict | None:
        """`intent_cfg` of policies/intent_act.IntentACT (None: plain LeRobot ACT)."""
        if not (self.use_intent and self.intent_arch == 'tokens'):
            return None
        s = self.intent_schema
        return dict(n_obj=len(s['obj_vocab']), n_act=len(s['act_vocab']), n_waypoints=int(s['n_waypoints']),
                    n_joints=len(s['keypoints']), kp_dim=int(s['keypoint_dim']), components=list(self.intent_components),
                    in_cvae=self.intent_in_cvae, film=self.intent_film, tte_max=float(s['tte_max_s']))
