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
INTENT_GROUPS = ('semantic', 'spatial', 'memory', 'time', 'motion')
INTENT_SOURCES = ('hindsight', 'perfect', 'predicted', 'device')
INTENT_VERSIONS = ('v2', 'v32')
TOKENS_V32 = ('human', 'sem', 'target', 'c_target', 'time', 'motion')     # = policies/intent_encoder_v32.TOKENS_V32
SCHEMA_KEYS = ('targets', 'who', 'phases', 'tasks', 'n_waypoints', 'keypoints', 'keypoint_dim', 'tte_max_s')


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
    # 'tokens' (INTENT_ACT_GUIDE_v2.md): the intent contract (dataset keys intent_<src>.*, read as intent.*) encoded as
    # extra tokens of the ACT transformer encoder and of the CVAE encoder (policies/intent_act.py).
    use_intent: bool = False
    intent_arch: str = 'fusion'
    intent_branches: dict = field(default_factory=dict)      # branch name -> list of observation.oracle.* keys
    intent_branch_dims: dict = field(default_factory=dict)   # branch name -> feature width (set from the dataset)
    intent_embed_dim: int = 64
    intent_hidden_dim: int = 128
    # Capacity-matched No-Information control: same encoders/fusion, intention inputs replaced by zeros.
    intent_mask: bool = False
    # tokens: configs/intent_schema.yaml (contract sizes), the intent source the model was trained on (metadata: the
    # model itself never knows the source), extra tokens also in the CVAE encoder, group dropout (training only),
    # groups kept at evaluation (ablation by information type), FiLM fallback (§4.5).
    intent_schema: dict = field(default_factory=dict)
    intent_source: str = 'hindsight'
    intent_in_cvae: bool = True
    intent_p_drop_group: float = 0.15
    intent_p_drop_all: float = 0.1
    intent_keep: list[str] = field(default_factory=lambda: list(INTENT_GROUPS))
    intent_film: bool = False
    # v3.2 conditioning study (INTENT_ACT_v3.2.md §5.3): intent_film = FiLM of every ResNet stage by the intent tokens;
    # image_p_drop = probability of masking all image tokens of a training sample (modality dropout, ModDrop).
    image_p_drop: float = 0.0
    # F1 (§5.3): initial scale of the target embeddings E_target / committed: 'small' = std 0.02 (D0), 'type' = N(0,1),
    # the scale of the token type embedding (nn.Embedding default) they are summed with.
    intent_target_init: str = 'small'
    # v3.2 contract encoded as the 6 tokens ('tokens') or as ONE token u_t from the device's output only ('unified':
    # target belief + hold, policies/intent_encoder_v32.IntentEncoderUnified; trial of 2026-10-18)
    intent_repr: str = 'tokens'
    # Contract / token layout: 'v2' (13 tokens, INTENT_ACT_GUIDE_v2.md) or 'v32' (6 tokens + key padding mask,
    # INTENT_ACT_v3.2.md §5; source `device`). v32: intent_p_drop_group drops single tokens, intent_keep lists tokens.
    intent_version: str = 'v2'
    # Task token (T1-T4, §4.1): conditions every model on the task id when set (needs intent_schema['tasks']).
    use_task_token: bool = False
    # Task instruction (intent/instruction.py): the object / destination kind of the spoken command, added to the task
    # token (needs use_task_token and intent_schema['instruction']).
    use_instruction: bool = False

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
            if self.intent_version not in INTENT_VERSIONS:
                raise ValueError(f'intent_version must be one of {INTENT_VERSIONS}')
            if self.intent_version == 'v32' and list(self.intent_keep) == list(INTENT_GROUPS):
                self.intent_keep = list(TOKENS_V32)             # the default (all groups) -> all v3.2 tokens
            groups = TOKENS_V32 if self.intent_version == 'v32' else INTENT_GROUPS
            if set(self.intent_keep) - set(groups):
                raise ValueError(f'intent_keep must be a subset of {groups}')
            if self.intent_target_init not in ('small', 'type') or (self.intent_target_init != 'small' and self.intent_version != 'v32'):
                raise ValueError("intent_target_init is 'small' or 'type' (intent v3.2 only)")
            if self.intent_repr not in ('tokens', 'unified') or (self.intent_repr != 'tokens' and self.intent_version != 'v32'):
                raise ValueError("intent_repr is 'tokens' or 'unified' (intent v3.2 only)")
            if not 0.0 <= self.image_p_drop < 1.0 or (self.image_p_drop and self.intent_version != 'v32'):
                raise ValueError('image_p_drop needs intent v3.2 and lies in [0, 1)')
            if self.intent_source not in INTENT_SOURCES:
                raise ValueError(f'intent_source must be one of {INTENT_SOURCES}')
            if self.intent_mask:
                raise ValueError('intent_mask is a fusion control; not implemented for intent tokens')
        if self.use_task_token and 'tasks' not in self.intent_schema:
            raise ValueError('use_task_token needs intent_schema with the task list (configs/intent_schema.yaml)')
        if self.use_instruction and not (self.use_task_token and 'instruction' in self.intent_schema):
            raise ValueError('use_instruction needs use_task_token and intent_schema["instruction"]')
        if self.use_intent and self.intent_arch == 'fusion':
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
        out = dict(n_targets=len(s['targets']), n_who=len(s['who']), n_phases=len(s['phases']),
                   n_waypoints=int(s['n_waypoints']), n_joints=len(s['keypoints']), kp_dim=int(s['keypoint_dim']),
                   in_cvae=self.intent_in_cvae, film=self.intent_film, tte_max=float(s['tte_max_s']),
                   version=self.intent_version)
        if self.intent_version == 'v32':
            from intent_policy.intent.contract import HAND_HIST
            out.update(n_gestures=len(s['gestures']) + 1, hand_hist=HAND_HIST,
                       target_init_std=1.0 if self.intent_target_init == 'type' else 0.02, repr=self.intent_repr)
        return out

    @property
    def n_tasks(self) -> int:
        return len(self.intent_schema['tasks']) if self.use_task_token else 0

    @property
    def instruction_sizes(self) -> tuple[int, int] | None:
        """(object kinds, destination kinds) of the task instruction, None without it."""
        if not self.use_instruction:
            return None
        ins = self.intent_schema['instruction']
        return len(ins['objects']), len(ins['dests'])
