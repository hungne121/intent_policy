"""LeRobot ACT policy with a selectable action head and optional intention fusion.

The ACT network (`self.model`, class `lerobot...ACT`) is used unmodified. The policy latent
h is the ACT decoder output (B, chunk, D) — exactly the tensor LeRobot feeds to its
continuous `action_head`. A forward pre-hook on that head captures h (and, with `use_intent`,
replaces it by the fused latent h'), so both heads share one backbone and one forward pass:

    observation -> LeRobot ACT (backbone + transformer) -> h -> [IntentFusion(h, oracle branches)] -> h'
                                                                  h' -+-> action_head (continuous chunk)
                                                                      +-> restricted head (9 logits / step)
"""
from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.utils.constants import ACTION, OBS_IMAGES

from intent_policy.policies.hri_act.configuration_hri_act import HRIACTConfig
from intent_policy.policies.intent_fusion import IntentFusion
from intent_policy.policies.restricted_action_head import RestrictedActionHead


class HRIACTPolicy(ACTPolicy):
    config_class = HRIACTConfig
    name = 'hri_act'

    def __init__(self, config: HRIACTConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.restricted_head = RestrictedActionHead(config.dim_model, config.num_restricted_actions,
                                                    config.restricted_head_hidden_dim)
        self.intent_fusion = IntentFusion(config.dim_model, dict(config.intent_branch_dims), config.intent_hidden_dim,
                                          config.intent_embed_dim) if config.use_intent else None
        self._latent: Tensor | None = None
        self._intent_input: dict | None = None
        self.model.action_head.register_forward_pre_hook(self._capture_latent)

    def _capture_latent(self, module, args):
        h = args[0]
        if self.intent_fusion is not None:
            if self._intent_input is None:
                raise RuntimeError('intent features were not set for this forward pass')
            h = self.intent_fusion(h, self._intent_input, mask=self.config.intent_mask)
        self._latent = h
        return (h,)

    def _set_intent(self, batch: dict[str, Tensor]) -> None:
        if self.intent_fusion is None:
            return
        b = batch['observation.state'].shape[0]
        dev = batch['observation.state'].device
        if self.config.intent_mask:
            self._intent_input = {k: torch.zeros(b, d, device=dev) for k, d in self.config.intent_branch_dims.items()}
        else:
            self._intent_input = {k: torch.cat([batch[f].reshape(b, -1).float() for f in keys], dim=-1)
                                  for k, keys in self.config.intent_branches.items()}

    def reset(self):
        super().reset()
        self._restricted_queue = deque([], maxlen=self.config.n_action_steps)
        coeff = self.config.temporal_ensemble_coeff
        self._restricted_ensembler = ACTTemporalEnsembler(coeff, self.config.chunk_size) if coeff is not None else None

    def set_temporal_ensemble(self, coeff: float | None) -> None:
        """Inference-time temporal ensembling (ACT, Zhao et al. 2023, Alg. 2); needs n_action_steps == 1."""
        if coeff is not None and self.config.n_action_steps != 1:
            raise ValueError('temporal ensembling needs n_action_steps == 1')
        self.config.temporal_ensemble_coeff = coeff
        if coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(coeff, self.config.chunk_size)
        self.reset()

    # ------------------------------------------------------------------ shared forward
    def _run_act(self, batch: dict[str, Tensor]):
        """One LeRobot ACT forward. Returns continuous chunk, (mu, log_sigma_x2) and latent h (fused h')."""
        self._set_intent(batch)
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        actions, dist = self.model(batch)
        return actions, dist, self._latent

    def restricted_logits(self, batch: dict[str, Tensor]) -> Tensor:
        _, _, latent = self._run_act(batch)
        return self.restricted_head(latent)

    # ------------------------------------------------------------------ training
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.config.action_mode == 'continuous':
            self._set_intent(batch)
            return super().forward(batch)
        actions_hat, (mu, log_sigma_x2), latent = self._run_act(batch)
        logits = self.restricted_head(latent)                                   # (B, S, 9)
        key = self.config.restricted_label_key
        labels = batch[key].long().reshape(logits.shape[:2])                    # (B, S)
        valid = ~batch[f'{key}_is_pad']
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction='none')
        ce = (ce * valid.reshape(-1)).sum() / valid.sum().clamp_min(1)
        with torch.no_grad():
            acc = ((logits.argmax(-1) == labels) & valid).sum() / valid.sum().clamp_min(1)
            acc0 = ((logits[:, 0].argmax(-1) == labels[:, 0]) & valid[:, 0]).sum() / valid[:, 0].sum().clamp_min(1)
        loss = ce
        info = {'ce_loss': ce.item(), 'chunk_accuracy': acc.item(), 'step0_accuracy': acc0.item()}
        if self.config.continuous_aux_weight > 0 and ACTION in batch:
            l1 = (F.l1_loss(batch[ACTION], actions_hat, reduction='none') * ~batch['action_is_pad'].unsqueeze(-1)).mean()
            loss = loss + self.config.continuous_aux_weight * l1
            info['aux_l1_loss'] = l1.item()
        if self.config.use_vae and log_sigma_x2 is not None:
            kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()
            loss = loss + kld * self.config.kl_weight
            info['kld_loss'] = kld.item()
        return loss, info

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_restricted_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Restricted logits for the whole chunk: (B, chunk_size, 9)."""
        self.eval()
        return self.restricted_logits(batch)

    @torch.no_grad()
    def select_restricted_action(self, batch: dict[str, Tensor]) -> dict:
        """Chunk execution with the same schemes as ACT's `select_action`: temporal ensembling of the
        predicted chunks (log-probabilities averaged with ACT's weights) when `temporal_ensemble_coeff` is
        set, otherwise a queue of `n_action_steps` predicted steps.

        Returns action_logits / action_probabilities (B, 9) and selected_action (B,).
        """
        self.eval()
        if self._restricted_ensembler is not None:
            logits = self._restricted_ensembler.update(self.predict_restricted_chunk(batch).log_softmax(-1))
            probs = logits.softmax(-1)
            return {'action_logits': logits, 'action_probabilities': probs, 'selected_action': probs.argmax(-1)}
        if len(self._restricted_queue) == 0:
            logits = self.predict_restricted_chunk(batch)[:, : self.config.n_action_steps]
            self._restricted_queue.extend(logits.transpose(0, 1))
        logits = self._restricted_queue.popleft()
        probs = logits.softmax(-1)
        return {'action_logits': logits, 'action_probabilities': probs, 'selected_action': probs.argmax(-1)}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self._set_intent(batch)
        return super().predict_action_chunk(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        if self.config.action_mode != 'continuous':
            raise RuntimeError('select_action returns continuous actions; use select_restricted_action in restricted mode')
        return super().select_action(batch)
