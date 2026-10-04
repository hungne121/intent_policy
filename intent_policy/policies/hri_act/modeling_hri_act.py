"""LeRobot ACT policy with a selectable action head and optional intention input.

The ACT network (`self.model`, class `lerobot...ACT`) is used unmodified. The policy latent
h is the ACT decoder output (B, chunk, D) — exactly the tensor LeRobot feeds to its
continuous `action_head`. A forward pre-hook on that head captures h (and, with intent fusion,
replaces it by the fused latent h'), so both heads share one backbone and one forward pass:

    observation -> LeRobot ACT (backbone + transformer) -> h -> [IntentFusion(h, oracle branches)] -> h'
                                                                  h' -+-> action_head (continuous chunk)
                                                                      +-> restricted head (9 logits / step)

With intent tokens (`intent_arch: tokens`, INTENT_ACT_GUIDE.md) the network is `IntentACT`, LeRobot's ACT with the
intent tokens in its transformer / CVAE encoders (policies/intent_act.py); the batch carries the labels intent.obj,
intent.act, intent.phase, intent.tte, intent.xi (raw xi, normalised here with the training-split statistics).
"""
from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.utils.constants import ACTION, OBS_IMAGES

from intent_policy.policies.hri_act.configuration_hri_act import INTENT_GROUPS, HRIACTConfig
from intent_policy.policies.intent_act import IntentACT
from intent_policy.policies.intent_fusion import IntentFusion
from intent_policy.policies.restricted_action_head import RestrictedActionHead


def _label(x: Tensor, n_cls: int, b: int, device) -> Tensor:
    """Intent class labels: hard (B,) long, or soft probabilities (B, n_cls) float (e.g. from a predictor)."""
    x = x.to(device)
    if x.is_floating_point() and x.dim() == 2 and x.shape[-1] == n_cls > 1:
        return x.float()
    return x.reshape(b).long()


class HRIACTPolicy(ACTPolicy):
    config_class = HRIACTConfig
    name = 'hri_act'

    def __init__(self, config: HRIACTConfig, **kwargs):
        super().__init__(config, **kwargs)
        if config.intent_token_cfg is not None or config.use_task_token:
            self.model = IntentACT(config, config.intent_token_cfg, config.n_tasks, config.instruction_sizes)
        self.restricted_head = RestrictedActionHead(config.dim_model, config.num_restricted_actions,
                                                    config.restricted_head_hidden_dim)
        self.intent_fusion = IntentFusion(config.dim_model, dict(config.intent_branch_dims), config.intent_hidden_dim,
                                          config.intent_embed_dim) if config.use_intent and config.intent_arch == 'fusion' else None
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

    @property
    def intent_encoder(self):
        return self.model.intent_encoder if isinstance(self.model, IntentACT) and self.model.use_intent else None

    def _with_intent(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Task / intent tokens: batch['task_id'] from hri.task_id, batch['intent'] from the contract fields intent.*
        (xi normalised; unless given) and batch['intent_keep']: group dropout in training mode, the configured
        `intent_keep` groups in evaluation mode (unless given)."""
        cfg = self.config
        if not isinstance(self.model, IntentACT) or not (self.model.use_intent or self.model.use_task):
            return batch
        batch = dict(batch)
        b, dev = batch['observation.state'].shape[0], batch['observation.state'].device
        if self.model.use_task and 'task_id' not in batch:
            batch['task_id'] = batch['hri.task_id'].to(dev).reshape(b).long()
        if self.model.use_instruction and 'instr_object' not in batch:
            batch['instr_object'] = batch['hri.instr_object'].to(dev).reshape(b).long()
            batch['instr_dest'] = batch['hri.instr_dest'].to(dev).reshape(b).long()
        enc = self.intent_encoder
        if enc is None:
            return batch
        if 'intent' not in batch:
            g = lambda k, n: batch[f'intent.{k}'].to(dev).float().reshape(b, n)
            S = cfg.intent_schema
            K, W, P = len(S['targets']), len(S['who']), len(S['phases'])
            batch['intent'] = dict(p_who=g('p_who', W), c_who=g('c_who', W), p_target=g('p_target', K),
                                   c_target=g('c_target', K), occupancy=g('occupancy', 7), tte=g('tte', 1),
                                   tte_std=g('tte_std', 1), phase=g('phase', P), confidence=g('confidence', 2),
                                   xi=enc.normalize_xi(batch['intent.xi'].to(dev).float()))
        if batch.get('intent_keep') is None:
            if self.training and (cfg.intent_p_drop_group > 0 or cfg.intent_p_drop_all > 0):
                batch['intent_keep'] = enc.sample_keep(b, dev, cfg.intent_p_drop_group, cfg.intent_p_drop_all)
            elif not self.training and set(cfg.intent_keep) != set(INTENT_GROUPS):
                batch['intent_keep'] = {gr: torch.full((b,), gr in cfg.intent_keep, dtype=torch.bool, device=dev)
                                        for gr in INTENT_GROUPS}
        return batch

    def _maybe_reset_ensembler(self, batch: dict) -> None:
        """§4.3: with temporal ensembling, clear the chunk buffer when the committed target or who changes, or when
        the target confidence first exceeds 0.8 after having been below it."""
        intent = batch.get('intent')
        ens = self._restricted_ensembler if self.config.action_mode == 'restricted' else getattr(self, 'temporal_ensembler', None)
        if intent is None or self.config.temporal_ensemble_coeff is None or ens is None:
            return
        key = (int(intent['c_target'][0].argmax()), int(intent['c_who'][0].argmax()))
        conf = float(intent['confidence'][0, 0])
        reset = self._intent_key is not None and key != self._intent_key
        if conf >= 0.8 and self._conf_armed:
            reset, self._conf_armed = True, False
        elif conf < 0.8:
            self._conf_armed = True
        if reset:
            ens.reset()
            self.ensembler_resets += 1
        self._intent_key = key

    def reset(self):
        super().reset()
        self._restricted_queue = deque([], maxlen=self.config.n_action_steps)
        coeff = self.config.temporal_ensemble_coeff
        self._restricted_ensembler = ACTTemporalEnsembler(coeff, self.config.chunk_size) if coeff is not None else None
        self._intent_key, self._conf_armed, self.ensembler_resets = None, True, 0

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
        batch = self._with_intent(batch)
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
            return super().forward(self._with_intent(batch))
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
            batch = self._with_intent(batch)
            self._maybe_reset_ensembler(batch)
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
        self.eval()
        self._set_intent(batch)
        return super().predict_action_chunk(self._with_intent(batch))

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        if self.config.action_mode != 'continuous':
            raise RuntimeError('select_action returns continuous actions; use select_restricted_action in restricted mode')
        if self.config.temporal_ensemble_coeff is not None:
            self.eval()
            batch = self._with_intent(batch)
            self._maybe_reset_ensembler(batch)
        return super().select_action(batch)
