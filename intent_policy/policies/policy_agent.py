"""Adapter: a trained HRI-ACT checkpoint as a benchmark `Agent` (observation -> decision).

The policy receives the robot observation (proprioception + camera images) and, only if its
config reads oracle features (`use_intent` without `intent_mask`), the oracle intention features
computed at the same instant (intent.oracle, under the information-timing contract) after the
evaluation condition's corruption (intent.corruption). The observation itself is never altered.
"""
from pathlib import Path
import json
import numpy as np
import torch
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import (batch_to_transition, policy_action_to_transition,
                                          transition_to_batch, transition_to_policy_action)
from intent_policy.benchmark.runner import Agent, Decision
from intent_policy.intent.corruption import make_corruption
from intent_policy.intent.instruction import INSTR_DEST, INSTR_OBJECT, instruction_of
from intent_policy.intent.online import OnlineHindsight
from intent_policy.intent.oracle import HORIZONS, OracleIntentProvider
from intent_policy.intent.representation import features as oracle_values
from intent_policy.policies.hri_act import HRIACTPolicy
from intent_policy.utils import obs_to_batch


def load_policy(checkpoint: str | Path, device: str | None = None):
    checkpoint = Path(checkpoint)
    policy = HRIACTPolicy.from_pretrained(checkpoint)
    if device:
        policy.to(device)
        policy.config.device = device
    overrides = {'device_processor': {'device': str(policy.config.device)}}
    pre = PolicyProcessorPipeline.from_pretrained(checkpoint, config_filename='policy_preprocessor.json',
                                                  overrides=overrides, to_transition=batch_to_transition,
                                                  to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(checkpoint, config_filename='policy_postprocessor.json',
                                                   to_transition=policy_action_to_transition,
                                                   to_output=transition_to_policy_action)
    return policy, pre, post


class PolicyAgent(Agent):
    def __init__(self, checkpoint: str | Path, device: str | None = None, oracle_condition: dict | None = None,
                 eval_seeds: list[int] | None = None, controller_cfg=None, temporal_ensemble_coeff: float | str | None = 'config'):
        self.checkpoint = Path(checkpoint)
        self.policy, self.pre, self.post = load_policy(self.checkpoint, device)
        cfg = self.policy.config
        self.tokens = cfg.use_intent and cfg.intent_arch == 'tokens'
        self.online_intent = None
        if self.tokens:
            if cfg.intent_source != 'hindsight':
                raise NotImplementedError(f'closed-loop intent source {cfg.intent_source!r}: only hindsight (online '
                                          'from the simulator plan, intent/online.py) is implemented')
            self.online_intent = OnlineHindsight(cfg.intent_schema)
        if temporal_ensemble_coeff != 'config':          # inference-time setting of the evaluation protocol
            self.policy.set_temporal_ensemble(temporal_ensemble_coeff)
        self.action_mode = self.policy.config.action_mode
        self.device = str(self.policy.config.device)
        meta = self.checkpoint / 'training_metadata.json'
        self.training_metadata = json.loads(meta.read_text()) if meta.exists() else {}
        self.expected_keys = set(self.policy.config.input_features)
        self.reads_oracle = bool(self.policy.config.intent_keys) and not self.tokens
        self.eval_seeds = list(eval_seeds) if eval_seeds is not None else None
        self.oracle_condition = dict(oracle_condition or {'condition': 'correct'}) if self.reads_oracle else {'condition': 'none'}
        if self.reads_oracle:
            horizons = tuple(self.training_metadata.get('experiment', {}).get('intent', {}).get('horizons_s', HORIZONS))
            self.corruption, lead = make_corruption(self.oracle_condition, eval_seeds, controller_cfg)
            self.provider = OracleIntentProvider(horizons, lead_s=lead)
            self.reference = OracleIntentProvider(horizons)            # uncorrupted oracle, logged for analysis
        self.scenario = None

    def reset(self, scenario, mapper) -> None:
        self.policy.reset()
        self.scenario = scenario
        self.step_index = 0
        if self.reads_oracle:
            self.corruption.reset(scenario, scenario.seed)
        if self.online_intent is not None:
            self.online_intent.reset()

    def act(self, obs: dict) -> Decision:
        extras = {}
        obs = dict(obs)
        if self.reads_oracle:
            given = self.corruption.apply(self.provider.record(self.scenario))
            obs.update(oracle_values(given, self.scenario.tcp()))
            extras = {'oracle_given': given.compact(), 'oracle_true': self.reference.record(self.scenario).compact()}
        intent = None
        if self.online_intent is not None:
            intent = self.online_intent.observe(self.scenario, self.step_index, float(np.asarray(obs['observation.state'])[6]))
            extras['intent'] = dict(p_target=int(intent['p_target'].argmax()), c_target=int(intent['c_target'].argmax()),
                                    p_who=int(intent['p_who'].argmax()), phase=int(intent['phase'].argmax()),
                                    tte=round(float(intent['tte'][0]), 3))
        self.step_index += 1
        obs = {k: v for k, v in obs.items() if k in self.expected_keys}
        if set(obs) != self.expected_keys:
            raise KeyError(f'observation keys {sorted(obs)} != policy inputs {sorted(self.expected_keys)}')
        batch = self.pre(obs_to_batch(obs, self.device))
        cfg = self.policy.config
        if cfg.use_task_token:
            task, obj, dest = instruction_of(self.scenario.variation['spec'], self.scenario.cfg.id, cfg.intent_schema) \
                if cfg.use_instruction else (cfg.intent_schema['tasks'].index(
                    cfg.intent_schema['task_of_scenario'][self.scenario.cfg.id]), 0, 0)
            batch['hri.task_id'] = torch.tensor([task], device=self.device)
            if cfg.use_instruction:
                batch[INSTR_OBJECT], batch[INSTR_DEST] = torch.tensor([obj], device=self.device), torch.tensor([dest], device=self.device)
        if intent is not None:
            batch.update({f'intent.{k}': torch.as_tensor(v, device=self.device)[None] for k, v in intent.items()})
        if self.action_mode == 'restricted':
            out = self.policy.select_restricted_action(batch)
            return Decision(action_id=int(out['selected_action'][0]),
                            logits=out['action_logits'][0].float().cpu().numpy().round(5).tolist(),
                            probabilities=out['action_probabilities'][0].float().cpu().numpy().round(5).tolist(),
                            extras=extras)
        action = self.post(self.policy.select_action(batch))
        return Decision(joint_target=np.asarray(action[0].cpu().numpy(), float), extras=extras)

    def describe(self) -> dict:
        cfg = self.policy.config
        return dict(type='lerobot_act', policy_class='HRIACTPolicy', checkpoint=str(self.checkpoint), device=self.device,
                    action_mode=cfg.action_mode, use_intent=cfg.use_intent, use_task_token=cfg.use_task_token,
                    online_intent=self.online_intent.name if self.online_intent is not None else None, intent_arch=cfg.intent_arch, intent_mask=cfg.intent_mask,
                    intent_provider='oracle' if self.reads_oracle else 'none',
                    oracle_condition=self.corruption.describe() | {'lead_s': self.provider.lead_s} if self.reads_oracle else None,
                    oracle_spec=self.oracle_condition if self.reads_oracle else None,
                    eval_seeds=self.eval_seeds if self.reads_oracle else None,
                    intent_branches=dict(cfg.intent_branches) if cfg.use_intent else None,
                    input_features=sorted(cfg.input_features), chunk_size=cfg.chunk_size,
                    n_action_steps=cfg.n_action_steps, temporal_ensemble_coeff=cfg.temporal_ensemble_coeff, training_step=self.training_metadata.get('step'),
                    training_seed=self.training_metadata.get('seed'), dataset_root=self.training_metadata.get('dataset_root'))
