"""Episode runner: agent -> (restricted mapper | continuous passthrough) -> scenario.

Benchmark-side code: it records the complete episode (configuration, variation, per-step
robot/human/object state, the policy's logits/probabilities/selected action, the mapped
controller command, all events and metrics) without knowing how the agent works.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import gzip
import json
import time
import numpy as np
from intent_policy.benchmark.events import EventType as E
from intent_policy.benchmark.metrics import compute_episode_metrics
from intent_policy.sim.restricted_action import RestrictedActionConfig, RestrictedActionMapper, RestrictedAction
from intent_policy.experts.scripted_expert import ScriptedExpert

SCHEMA = 'hri_episode_v1'
TRANSLATIONS = {int(a) for a in (RestrictedAction.MOVE_FORWARD, RestrictedAction.MOVE_BACKWARD, RestrictedAction.MOVE_LEFT,
                                 RestrictedAction.MOVE_RIGHT, RestrictedAction.MOVE_UP, RestrictedAction.MOVE_DOWN)}


@dataclass
class ObservationConfig:
    cameras: tuple = ('high', 'wrist')
    width: int = 128
    height: int = 96

    def to_dict(self):
        return dict(cameras=list(self.cameras), width=self.width, height=self.height)


@dataclass
class Decision:
    action_id: int | None = None                 # restricted mode
    joint_target: np.ndarray | None = None       # continuous mode, [7]
    logits: list | None = None
    probabilities: list | None = None
    extras: dict = field(default_factory=dict)


class Agent:
    """Minimal agent contract used by the runner."""
    action_mode = 'restricted'
    needs_observation = True

    def reset(self, scenario, mapper) -> None: ...
    def act(self, obs: dict | None) -> Decision: ...
    def describe(self) -> dict: return {}


class ExpertAgent(Agent):
    """Privileged scripted demonstrator (restricted actions); data generation only."""
    needs_observation = False

    def reset(self, scenario, mapper):
        self.expert = ScriptedExpert(scenario, mapper, rng_seed=scenario.seed)
        self.expert.reset()

    def act(self, obs):
        action = self.expert.act()
        return Decision(action_id=action, extras={'expert_stage': self.expert.stage, 'expert_commit': self.expert.pop_commit()})

    def describe(self):
        return dict(type='scripted_expert', action_mode='restricted', privileged=True)


class NoisyExpertAgent(ExpertAgent):
    """DART-style noise injection for demonstrations (Laskey et al., 2017).

    With probability `burst_prob` per decision (only while the expert is translating) a burst of
    `burst_ticks` random translations is executed instead of the expert's action. The expert
    re-targets from wherever the arm ends up, so the data covers off-nominal states and the way back.
    `extras['expert_action']` / `extras['clean_joint_target']` hold the clean label to record.
    """

    def __init__(self, burst_prob: float = 0.04, burst_ticks: tuple[int, int] = (3, 8)):
        self.burst_prob, self.burst_ticks = float(burst_prob), tuple(int(b) for b in burst_ticks)

    def reset(self, scenario, mapper):
        super().reset(scenario, mapper)
        self.mapper = mapper
        self.rng = np.random.default_rng(int(scenario.seed) * 7919 + 17)
        self.burst, self.burst_action = 0, None

    def act(self, obs):
        d = super().act(obs)
        clean = int(d.action_id)
        translating = clean in TRANSLATIONS
        if not translating:
            self.burst = 0
        elif self.burst == 0 and self.rng.random() < self.burst_prob:
            self.burst = int(self.rng.integers(self.burst_ticks[0], self.burst_ticks[1] + 1))
            self.burst_action = int(self.rng.choice(sorted(TRANSLATIONS)))
        d.extras['expert_action'] = clean
        d.extras['noise'] = self.burst > 0
        if self.burst > 0:
            self.burst -= 1
            d.extras['clean_joint_target'] = self.mapper.preview(clean).joint_target
            d.action_id = self.burst_action
        return d

    def describe(self):
        return dict(type='scripted_expert', action_mode='restricted', privileged=True, noise='dart',
                    burst_prob=self.burst_prob, burst_ticks=list(self.burst_ticks))


def run_episode(scenario, agent: Agent, seed: int, controller_cfg: RestrictedActionConfig | None = None,
                obs_cfg: ObservationConfig | None = None, on_frame=None, keep_trace: bool = True,
                episode_id: str | None = None, spec: dict | None = None) -> dict:
    """Run one episode to termination and return the full episode record. `spec`: discrete episode spec from a
    scenario list (None: drawn from the seed)."""
    controller_cfg = controller_cfg or RestrictedActionConfig.load()
    obs_cfg = obs_cfg or ObservationConfig()
    wall = time.monotonic()
    scenario.reset(seed, episode_id=episode_id, spec=spec)
    mapper = RestrictedActionMapper(scenario.env, controller_cfg)
    agent.reset(scenario, mapper)
    need_obs = agent.needs_observation or on_frame is not None
    trace, previous, done, step = [], None, False, 0
    while not done:
        state = scenario.snapshot() if keep_trace else None
        obs = scenario.get_observation(obs_cfg.cameras, obs_cfg.width, obs_cfg.height) if need_obs else None
        decision = agent.act(obs)
        if agent.action_mode == 'restricted':
            command = mapper.map(decision.action_id)
            target, ticks, cmd_record = command.joint_target, command.execution_ticks, command.to_dict()
            label = RestrictedAction(decision.action_id).name
        else:
            target = np.asarray(decision.joint_target, float)
            ticks, cmd_record, label = 1, dict(joint_target=target.round(6).tolist(), execution_ticks=1), 'continuous'
        if agent.action_mode == 'restricted' and decision.action_id != previous:
            scenario.log(E.ROBOT_ACTION_CHANGE, 'policy', previous=None if previous is None else RestrictedAction(previous).name,
                         action=label)
            previous = decision.action_id
        if on_frame is not None:
            on_frame(obs, decision, target, scenario)
        for _ in range(ticks):
            done = scenario.step(target)
            if done:
                break
        if keep_trace:
            trace.append(dict(step=step, t=state['t'], selected_action=decision.action_id, selected_action_name=label,
                              action_logits=decision.logits, action_probabilities=decision.probabilities,
                              mapped_controller_command=cmd_record, extras=decision.extras, state=state))
        step += 1
    events = scenario.get_events()
    cfg = scenario.cfg
    metrics = compute_episode_metrics(events, cfg.applicable_metrics, cfg.protocol_steps, cfg.metric_params)
    return dict(schema=SCHEMA, episode_id=scenario.episode_id, scenario_id=cfg.id, role=cfg.role, seed=int(seed),
                task=scenario.variation['task'], spec=scenario.variation['spec'],
                scenario_config=cfg.to_dict(), scenario_source=cfg.source, variation=scenario.variation,
                human_trajectory_variant=scenario.variation['human'], policy=agent.describe(),
                controller=controller_cfg.to_dict(), observation=obs_cfg.to_dict(),
                success=scenario.success, failure=scenario.failure, n_decisions=step,
                duration_s=scenario.time, min_human_robot_distance=float(scenario.min_human_distance),
                protocol_steps_completed=list(scenario.protocol_done), metrics=metrics, events=events,
                trace=trace, wall_time_s=round(time.monotonic() - wall, 2))


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))


def save_record(record: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, default=_default)
    if path.suffix == '.gz':
        with gzip.open(path, 'wt') as f:
            f.write(text)
    else:
        path.write_text(text)
    return path


def load_record(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    return json.loads(path.read_text())
