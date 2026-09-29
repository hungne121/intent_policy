"""Evaluation-time corruption of the oracle information (the observation is never touched).

  correct    unchanged
  wrong      numeric target swap: the other candidate's identity AND its true position (regions without
             an alternative are displaced by `displacement_m`), motion reversed (approaching <-> withdrawing:
             future = current - (future - current), velocities negated)
  shuffled   the oracle stream of another episode of the same scenario (donor seed), aligned by tick
  noisy      Gaussian noise on positions / velocities; the target identity is swapped with probability
             `flip_prob`, drawn once per new target
  delayed    the record from `delay_s` ago (world frame; converted with the current TCP later)
  lead       information `lead_s` earlier (handled by OracleIntentProvider.lead_s)
Every condition is deterministic given (condition parameters, episode seed).
"""
from __future__ import annotations
from collections import deque
import numpy as np
from intent_policy.intent.oracle import OracleIntentProvider, OracleRecord, empty_record

CONDITIONS = ('none', 'correct', 'wrong', 'shuffled', 'noisy', 'delayed', 'lead')


class Corruption:
    name = 'correct'

    def reset(self, scenario, seed: int) -> None:
        self.scenario = scenario

    def apply(self, rec: OracleRecord) -> OracleRecord:
        return rec

    def describe(self) -> dict:
        return {'condition': self.name}


def _reverse_motion(rec: OracleRecord) -> None:
    for h in rec.future_pos:
        rec.future_pos[h] = rec.hand_pos - (rec.future_pos[h] - rec.hand_pos)
        rec.future_vel[h] = -rec.future_vel[h]
    rec.hand_vel = -rec.hand_vel


class Wrong(Corruption):
    name = 'wrong'

    def __init__(self, displacement_m: float = 0.2, reverse_motion: bool = True):
        self.displacement_m, self.reverse_motion = float(displacement_m), bool(reverse_motion)

    def _swap(self, key, pos):
        if key is None:
            return key, pos
        alts = self.scenario.oracle_alternatives(key)
        if alts:
            return alts[0], np.asarray(self.scenario.oracle_position(alts[0]), float)
        shift = np.array([0.0, -np.sign(pos[1]) * self.displacement_m if abs(pos[1]) > 1e-3 else self.displacement_m, 0.0])
        return key, pos + shift

    def apply(self, rec):
        rec = rec.copy()
        rec.object_key, rec.object_pos = self._swap(rec.object_key, rec.object_pos)
        rec.region_key, rec.region_pos = self._swap(rec.region_key, rec.region_pos)
        if self.reverse_motion:
            _reverse_motion(rec)
        return rec

    def describe(self):
        return dict(condition=self.name, displacement_m=self.displacement_m, reverse_motion=self.reverse_motion)


class Noisy(Corruption):
    name = 'noisy'

    def __init__(self, sigma_pos_m: float = 0.05, sigma_vel_m_s: float = 0.1, flip_prob: float = 0.2):
        self.sigma_pos, self.sigma_vel, self.flip_prob = float(sigma_pos_m), float(sigma_vel_m_s), float(flip_prob)

    def reset(self, scenario, seed):
        super().reset(scenario, seed)
        self.rng = np.random.default_rng(int(seed) * 104729 + 3)
        self.flips: dict = {}

    def _maybe_flip(self, slot, key, pos):
        if key is None:
            return key, pos
        if (slot, key) not in self.flips:
            alts = self.scenario.oracle_alternatives(key)
            self.flips[(slot, key)] = alts[int(self.rng.integers(len(alts)))] if alts and self.rng.random() < self.flip_prob else None
        alt = self.flips[(slot, key)]
        return (alt, np.asarray(self.scenario.oracle_position(alt), float)) if alt else (key, pos)

    def apply(self, rec):
        rec = rec.copy()
        n = lambda s: self.rng.normal(0.0, s, 3)
        rec.object_key, rec.object_pos = self._maybe_flip('object', rec.object_key, rec.object_pos)
        rec.region_key, rec.region_pos = self._maybe_flip('region', rec.region_key, rec.region_pos)
        if rec.object_pos is not None:
            rec.object_pos = rec.object_pos + n(self.sigma_pos)
        if rec.region_pos is not None:
            rec.region_pos = rec.region_pos + n(self.sigma_pos)
        rec.hand_pos = rec.hand_pos + n(self.sigma_pos)
        rec.hand_vel = rec.hand_vel + n(self.sigma_vel)
        for h in rec.future_pos:
            rec.future_pos[h] = rec.future_pos[h] + n(self.sigma_pos)
            rec.future_vel[h] = rec.future_vel[h] + n(self.sigma_vel)
        return rec

    def describe(self):
        return dict(condition=self.name, sigma_pos_m=self.sigma_pos, sigma_vel_m_s=self.sigma_vel, flip_prob=self.flip_prob)


class Delayed(Corruption):
    name = 'delayed'

    def __init__(self, delay_s: float):
        self.delay_s = float(delay_s)

    def reset(self, scenario, seed):
        super().reset(scenario, seed)
        self.buffer: deque = deque()

    def apply(self, rec):
        self.buffer.append(rec.copy())
        while len(self.buffer) > 1 and self.buffer[1].t <= rec.t - self.delay_s + 1e-9:
            self.buffer.popleft()
        old = self.buffer[0]
        if old.t > rec.t - self.delay_s + 1e-9:      # nothing that old yet: no information
            return empty_record(rec.t, old.hand_pos, np.zeros(3), tuple(rec.future_pos))
        return old.copy()

    def describe(self):
        return dict(condition=self.name, delay_s=self.delay_s)


class Shuffled(Corruption):
    """Oracle stream of a donor episode (same scenario, another seed), recorded with the expert."""
    name = 'shuffled'

    def __init__(self, donor_of, controller_cfg=None):
        self.donor_of, self.controller_cfg = donor_of, controller_cfg
        self.cache: dict = {}

    def reset(self, scenario, seed):
        super().reset(scenario, seed)
        self.donor = int(self.donor_of(int(seed)))
        key = (scenario.cfg.id, self.donor)
        if key not in self.cache:
            self.cache[key] = record_stream(scenario.cfg.to_dict(), self.donor, self.controller_cfg)
        self.stream = self.cache[key]
        self.k = 0

    def apply(self, rec):
        donor = self.stream[min(self.k, len(self.stream) - 1)].copy()
        self.k += 1
        donor.t = rec.t
        return donor

    def describe(self):
        return dict(condition=self.name, donor_rule='derangement of evaluation seeds')


def record_stream(scenario_config: dict, seed: int, controller_cfg=None) -> list[OracleRecord]:
    """Oracle records of one expert episode (no rendering), one per decision."""
    from intent_policy.benchmark.runner import ExpertAgent, run_episode
    from intent_policy.scenarios.scenario_registry import make_scenario
    provider, out = OracleIntentProvider(), []
    sc = make_scenario(scenario_config)
    try:
        run_episode(sc, ExpertAgent(), seed, controller_cfg, keep_trace=False,
                    on_frame=lambda obs, d, target, s: out.append(provider.record(s)))
    finally:
        sc.close()
    return out


def derangement(seeds: list[int], rng_seed: int = 0) -> dict[int, int]:
    """Fixed seed -> donor seed map without fixed points (a cyclic shift of a shuffled order)."""
    order = list(np.random.default_rng(rng_seed).permutation(seeds))
    return {int(a): int(order[(i + 1) % len(order)]) for i, a in enumerate(order)}


def make_corruption(spec: dict, seeds: list[int] | None = None, controller_cfg=None) -> tuple[Corruption, float]:
    """Condition spec {'condition': ..., params} -> (corruption, provider lead_s)."""
    c = spec.get('condition', 'correct')
    if c == 'correct':
        return Corruption(), 0.0
    if c == 'wrong':
        return Wrong(spec.get('displacement_m', 0.2), spec.get('reverse_motion', True)), 0.0
    if c == 'noisy':
        return Noisy(spec.get('sigma_pos_m', 0.05), spec.get('sigma_vel_m_s', 0.1), spec.get('flip_prob', 0.2)), 0.0
    if c == 'delayed':
        return Delayed(spec['delay_s']), 0.0
    if c == 'lead':
        return Corruption(), float(spec['lead_s'])
    if c == 'shuffled':
        donors = derangement(list(seeds), spec.get('rng_seed', 0))
        return Shuffled(lambda s: donors[s], controller_cfg), 0.0
    raise ValueError(f'unknown oracle condition {c!r}')
