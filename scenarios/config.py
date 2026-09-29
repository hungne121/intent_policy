"""Scenario configuration contract and deterministic per-episode variation sampling.

Nothing here knows about any policy. A variation is a plain JSON-serialisable dict so it can
be stored in the episode record and replayed exactly.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import copy
import numpy as np
import yaml
from env.base_env import ROOT

CONFIG_DIR = ROOT / 'configs/scenarios'
ROLES = ('instructor', 'collaborator', 'intruder')


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


@dataclass
class ScenarioConfig:
    id: str
    role: str
    goal: str
    seed: int
    scene: dict
    scene_variation: dict
    human_behavior: dict
    timing: dict
    success_conditions: dict
    safety_constraints: dict
    protocol_steps: list[str]
    applicable_metrics: list[str]
    metric_params: dict = field(default_factory=dict)
    task: str = ''                     # natural-language LeRobot `task` (generic; never names the episode's target)
    # Demonstrator settings (teacher only, never a policy input). trigger: cue_complete (Phase 1) | cue_onset.
    expert: dict = field(default_factory=dict)
    # Optional mid-episode change of the human's intention (Phase 2); probability 0 disables it.
    intention_change: dict = field(default_factory=dict)
    source: str = ''
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path_or_id: str | Path, overrides: dict | None = None) -> 'ScenarioConfig':
        """Load a scenario YAML; `overrides` (e.g. from an experiment config) are deep-merged into it."""
        path = Path(path_or_id)
        if not path.suffix:
            path = CONFIG_DIR / f'{path_or_id}.yaml'
        raw = yaml.safe_load(path.read_text())
        if overrides:
            raw = _deep_merge(raw, overrides)
        return cls.from_dict(raw, source=str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path))

    @classmethod
    def from_dict(cls, raw: dict, source: str = '') -> 'ScenarioConfig':
        """Build from a raw scenario dict (as stored in episode records)."""
        scene = dict(raw['scene'])
        base_ref = scene.pop('base', None)
        if base_ref is not None:
            base = yaml.safe_load((CONFIG_DIR / base_ref).resolve().read_text())
            scene = _deep_merge(base, scene)
        if raw['role'] not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {raw['role']!r}")
        return cls(id=raw['scenario_id'], role=raw['role'], goal=' '.join(raw['goal'].split()),
                   seed=int(raw.get('seed', 0)), scene=scene, scene_variation=raw['scene_variation'],
                   human_behavior=raw['human_behavior'], timing=raw['timing'],
                   success_conditions=raw['success_conditions'], safety_constraints=raw['safety_constraints'],
                   protocol_steps=list(raw['protocol_steps']), applicable_metrics=list(raw['applicable_metrics']),
                   metric_params=dict(raw.get('metric_params', {})), task=' '.join(raw.get('task', '').split()),
                   expert=dict(raw.get('expert') or {}), intention_change=dict(raw.get('intention_change') or {}),
                   source=source, raw=copy.deepcopy(raw))

    def to_dict(self) -> dict:
        """Complete, self-contained configuration (base scene already merged)."""
        raw = copy.deepcopy(self.raw)
        raw['scene'] = copy.deepcopy(self.scene)
        return raw


def _uniform(rng, bounds) -> float:
    lo, hi = bounds
    return float(rng.uniform(lo, hi))


def sample_variation(cfg: ScenarioConfig, seed: int) -> dict:
    """Deterministic episode variation. Same (config, seed) => identical dict.

    Every random draw uses one generator in a fixed order, so adding a field at the end of
    this function never changes previously sampled fields.
    """
    sv, hb = cfg.scene_variation, cfg.human_behavior
    # Twin pairs: seeds 2k and 2k+1 share every draw (layout, human timing, changes) and differ only in the
    # human's choice, so the pair is a clean counterfactual in the data.
    twin = bool(sv.get('twin_pairs'))
    rng = np.random.default_rng(seed // 2 if twin else seed)
    objects = list(cfg.scene.get('objects', {}))
    var: dict = {'seed': int(seed)}

    slots = np.array(sv['object_slots'], float)
    order = rng.permutation(len(slots))
    var['object_xy'] = {k: (slots[order[i]] + rng.uniform(-1, 1, 2) * sv['object_jitter_m']).tolist()
                        for i, k in enumerate(objects)}
    var['object_yaw'] = {k: float(rng.uniform(-1, 1) * sv.get('object_yaw_rad', 0.2)) for k in objects}
    regions = list(cfg.scene.get('regions', {}))
    if regions:
        rslots = np.array(sv['region_slots'], float)
        # Intruder pairs region slot with object slot (place on the opposite side); others shuffle.
        rorder = order if cfg.role == 'intruder' else rng.permutation(len(rslots))
        var['region_xy'] = {k: (rslots[rorder[i]] + rng.uniform(-1, 1, 2) * sv['region_jitter_m']).tolist()
                            for i, k in enumerate(regions)}
    bowls = list(cfg.scene.get('bowls', {}))
    if bowls:
        bslots = np.array(sv['bowl_slots'], float)
        border = rng.permutation(len(bslots))
        var['bowl_xy'] = {k: (bslots[border[i]] + rng.uniform(-1, 1, 2) * sv['bowl_jitter_m']).tolist()
                          for i, k in enumerate(bowls)}
    var['robot_home_offset'] = (rng.uniform(-1, 1, 6) * sv['robot_home_jitter_rad']).tolist()

    colors = {k: o['color'] for k, o in cfg.scene.get('objects', {}).items()}
    rcolors = {k: r['color'] for k, r in cfg.scene.get('regions', {}).items()}
    if sv.get('appearance_permutation'):
        perm = rng.permutation(len(colors))
        values = list(colors.values())
        colors = {k: values[perm[i]] for i, k in enumerate(colors)}
        rperm = rng.permutation(len(rcolors))
        rvalues = list(rcolors.values())
        rcolors = {k: rvalues[rperm[i]] for i, k in enumerate(rcolors)}
    var['object_color'] = colors
    var['region_color'] = rcolors

    # Human task choice (ground truth): which object / target the human wants.
    flip = twin and seed % 2 == 1
    other = lambda options, value: options[(options.index(value) + 1) % len(options)] if flip else value
    if 'requested_object' in sv:
        var['requested_object'] = other(list(sv['requested_object']), str(rng.choice(sv['requested_object'])))
        var['requested_target'] = other(list(sv['requested_target']), str(rng.choice(sv['requested_target'])))
    if 'selected_object' in sv:
        var['selected_object'] = other(list(sv['selected_object']), str(rng.choice(sv['selected_object'])))

    # Human timing / trajectory variant.
    human = {'speed_scale': _uniform(rng, hb['speed_scale']), 'cue_onset_s': None}
    t = hb['type']
    if t == 'instructor_pointing':
        human.update(cue_onset_s=_uniform(rng, hb['cue_onset_s']), point_object_s=_uniform(rng, hb['point_object_s']),
                     point_target_s=_uniform(rng, hb['point_target_s']))
    elif t == 'handover_receiver':
        human.update(cue_onset_s=_uniform(rng, hb['cue_onset_s']), reach_s=_uniform(rng, hb['reach_s']),
                     receive_move_s=_uniform(rng, hb['receive_move_s']),
                     receive_pose=(np.array(hb['receive_pose']) + rng.uniform(-1, 1, 3) * hb['receive_jitter_m']).tolist())
    elif t == 'bowl_requester':
        human.update(cue_onset_s=_uniform(rng, hb['cue_onset_s']), reach_s=_uniform(rng, hb['reach_s']),
                     lift_s=_uniform(rng, hb['lift_s']), drop_move_s=_uniform(rng, hb['drop_move_s']))
    elif t == 'intruder':
        human.update(trigger=str(rng.choice(hb['trigger'])), trigger_delay_s=_uniform(rng, hb['trigger_delay_s']),
                     approach_s=_uniform(rng, hb['approach_s']), dwell_s=_uniform(rng, hb['dwell_s']),
                     withdraw_s=_uniform(rng, hb['withdraw_s']),
                     intrusion_offset=list(map(float, hb['intrusion_offset'][int(rng.integers(len(hb['intrusion_offset'])))])))
    else:
        raise ValueError(f'unknown human behaviour {t!r}')
    human['type'] = t
    var['human'] = human

    # Mid-episode intention change. Drawn last so earlier fields keep their values; absent when disabled
    # (probability 0) so Phase-1 variations stay unchanged.
    ic = cfg.intention_change
    if ic.get('type') and float(ic.get('probability', 0.0)) > 0.0:
        change = {'type': ic['type'], 'enabled': bool(rng.random() < float(ic['probability']))}
        if ic['type'] == 'target_object':
            change.update(delay_after_cue_s=_uniform(rng, ic['delay_after_cue_s']), reach_s=_uniform(rng, ic['reach_s']))
        elif ic['type'] == 'target_region':
            change.update(delay_after_lift_s=_uniform(rng, ic['delay_after_lift_s']))
        else:
            raise ValueError(f"unknown intention change type {ic['type']!r}")
        var['intention_change'] = change
    if twin:
        var['twin'] = {'base_seed': int(seed // 2), 'member': int(seed % 2)}
    return var
