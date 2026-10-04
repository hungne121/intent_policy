"""Scenario configuration contract and deterministic per-episode variation.

Nothing here knows about any policy. An episode is defined by a discrete *spec* (task code, target, slot layout,
place / hand zone, identical pair, timing variant, T3 cube order in U, interrupt phase, change of mind; docs/requirements/
scence_construct.md §4-5) plus continuous draws from the seed (position jitter, human timing). Specs come from the
balanced scenario lists (scripts/generate_scenarios.py) or, without a list, are drawn from the seed
(`random_spec`). A variation is a plain JSON-serialisable dict stored in the episode record and replayed exactly.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import copy
import numpy as np
import yaml
from intent_policy.sim.base_env import ROOT

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
    task_code: str = ''                # T1..T4 (scence_construct.md §4.4); T5 = T1-T3 with a `change` spec
    # Demonstrator settings (teacher only, never a policy input). trigger: cue_complete | evidence | cue_onset.
    expert: dict = field(default_factory=dict)
    # Change of mind (T5): timing parameters; `probability` > 0 adds changes to random (list-free) episodes.
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
        if isinstance(scene.get('objects'), list):          # catalog keys -> object definitions
            scene['objects'] = {k: copy.deepcopy(scene['objects_catalog'][k]) for k in scene['objects']}
        if raw['role'] not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {raw['role']!r}")
        return cls(id=raw['scenario_id'], role=raw['role'], goal=' '.join(raw['goal'].split()),
                   seed=int(raw.get('seed', 0)), scene=scene, scene_variation=raw['scene_variation'],
                   human_behavior=raw['human_behavior'], timing=raw['timing'],
                   success_conditions=raw['success_conditions'], safety_constraints=raw['safety_constraints'],
                   protocol_steps=list(raw['protocol_steps']), applicable_metrics=list(raw['applicable_metrics']),
                   metric_params=dict(raw.get('metric_params', {})), task=' '.join(raw.get('task', '').split()),
                   task_code=str(raw.get('task_code', '')),
                   expert=dict(raw.get('expert') or {}), intention_change=dict(raw.get('intention_change') or {}),
                   source=source, raw=copy.deepcopy(raw))

    def to_dict(self) -> dict:
        """Complete, self-contained configuration (base scene already merged)."""
        raw = copy.deepcopy(self.raw)
        raw['scene'] = copy.deepcopy(self.scene)
        return raw




# ---------------------------------------------------------------------------------------------- episode specs
TASK_CODES = ('T1', 'T2', 'T3', 'T4')
TWINS = (('B1', 'B1p'), ('B2', 'B2p'), ('B3', 'B3p'), ('C1', 'C2'))   # identical-looking pairs (§3.1; B2', B3': plan v5)
CHANGE_TIMINGS = ('early', 'late', 'free')     # free: the delay comes from intention_change.delay_s / the spec's delay_range


def twin_of(key: str) -> str | None:
    for a, b in TWINS:
        if key in (a, b):
            return b if key == a else a
    return None


def is_cup(obj: dict) -> bool:
    return obj['shape'] == 'cup'


_LAYOUTS: dict = {}


def layout_of(scene: dict) -> dict:
    """configs/layout.yaml (zones S/P/H/U/U_cup, robot zone), cached per path."""
    path = scene['layout']
    if path not in _LAYOUTS:
        _LAYOUTS[path] = yaml.safe_load((ROOT / path).read_text())
    return _LAYOUTS[path]


def zones_of(scene: dict, kind: str) -> list[str]:
    return [k for k, z in layout_of(scene)['zones'].items() if z['kind'] == kind]


def _pick(rng, options):
    return options[int(rng.integers(len(options)))]


def row_neighbours(scene: dict) -> set[frozenset]:
    """Slot pairs side by side in the same S row."""
    lay = layout_of(scene)['zones']
    rows: dict = {}
    for k in zones_of(scene, 'slot'):
        rows.setdefault(round(float(lay[k]['xy'][1]), 3), []).append(k)
    out = set()
    for ks in rows.values():
        ks.sort(key=lambda k: lay[k]['xy'][0])
        out |= {frozenset(pair) for pair in zip(ks, ks[1:])}
    return out


def cup_crowded(scene: dict, layout: dict) -> bool:
    """A cup on a slot next to another object in the same row: the open gripper (fingers ~6.5 cm either side of the
    TCP along X) grasping the neighbour low (a cube) or at the rim (a cup) would hit the 8.5 cm high cup wall (a cup
    grasped at its rim passes above a neighbouring cube; objects in the other row are far enough)."""
    objs, nb = scene['objects'], row_neighbours(scene)
    cups = [s for k, s in layout.items() if is_cup(objs[k])]
    return any(frozenset((c, s)) in nb for c in cups for s in layout.values() if s != c)


def _slots_for(rng, scene: dict, keys: list[str], slots: list[str]) -> dict:
    """Random distinct slots for `keys`, no cup next to another object in a row."""
    while True:
        layout = dict(zip(keys, [slots[i] for i in rng.permutation(len(slots))[:len(keys)]]))
        if not cup_crowded(scene, layout):
            return layout


def random_spec(cfg: ScenarioConfig, rng) -> dict:
    """Random discrete episode spec (used when no scenario list is given)."""
    sv, objs, scene = cfg.scene_variation, cfg.scene['objects'], cfg.scene
    slots = zones_of(scene, 'slot')
    code = cfg.task_code
    spec: dict = {'task': code}
    if code in ('T1', 'T2', 'T4'):
        pool = list(objs)
        pair = bool(rng.random() < float(sv.get('pair_probability', 0.0)))
        if pair:                                     # the target is one of an identical pair (§5.2 rule 3)
            a, b = _pick(rng, [p for p in TWINS if set(p) <= set(pool)])
            target = _pick(rng, [a, b])
            others = [a if target == b else b]
            candidates = [k for k in pool if k not in (a, b)]
        else:                                        # no identical objects on the table: the second twins unused
            candidates = [k for k in pool if k not in {b for _, b in TWINS}]
            cup = code == 'T2' and rng.random() < float(sv.get('cup_target_probability', 0.0))
            target = _pick(rng, [k for k in candidates if is_cup(objs[k]) == cup])
            others = []
        while len(others) < int(sv['n_objects']) - 1:
            others.append(_pick(rng, [k for k in candidates if k != target and k not in others]))
        spec.update(target=target, layout=_slots_for(rng, scene, [target, *others], slots), pair=pair)
        if code in ('T1', 'T4'):              # T4: a fixed place zone (no instruction; scene_variation.place_zone)
            spec['place'] = sv.get('place_zone') or _pick(rng, zones_of(scene, 'place'))
        if code == 'T2':
            spec['hand'] = _pick(rng, zones_of(scene, 'hand'))
            spec['timing'] = _pick(rng, list(sv['timings']))
        if code == 'T4':
            spec['phase'] = _pick(rng, list(sv['phases']))
            spec['hold_s'] = float(_pick(rng, list(sv['hold_s'])))
            spec['negative'] = bool(rng.random() < float(sv.get('negative_probability', 0.0)))
            spec['neg_target'] = _pick(rng, list(sv['negative_targets'])) if spec['negative'] else None
    elif code == 'T3':                              # two cubes in U, their paired cups on two slots
        pairs = sv['pairs']
        blocks, cups = list(pairs), [pairs[k] for k in pairs]
        spec.update(target=_pick(rng, cups), layout=_slots_for(rng, scene, cups, slots),
                    u_blocks=[blocks[i] for i in rng.permutation(len(blocks))])
    else:
        raise ValueError(f'unknown task code {code!r}')
    ic = cfg.intention_change
    if code in ('T1', 'T2', 'T3') and float(ic.get('probability', 0.0)) > 0 and rng.random() < float(ic['probability']):
        candidates = [k for k in spec['layout'] if k != spec['target']]
        spec['change'] = {'timing': _pick(rng, list(CHANGE_TIMINGS)), 'old': _pick(rng, candidates)}
    return spec


def validate_spec(cfg: ScenarioConfig, spec: dict) -> None:
    objs, scene = cfg.scene['objects'], cfg.scene
    code = spec['task']
    if code != cfg.task_code:
        raise ValueError(f"spec task {code} does not match scenario {cfg.id} ({cfg.task_code})")
    layout = spec['layout']
    slots = set(zones_of(scene, 'slot'))
    if not set(layout) <= set(objs) or not set(layout.values()) <= slots or len(set(layout.values())) != len(layout):
        raise ValueError(f'bad slot layout {layout}')
    if spec['target'] not in layout:
        raise ValueError('the target must stand on a slot')
    if cup_crowded(scene, layout):
        raise ValueError(f'a cup next to another object in a row: {layout}')
    need = {'T1': {'place': 'place'}, 'T4': {'place': 'place'}, 'T2': {'hand': 'hand'}}.get(code, {})
    for field_, kind in need.items():
        if spec.get(field_) not in zones_of(scene, kind):
            raise ValueError(f'{field_} must be one of {zones_of(scene, kind)}')
    if code == 'T2' and spec.get('timing') not in ('early', 'on_time', 'late', 'free'):
        raise ValueError('T2 timing must be early | on_time | late | free')
    if code == 'T3':
        pairs = cfg.scene_variation['pairs']
        if sorted(spec.get('u_blocks') or []) != sorted(pairs) or sorted(layout) != sorted(pairs.values()):
            raise ValueError(f'T3 needs the cubes {list(pairs)} in U and their paired cups {list(pairs.values())} on slots')
    if code == 'T4' and len(layout) != 1:
        raise ValueError('T4 has a single cube on the table')
    change = spec.get('change')
    if change:
        if code == 'T4' or change['timing'] not in CHANGE_TIMINGS or change['old'] not in layout or change['old'] == spec['target']:
            raise ValueError(f'bad change {change}')


def _uniform(rng, bounds) -> float:
    lo, hi = bounds
    return float(rng.uniform(lo, hi))


def sample_variation(cfg: ScenarioConfig, seed: int, spec: dict | None = None) -> dict:
    """Deterministic episode variation. Same (config, seed, spec) => identical dict.

    Continuous draws use one generator seeded by `seed` in a fixed order; the discrete spec comes from the
    scenario list or, if absent, from a second generator of the same seed.
    """
    sv, hb, ic = cfg.scene_variation, cfg.human_behavior, cfg.intention_change
    spec = copy.deepcopy(spec) if spec is not None else random_spec(cfg, np.random.default_rng([int(seed), 1]))
    validate_spec(cfg, spec)
    rng = np.random.default_rng(int(seed))
    lay = layout_of(cfg.scene)['zones']
    objs = cfg.scene['objects']
    var: dict = {'seed': int(seed), 'task': spec['task'], 'spec': spec}
    xy = {k: (np.asarray(lay[s]['xy'], float) + rng.uniform(-1, 1, 2) * sv['object_jitter_m']).tolist()
          for k, s in spec['layout'].items()}
    u_blocks = list(spec.get('u_blocks') or [])
    if u_blocks:                                     # T3: the two cubes on their spots in U (config order)
        u = next(k for k, z in lay.items() if z['kind'] == 'human_store')
        for k, spot in zip(u_blocks, lay[u]['blocks']):
            xy[k] = (np.asarray(spot, float) + rng.uniform(-1, 1, 2) * 0.005).tolist()
    var['object_xy'] = xy
    var['object_yaw'] = {k: 0.0 if is_cup(objs[k]) else float(rng.uniform(-1, 1) * sv.get('object_yaw_rad', 0.2)) for k in xy}
    var['on_table'] = list(xy)
    var['robot_home_offset'] = (rng.uniform(-1, 1, 6) * sv['robot_home_jitter_rad']).tolist()
    var['target_object'] = spec['target']
    var['target_zone'] = spec.get('place') or spec.get('hand') or \
        (next(k for k, z in lay.items() if z['kind'] == 'support') if spec['task'] == 'T3' else None)
    var['change'] = {**spec['change'], 'late_trigger_m': float(ic['late_trigger_m'])} if spec.get('change') else None
    if spec['task'] == 'T3':
        var['pairs'] = dict(sv['pairs'])            # cube in U -> its cup

    # Human timing / trajectory variant (fixed draw order; unused fields are still drawn).
    human = {'type': hb['type'], 'speed_scale': _uniform(rng, hb['speed_scale']),
             'cue_onset_s': _uniform(rng, hb['cue_onset_s']), 'point_s': _uniform(rng, hb['point_s']),
             'point_dwell_s': [_uniform(rng, hb['point_dwell_s']) for _ in range(3)],
             'rest_s': _uniform(rng, hb.get('rest_s', [1.0, 1.0])),
             'withdraw_change_s': _uniform(rng, ic.get('withdraw_s', [0.7, 0.7]))}
    t = hb['type']
    if t == 't2_receiver':
        human.update(reach_out_s=_uniform(rng, hb['reach_out_s']), late_wait_s=_uniform(rng, hb['late_wait_s']))
    elif t == 't3_requester':
        human.update(u_reach_s=_uniform(rng, hb['u_reach_s']), hold_s=_uniform(rng, hb['hold_s']),
                     drop_move_s=_uniform(rng, hb['drop_move_s']), put_back_s=_uniform(rng, hb['put_back_s']))
    elif t == 't4_intruder':
        human.update(trigger_delay_s=_uniform(rng, hb['trigger_delay_s']), approach_s=_uniform(rng, hb['approach_s']),
                     withdraw_s=_uniform(rng, hb['withdraw_s']))
    elif t != 't1_instructor':
        raise ValueError(f'unknown human behaviour {t!r}')
    # early change (T5): the first pointing gesture is held only briefly ("ngay sau ra hiệu đầu")
    human['early_dwell_s'] = _uniform(rng, ic.get('early_dwell_s', [0.5, 0.5]))
    human['change_pause_s'] = _uniform(rng, ic.get('pause_s', [0.0, 0.0]))     # pause at rest before the new target
    # fixed (no draw, the variation of older lists is unchanged): a pointing is held, looking at the target, until the
    # robot heads for it, at most point_hold_max_s (0: dwell only); T1 waits at most pick_wait_max_s for the pick
    human['point_hold_max_s'] = float(hb.get('point_hold_max_s', 0.0))
    human['pick_wait_max_s'] = float(hb.get('pick_wait_max_s', 1e9))
    # (2026-10-03) free human timing, drawn last (the earlier draws of older lists are unchanged): T2 holds the hand
    # out after a random delay (lowering the pointing hand first or not), T4 intrudes at a random moment for a random time
    # Plan v5: a scenario-list spec may fix the stratum of a timing (uniform within it) and the T2 lowering choice.
    if t == 't2_receiver' and 'reach_delay_s' in hb:
        human['reach_delay_s'] = _uniform(rng, spec.get('reach_delay_range') or hb['reach_delay_s'])
        lower = bool(rng.random() < float(hb.get('lower_first_prob', 0.5)))
        human['lower_first'] = bool(spec['lower_first']) if 'lower_first' in spec else lower
    if 'delay_s' in ic:                    # T5 (2026-10-03): the change of mind comes a random time after the first
        human['change_delay_s'] = _uniform(rng, (spec.get('change') or {}).get('delay_range') or ic['delay_s'])
        human['stop_s'] = _uniform(rng, ic['stop_s'])               # starting with a 'wait' signal (open hand, palm down,
        human['stop_hold_s'] = _uniform(rng, ic['stop_hold_s'])     # held out towards the robot), then pointing again
    if t == 't4_intruder' and 'intrusion_time_s' in hb:
        human['intrusion_time_s'] = _uniform(rng, spec.get('intrusion_time_range') or hb['intrusion_time_s'])
        human['intrusion_hold_s'] = _uniform(rng, spec.get('hold_range') or hb['intrusion_hold_s'])
    var['human'] = human
    return var


def load_scenario_list(path) -> list[dict]:
    """Read a scenario list (.jsonl from scripts/generate_scenarios.py): {index, task, base, scenario_id, seed, spec}."""
    import json
    p = Path(path) if Path(path).is_absolute() else ROOT / path
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
