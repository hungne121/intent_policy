"""Counterfactual intent swaps and their correct actions (INTENT_ACT_GUIDE_v2.md §6, §9.8; M0 answer on CGR).

  spatial swap   p_target and c_target move to a valid other place of the same kind: an object slot to another
                 occupied slot holding the same kind of object (cube <-> cube, cup <-> cup; the T3 cubes U1 <-> U2),
                 P1 <-> P2, H1 <-> H2; other places have no partner (not swapped)
  semantic swap  p_who and c_who exchange robot <-> human
Correct actions after a swap:
  (a) rule        spatial: every translation that reduces the error from the end effector at t to the new target by
                  more than the controller tolerance (half a step, the expert's `_toward` rule) on an axis where the
                  error exceeds it; the target is the place at travel height while the horizontal error is above
                  2 cm, then the place itself. semantic: robot -> human => HOLD (yield); human -> robot => the
                  spatial rule towards the current target
  (b) re-simulation  the scripted expert replanned to the new target from the exact simulator state at t (episode
                  replayed with the recorded actions): used to check (a) on sample frames
"""
from __future__ import annotations
import numpy as np
from intent_policy.sim.restricted_action import DIRECTIONS, RestrictedAction as A

HORIZONTAL_FIRST_M = 0.02
TRAVEL_DZ = 0.13


def object_shape(key: str) -> str:
    return 'cup' if key.startswith('C') else 'cube'


def swap_partners(spec: dict, schema: dict) -> dict[str, str]:
    """{place: partner} for the spatial swap of one episode: a permutation (the next same-kind occupied slot in name
    order, cyclic; P1 <-> P2; H1 <-> H2)."""
    out = {}
    occupied = {slot: object_shape(k) for k, slot in (spec.get('layout') or {}).items()}
    for i, k in enumerate(spec.get('u_blocks') or []):
        occupied[f'U{i + 1}'] = 'cube'
    for slot, shape in occupied.items():
        same = sorted(s for s, sh in occupied.items() if sh == shape and s != slot)
        if same:
            out[slot] = next((s for s in same if s > slot), same[0])
    for a, b in (('P1', 'P2'), ('H1', 'H2')):
        out[a], out[b] = b, a
    return out


def permute_targets(p: np.ndarray, partners: dict[str, str], targets: list[str]) -> np.ndarray:
    """q[partner(k)] = p[k]: `partners` is a permutation of its places (cycles among same-kind slots, P1 <-> P2,
    H1 <-> H2), so the mass is conserved; places without a partner keep theirs."""
    q = p.copy()
    for a, b in partners.items():
        q[..., targets.index(b)] = p[..., targets.index(a)]
    return q.astype(np.float32)


def swap_who(p: np.ndarray, who: list[str]) -> np.ndarray:
    q = p.copy()
    r, h = who.index('robot'), who.index('human')
    q[..., r], q[..., h] = p[..., h], p[..., r]
    return q


def approach_point(tcp: np.ndarray, place: np.ndarray, table_z: float = 0.62) -> np.ndarray:
    if np.linalg.norm(place[:2] - tcp[:2]) > HORIZONTAL_FIRST_M:
        return np.r_[place[:2], max(table_z + TRAVEL_DZ, place[2])]
    return place


def correct_moves(tcp: np.ndarray, target: np.ndarray, tol: float = 0.005) -> set[int]:
    """Translations (restricted action ids) that reduce the error to `target` on an axis where it exceeds `tol`
    (world-aligned control frame, configs/controller/restricted_action.yaml)."""
    err = np.asarray(target, float) - np.asarray(tcp, float)
    return {int(a) for a, (axis, sign) in DIRECTIONS.items() if abs(err[axis]) > tol and np.sign(err[axis]) == sign}


def correct_actions_spatial(tcp, new_target: str, places: dict) -> set[int] | None:
    if new_target not in places:
        return None
    moves = correct_moves(tcp, approach_point(np.asarray(tcp, float), places[new_target]))
    return moves or None


def correct_actions_semantic(tcp, new_who: str, target: str | None, places: dict) -> set[int] | None:
    if new_who == 'human':
        return {int(A.HOLD)}
    if new_who == 'robot' and target is not None:
        return correct_actions_spatial(tcp, target, places)
    return None


# ---------------------------------------------------------------------- (b) re-simulation
def expert_counterfactual(exp: dict, episode_meta: dict, actions: np.ndarray, t: int, new_place: str) -> dict:
    """Replay the episode's recorded restricted actions up to frame t, then ask the scripted expert, replanned to grasp
    the object standing on `new_place` (or to go to that zone), for its next action. Returns action, TCP, setpoint."""
    from intent_policy.experts.scripted_expert import ScriptedExpert
    from intent_policy.scenarios.scenario_registry import make_scenario, scenario_overrides
    from intent_policy.sim.restricted_action import RestrictedActionMapper
    from intent_policy.utils import controller_config

    sid, spec = episode_meta['scenario_id'], episode_meta['spec']
    sc = make_scenario(sid, scenario_overrides(exp, sid))
    try:
        sc.reset(int(episode_meta['seed']), spec=spec)
        mapper = RestrictedActionMapper(sc.env, controller_config(exp))
        for a in actions[:t]:
            cmd = mapper.map(int(a))
            for _ in range(cmd.execution_ticks):
                sc.step(cmd.joint_target)
        ex = ScriptedExpert(sc, mapper, rng_seed=sc.seed)
        ex.reset()
        ex.trigger, ex.anticipate, ex.n_changes = 'cue_complete', False, len(sc.intention_changes)
        obj = next((k for k, s in (spec.get('layout') or {}).items() if s == new_place), None)
        if obj is None and new_place.startswith('U') and new_place[1:].isdigit():
            obj = spec['u_blocks'][int(new_place[1:]) - 1]
        if obj is not None:
            ex.obj, ex.zone = obj, None
            plan = ex._grasp_steps(obj)
        else:
            ex.zone = new_place
            from intent_policy.experts.scripted_expert import Step
            plan = [Step('goto', lambda: np.r_[sc.zone_pos(new_place)[:2], ex.travel], name='to_zone')]
        ex._set_plan(plan)
        a = int(ex.act())
        return dict(action=a, tcp=sc.tcp().round(5).tolist(), setpoint=mapper.setpoint.round(5).tolist(), stage=ex.stage)
    finally:
        sc.close()
