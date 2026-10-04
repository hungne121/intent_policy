"""Task scenarios T1-T5 (docs/requirements/scence_construct.md): configuration, episode specs, seed reproducibility,
success / failure predicates, ground truth, balanced scenario lists."""
import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from intent_policy.benchmark.metrics import ALL_METRICS
from intent_policy.benchmark.runner import ExpertAgent, run_episode
from intent_policy.sim.base_env import ROOT
from intent_policy.scenarios.config import (ScenarioConfig, load_scenario_list, random_spec, sample_variation,
                                            validate_spec, twin_of)
from intent_policy.scenarios.scenario_registry import SCENARIO_OF_TASK, SCENARIOS, TASK_SCENARIOS, make_scenario
from intent_policy.utils import load_yaml
from conftest import hold_agent_steps

ROLES = {'t1_pick_place': ('instructor', 'T1'), 't2_handover': ('collaborator', 'T2'),
         't3_assist': ('collaborator', 'T3'), 't4_interrupt': ('intruder', 'T4')}
OK_SEED = 3                         # a seed whose random spec the expert completes in every scenario


class PatchedExpert(ExpertAgent):
    """Expert with a patch applied after reset (failure probes; the scenario keeps the true intention)."""
    def __init__(self, patch):
        self.patch = patch

    def reset(self, scenario, mapper):
        super().reset(scenario, mapper)
        self.patch(self.expert, scenario)


def wrong_object(e, sc):
    def start():
        e.obj = next(k for k in sc.variation['spec']['layout'] if k != sc.human.selected_object)
        e.zone = e._known_zone()
        e._commit(e.obj, e._task_steps())
    e.plan[1].target = start


def test_four_task_scenarios_with_one_primary_role():
    assert set(TASK_SCENARIOS) == set(ROLES)
    for sid, (role, code) in ROLES.items():
        cfg = ScenarioConfig.load(sid)
        assert cfg.id == sid and cfg.role == role and cfg.task_code == code and SCENARIO_OF_TASK[code] == sid
        assert cfg.success_conditions and cfg.protocol_steps and cfg.applicable_metrics
        assert set(cfg.applicable_metrics) <= set(ALL_METRICS)
        for section in ('scene_variation', 'human_behavior', 'timing', 'safety_constraints'):
            assert getattr(cfg, section)
        assert Path(ROOT / cfg.source).exists()


def test_object_catalog_and_identical_pairs():
    objs = ScenarioConfig.load('t2_handover').scene['objects']
    assert set(objs) == {'B1', 'B2', 'B3', 'B1p', 'C1', 'C2', 'C3'}
    same = lambda a, b: (objs[a]['shape'], objs[a]['size'], objs[a]['color']) == (objs[b]['shape'], objs[b]['size'], objs[b]['color'])
    assert same('B1', 'B1p') and same('C1', 'C2') and not same('C1', 'C3') and not same('B1', 'B2')
    assert twin_of('B1') == 'B1p' and twin_of('C2') == 'C1' and twin_of('B3') == 'B3p' and twin_of('C3') is None
    r, h = objs['C1']['size']
    assert 2 * (r - 0.004) > np.hypot(0.04, 0.04)          # a 4 cm cube fits into a cup (T3)


def test_scenario_code_is_independent_of_policy_code():
    for pkg in ('scenarios', 'benchmark', 'sim'):
        for f in (ROOT / 'intent_policy' / pkg).glob('*.py'):
            tree = ast.parse(f.read_text())
            mods = {n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
            mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            assert not any(m.startswith(('intent_policy.policies', 'lerobot', 'torch')) for m in mods), f'{f} imports policy code'


def test_registry_selects_environments_not_policies():
    for cls in SCENARIOS.values():
        assert not cls.__module__.startswith('intent_policy.policies') and not hasattr(cls, 'select_action')


@pytest.mark.parametrize('sid', TASK_SCENARIOS)
def test_random_specs_are_valid_and_follow_the_task_rules(sid):
    cfg = ScenarioConfig.load(sid)
    specs = [random_spec(cfg, np.random.default_rng(s)) for s in range(40)]
    for spec in specs:
        validate_spec(cfg, spec)
        on_table = list(spec['layout'])
        twins = [k for k in on_table if twin_of(k) in on_table]
        if spec.get('pair'):
            assert spec['target'] in twins and len(twins) == 2
        elif cfg.task_code != 'T3':
            assert not twins
    assert len({spec['layout'][spec['target']] for spec in specs}) == 6          # every slot can be the target
    with pytest.raises(ValueError):
        validate_spec(cfg, dict(specs[0], target='nope'))


@pytest.mark.parametrize('sid', TASK_SCENARIOS)
def test_same_seed_same_scene_and_human_trajectory(scenarios, sid):
    sc = scenarios[sid]
    traces = []
    for _ in range(2):
        sc.reset(11)
        poses = {k: sc.pos(k).round(6).tolist() for k in sc.graspables}
        hand = []
        for _ in range(60):
            sc.step(sc.env.hold())
            hand.append(sc.human.pos.copy())
        traces.append((sc.variation, poses, np.array(hand), [(e['event_type'], e['timestamp']) for e in sc.get_events()]))
    (v1, p1, h1, e1), (v2, p2, h2, e2) = traces
    assert v1 == v2 and p1 == p2 and e1 == e2
    np.testing.assert_array_equal(h1, h2)


@pytest.mark.parametrize('sid', TASK_SCENARIOS)
def test_different_seeds_give_controlled_variation(sid):
    cfg = ScenarioConfig.load(sid)
    vs = [sample_variation(cfg, s) for s in range(12)]
    assert len({str(v['object_xy']) for v in vs}) == 12
    assert len({round(v['human']['speed_scale'], 6) for v in vs}) == 12
    assert len({v['target_object'] for v in vs}) > 1
    assert sample_variation(cfg, 5) == sample_variation(cfg, 5)
    spec = vs[0]['spec']
    assert sample_variation(cfg, 7, spec)['spec'] == spec          # a list spec overrides the seed's own spec


@pytest.mark.parametrize('sid', TASK_SCENARIOS)
def test_unused_objects_are_parked_out_of_sight(scenarios, sid):
    sc = scenarios[sid]
    sc.reset(OK_SEED)
    for k in sc.objects:
        z = sc.pos(k)[2]
        assert (abs(z - sc.rest_height[k]) < 1e-6 and z > sc.table_z) if k in sc.on_table else z < sc.table_z - 0.5


@pytest.mark.parametrize('sid', TASK_SCENARIOS)
def test_expert_reaches_success_predicate(scenarios, sid):
    rec = run_episode(scenarios[sid], ExpertAgent(), OK_SEED, keep_trace=False)
    assert rec['success'] and rec['failure'] is None, rec['failure']
    assert set(scenarios[sid].cfg.protocol_steps) <= set(rec['protocol_steps_completed'])
    assert rec['metrics']['CSR'] == 1.0
    assert rec['metrics']['OC'] == (1.0 if 'OC' in scenarios[sid].cfg.applicable_metrics else 'n/a')


@pytest.mark.parametrize('sid', TASK_SCENARIOS)
def test_idle_robot_times_out(scenarios, sid):
    sc = scenarios[sid]
    sc.reset(2)
    hold_agent_steps(sc, 2000)
    assert sc.is_failure() and sc.failure == 'timeout' and not sc.is_success()
    assert sc.get_events()[-1]['event_type'] == 'episode_end'


@pytest.mark.parametrize('sid', ['t1_pick_place', 't2_handover', 't3_assist'])
def test_wrong_object_fails(scenarios, sid):
    rec = run_episode(scenarios[sid], PatchedExpert(wrong_object), OK_SEED, keep_trace=False)
    assert not rec['success'] and rec['failure'] in ('touched_other_object', 'wrong_object_manipulated')


def test_wrong_place_zone_fails(scenarios):
    def patch(e, sc):
        e._known_zone = lambda: next(z for z in sc.places if z != sc.place_zone)
    rec = run_episode(scenarios['t1_pick_place'], PatchedExpert(patch), OK_SEED, keep_trace=False)
    assert not rec['success'] and rec['failure'] == 'wrong_place_zone'


def let_go_above_handover_point(dx: float, dz: float):
    """Expert patch: stop dz above (dx beside) the handover point and open at once, before the human closes the hand."""
    def patch(e, sc):
        steps = e._handover_steps

        def patched(obj):
            out = steps(obj)
            for s in out:
                if s.name == 'to_handover':
                    s.target = (lambda f: lambda: f() + [dx, 0.0, dz])(s.target)
                elif s.name == 'wait_pull':
                    s.until = lambda: True
            return out
        e._handover_steps = patched
    return patch


@pytest.mark.parametrize('dx, received', [(0.0, True), (0.12, False)])
def test_object_let_go_into_the_waiting_palm_rests_on_it(scenarios, dx, received):
    """Let go 3 cm above the waiting palm: the object lands on it and the human closes the hand on it (palm support);
    let go beside the hand: it falls to the table."""
    spec = {'task': 'T2', 'target': 'B1', 'layout': {'B1': 'S5', 'B1p': 'S2', 'B3': 'S3'}, 'pair': True, 'hand': 'H1',
            'timing': 'early'}
    rec = run_episode(scenarios['t2_handover'], PatchedExpert(let_go_above_handover_point(dx, 0.03)), OK_SEED,
                      keep_trace=False, spec=spec)
    if received:
        assert rec['success'], rec['failure']
        t = lambda kind, actor: next(e['timestamp'] for e in rec['events'] if e['event_type'] == kind
                                     and e['entity_id'] == actor and e['payload'].get('object') == 'B1')
        assert t('object_release', 'robot') < t('object_grasp', 'human')     # caught in the palm, not taken from the gripper
    else:
        assert not rec['success'] and rec['failure'] == 'object_dropped'


def test_robot_that_does_not_yield_is_not_successful(scenarios):
    def patch(e, sc):
        e._yield = lambda: False
    sc = scenarios['t4_interrupt']
    spec = dict(random_spec(sc.cfg, np.random.default_rng(0)), phase='carry', hold_s=4.0, negative=False, neg_target=None)
    rec = run_episode(sc, PatchedExpert(patch), OK_SEED, keep_trace=False, spec=spec)
    assert not rec['success']
    kinds = [e['event_type'] for e in rec['events']]
    assert 'human_robot_contact' in kinds or any(e['payload'].get('kind') == 'yield' for e in rec['events'])


def test_interrupt_disruption_and_recovery_events(scenarios):
    rec = run_episode(scenarios['t4_interrupt'], ExpertAgent(), OK_SEED, keep_trace=False)
    kinds = [e['event_type'] for e in rec['events']]
    order = [kinds.index(k) for k in ('disruption_start', 'disruption_end', 'recovery_start', 'recovery_complete', 'task_success')]
    assert order == sorted(order)
    assert rec['metrics']['DSR'] == 1.0 and rec['metrics']['CFR'] == 1.0


def test_negative_interrupt_robot_must_not_stop(scenarios):
    sc = scenarios['t4_interrupt']
    spec = next(e['spec'] for e in load_scenario_list('configs/scenario_lists/demo_v1.jsonl') if e['task'] == 'T4neg')
    rec = run_episode(sc, ExpertAgent(), 1, keep_trace=False, spec=spec)
    assert rec['success'], rec['failure']
    assert 'disruption_start' not in [e['event_type'] for e in rec['events']]

    def stops(e, s):                   # a robot that stops for any nearby hand fails T4-neg
        e._yield = lambda: s.human.stage in ('reach_near', 'dwelling')
    rec = run_episode(sc, PatchedExpert(stops), 1, keep_trace=False, spec=spec)
    assert not rec['success'] and rec['failure'] == 'unnecessary_stop'


def test_t3_robot_brings_the_cup_paired_with_the_picked_cube(scenarios):
    """T3 is a learned workflow: the human only picks a cube (no pointing), the robot brings that cube's cup."""
    sc = scenarios['t3_assist']
    pairs = sc.cfg.scene_variation['pairs']
    for target in pairs.values():
        spec = dict(random_spec(sc.cfg, np.random.default_rng(2)), target=target)
        rec = run_episode(sc, ExpertAgent(), OK_SEED, keep_trace=False, spec=spec)
        assert rec['success'], rec['failure']
        picked = next(e['payload']['object'] for e in rec['events']
                      if e['event_type'] == 'object_grasp' and e['payload'].get('actor') == 'human')
        assert pairs[picked] == target
        assert not any(e['event_type'] == 'human_motion_start' and e['payload']['label'].startswith('point')
                       for e in rec['events'])


def test_t4_robot_starts_without_instruction(scenarios):
    """T4: one cube, fixed place zone; the robot starts by itself and the human only intrudes."""
    sc = scenarios['t4_interrupt']
    rec = run_episode(sc, ExpertAgent(), OK_SEED, keep_trace=False)
    assert rec['success'], rec['failure']
    assert len(rec['spec']['layout']) == 1 and rec['spec']['place'] == sc.cfg.scene_variation['place_zone']
    events = rec['events']
    first_move = next(e['timestamp'] for e in events if e['event_type'] == 'robot_action_change'
                      and e['payload']['action'].startswith('MOVE'))
    human = [e for e in events if e['event_type'] == 'human_motion_start']
    assert first_move < human[0]['timestamp'] and human[0]['payload']['label'] in ('intrude', 'reach_near', 'intrude_approach')
    assert 'human_cue_onset' not in [e['event_type'] for e in events]


def test_no_cup_next_to_another_object_in_a_row():
    """The open gripper would hit a cup standing on the neighbouring slot of the same row."""
    cfg = ScenarioConfig.load('t2_handover')
    ok = {'task': 'T2', 'target': 'C1', 'layout': {'C1': 'S5', 'B2': 'S1', 'B3': 'S3'}, 'pair': False, 'hand': 'H1',
          'timing': 'early'}
    validate_spec(cfg, ok)
    with pytest.raises(ValueError):
        validate_spec(cfg, dict(ok, layout={'C1': 'S5', 'B2': 'S4', 'B3': 'S1'}))


@pytest.mark.parametrize('sid', TASK_SCENARIOS)
def test_ground_truth_available_but_not_observed(scenarios, sid):
    sc = scenarios[sid]
    sc.reset(8)
    hold_agent_steps(sc, 40)
    gt = sc.get_ground_truth_intention_information()
    for k in ('selected_object', 'hand_position', 'hand_velocity', 'future_hand_position', 'future_hand_velocity',
              'target_position', 'frame'):
        assert k in gt
    assert gt['selected_object'] is not None
    hs = sc.get_human_state()
    assert hs.hand_position.shape == (3,) and hs.hand_velocity.shape == (3,) and hs.torso_position.shape == (3,)
    obs = sc.get_observation(('high', 'wrist'), 32, 24)
    assert set(obs) == {'observation.state', 'observation.images.high', 'observation.images.wrist'}
    assert obs['observation.state'].shape == (10,) and obs['observation.images.high'].shape == (24, 32, 3)


def test_make_scenario_accepts_saved_config_dict(scenarios):
    raw = scenarios['t1_pick_place'].cfg.to_dict()
    sc = make_scenario(raw)
    try:
        assert sc.cfg.id == 't1_pick_place' and sc.cfg.scene == scenarios['t1_pick_place'].cfg.scene
    finally:
        sc.close()


@pytest.mark.parametrize('split', ['demo', 'eval'])
def test_scenario_lists_follow_the_generation_rules(split):
    gcfg = load_yaml('configs/scenario_lists/generator.yaml')
    eps = load_scenario_list(f"configs/scenario_lists/{split}_v{gcfg['version']}.jsonl")
    assert dict(Counter(e['task'] for e in eps)) == gcfg['counts'][split]
    cfgs = {code: ScenarioConfig.load(sid) for code, sid in SCENARIO_OF_TASK.items()}
    slot = lambda e: e['spec']['layout'][e['spec']['target']]
    for e in eps:
        validate_spec(cfgs[e['base']], e['spec'])
    assert all(slot(a) != slot(b) for a, b in zip(eps, eps[1:]))                    # rule 4
    assert len({e['seed'] for e in eps}) == len(eps)
    for task in ('T1', 'T2'):                                                         # rule 3
        s = [e['spec'] for e in eps if e['task'] == task]
        assert sum(x['pair'] for x in s) == round(len(s) * gcfg['pair_fraction'])
    t1 = Counter((slot(e), e['spec']['place']) for e in eps if e['task'] == 'T1')    # rule 1
    assert len(t1) == 12 and max(t1.values()) - min(t1.values()) <= 1
    t5 = [e for e in eps if e['task'] == 'T5']
    assert all(e['spec']['change']['old'] != e['spec']['target'] for e in t5)
    assert Counter(e['spec']['change']['timing'] for e in t5) == {'early': len(t5) // 2, 'late': len(t5) // 2}
