"""Scenario configuration, seed reproducibility, success/failure predicates, ground truth."""
import ast
from pathlib import Path

import numpy as np
import pytest

from benchmark.metrics import ALL_METRICS
from benchmark.runner import ExpertAgent, run_episode
from env.base_env import ROOT
from scenarios.config import ScenarioConfig, sample_variation
from scenarios.scenario_registry import PHASE1_SCENARIOS, SCENARIOS, make_scenario
from conftest import hold_agent_steps

ROLES = {'instructor_object_to_target': 'instructor', 'collaborator_object_handover': 'collaborator',
         'collaborator_bowl_assistance': 'collaborator', 'intruder_pick_place_interruption': 'intruder'}


class WrongChoiceExpert(ExpertAgent):
    """Expert that acts on a different object/bowl than the human asked for (failure probe)."""
    def reset(self, scenario, mapper):
        v = scenario.variation
        key = 'requested_object' if 'requested_object' in v else 'selected_object'
        others = [o for o in scenario.objects if o != v[key]]
        v[key] = others[0]             # only the expert sees this; the scenario keeps the true choice
        super().reset(scenario, mapper)


class NoYieldExpert(ExpertAgent):
    def reset(self, scenario, mapper):
        super().reset(scenario, mapper)
        self.expert._yield = lambda: False


def test_four_phase1_scenarios_with_one_primary_role():
    assert set(PHASE1_SCENARIOS) == set(ROLES)
    for sid, role in ROLES.items():
        cfg = ScenarioConfig.load(sid)
        assert cfg.id == sid and cfg.role == role
        assert cfg.success_conditions and cfg.protocol_steps and cfg.applicable_metrics
        assert set(cfg.applicable_metrics) <= set(ALL_METRICS)
        for section in ('scene_variation', 'human_behavior', 'timing', 'safety_constraints'):
            assert getattr(cfg, section)
        assert Path(ROOT / cfg.source).exists()


def test_instructor_scene_contents():
    cfg = ScenarioConfig.load('instructor_object_to_target')
    objs = cfg.scene['objects']
    assert len(objs) == 3 and len(cfg.scene['regions']) == 2
    assert len({o['shape'] for o in objs.values()}) == 3 and len({o['color'] for o in objs.values()}) == 3
    assert len({r['color'] for r in cfg.scene['regions'].values()}) == 2
    bowl = ScenarioConfig.load('collaborator_bowl_assistance')
    assert {b['pairs_with'] for b in bowl.scene['bowls'].values()} == set(bowl.scene['objects'])


def test_scenario_code_is_independent_of_policy_code():
    for pkg in ('scenarios', 'benchmark', 'human', 'env', 'controllers'):
        for f in (ROOT / pkg).glob('*.py'):
            tree = ast.parse(f.read_text())
            mods = {n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
            mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            assert not any(m.startswith(('policies', 'lerobot', 'torch')) for m in mods), f'{f} imports policy code'


def test_registry_selects_environments_not_policies():
    for cls in SCENARIOS.values():
        assert 'policy' not in cls.__module__ and not hasattr(cls, 'select_action')


@pytest.mark.parametrize('sid', PHASE1_SCENARIOS)
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


@pytest.mark.parametrize('sid', PHASE1_SCENARIOS)
def test_different_seeds_give_controlled_variation(sid):
    cfg = ScenarioConfig.load(sid)
    vs = [sample_variation(cfg, s) for s in range(12)]
    assert len({str(v['object_xy']) for v in vs}) == 12
    assert len({round(v['human']['speed_scale'], 6) for v in vs}) == 12
    key = 'requested_object' if 'requested_object' in vs[0] else 'selected_object'
    if len(cfg.scene_variation[key]) > 1:
        assert len({v[key] for v in vs}) > 1
    assert sample_variation(cfg, 5) == sample_variation(cfg, 5)


@pytest.mark.parametrize('sid', PHASE1_SCENARIOS)
def test_expert_reaches_success_predicate(scenarios, sid):
    rec = run_episode(scenarios[sid], ExpertAgent(), 3, keep_trace=False)
    assert rec['success'] and rec['failure'] is None
    assert rec['protocol_steps_completed'][:1] and set(scenarios[sid].cfg.protocol_steps) <= set(rec['protocol_steps_completed'])
    assert rec['metrics']['CSR'] == 1.0
    assert rec['metrics']['OC'] == (1.0 if 'OC' in scenarios[sid].cfg.applicable_metrics else 'n/a')


@pytest.mark.parametrize('sid', PHASE1_SCENARIOS)
def test_idle_robot_times_out(scenarios, sid):
    sc = scenarios[sid]
    sc.reset(2)
    hold_agent_steps(sc, 2000)
    assert sc.is_failure() and sc.failure == 'timeout' and not sc.is_success()
    assert sc.get_events()[-1]['event_type'] == 'episode_end'


@pytest.mark.parametrize('sid', ['instructor_object_to_target', 'collaborator_object_handover', 'collaborator_bowl_assistance'])
def test_wrong_object_or_bowl_fails(scenarios, sid):
    rec = run_episode(scenarios[sid], WrongChoiceExpert(), 4, keep_trace=False)
    assert not rec['success'] and rec['failure'] == 'wrong_object_manipulated'


def test_instructor_wrong_target_region_fails(scenarios):
    class WrongTarget(ExpertAgent):
        def reset(self, scenario, mapper):
            v = scenario.variation
            v['requested_target'] = [r for r in scenario.regions if r != v['requested_target']][0]
            super().reset(scenario, mapper)
    rec = run_episode(scenarios['instructor_object_to_target'], WrongTarget(), 4, keep_trace=False)
    assert not rec['success'] and rec['failure'] == 'wrong_target_region'


def test_intruder_robot_that_does_not_yield_is_not_successful(scenarios):
    rec = run_episode(scenarios['intruder_pick_place_interruption'], NoYieldExpert(), 5, keep_trace=False)
    assert not rec['success']
    kinds = [e['event_type'] for e in rec['events']]
    assert 'human_robot_contact' in kinds or any(e['payload'].get('kind') == 'yield' for e in rec['events'])


def test_intruder_disruption_and_recovery_events(scenarios):
    rec = run_episode(scenarios['intruder_pick_place_interruption'], ExpertAgent(), 6, keep_trace=False)
    kinds = [e['event_type'] for e in rec['events']]
    order = [kinds.index(k) for k in ('disruption_start', 'disruption_end', 'recovery_start', 'recovery_complete', 'task_success')]
    assert order == sorted(order)
    assert rec['metrics']['DSR'] == 1.0 and rec['metrics']['CFR'] == 1.0


@pytest.mark.parametrize('sid', PHASE1_SCENARIOS)
def test_ground_truth_available_but_not_observed(scenarios, sid):
    sc = scenarios[sid]
    sc.reset(8)
    hold_agent_steps(sc, 40)
    gt = sc.get_ground_truth_intention_information()
    for k in ('selected_object', 'hand_position', 'hand_velocity', 'future_hand_position', 'future_hand_velocity',
              'target_position', 'frame'):
        assert k in gt
    assert gt['selected_object'] is not None or sid == 'intruder_pick_place_interruption'
    hs = sc.get_human_state()
    assert hs.hand_position.shape == (3,) and hs.hand_velocity.shape == (3,) and hs.torso_position.shape == (3,)
    obs = sc.get_observation(('scene', 'wrist'), 32, 24)
    assert set(obs) == {'observation.state', 'observation.images.scene', 'observation.images.wrist'}
    assert obs['observation.state'].shape == (10,) and obs['observation.images.scene'].shape == (24, 32, 3)


def test_bowl_scenario_needs_human_information():
    """Before the human acts, the two bowl cases are observationally symmetric in layout rules."""
    cfg = ScenarioConfig.load('collaborator_bowl_assistance')
    choices = {sample_variation(cfg, s)['selected_object'] for s in range(20)}
    assert choices == set(cfg.scene['objects'])


def test_make_scenario_accepts_saved_config_dict(scenarios):
    raw = scenarios['instructor_object_to_target'].cfg.to_dict()
    sc = make_scenario(raw)
    try:
        assert sc.cfg.id == 'instructor_object_to_target'
        assert sc.cfg.scene == scenarios['instructor_object_to_target'].cfg.scene
    finally:
        sc.close()
