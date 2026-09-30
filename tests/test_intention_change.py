"""T5 change of mind on T1-T3 (withdraw, then point at the new target; before the robot grasps), expert triggers
(cue_complete / evidence / cue_onset), scenario overrides."""
import numpy as np
import pytest

from intent_policy.benchmark.runner import ExpertAgent, run_episode
from intent_policy.scenarios.config import ScenarioConfig, random_spec, sample_variation
from intent_policy.scenarios.scenario_registry import make_scenario, scenario_overrides
from scripts.reproduce_episode import compare

CHANGE = {'intention_change': {'probability': 1.0}}
BASES = ('t1_pick_place', 't2_handover', 't3_assist')


class IgnoreChangeExpert(ExpertAgent):
    """Keeps executing the original plan after the human changed their mind (failure probe)."""
    def reset(self, scenario, mapper):
        super().reset(scenario, mapper)
        self.expert._follow_intention_change = lambda: None


def change_spec(sid: str, timing: str, seed: int = 1) -> dict:
    cfg = ScenarioConfig.load(sid)
    spec = random_spec(cfg, np.random.default_rng(seed))
    if spec['task'] == 'T2':
        spec['timing'] = 'on_time'
    old = next(k for k in spec['layout'] if k != spec['target'])
    return dict(spec, change=dict(timing=timing, old=old))


def test_overrides_are_merged_and_stored():
    exp = {'scenario_overrides': {'all': {'expert': {'trigger': 'evidence'}}, 't4_interrupt': {'expert': {'yield_prediction_horizon_s': 0.5}}}}
    ov = scenario_overrides(exp, 't4_interrupt')
    assert ov['expert'] == {'trigger': 'evidence', 'yield_prediction_horizon_s': 0.5}
    cfg = ScenarioConfig.load('t4_interrupt', ov)
    assert cfg.expert['trigger'] == 'evidence' and cfg.to_dict()['expert']['yield_prediction_horizon_s'] == 0.5
    assert ScenarioConfig.load('t4_interrupt').expert['trigger'] == 'cue_complete'


@pytest.mark.parametrize('sid', BASES)
def test_change_is_off_by_default_and_part_of_the_spec(sid):
    base, changed = ScenarioConfig.load(sid), ScenarioConfig.load(sid, CHANGE)
    for seed in range(5):
        v0, v1 = sample_variation(base, seed), sample_variation(changed, seed)
        assert v0['change'] is None and 'change' not in v0['spec']
        ch = v1['change']
        assert ch['timing'] in ('early', 'late') and ch['old'] in v1['spec']['layout'] and ch['old'] != v1['target_object']
    with pytest.raises(ValueError):
        sample_variation(ScenarioConfig.load('t4_interrupt'), 0, dict(random_spec(ScenarioConfig.load('t4_interrupt'),
                                                                               np.random.default_rng(0)), change={'timing': 'early', 'old': 'B2'}))


@pytest.mark.parametrize('trigger', ['cue_complete', 'evidence'])
@pytest.mark.parametrize('timing', ['early', 'late'])
@pytest.mark.parametrize('sid', BASES)
def test_expert_follows_the_change_of_mind(sid, timing, trigger):
    sc = make_scenario(sid, {'expert': {'trigger': trigger}})
    try:
        spec = change_spec(sid, timing)
        rec = run_episode(sc, ExpertAgent(), 1, keep_trace=False, spec=spec)
        changes = [e for e in rec['events'] if e['event_type'] == 'human_intention_change']
        assert rec['success'], rec['failure']
        assert len(changes) == 1 and changes[0]['payload']['previous'] == spec['change']['old']
        assert changes[0]['payload']['current'] == spec['target'] == sc.robot_target()
        assert not any(e['event_type'] == 'object_grasp' and e['entity_id'] == 'robot' and e['payload']['object'] == spec['change']['old']
                       for e in rec['events'])
        steps = rec['protocol_steps_completed']
        assert 'change_indicated' in steps
        assert not any(e['event_type'] in ('human_robot_contact',) for e in rec['events'])
    finally:
        sc.close()


@pytest.mark.parametrize('sid', BASES)
def test_ignoring_a_late_change_fails(sid):
    sc = make_scenario(sid)
    try:
        rec = run_episode(sc, IgnoreChangeExpert(), 1, keep_trace=False, spec=change_spec(sid, 'late'))
    finally:
        sc.close()
    assert not rec['success'] and rec['failure'] in ('touched_other_object', 'wrong_object_manipulated')


def test_anticipating_triggers_start_earlier():
    records = {}
    for trigger in ('cue_complete', 'evidence', 'cue_onset'):
        sc = make_scenario('t1_pick_place', {'expert': {'trigger': trigger}})
        try:
            records[trigger] = run_episode(sc, ExpertAgent(), 5, keep_trace=False)
        finally:
            sc.close()
    lag = {}
    for trigger, rec in records.items():
        assert rec['success'], (trigger, rec['failure'])
        cue = next(e['timestamp'] for e in rec['events'] if e['event_type'] == 'human_cue_onset')
        lag[trigger] = next(e['timestamp'] for e in rec['events'] if e['event_type'] == 'robot_motion_start' and e['timestamp'] >= cue) - cue
    assert lag['cue_onset'] < 0.3 < 2.0 < lag['cue_complete']
    assert records['evidence']['metrics']['CT'] < records['cue_complete']['metrics']['CT']


def test_predictive_yield_expert_succeeds_without_contact():
    sc = make_scenario('t4_interrupt', {'expert': {'trigger': 'evidence', 'yield_prediction_horizon_s': 0.5}})
    try:
        rec = run_episode(sc, ExpertAgent(), 3, keep_trace=False)
    finally:
        sc.close()
    assert rec['success'] and rec['metrics']['DSR'] == 1.0 and rec['metrics']['CFR'] == 1.0


def test_change_episode_reproduces_from_saved_config():
    sc = make_scenario('t2_handover', {'expert': {'trigger': 'evidence'}})
    try:
        rec = run_episode(sc, ExpertAgent(), 1, spec=change_spec('t2_handover', 'late'))
    finally:
        sc.close()
    sc = make_scenario(rec['scenario_config'])
    try:
        again = run_episode(sc, ExpertAgent(), rec['seed'], episode_id=rec['episode_id'], spec=rec['spec'])
    finally:
        sc.close()
    assert rec['spec']['change']['timing'] == 'late' and compare(rec, again) == []
