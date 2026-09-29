"""Phase-2 scenario extensions: cue-onset expert, intention-change variants, scenario overrides."""
import pytest

from benchmark.runner import ExpertAgent, run_episode
from scenarios.config import ScenarioConfig, sample_variation
from scenarios.scenario_registry import make_scenario, scenario_overrides
from scripts.reproduce_episode import compare

ONSET = {'expert': {'trigger': 'cue_onset'}}
CHANGE = {'expert': {'trigger': 'cue_onset'}, 'intention_change': {'probability': 1.0}}
CHANGE_SCENARIOS = ('collaborator_object_handover', 'instructor_object_to_target')


class IgnoreChangeExpert(ExpertAgent):
    """Keeps executing the original plan after the human changed their mind (failure probe)."""
    def reset(self, scenario, mapper):
        super().reset(scenario, mapper)
        self.expert._follow_intention_change = lambda: None


@pytest.fixture(scope='module')
def change_scenarios():
    made = {sid: make_scenario(sid, CHANGE) for sid in CHANGE_SCENARIOS}
    yield made
    for sc in made.values():
        sc.close()


def test_overrides_are_merged_and_stored():
    exp = {'scenario_overrides': {'all': ONSET, 'intruder_pick_place_interruption': {'expert': {'yield_prediction_horizon_s': 0.5}}}}
    ov = scenario_overrides(exp, 'intruder_pick_place_interruption')
    assert ov['expert'] == {'trigger': 'cue_onset', 'yield_prediction_horizon_s': 0.5}
    cfg = ScenarioConfig.load('intruder_pick_place_interruption', ov)
    assert cfg.expert['trigger'] == 'cue_onset' and cfg.to_dict()['expert']['yield_prediction_horizon_s'] == 0.5
    assert ScenarioConfig.load('intruder_pick_place_interruption').expert['trigger'] == 'cue_complete'


@pytest.mark.parametrize('sid', CHANGE_SCENARIOS)
def test_intention_change_variation_is_appended_and_disabled_by_default(sid):
    base = ScenarioConfig.load(sid)
    changed = ScenarioConfig.load(sid, CHANGE)
    for seed in range(5):
        v0, v1 = sample_variation(base, seed), sample_variation(changed, seed)
        assert 'intention_change' not in v0
        assert v1['intention_change']['enabled'] and v1['intention_change']['type'] == changed.intention_change['type']
        assert {k: v for k, v in v1.items() if k != 'intention_change'} == v0     # earlier draws unchanged


@pytest.mark.parametrize('sid', CHANGE_SCENARIOS)
def test_expert_follows_the_changed_intention(change_scenarios, sid):
    sc = change_scenarios[sid]
    rec = run_episode(sc, ExpertAgent(), 7, keep_trace=False)
    changes = [e for e in rec['events'] if e['event_type'] == 'human_intention_change']
    assert rec['success'] and len(changes) == 1
    payload = changes[0]['payload']
    assert payload['current'] != payload['previous']
    if sid == 'collaborator_object_handover':
        assert payload['kind'] == 'target_object' and sc.robot_target() == payload['current']
        assert sc.human.holding == payload['current']
    else:
        assert payload['kind'] == 'target_region' and sc.req_target == payload['current']
        placed = [e for e in rec['events'] if e['event_type'] == 'protocol_step_complete'
                  and e['payload']['step'] == 'object_released_in_target']
        assert placed[0]['payload']['target'] == payload['current']
    assert not any(e['event_type'] in ('human_robot_contact', 'safety_distance_violation') for e in rec['events'])


@pytest.mark.parametrize('sid,failure', [('collaborator_object_handover', 'wrong_object_manipulated'),
                                         ('instructor_object_to_target', 'wrong_target_region')])
def test_ignoring_the_change_fails(change_scenarios, sid, failure):
    rec = run_episode(change_scenarios[sid], IgnoreChangeExpert(), 7, keep_trace=False)
    assert not rec['success'] and rec['failure'] == failure


def test_cue_onset_expert_starts_at_the_cue():
    records = {}
    for trigger in ('cue_complete', 'cue_onset'):
        sc = make_scenario('instructor_object_to_target', {'expert': {'trigger': trigger}})
        try:
            records[trigger] = run_episode(sc, ExpertAgent(), 5, keep_trace=False)
        finally:
            sc.close()
    lag = {}
    for trigger, rec in records.items():
        assert rec['success']
        cue = next(e['timestamp'] for e in rec['events'] if e['event_type'] == 'human_cue_onset')
        lag[trigger] = next(e['timestamp'] for e in rec['events'] if e['event_type'] == 'robot_motion_start' and e['timestamp'] >= cue) - cue
    assert lag['cue_onset'] < 0.3 < 2.0 < lag['cue_complete']
    assert records['cue_onset']['metrics']['CT'] < records['cue_complete']['metrics']['CT']


def test_predictive_yield_expert_succeeds_without_violations():
    sc = make_scenario('intruder_pick_place_interruption', {'expert': {'trigger': 'cue_onset', 'yield_prediction_horizon_s': 0.5}})
    try:
        rec = run_episode(sc, ExpertAgent(), 6, keep_trace=False)
    finally:
        sc.close()
    assert rec['success'] and rec['metrics']['DSR'] == 1.0 and rec['metrics']['HCS'] == 1.0


def test_change_episode_reproduces_from_saved_config(change_scenarios):
    rec = run_episode(change_scenarios['collaborator_object_handover'], ExpertAgent(), 9)
    sc = make_scenario(rec['scenario_config'])
    try:
        again = run_episode(sc, ExpertAgent(), rec['seed'], episode_id=rec['episode_id'])
    finally:
        sc.close()
    assert rec['scenario_config']['intention_change']['probability'] == 1.0
    assert compare(rec, again) == []
