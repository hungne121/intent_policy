"""Event stream contract, episode records, serialisation and reproduction from config + seed."""
import pytest

from benchmark.events import EventType, REQUIRED_EVENT_TYPES
from benchmark.logger import EpisodeEventLogger, EventOrderError
from benchmark.runner import ExpertAgent, load_record, run_episode, save_record
from scenarios.scenario_registry import PHASE1_SCENARIOS, make_scenario
from scripts.reproduce_episode import compare
from conftest import assert_monotonic

MINIMUM_EVENTS = {
    'episode_start', 'episode_end', 'human_motion_start', 'human_motion_end', 'human_cue_onset',
    'human_intention_change', 'robot_motion_start', 'robot_motion_end', 'robot_action_change',
    'interaction_window_start', 'interaction_window_end', 'object_grasp', 'object_release',
    'human_robot_contact', 'safety_distance_violation', 'protocol_step_complete', 'disruption_start',
    'disruption_end', 'recovery_start', 'recovery_complete', 'task_success', 'task_failure'}
EXPECTED_PER_SCENARIO = {
    'instructor_object_to_target': {'human_cue_onset', 'object_grasp', 'object_release', 'robot_motion_start',
                                    'robot_action_change', 'protocol_step_complete', 'task_success'},
    'collaborator_object_handover': {'human_cue_onset', 'interaction_window_start', 'interaction_window_end',
                                     'object_grasp', 'object_release', 'task_success'},
    'collaborator_bowl_assistance': {'human_cue_onset', 'interaction_window_start', 'interaction_window_end',
                                     'object_grasp', 'object_release', 'task_success'},
    'intruder_pick_place_interruption': {'disruption_start', 'disruption_end', 'recovery_start',
                                         'recovery_complete', 'object_grasp', 'object_release', 'task_success'},
}


@pytest.fixture(scope='module')
def expert_records(scenarios):
    return {sid: run_episode(scenarios[sid], ExpertAgent(), 21) for sid in PHASE1_SCENARIOS}


def test_event_vocabulary_is_complete():
    assert MINIMUM_EVENTS <= set(REQUIRED_EVENT_TYPES)


def test_logger_contract():
    log = EpisodeEventLogger('s', 'collaborator')
    with pytest.raises(EventOrderError):
        log.log('task_success', 0.0)                     # outside an open episode
    log.start('ep', 0.0)
    log.log(EventType.HUMAN_CUE_ONSET, 1.0, entity_id='human', selected_object='object_a')
    with pytest.raises(ValueError):
        log.log('not_an_event', 1.0)
    with pytest.raises(EventOrderError):
        log.log('robot_motion_start', 0.5)                # time went backwards
    log.log('interaction_window_start', 2.0)
    log.end(3.0, success=False)
    types = [e.event_type for e in log.events]
    assert types[0] == 'episode_start' and types[-1] == 'episode_end'
    assert types.count('interaction_window_start') == types.count('interaction_window_end')
    e = log.events[1]
    assert e.episode_id == 'ep' and e.scenario_id == 's' and e.role == 'collaborator' and e.entity_id == 'human'
    assert e.payload == {'selected_object': 'object_a'}


@pytest.mark.parametrize('sid', PHASE1_SCENARIOS)
def test_required_events_emitted_in_order(expert_records, sid):
    rec = expert_records[sid]
    events = rec['events']
    types = [e['event_type'] for e in events]
    assert types[0] == 'episode_start' and types[-1] == 'episode_end'
    assert types.count('episode_start') == 1 and types.count('episode_end') == 1
    assert EXPECTED_PER_SCENARIO[sid] <= set(types), EXPECTED_PER_SCENARIO[sid] - set(types)
    assert_monotonic(events)
    assert types.count('interaction_window_start') == types.count('interaction_window_end')
    assert types.count('disruption_start') == types.count('disruption_end')
    motion = [t for t in types if t in ('robot_motion_start', 'robot_motion_end')]
    assert all(t == ('robot_motion_start' if i % 2 == 0 else 'robot_motion_end') for i, t in enumerate(motion))
    assert all(e['episode_id'] == rec['episode_id'] and e['scenario_id'] == sid and e['role'] == rec['role']
               for e in events)
    assert [e['seq'] for e in events] == list(range(len(events)))


@pytest.mark.parametrize('sid', PHASE1_SCENARIOS)
def test_episode_record_contains_required_data(expert_records, sid):
    rec = expert_records[sid]
    for k in ('scenario_id', 'role', 'seed', 'variation', 'human_trajectory_variant', 'policy', 'controller',
              'scenario_config', 'events', 'success', 'failure', 'metrics', 'trace'):
        assert k in rec
    step = rec['trace'][5]
    for k in ('t', 'selected_action', 'selected_action_name', 'action_logits', 'action_probabilities',
              'mapped_controller_command', 'state'):
        assert k in step
    for k in ('robot', 'human', 'objects'):
        assert k in step['state']
    cmd = step['mapped_controller_command']
    assert len(cmd['joint_target']) == 7 and 'delta_world' in cmd
    assert set(rec['metrics']) >= {'CSR', 'CT', 'CFR', 'HCS'}


def test_record_serialisation_roundtrip(expert_records, tmp_path):
    rec = expert_records['collaborator_object_handover']
    path = save_record(rec, tmp_path / 'ep.json.gz')
    back = load_record(path)
    assert back['events'] == rec['events'] and back['variation'] == rec['variation']
    assert back['trace'][10]['state']['human']['hand_position'] == rec['trace'][10]['state']['human']['hand_position']


@pytest.mark.parametrize('sid', PHASE1_SCENARIOS)
def test_episode_reproduces_from_saved_config_and_seed(expert_records, tmp_path, sid):
    saved = load_record(save_record(expert_records[sid], tmp_path / 'rec.json.gz'))
    sc = make_scenario(saved['scenario_config'])            # rebuilt from the saved configuration
    try:
        again = run_episode(sc, ExpertAgent(), saved['seed'], episode_id=saved['episode_id'])
    finally:
        sc.close()
    assert compare(saved, again) == []
