import numpy as np
from intent_policy.scenarios.scenario_registry import make_scenario
from intent_policy.benchmark.runner import run_episode, ExpertAgent
from intent_policy.sim.restricted_action import RestrictedActionConfig


def test_optional_withdrawal_raise_is_before_return_to_rest():
    sc = make_scenario('i1_coaster_sequence', {'human_behavior': {
        'withdraw_raise_height_m': .25, 'withdraw_raise_s': .5}})
    try:
        sc.reset(300000)
        gestures = sc.human.placing()
        labels = [g.label for g in gestures]
        assert labels.index('withdraw_raise') < labels.index('withdraw_clear_sources') < labels.index('withdraw')
    finally:
        sc.close()


def test_sequence_samples_separated_continuous_destinations_reproducibly():
    sc = make_scenario('i1_coaster_sequence')
    try:
        sc.reset(300000)
        first = sc.variation['coaster_sequence']
        assert len(sc.on_table) == 6
        sc.reset(300000)
        assert sc.variation['coaster_sequence'] == first
        p = np.array([r['destination_xy'] for r in first])
        distances = np.linalg.norm(p[:, None] - p[None], axis=2)
        assert np.min(distances[np.triu_indices(3, 1)]) >= .115
        sc.reset(300001)
        assert sc.variation['coaster_sequence'] != first
    finally:
        sc.close()


def test_three_cups_complete_without_scene_reset():
    sc = make_scenario('i1_coaster_sequence')
    agent = ExpertAgent()
    try:
        rec = run_episode(sc, agent, 300000,
            RestrictedActionConfig.load('configs/controller/continuous_motion.yaml'), keep_trace=False)
        assert rec['success'], rec['failure']
        assert sc.completed == ['C1', 'C2', 'C3']
        assert sc.human.stage == 'done'
        assert len([e for e in rec['events'] if e['event_type'] == 'episode_start']) == 1
        for item in sc.rounds:
            np.testing.assert_allclose(sc.pos(item['cup'])[:2],sc.pos(item['coaster'])[:2],atol=.02)
    finally:
        sc.close()


def test_human_places_all_coasters_without_waiting_for_robot():
    """The human's schedule (incl. the sampled pauses) never waits for the robot's rounds: an idle robot still sees
    all three coasters placed."""
    sc = make_scenario('i1_coaster_sequence')
    try:
        sc.reset(300000)
        for _ in range(int(40 / sc.dt)):
            if sc.step(sc.env.hold()):
                break
        assert all('coaster_placed_' + r['coaster'] in sc.protocol_done for r in sc.rounds)
        assert sc.human.stage == 'done'
    finally:
        sc.close()
