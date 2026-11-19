"""Integration tests of the event pipeline (11/QĐ-KT integration items 2-4): one real expert episode (C4, N1 val seed
300800, forecast) through the same detector that drives the person and scores the episode."""
import mujoco
import numpy as np
import pytest

from intent_policy.shared.events import CFG, DETECTOR_VERSION, METRIC_SCHEMA_VERSION, SCHEMA, metric_record


@pytest.fixture(scope='module')
def c4_episode():
    from intent_policy.shared.episode import Episode
    from intent_policy.shared.manicast_adapter import load_predictor
    from intent_policy.shared.variation import sample_variation, timed_human
    from intent_policy.experts import expert_class
    from intent_policy.scenarios import scenario
    from tools.collect import task_cfg, CFG as CCFG
    sc, seed = scenario('C4'), 300800
    scene, positions, xml, info = sc.build(seed, 'nominal')
    variation = sample_variation(seed, task_cfg('n1', 'C4'))
    ep = Episode('C4', 'forecast', seed, scene, positions, xml, load_predictor(), out=None, max_ticks=sc.MAX_TICKS,
                 review_camera=CCFG['review_camera'], review_hw=CCFG['review_hw'], variant='nominal', variation=variation)
    expert = expert_class('C4')(ep)
    ep.event('task_active', **{k + '_evaluator_only': v for k, v in info.items() if k != 'variant'})
    ep.variation_log['human'] = {}
    human = timed_human(sc.human(ep, info), variation, ep.variation_log['human'])
    ep.run(human, expert.run(), done=lambda: sc.human_done(ep))
    ok, _ = expert.success()
    s = ep.finish(ok, None, {})
    return ep, s, info, metric_record(ep, ok, ep.t, s)


def test_expert_regression_c4(c4_episode):
    ep, s, info, rec = c4_episode
    assert s['success'] and rec['success']
    cues = [e for e in ep.events if e['event'] == 'human_cue']
    assert [c['socket'] for c in cues if c['cue'] == 'part_on_palm'] == info['order']
    assert [c['socket'] for c in cues if c['cue'] == 'robot_clear'] == info['order']
    assert not [e for e in ep.events if e['event'] == 'human_grasp_timeout']


def test_schema_locked(c4_episode):
    rec = c4_episode[3]
    assert rec['detector_version'] == DETECTOR_VERSION and rec['schema_version'] == METRIC_SCHEMA_VERSION
    assert set(SCHEMA['per_episode']) <= set(rec)
    assert all(set(SCHEMA['rounds']) == set(r) for r in rec['rounds'])
    assert all(set(SCHEMA['anticipation']) == set(r) for r in rec['anticipation'])
    assert set(SCHEMA['safety_audit']) <= set(rec['safety_audit'])
    assert all(set(SCHEMA['violation']) <= set(v) for v in rec['safety_audit']['violations'])


def test_round_timing(c4_episode):
    ep, s, info, rec = c4_episode
    assert [r['round_id'] for r in rec['rounds']] == [0, 1, 2]
    assert [r['step'] for r in rec['rounds']] == [f'part{i}_installed' for i in info['order']]
    done = dict((k, v) for k, v in rec['steps_done'])
    prev_done = 0.
    for r in rec['rounds']:
        assert r['t_trigger'] is not None and r['t_response'] is not None and not r['response_missing']
        assert r['t_response'] >= r['t_trigger']                       # Rsp never negative (11/QĐ-KT C4)
        assert prev_done <= r['t_trigger'] <= done[r['step']]          # the pair belongs to this round
        a = rec['anticipation'][r['round_id']]
        assert a['t_useful_onset'] is not None and a['t_useful_onset'] <= r['t_response']
        prev_done = done[r['step']]
    assert rec['step_predecessors'][rec['steps_required'][2]] == rec['steps_required'][:2]


def test_close_interaction_phase(c4_episode):
    ep, s, info, rec = c4_episode
    saf = rec['safety_audit']
    assert saf['interaction_phase_ticks'] > 0
    assert set(saf['exemptions']) == {f'part{i}' for i in info['order']}
    grasps = {e['object']: e['t'] for e in ep.events if e['event'] == 'human_grasp'}
    for part, ex in saf['exemptions'].items():
        assert ex['first_t'] <= grasps[part] <= ex['last_t']          # the transfer lies inside the phase
        assert ex['last_t'] - ex['first_t'] < 5.                       # a hand-over, not the whole task
    ticks = np.array([[row[0], row[4]] for row in saf['ticks']])     # columns: safety_audit.ticks_columns
    assert 0 < ticks[:, 1].sum() < .5 * len(ticks)


def test_contact_classification_consistent(c4_episode):
    ep, s, info, rec = c4_episode
    bad = sorted(k for k, v in s['contact_classes'].items() if v['category'] == 'invalid_contact')
    assert rec['human_robot_contact'] == bool(bad)
    saf = rec['safety_audit']
    assert saf['unsafe_contacts'] == bad == rec['human_robot_contact_classes']
    assert sorted(v['class'] for v in saf['violations'] if v['kind'] == 'contact') == bad
    assert not any(v['exempt'] for v in saf['violations'] if v['kind'] == 'contact')
    assert saf['min_distance_m'] <= saf['min_nonexempt_m']


def test_no_distance_cutoff():
    """mj_geomDistance with the 0.25 m bound of robot_near_human reports 0.25 for a 0.5 m pair (cut off); the audit
    bound returns the true distance."""
    m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".05" pos="0 0 0"/>'
                                       '<geom type="capsule" size=".03 .1" pos=".58 0 0"/></worldbody></mujoco>')
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    ft = np.zeros(6)
    assert mujoco.mj_geomDistance(m, d, 0, 1, .25, ft) == pytest.approx(.25)
    assert mujoco.mj_geomDistance(m, d, 0, 1, CFG['metric']['safety']['distmax_m'], ft) == pytest.approx(.5, abs=1e-6)
