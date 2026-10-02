"""Intent contract and sources (INTENT_ACT_GUIDE_v2.md M1-M2): schema, simulator segments, validate_intent,
hindsight / perfect / predicted outputs on recorded episodes, causality, and the labelled LeRobot dataset."""
import json

import numpy as np
import pytest
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from intent_policy.intent.contract import FIELDS, field_shapes, one_hot, validate_intent
from intent_policy.intent.labels import (event_frame, future_waypoints, load_schema, location_of, motions_from_events,
                                         segments_from_sim, waypoint_offsets)
from intent_policy.intent.perception import PerfectPerceptionSource
from intent_policy.intent.sources import HindsightSource, HumanObs
from scripts.build_intent_labels import TASK_ID, run_source

SCHEMA = load_schema()


def test_schema_is_consistent():
    s = SCHEMA
    assert s['targets'][0] == s['who'][0] == 'none' and s['gestures'][0] == 'rest' and s['phases'][0] == 'rest'
    assert len(s['targets']) == 17 and len(s['who']) == 4 and len(s['phases']) == 5 and len(s['keypoints']) == 6
    kinds = [k for ks in s['target_kinds'].values() for k in ks]
    assert sorted(kinds) == sorted(s['targets'][1:])                      # every place has exactly one kind
    for label, rule in s['sim_segments'].items():
        assert rule['gesture'] in s['gestures'] and rule['who'] in s['who'], label
    assert set(s['task_of_scenario'].values()) == set(s['tasks']) == {'T1', 'T2', 'T3', 'T4'}


def test_xi_waypoints_stay_in_range_at_the_end():
    T, J = 30, 6
    kp = np.cumsum(np.ones((T, J, 3), np.float32) * 0.01, axis=0)            # constant velocity 0.2 m/s
    xi = future_waypoints(kp, 20, 1.2, 8)
    k = waypoint_offsets(20, 1.2, 8)
    assert xi.shape == (T, 8, J, 3) and k.tolist() == [3, 6, 9, 12, 15, 18, 21, 24]
    assert np.allclose(xi[0, :, 0, 0], 0.01 * k, atol=1e-5) and np.allclose(xi[-1], 0.0)
    assert future_waypoints(kp[:3], 20, 1.2, 8).shape == (3, 8, J, 3)


def test_motions_pair_starts_with_ends_and_replacements():
    ev = lambda t, kind, **p: dict(timestamp=t, event_type=kind, payload=p)
    events = [ev(1.0, 'human_motion_start', label='point_object', target_key='B1'),
              ev(2.0, 'human_motion_end', label='point_object'),
              ev(3.0, 'human_motion_start', label='intrude', target_key='robot_zone'),
              ev(3.5, 'human_motion_start', label='withdraw', target_key=None),
              ev(4.0, 'human_motion_end', label='withdraw', stopped_for_clearance=True)]
    m = motions_from_events(events, 20)
    assert [(x['label'], x['start'], x['end']) for x in m] == [('point_object', 21, 41), ('intrude', 61, 71), ('withdraw', 71, 81)]
    assert m[2]['stopped'] and event_frame(1.0, 20) == 21


def test_location_of_maps_keys_to_places():
    spec = dict(layout={'B1': 'S2', 'C3': 'S5'}, u_blocks=['B2', 'B1'])
    assert location_of('B1', spec, SCHEMA) == 'S2' and location_of('C3', spec, SCHEMA) == 'S5'
    assert location_of('B2', dict(spec, layout={}), SCHEMA) == 'U1'
    assert location_of('P1', spec, SCHEMA) == 'P1' and location_of('edge', spec, SCHEMA) == 'zone_edge'


def test_sim_segments_merge_hold_and_retract():
    stages = [dict(name='idle', start=0, end=9), dict(name='intrude_approach', start=10, end=19),
              dict(name='intrude', start=20, end=29), dict(name='dwelling', start=30, end=49),
              dict(name='withdraw', start=50, end=59)]
    motions = [dict(label='intrude_approach', key='robot_zone', start=10, end=20, stopped=False),
               dict(label='intrude', key='robot_zone', start=20, end=28, stopped=True),
               dict(label='withdraw', key=None, start=50, end=60, stopped=False)]
    segs = segments_from_sim(motions, stages, {'layout': {}}, SCHEMA, 60)
    assert segs == [dict(t_onset=10, t_clear=28, t_event=29, t_end=49, target='robot_zone', gesture='reach', who='human',
                         source='intrude_approach+intrude', t_evident=17, t_retract_start=50, t_retract=59)]


def frame(**over):
    K = len(SCHEMA['targets'])
    d = {k: np.zeros(v, np.float32) for k, v in field_shapes(SCHEMA).items()}
    d.update(p_who=one_hot(0, 4), p_target=one_hot(0, K), c_who=one_hot(0, 4), c_target=one_hot(0, K),
             phase=one_hot(0, 5), confidence=np.float32([1, 1]), tte=np.float32([3.0]))
    d.update(over)
    return d


def test_validate_intent():
    validate_intent(frame(), SCHEMA)
    validate_intent({k: v[None].repeat(4, 0) for k, v in frame().items()}, SCHEMA, batched=True)
    for bad in (dict(p_target=np.zeros(17, np.float32)),                   # all-zero distribution
                dict(p_who=np.float32([0.5, 0.6, 0, 0])), dict(phase=np.ones(4, np.float32) / 4),
                dict(tte=np.float32([3.5])), dict(xi=np.full((8, 6, 3), np.nan, np.float32)),
                dict(occupancy=np.float32([0, 0, 0, 0.1, 0.1, 0.1, 1.5])), dict(c_target=np.zeros(17))):
        with pytest.raises(ValueError):
            validate_intent(frame(**bad), SCHEMA)


# ---------------------------------------------------------------------- recorded episodes (conftest.labelled_dataset)
def test_sim_segments_of_recorded_episodes(labelled_dataset):
    root, res, meta = labelled_dataset
    t1 = meta[0]['spec']
    segs = res['segments']
    assert [(s['gesture'], s['target'], s['who']) for s in segs[0]] == [('point', t1['layout'][t1['target']], 'robot'),
                                                                        ('point', t1['place'], 'robot')]
    assert [(s['gesture'], s['target'], s['who']) for s in segs[1]] == [('point', 'S5', 'robot'), ('palm_up', 'H1', 'joint')]
    for ss in segs.values():
        for s in ss:
            assert s['t_onset'] <= s['t_evident'] <= s['t_clear'] <= s['t_event'] <= s['t_end']


def test_hindsight_memory_and_phases_on_t1(labelled_dataset):
    root, res, meta = labelled_dataset
    hs, seg = res['outputs']['intent_hs'][0], res['segments'][0]
    targets, phases = SCHEMA['targets'], SCHEMA['phases']
    c = [targets[i] for i in hs['c_target'].argmax(1)]
    p = [targets[i] for i in hs['p_target'].argmax(1)]
    obj, place = seg[0]['target'], seg[1]['target']
    after = seg[1]['t_end'] + 5                                             # pointing over, the robot works from memory
    assert p[after] == 'none' and c[after] in (obj, place)
    assert c[seg[0]['t_onset'] + SCHEMA['tracker']['hold_frames']] == obj
    assert c[-1] == 'none'                                                  # placed: nothing left to do
    ph = [phases[i] for i in hs['phase'].argmax(1)]
    s0 = seg[0]
    assert ph[s0['t_onset']] == 'prepare' and ph[s0['t_evident']] == 'stroke' and ph[s0['t_clear']] == 'hold'
    assert ph[0] == 'rest'
    assert np.isclose(hs['tte'][s0['t_onset'], 0], (s0['t_clear'] - s0['t_onset']) / 20) and hs['tte'][s0['t_clear'], 0] == 0


def test_every_source_follows_the_contract(labelled_dataset):
    root, res, _ = labelled_dataset
    assert set(res['outputs']) == {'intent_hs', 'intent_pp', 'intent_pr0', 'intent_pr1', 'intent_pr2'}
    for prefix, per_ep in res['outputs'].items():
        for out in per_ep.values():
            validate_intent(out, SCHEMA, batched=True)


def test_perfect_source_is_causal(labelled_dataset):
    """Changing the future keypoints never changes earlier outputs (hindsight does change: it knows the future)."""
    root, res, _ = labelled_dataset
    ep, robots, segs = res['episodes'][1], res['robots'][1], res['segments'][1]
    t0 = segs[0]['t_onset'] + 3
    junk = ep['keypoints'].copy()
    junk[t0:] += np.random.default_rng(0).normal(0, 0.1, junk[t0:].shape).astype(np.float32)
    meta = dict(segments=segs, keypoints=ep['keypoints'])
    a = run_source(PerfectPerceptionSource(SCHEMA), meta, ep, robots)
    b = run_source(PerfectPerceptionSource(SCHEMA), dict(meta, keypoints=junk), dict(ep, keypoints=junk), robots)
    for k in FIELDS:
        assert np.array_equal(a[k][:t0], b[k][:t0]), k
    h1 = run_source(HindsightSource(SCHEMA), meta, ep, robots)
    h2 = run_source(HindsightSource(SCHEMA), dict(meta, keypoints=junk), dict(ep, keypoints=junk), robots)
    assert not np.array_equal(h1['xi'][:t0], h2['xi'][:t0])


def test_perfect_source_finds_the_targets(labelled_dataset):
    root, res, _ = labelled_dataset
    for e, segs in res['segments'].items():
        pp = res['outputs']['intent_pp'][e]
        for s in segs:
            t = s['t_event']
            assert SCHEMA['targets'][int(pp['p_target'][t].argmax())] == s['target'], (e, s)
            assert SCHEMA['who'][int(pp['p_who'][t].argmax())] == s['who'], (e, s)


def test_dataset_carries_the_contract(labelled_dataset):
    root, res, _ = labelled_dataset
    ds = LeRobotDataset('local/intent', root=root)
    K, M, J = len(SCHEMA['targets']), SCHEMA['n_waypoints'], len(SCHEMA['keypoints'])
    b = next(iter(torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)))
    assert b['intent_hs.p_target'].shape == (4, K) and b['intent_pp.p_who'].shape == (4, 4)
    assert b['intent_pr0.xi'].shape == (4, M, J, 3) and b['intent_hs.tte'].shape in ((4,), (4, 1))
    assert b[TASK_ID].shape == (4,) and set(b[TASK_ID].tolist()) <= {0, 1}
    assert b['observation.state'].shape == (4, 10)
    info = json.loads((root / 'meta/intent_schema.json').read_text())
    assert info['sources'] == list(res['outputs'])
