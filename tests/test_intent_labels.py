"""Intent labels (INTENT_ACT_GUIDE.md M1): per-timestep labels, phase/tte, xi waypoints, simulator segments and the
labelled LeRobot dataset."""
import json

import numpy as np
import pytest
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from intent_policy.intent.labels import (event_frame, future_waypoints, load_schema, location_of, motions_from_events,
                                         per_timestep_labels, phase_tte, segments_from_sim, waypoint_offsets)
from scripts.build_intent_labels import INTENT_FEATURES, build_labels

SCHEMA = load_schema()
SEG = dict(t_onset=10, t_clear=16, t_event=20, obj='S3', act='point_command')


def test_schema_is_consistent():
    assert SCHEMA['obj_vocab'][0] == SCHEMA['act_vocab'][0] == 'none'
    assert len(set(SCHEMA['obj_vocab'])) == len(SCHEMA['obj_vocab']) and len(set(SCHEMA['act_vocab'])) == len(SCHEMA['act_vocab'])
    for label, rule in SCHEMA['sim_segments'].items():
        assert rule['act'] in SCHEMA['act_vocab'], label
    assert set(SCHEMA['neg_target_obj'].values()) <= set(SCHEMA['obj_vocab'])


def test_labels_are_none_outside_the_segment():
    obj, act = per_timestep_labels(40, [SEG], SCHEMA['obj_vocab'], SCHEMA['act_vocab'])
    inside = np.zeros(40, bool)
    inside[10:21] = True
    assert (obj[~inside] == 0).all() and (act[~inside] == 0).all()
    assert (obj[inside] == SCHEMA['obj_vocab'].index('S3')).all()
    assert (act[inside] == SCHEMA['act_vocab'].index('point_command')).all()
    obj, _ = per_timestep_labels(40, [dict(SEG, t_end=30)], SCHEMA['obj_vocab'], SCHEMA['act_vocab'])
    assert (obj[10:31] > 0).all() and (obj[31:] == 0).all() and (obj[:10] == 0).all()


def test_phase_is_monotonic_and_one_at_the_event():
    phase, tte = phase_tte(40, [SEG], 20, 3.0)
    assert (np.diff(phase) >= 0).all()
    assert phase[SEG['t_event']] == 1.0 and (phase[:SEG['t_onset'] + 1] == 0).all()
    assert (phase[SEG['t_event']:] == 1).all() and (tte[SEG['t_event']:] == 0).all()     # after the event
    assert (tte[:SEG['t_onset']] == 3.0).all()                                             # before any segment
    assert np.isclose(tte[SEG['t_onset']], (SEG['t_event'] - SEG['t_onset']) / 20)
    second = dict(t_onset=30, t_clear=33, t_event=35, obj='P1', act='point_command')
    phase, tte = phase_tte(40, [SEG, second], 20, 3.0)
    assert phase[29] == 1.0 and phase[30] == 0.0 and phase[35] == 1.0 and np.isclose(tte[30], 0.25)


def test_xi_waypoints_stay_in_range_at_the_end():
    T, J = 30, 4
    kp = np.cumsum(np.ones((T, J, 3), np.float32) * 0.01, axis=0)            # constant velocity 0.2 m/s
    xi = future_waypoints(kp, 20, 1.2, 8)
    assert xi.shape == (T, 8, J, 3) and np.isfinite(xi).all()
    k = waypoint_offsets(20, 1.2, 8)
    assert k.tolist() == [3, 6, 9, 12, 15, 18, 21, 24]
    assert np.allclose(xi[0, :, 0, 0], 0.01 * k, atol=1e-5)                   # linear motion survives smoothing
    assert np.allclose(xi[-1], 0.0)                                            # clipped to T-1: no motion left
    assert np.allclose(xi[T - 4, 0], xi[T - 4, -1])                            # every waypoint beyond the end
    assert future_waypoints(kp[:3], 20, 1.2, 8).shape == (3, 8, J, 3)         # shorter than the filter window


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
    assert location_of(None, spec, SCHEMA) == 'none'


def test_sim_segments_merge_and_hold():
    stages = [dict(name='idle', start=0, end=9), dict(name='intrude_approach', start=10, end=19),
              dict(name='intrude', start=20, end=29), dict(name='dwelling', start=30, end=49),
              dict(name='withdraw', start=50, end=59)]
    motions = [dict(label='intrude_approach', key='robot_zone', start=10, end=20, stopped=False),
               dict(label='intrude', key='robot_zone', start=20, end=28, stopped=True),
               dict(label='withdraw', key=None, start=50, end=60, stopped=False)]
    segs = segments_from_sim(motions, stages, {'layout': {}}, SCHEMA, 60)
    assert segs == [dict(t_onset=10, t_clear=28, t_event=29, t_end=49, obj='robot_zone', act='reach_into',
                         source='intrude_approach+intrude')]


# ---------------------------------------------------------------------- recorded episodes (conftest.labelled_dataset)
def test_sim_segments_of_recorded_episodes(labelled_dataset):
    root, segments, labels, meta = labelled_dataset
    t1, t2 = meta[0]['spec'], meta[1]['spec']
    assert [(s['act'], s['obj']) for s in segments[0]] == [('point_command', t1['layout'][t1['target']]),
                                                           ('point_command', t1['place'])]
    assert [(s['act'], s['obj']) for s in segments[1]] == [('point_command', 'S5'), ('receive_from_robot', 'H1')]
    for segs in segments.values():
        for s in segs:
            assert s['t_onset'] < s['t_clear'] <= s['t_event'] <= s['t_end']
    point, receive = segments[1]
    assert receive['t_end'] > receive['t_event']                    # the hand stays out until the object is taken
    lab = labels[1]
    assert lab['act'][receive['t_end']] == SCHEMA['act_vocab'].index('receive_from_robot')
    assert lab['act'][receive['t_end'] + 1] == 0


def test_dataset_carries_intent_with_the_right_shapes(labelled_dataset):
    root, segments, labels, _ = labelled_dataset
    info = json.loads((root / 'meta/info.json').read_text())
    assert set(INTENT_FEATURES) <= set(info['features'])
    ds = LeRobotDataset('local/intent', root=root)
    M, J = SCHEMA['n_waypoints'], len(SCHEMA['keypoints'])
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)))
    assert batch['intent.obj'].shape == (4,) and batch['intent.obj'].dtype == torch.int64
    assert batch['intent.act'].shape == (4,) and batch['intent.phase'].shape == (4,) and batch['intent.tte'].shape == (4,)
    assert batch['intent.xi'].shape == (4, M, J, 3) and batch['intent.xi'].dtype == torch.float32
    assert batch['human.keypoints'].shape == (4, J, 3)
    assert batch['observation.state'].shape == (4, 10)                 # the rest of the dataset is untouched
    item = ds[len(ds) - 1]                                             # last frame of the last episode
    assert torch.allclose(item['intent.xi'], torch.zeros(M, J, 3), atol=1e-6)
    i = int(ds.meta.episodes[1]['dataset_from_index']) + 30
    assert int(ds[i]['intent.act']) == int(labels[1]['act'][30])
    assert np.allclose(ds[i]['intent.xi'].numpy(), labels[1]['xi'][30], atol=1e-6)
    with pytest.raises(SystemExit):
        build_labels(root)                                             # refuses to overwrite silently
    build_labels(root, overwrite=True)
