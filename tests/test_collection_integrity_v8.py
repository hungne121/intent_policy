from types import SimpleNamespace
import numpy as np
import pytest
from intent_policy.benchmark.collection_integrity import motion_source_hash
from intent_policy.benchmark.paired_collection import collect_pairs, IncompleteQuota
from intent_policy.sim.scripted_human import ScriptedHuman
from intent_policy.scenarios.c3_cloth_bowl import ClothSpreader


def record(crop=0, time=1):
    return dict(success=True, collection_quality={'passed': True}, exogenous_condition_sha256='same',
                human_motion_sources=[dict(t=time, label='carry', source=dict(kind='hymotion', clip_id='x',
                    start_frame=crop, end_frame_exclusive=30, hand='right', split='train'))])


def test_crop_mismatch_rejects_even_with_same_seed_and_clip():
    with pytest.raises(IncompleteQuota):
        collect_pairs([{'pair_id': 1}], 1, lambda mode,c: record(crop=int(mode=='early')),
                      lambda *a: pytest.fail('mismatched crop written'), lambda e: None, require_motion_match=True)


def test_reactive_wait_does_not_change_source_identity():
    assert motion_source_hash(record(time=1)) == motion_source_hash(record(time=5))
    assert motion_source_hash({}) is None


def human():
    h = ScriptedHuman({'human': {'hand_rest': [0,0,1]}}, {'motion_bank': dict(
        paired_source_selection=True, labels={'carry':['x']}, min_dist_m=.01, min_cos=0,
        extent_ratio=[.1,10], holding_margin_m=0, table_margin_m=0, table_edge_y=-1,
        max_dev_ratio=1, robot_clear_m=.15)}, {'seed':1,'human':{'speed_scale':1}})
    records=[SimpleNamespace(path=np.array([[0,0,1],[.1,0,1],[.2,0,1]]),source={'clip_id':str(i)}) for i in range(5)]
    h.bank=SimpleNamespace(pool_records=lambda keys: records)
    h.bank_rng=np.random.default_rng(999)
    return h


def test_robot_pose_and_repeated_yield_probes_cannot_reselect_clip():
    h=human(); target=np.array([.2,0,1.])
    h._bank_path('carry',target); first=h._last_motion_source['clip_id']
    h.last_ctx={'robot_chain':[np.array([0,0,1]),np.array([.2,0,1])]}
    for _ in range(20):
        h._bank_path('carry',target)
        assert h._last_motion_source['clip_id']==first
    h.move(0,target,1,'carry')
    assert h._last_motion_source['clip_id']==first
    assert h._motion_draws['carry']==1


def test_c3_human_clock_does_not_wait_for_bowl_before_deciding():
    h=object.__new__(ClothSpreader)
    h.behavior={'carry_height_m':.1}; h.params={'take_reach_s':1,'carry_s':1,'lower_s':.5,'rest_s':1}
    h.coaster='cloth';h.selected_target='B';h.decoy='A';h.rest=np.zeros(3)
    h.switch={'kind':'after','clock':'human_arrival','delay_s':.5}
    labels=[g.label for g in h.placing()]
    assert 'wait_tray_clear' not in labels[:labels.index('decide')]
    assert 'wait_tray_clear' in labels[labels.index('decide')+1:]


def test_source_priority_does_not_depend_on_pool_order():
    h=human(); target=np.array([.2,0,1.])
    h._bank_path('carry',target); first=h._last_motion_source['clip_id']
    records=h.bank.pool_records(['x'])
    h.bank.pool_records=lambda keys: list(reversed(records))
    h._bank_path('carry',target)
    assert h._last_motion_source['clip_id']==first
