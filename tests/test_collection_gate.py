from types import SimpleNamespace
import mujoco
import numpy as np
import pytest
from intent_policy.sim.collection_gate import CollectionGate


def scene():
    m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body name="item"><freejoint/><geom name="item_geom" type="sphere" size=".05"/></body>
    <body pos=".3 0 0"><geom name="other_geom" type="sphere" size=".05"/></body>
    </worldbody></mujoco>''')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    sc = SimpleNamespace(env=SimpleNamespace(model=m, data=d, physics_observers=[], pre_physics_observers=[]),
        human=SimpleNamespace(stage='rest'), pos=lambda k: d.qpos[:3], quat=lambda k: d.qpos[3:7])
    ag = SimpleNamespace(expert=SimpleNamespace(stage='approach'))
    cfg = dict(scene_coverage_reviewed=True,
        pairs=[dict(geom_a='item_geom', geom_b='other_geom', clearance_m=.01, max_contact_penetration_m=.001)],
        protected_objects=['item'], protected_translation_m=.001, protected_rotation_rad=.01,
        require_hymotion_labels=['reach'])
    return sc, ag, cfg


def source(kind='hymotion'):
    return [dict(event_type='human_motion_start', timestamp=0,
                 payload=dict(label='reach', motion_source=dict(kind=kind, clip_id='source_a')))]


def test_gate_rejects_displacement_and_detaches_observer():
    sc, ag, cfg = scene()
    gate = CollectionGate(sc, ag, cfg)
    sc.env.data.qpos[1] += .002
    mujoco.mj_forward(sc.env.model, sc.env.data)
    gate.observe(sc.env.model, sc.env.data)
    r = gate.finish(source())
    assert not r['passed']
    assert r['movement_violation']['object'] == 'item'
    assert sc.env.physics_observers == []
    assert sc.env.pre_physics_observers == []


@pytest.mark.parametrize('events', [[], source('minimum_jerk')])
def test_required_hymotion_cannot_silently_fallback(events):
    sc, ag, cfg = scene()
    r = CollectionGate(sc, ag, cfg).finish(events)
    assert not r['passed'] and r['missing_hymotion']


def test_unreviewed_scene_cannot_collect():
    sc, ag, cfg = scene()
    cfg['scene_coverage_reviewed'] = False
    with pytest.raises(ValueError, match='reviewed'):
        CollectionGate(sc, ag, cfg)


def test_valid_sample_passes_only_configured_checks():
    sc, ag, cfg = scene()
    r = CollectionGate(sc, ag, cfg).finish(source())
    assert r['passed']


def table_scene(finger_z=0.10, item_z=0.10, held=False, stage='approach', rules=(), table_pairs=True):
    """A table, a moving robot finger (slide joint) and an object (free joint) at the given heights above the table
    top (z = 0): penetration when the finger's half height 0.01 / the item's 0.02 exceed it (PIF §8.13 03/11)."""
    m = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
    <geom name="table" type="box" size=".5 .5 .02" pos="0 0 -.02"/>
    <body name="base_link" pos="-.3 0 .05"><geom name="base_geom" type="box" size=".05 .05 .05"/>
      <body name="sus2f_finger_l_link" pos=".1 0 {finger_z - .05}"><joint type="slide" axis="0 0 1"/>
        <geom name="finger_geom" type="box" size=".01 .01 .01"/></body></body>
    <body name="item" pos=".2 0 {item_z}"><freejoint/><geom name="item_geom" type="box" size=".02 .02 .02"/></body>
    </worldbody></mujoco>''')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    gid = lambda n: m.geom(n).id
    sc = SimpleNamespace(env=SimpleNamespace(model=m, data=d, physics_observers=[], pre_physics_observers=[]),
        human=SimpleNamespace(stage='rest', holding=None), pos=lambda k: d.qpos[1:4], quat=lambda k: d.qpos[4:8],
        robot_geoms=[gid('base_geom'), gid('finger_geom')], on_table=['item'], object_geom_ids={'item': [gid('item_geom')]},
        scene={}, grasp_state={'item': held}, speed=lambda k: 0.0, radius={'item': .02})
    ag = SimpleNamespace(expert=SimpleNamespace(stage=stage))
    cfg = dict(scene_coverage_reviewed=True, pairs='all_scene_pairs', pair_clearance_m=0.0, pair_max_penetration_m=.001,
               table_pairs=table_pairs, contact_rules=[dict(r) for r in rules], protected_objects=[],
               protected_translation_m=.005, protected_rotation_rad=.05, require_hymotion_labels=[])
    return sc, ag, cfg


def table_result(**kw):
    sc, ag, cfg = table_scene(**kw)
    gate = CollectionGate(sc, ag, cfg)
    gate.observe(sc.env.model, sc.env.data)
    r = gate.finish([])
    v = r['geometry']['first_violation']
    names = v and {sc.env.model.geom(v['geom_a']).name, sc.env.model.geom(v['geom_b']).name}
    return r['passed'], names


SUPPORT = [dict(kind='table_support', max_penetration_m=.001, robot_stages=['lower_home'])]


def test_robot_into_table_is_rejected_and_not_without_table_pairs():
    assert table_result(finger_z=.008) == (False, {'finger_geom', 'table'})      # 2 mm into the table
    assert table_result(finger_z=.008, table_pairs=False)[0]                     # the old coverage missed it
    assert table_result(finger_z=.012)[0]                                         # 2 mm above: clear
    assert table_result(finger_z=.008, rules=SUPPORT, stage='lower_home') == (False, {'finger_geom', 'table'})


def test_mounted_base_on_table_is_not_a_pair():
    sc, ag, cfg = table_scene()
    gate = CollectionGate(sc, ag, cfg)
    pairs = {frozenset((sc.env.model.geom(a).name, sc.env.model.geom(b).name)) for a, b in gate.audit.pairs}
    assert frozenset(('base_geom', 'table')) not in pairs
    assert frozenset(('finger_geom', 'table')) in pairs and frozenset(('item_geom', 'table')) in pairs


def test_object_on_table_needs_explicit_rule():
    resting = dict(item_z=.0195)                                                  # 0.5 mm into the table
    assert table_result(**resting) == (False, {'item_geom', 'table'})            # no implicit whitelist
    assert table_result(**resting, rules=SUPPORT)[0]                              # resting, explicit rule
    assert table_result(**resting, rules=SUPPORT, held=True) == (False, {'item_geom', 'table'})   # robot-held
    assert table_result(**resting, rules=SUPPORT, held=True, stage='lower_home')[0]               # put down
    assert not table_result(item_z=.017, rules=SUPPORT, stage='lower_home', held=True)[0]         # 3 mm > bound


def test_stage_window_permission_expires():
    """An allowed_contacts entry with robot_stage_window_s holds only at the start of the stage (W3 lift_wipe)."""
    sc, ag, cfg = table_scene(finger_z=.0095, rules=SUPPORT, stage='lift_wipe')  # finger 0.5 mm into the table
    cfg['allowed_contacts'] = [dict(geom_a='finger_geom', geom_b='table', robot_stages=['lift_wipe'],
                                    robot_stage_window_s=.05)]
    d = sc.env.data
    d.time = 1.0
    gate = CollectionGate(sc, ag, cfg)                                           # stage entered at t 1.0
    d.time = 1.04
    gate.observe(sc.env.model, d)
    assert gate.audit.first_violation is None
    d.time = 1.06
    gate.observe(sc.env.model, d)
    assert gate.audit.first_violation is not None


def test_all_violations_reported_not_only_the_first():
    # finger 2 mm into the table and the unheld item 2 mm into it with no table_support rule: two violating pairs,
    # both kept (PIF §8.13 decision 04/11 08:34); the verdict is the same as with the first one only
    sc, ag, cfg = table_scene(finger_z=.008, item_z=.018)
    cfg['stop_on_rejection'] = False
    gate = CollectionGate(sc, ag, cfg)
    ag.expert.stage = 'close'
    gate.observe(sc.env.model, sc.env.data)
    r = gate.finish([])
    assert not r['passed']
    seen = {(v['body_a'], v['geom_b'], v['robot_stage']) for v in r['geometry']['all_violations']}
    assert ('sus2f_finger_l_link', 'table', 'approach') in seen and ('sus2f_finger_l_link', 'table', 'close') in seen
    assert ('item', 'table', 'close') in seen
    assert all(v['min_distance_m'] < -.0005 for v in r['geometry']['all_violations'])   # solver contact depth


def test_human_manipulation_robot_stages_separate_receiving_from_before():
    # the hand on the object in a permitted human stage: allowed only in the rule's robot_stages (C1 receiving phase,
    # PIF §8.13 decision 04/11 08:34), a collision while the robot is still above the hand
    rule = dict(kind='human_manipulation', max_penetration_m=.001, objects=['CAN'], stages=['reach_out'],
                robot_stages=['to_handover'])
    gate = SimpleNamespace(sc=SimpleNamespace(human=SimpleNamespace(stage='reach_out', holding=None, motion_key=None),
                                              env=SimpleNamespace(data=SimpleNamespace(time=1.0))),
                           human_hand={1}, _human_released_at={}, release_grace_s=0.0)
    rule['_stages'] = frozenset(rule['stages'])
    for stage, ok in (('to_handover', True), ('over_hand', False)):
        gate.agent = SimpleNamespace(expert=SimpleNamespace(stage=stage))
        gate._expert_stage = lambda: gate.agent.expert.stage
        assert CollectionGate._rule_allows(gate, rule, 1, 2, None, 'CAN') is ok
    del rule['robot_stages']
    assert CollectionGate._rule_allows(gate, rule, 1, 2, None, 'CAN')
