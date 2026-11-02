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
