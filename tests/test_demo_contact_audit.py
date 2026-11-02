import mujoco
import pytest
from intent_policy.sim.demo_contact_audit import DemoContactAudit, PairRule


def scene(mask=1):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="moving"><freejoint/><geom name="a" type="sphere" size=".05"
        contype="{mask}" conaffinity="{mask}"/></body>
      <body pos=".15 0 0"><geom name="b" type="sphere" size=".05"
        contype="{mask}" conaffinity="{mask}"/></body>
      </worldbody></mujoco>''')
    return model, mujoco.MjData(model)


def pose(model, data, x):
    data.qpos[0] = x
    mujoco.mj_forward(model, data)


def test_transient_penetration_is_retained_after_separation():
    m, d = scene()
    audit = DemoContactAudit(m, [PairRule('a', 'b', .01, .001)])
    for x in (0, .07, 0):
        pose(m, d, x)
        audit.observe(m, d)
    assert audit.report()['samples'] == 3
    assert audit.first_violation['kind'] == 'penetration'
    assert not audit.report()['passed_configured_pairs']


def test_permitted_contact_does_not_allow_deep_penetration():
    m, d = scene()
    audit = DemoContactAudit(m, [PairRule('a', 'b', .01, .001)], lambda *args: True)
    pose(m, d, .0505)
    audit.observe(m, d)
    assert audit.report()['passed_configured_pairs']
    pose(m, d, .06)
    audit.observe(m, d)
    assert audit.first_violation['contact_permitted']
    assert not audit.report()['passed_configured_pairs']


def test_disabled_masks_rejected_but_distance_can_still_detect_visual_overlap():
    m, d = scene(mask=0)
    with pytest.raises(ValueError, match='masks disabled'):
        DemoContactAudit(m, [PairRule('a', 'b', 0, 0, True)])
    audit = DemoContactAudit(m, [PairRule('a', 'b', 0, 0)])
    pose(m, d, .07)
    audit.observe(m, d)
    assert audit.first_violation['kind'] == 'penetration'


def test_no_samples_is_not_a_pass():
    m, _ = scene()
    audit = DemoContactAudit(m, [PairRule('a', 'b', .01, .001)])
    assert not audit.report()['passed_configured_pairs']


def test_physical_pair_with_world_is_valid():
    m, d = scene()
    audit = DemoContactAudit(m, [PairRule('a', 'b', 0, .001, True)])
    pose(m, d, 0)
    audit.observe(m, d)
    assert audit.report()['passed_configured_pairs']


def test_environment_notifies_every_physics_substep():
    from intent_policy.sim.base_env import ManipulationEnv
    with ManipulationEnv() as env:
        times = []
        env.physics_observers.append(lambda m, d: times.append(float(d.time)))
        env.step(env.hold())
        assert len(times) == env.substeps
        assert all(b > a for a, b in zip(times, times[1:]))


def test_nonfinite_distance_is_rejected(monkeypatch):
    m, d = scene()
    pose(m, d, .07)   # within the broad-phase bound, so the exact distance is evaluated
    audit = DemoContactAudit(m, [PairRule('a', 'b', 0, 0)])
    monkeypatch.setattr(mujoco, 'mj_geomDistance', lambda *args: float('nan'))
    audit.observe(m, d)
    assert audit.first_violation['kind'] == 'nonfinite_distance'
    assert not audit.report()['passed_configured_pairs']


def test_mocap_nonfinite_is_rejected():
    m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body mocap="true"><geom name="a" type="sphere" size=".05"/></body>
      <geom name="b" type="sphere" size=".05" pos="1 0 0"/>
      </worldbody></mujoco>''')
    d = mujoco.MjData(m)
    d.mocap_pos[0, 0] = float('nan')
    audit = DemoContactAudit(m, [PairRule('a', 'b', 0, 0)])
    audit.observe(m, d)
    assert audit.first_violation['kind'] == 'nonfinite_state'


def test_prephysics_observer_runs_before_integration_and_geoms_are_current():
    import numpy as np
    from intent_policy.sim.base_env import ManipulationEnv
    with ManipulationEnv() as env:
        order = []
        initial_time = float(env.data.time)
        def verify(m, d):
            fresh = mujoco.MjData(m)
            fresh.qpos[:] = d.qpos
            fresh.mocap_pos[:] = d.mocap_pos
            fresh.mocap_quat[:] = d.mocap_quat
            mujoco.mj_kinematics(m, fresh)
            np.testing.assert_allclose(d.geom_xpos, fresh.geom_xpos, atol=1e-12)
            order.append(('post', float(d.time)))
        env.pre_physics_observers.append(lambda m, d: order.append(('pre', float(d.time))))
        env.physics_observers.append(verify)
        target = env.hold().copy()
        target[0] += .05
        env.step(target)
        assert order[0] == ('pre', initial_time)
        assert len(order) == env.substeps + 1
        assert all(t > initial_time for _, t in order[1:])


def test_broad_phase_skips_only_provably_distant_pairs():
    m, d = scene()
    audit = DemoContactAudit(m, [PairRule('a', 'b', .01, .001)])
    pose(m, d, 0)          # 5 cm gap > limit 2 cm: culled, recorded at the cap, still counted as checked
    audit.observe(m, d)
    rep = audit.report()
    assert rep['passed_configured_pairs'] and rep['pairs'][0]['checked_samples'] == 1
    assert rep['pairs'][0]['minimum_distance_m'] == 0.02
    pose(m, d, .045)       # 5 mm gap < clearance 1 cm: must be measured and flagged
    audit.observe(m, d)
    assert audit.first_violation['kind'] == 'clearance'


def test_nonfinite_geometry_is_rejected():
    m, d = scene()
    pose(m, d, 0)
    d.geom_xpos[0, 0] = float('nan')
    audit = DemoContactAudit(m, [PairRule('a', 'b', 0, 0)])
    audit.observe(m, d)
    assert audit.first_violation['kind'] == 'nonfinite_geometry'


def test_sdf_of_primitives_and_surface_samples():
    import numpy as np
    from intent_policy.sim.demo_contact_audit import primitive_sdf, surface_samples
    p = np.array([[0, 0, 0], [0, 0, .2], [.3, 0, 0]], float)
    np.testing.assert_allclose(primitive_sdf(mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([.1, .05, 0]), p), [-.05, .15, .2])
    np.testing.assert_allclose(primitive_sdf(mujoco.mjtGeom.mjGEOM_BOX, np.array([.1, .1, .1]), p), [-.1, .1, .2])
    np.testing.assert_allclose(primitive_sdf(mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.1, 0, 0]), p), [-.1, .1, .2])
    np.testing.assert_allclose(primitive_sdf(mujoco.mjtGeom.mjGEOM_CAPSULE, np.array([.1, .05, 0]), p), [-.1, .05, .2])
    s = surface_samples(np.array([[0, 0, 0], [.01, 0, 0], [0, .01, 0]], float), np.array([[0, 1, 2]]), .0025)
    assert len(s) == 28 and s[:, 2].max() == 0     # hypotenuse 14.1 mm -> 6 subdivisions: 7*8/2 points, on the triangle


def test_primitive_pair_remeasured_and_numeric_permission_bound():
    import numpy as np
    from intent_policy.sim.demo_contact_audit import primitive_samples, primitive_sdf
    box = primitive_samples(mujoco.mjtGeom.mjGEOM_BOX, np.array([.01, .02, .03]), .0025)
    np.testing.assert_allclose(primitive_sdf(mujoco.mjtGeom.mjGEOM_BOX, np.array([.01, .02, .03]), box), 0, atol=1e-12)
    cyl = primitive_samples(mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([.04, .003, 0]), .0025)
    np.testing.assert_allclose(primitive_sdf(mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([.04, .003, 0]), cyl), 0, atol=1e-12)
    for mask in (1, 0):               # physical pair: solver contacts; non-physical: sampled primitive SDF
        m, d = scene(mask)
        pose(m, d, .0515)             # centres 98.5 mm apart: 1.5 mm overlap of the two 50 mm spheres
        audit = DemoContactAudit(m, [PairRule('a', 'b', 0, .001)], lambda *a: .002)
        audit.observe(m, d)
        assert audit.first_violation is None and audit.refined == (1 - mask)
        assert -0.0016 < audit.report()['pairs'][0]['minimum_distance_m'] < -0.0012
        audit = DemoContactAudit(m, [PairRule('a', 'b', 0, .001)], lambda *a: True)
        audit.observe(m, d)
        assert audit.first_violation['kind'] == 'penetration'


def test_physical_pair_without_solver_contact_is_not_penetrating(monkeypatch):
    m, d = scene()
    pose(m, d, .0515)
    d.ncon = 0                        # the solver found no contact for the pair at this step
    monkeypatch.setattr(mujoco, 'mj_geomDistance', lambda *args: -0.05)
    audit = DemoContactAudit(m, [PairRule('a', 'b', 0, 0)])
    audit.observe(m, d)
    assert audit.first_violation is None


def test_observer_contacts_match_current_geometry():
    import numpy as np
    from intent_policy.sim.base_env import ManipulationEnv
    with ManipulationEnv() as env:
        fresh = mujoco.MjData(env.model)
        def verify(m, d):
            fresh.qpos[:] = d.qpos
            fresh.mocap_pos[:] = d.mocap_pos
            fresh.mocap_quat[:] = d.mocap_quat
            mujoco.mj_kinematics(m, fresh)
            mujoco.mj_collision(m, fresh)
            assert d.ncon == fresh.ncon
            np.testing.assert_array_equal(d.contact.geom1, fresh.contact.geom1)
            np.testing.assert_array_equal(d.contact.geom2, fresh.contact.geom2)
            np.testing.assert_allclose(d.contact.dist, fresh.contact.dist, atol=1e-12)
        env.pre_physics_observers.append(verify)
        env.physics_observers.append(verify)
        action = env.hold().copy()
        action[0] += .1
        for _ in range(3):
            env.step(action)
