"""57/QĐ-DL: shared Predictive Minimum-Interruption Avoidance (W1-W3)."""
import mujoco
import numpy as np

from intent_policy.shared.predictive_avoidance import seg_dist, capsule_from_geom
from intent_policy.shared.sim_env import make_env, geom_sets
from intent_policy.scenarios import scenario


def test_segment_distance():
    a0, a1 = np.array([0., 0., 0.]), np.array([1., 0., 0.])
    assert np.isclose(seg_dist(a0, a1, np.array([.5, 1., 0.]), np.array([.5, 2., 0.])), 1.)      # T shape
    assert np.isclose(seg_dist(a0, a1, np.array([2., 0., 0.]), np.array([3., 0., 0.])), 1.)      # collinear gap
    assert np.isclose(seg_dist(a0, a1, np.array([0., 1., 0.]), np.array([1., 1., 0.])), 1.)      # parallel
    assert np.isclose(seg_dist(a0, a1, np.array([.5, -1., 1.]), np.array([.5, 1., 1.])), 1.)     # crossing above


def test_robot_capsules_never_overstate_clearance():
    """Near range (< 0.3 m): the capsule bound of every robot link is never farther from a nearby solid (the cans)
    than the true mesh distance (mj_geomDistance), so the planner never overstates its clearance."""
    scene, positions, xml, _ = scenario('W1').build(300100)
    env, _ = make_env(scene, positions, xml)
    m, d = env.model, env.data
    _, robot, _, _ = geom_sets(m, positions)
    robot = [g for g in robot if m.geom_contype[g] or m.geom_conaffinity[g]]
    cans = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith('can') and m.geom_contype[g]]
    rng = np.random.default_rng(0)
    checked, worst = 0, -np.inf
    for _ in range(25):
        c = d.xpos[m.body('can0').id]
        q = env.solve_ik(c + [rng.uniform(-.12, .12), rng.uniform(-.12, .12), rng.uniform(.06, .20)])
        d.qpos[env.qids] = q
        mujoco.mj_forward(m, d)
        for a in robot:
            A, B, r = capsule_from_geom(m, d, a)
            for b in cans:
                true = mujoco.mj_geomDistance(m, d, a, b, 1., None)
                if true > .3:
                    continue
                C, D, s_ = capsule_from_geom(m, d, b)
                approx = float(seg_dist(A, B, C, D)) - r - s_
                worst = max(worst, approx - true); checked += 1
    assert checked > 50
    assert worst <= .005, worst
