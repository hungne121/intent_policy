"""C2 dust removal is a world rule (Episode.world_tick), not an expert action (14/QĐ-KT, 09/QĐ-KT P0)."""
import mujoco
import numpy as np

from intent_policy.envs import HRCEnv
from intent_policy.scenarios import c2
from intent_policy.shared.episode import dirt_geoms


def _put_cloth(ep, xyz):
    m, d = ep.m, ep.d
    j = m.body_jntadr[m.body('cloth').id]
    assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    a = m.jnt_qposadr[j]
    d.qpos[a:a + 3] = xyz
    mujoco.mj_forward(m, d)


def test_dust_wiped_without_expert():
    env = HRCEnv('C2', 'nominal'); env.reset(seed=300800)
    ep = env.ep
    assert ep.world_tick is c2.world_tick                       # Episode picks the rule up from the scenario
    g = [x for x in dirt_geoms(ep.m) if ep.m.geom_rgba[x, 3] > 0]
    n0 = len(g); assert n0 > 0
    x, y = ep.m.geom_pos[g[0], :2]
    _put_cloth(ep, [x, y, c2.SHELF_FLOOR + c2.BRISTLE_HALF[2]])  # bristles on the floor, no expert running
    ep.world_tick(ep)
    assert ep.m.geom_rgba[g[0], 3] == 0 and sum(ep.m.geom_rgba[x_, 3] > 0 for x_ in g) < n0


def test_dust_kept_when_bristles_lifted():
    env = HRCEnv('C2', 'nominal'); env.reset(seed=300800)
    ep = env.ep
    g = [x for x in dirt_geoms(ep.m) if ep.m.geom_rgba[x, 3] > 0]
    x, y = ep.m.geom_pos[g[0], :2]
    _put_cloth(ep, [x, y, c2.SHELF_FLOOR + c2.BRISTLE_HALF[2] + .05])
    ep.world_tick(ep)
    assert all(ep.m.geom_rgba[x_, 3] > 0 for x_ in g)


def test_other_tasks_have_no_c2_rule():
    assert not hasattr(__import__('intent_policy.scenarios', fromlist=['scenario']).scenario('C4'), 'world_tick')
