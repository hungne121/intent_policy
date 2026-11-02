"""Speed and separation monitoring for the demonstrators (ISO/TS 15066 idea; the expert's T4 constants).

The mannequin's distance proxies (scene_builder: a small sphere at the palm) miss the fingers, which reach 85-130 mm
below the palm point (measured 2026-10-30): the gripper body hit a hand the proxy check did not see. Here the
visible hand mesh and the object the human holds are measured against every robot geom (MuJoCo convex hulls: a
conservative distance), besides the palm / held object against the TCP / the robot's held object.
"""
import mujoco
import numpy as np
from intent_policy.experts.scripted_expert import SEPARATION_M, SEPARATION_GEOM_M


def separation_hold(expert) -> bool:
    sc = expert.sc
    if sc.human_distance < SEPARATION_GEOM_M:
        return True
    human = [sc.human.pos] + ([sc.pos(sc.human.holding)] if sc.human.holding else [])
    held = expert._grasped_key()
    robot = [sc.tcp()] + ([sc.pos(held)] if held is not None else [])
    if min(float(np.linalg.norm(h - r)) for h in human for r in robot) < SEPARATION_M:
        return True
    m, d = sc.env.model, sc.env.data
    geoms = [sc.body_model.hand_geom] + (list(sc.object_geom_ids[sc.human.holding]) if sc.human.holding else [])
    if getattr(sc.human, 'bimanual', None):      # both hands on one object: the left hand too
        geoms.append(sc.body_model.idle_hand_geom)
    for r in sc.robot_geoms:
        for h in geoms:
            if mujoco.mj_geomDistance(m, d, r, h, SEPARATION_GEOM_M, None) < SEPARATION_GEOM_M:
                return True
    return False
