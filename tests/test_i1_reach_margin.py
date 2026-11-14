"""I1 must not command the cup transport through a straight-elbow singularity."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from i1_v2_preview import hook, CUPS, SPOTS, BASE_XY
from layout_gallery import make
from i3_collect import make_env


def test_i1_transport_waypoints_have_reach_margin():
    scene, positions, xml = hook(*make('I1'))
    env, human = make_env(scene, positions, xml)
    try:
        waypoints = [np.r_[xy, .80] for xy in CUPS.values()]
        waypoints += [np.r_[xy, .80] for xy in SPOTS]
        waypoints += [np.array([.28, .05, .82])]
        for p in waypoints:
            q = env.solve_ik(p)
            assert env.last_ik_error < .001, (p, env.last_ik_error)
            assert abs(np.sin(q[2])) > .15, (p, q[2])
    finally:
        env.close()
