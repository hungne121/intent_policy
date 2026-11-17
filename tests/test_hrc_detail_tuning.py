"""Regression: predictions prepare continuously; kinematic second-hand props stay attached."""
from types import SimpleNamespace
import numpy as np
import mujoco
from intent_policy.shared.episode import Episode
from intent_policy.experts.c4_expert import Expert


def test_c4_ambiguous_forecast_does_not_grasp_middle_part():
    class Fake:
        mode = 'forecast'
        delivered = None
        hand_speed = .2
        pose = {'palm': np.array([.28, -.08, .71])}
        initial = {f'part{i}': np.array([x,.12,.65]) for i,x in enumerate([.16,.28,.4])}
        def goto(self,*a,**kw):
            yield 'prepare'
        def wrist(self):
            return np.array([.3, -.3, .8])
        def pick(self,*a,**kw):
            raise AssertionError('A moving ambiguous hand must not commit to the middle part')
        def hold(self,*a): return 'hold'
    ep=Fake()
    ep.planner=SimpleNamespace(follow_point=lambda: np.array([.28,-.06,.71]),follow_step=lambda *a:'follow')
    gen=Expert(ep).run()
    assert next(gen)=='prepare'
    assert [next(gen) for _ in range(15)]==['follow']*15


def test_brush_follows_reached_idle_palm_across_physics_steps():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="cloth"><freejoint name="cloth_free"/><geom type="sphere" size=".01" mass=".02"/></body></worldbody></mujoco>''')
    data=mujoco.MjData(model)
    ep=SimpleNamespace(pose={'idle_palm':np.array([.1,-.2,.8])},idle_hand_objects={'cloth':np.array([0,0,-.047])})
    for i in range(80):
        ep.pose['idle_palm'][0]+=.001
        Episode._sync_idle_hand_objects(ep,model,data)
        mujoco.mj_step(model,data)
        Episode._sync_idle_hand_objects(ep,model,data)
        assert np.allclose(data.qpos[:3],ep.pose['idle_palm']+[0,0,-.047])
        assert np.allclose(data.qvel,0)
