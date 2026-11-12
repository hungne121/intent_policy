"""C2 nominal paired preview: the human hands a cloth to the robot, lifts the cup / clock in a
random order, the robot wipes the exposed marks and returns the cloth. Scripted expert, not ACT.
Wiping is a visual proxy (marks disappear under the cloth); no fabric physics."""
import numpy as np
from v2_preview_common import (Preview, move, hold, main_pair, wipe_marks, marks_left, TRAVEL_Z, GRIP_OPEN,
                               GRIP_CLOSED, GRASP_DZ)

HANDOVER = np.array([.28, -.24, .78])        # cloth centre where the human offers / receives it
HAND_OFF = np.array([0., -.055, .035])       # palm behind and above the held cloth (hand grips its near edge)
SHELF_FLOOR = .623
SPOTS = {'cup': [(.13, -.11), (.18, -.12), (.22, -.11)], 'clock': [(.33, -.11), (.38, -.12), (.43, -.11)]}
STANDBY = np.array([.28, .06, .80])


def hook(scene, positions, xml):
    scene['human']['stand_xy'] = [.30, -.56]
    scene['objects']['cloth']['size'] = [.03, .035, .004]
    positions['cloth'] = (.28, -.32, positions['cloth'][2])
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    g = root.find("./worldbody/body[@name='cloth']/geom")
    g.set('size', '0.03 0.035 0.004')
    return scene, positions, ET.tostring(root, encoding='unicode')


def human(pv, order):
    home = pv.human.hanging_palm(pv.human.active)
    cloth_top = pv.pos('cloth') + [0, -.03, .035]
    yield from hold(home, 1.0, phase='idle')
    yield from move(home, cloth_top, 1.0, lift=.05, shape='grasp', shape_from='relaxed', phase='reach_cloth')
    pv.human_grasp('cloth')
    offer = HANDOVER + HAND_OFF + [0, .03, 0]
    yield from move(cloth_top, offer, 1.2, lift=.04, shape='grasp', phase='offer_cloth')
    yield from hold(offer, until=lambda: pv.attached_name() == 'cloth', shape='grasp', phase='wait_robot_grip')
    pv.human_release('cloth'); pv.event('handover_complete')
    yield from move(offer, home, .9, lift=.05, shape='relaxed', shape_from='grasp', phase='withdraw')
    cur = home
    for obj in order:
        top = pv.pos(obj) + [0, 0, .045 if obj == 'cup' else .045]
        yield from move(cur, top + [0, 0, .06], 1.0, lift=.04, shape='grasp', shape_from='relaxed', phase=f'reach_{obj}')
        yield from move(top + [0, 0, .06], top, .35, shape='grasp', phase=f'grip_{obj}')
        pv.human_grasp(obj)
        up = top + [0, 0, .12]
        yield from move(top, up, .7, shape='grasp', phase=f'lift_{obj}'); pv.event('object_lifted', object=obj)
        # Hold the object up until the robot has wiped and withdrawn from the shelf region.
        yield from hold(up, until=lambda: pv.wiped.get(obj) and pv.tcp()[1] > -.02 and pv.tcp()[2] > .74,
                        shape='grasp', phase=f'hold_{obj}')
        yield from move(up, top, .7, shape='grasp', phase=f'replace_{obj}')
        pv.human_release(obj); pv.event('object_replaced', object=obj)
        yield from move(top, top + [0, 0, .06], .3, shape='relaxed', shape_from='grasp', phase=f'release_{obj}')
        cur = top + [0, 0, .06]
    yield from move(cur, home, .8, lift=.04, shape='relaxed', phase='rest')
    # Receive the cloth back once the robot presents it.
    yield from hold(home, until=lambda: pv.phase == 'present_cloth', phase='wait_cloth')
    recv = HANDOVER + HAND_OFF
    yield from move(home, recv, 1.0, lift=.04, shape='grasp', shape_from='relaxed', phase='reach_cloth_back')
    pv.human_grasp('cloth'); pv.event('human_holds_cloth')
    yield from hold(recv, until=lambda: pv.attached_name() is None, shape='grasp', phase='wait_robot_release')
    yield from move(recv, home + [0, 0, .05], 1.0, lift=.03, shape='grasp', phase='take_cloth')
    yield from hold(home + [0, 0, .05], 1.0, shape='grasp', phase='done')


def robot(pv, order_hint=None):
    pv.wiped = {}
    # 1. take the cloth where the human offers it (observed cloth pose; forecast of the offering hand in Intent)
    def offered(): return pv.human_phase in ('wait_robot_grip',) or (pv.human_phase == 'offer_cloth' and pv.human.active)
    yield from pv.wait(lambda: pv.human_phase in ('offer_cloth', 'wait_robot_grip'), 'wait_offer')
    if pv.mode == 'intent':
        # follow the forecast of the offering hand at human pace until the hand is still
        yield from pv.goto(lambda: np.r_[(pv.wrist_future()[-1] + [0, .085, 0])[:2], TRAVEL_Z + .03], 'approach',
                           following=True, until=lambda: pv.human_phase == 'wait_robot_grip')
    yield from pv.wait(lambda: pv.human_phase == 'wait_robot_grip', 'wait_offer_still')
    cloth = lambda: pv.pos('cloth')
    yield from pv.goto(lambda: np.r_[cloth()[:2] + [0, .012], cloth()[2] + .08], 'approach')
    yield from pv.goto(lambda: np.r_[cloth()[:2] + [0, .012], cloth()[2] + GRASP_DZ], 'descend')
    ok = yield from pv.close_on('cloth')
    yield from pv.wait(lambda: 'cloth' not in pv.held_by_human, 'wait_human_release', GRIP_CLOSED, timeout=40)
    yield from pv.goto(STANDBY, 'retreat', GRIP_CLOSED)
    for _ in range(2):
        # 2. which object is being lifted: observed lift, or in Intent the forecast hand heading
        def heading():
            if pv.forecast is None or pv.hand_speed < .03: return None
            f = pv.wrist_future()[-3:].mean(axis=0)[:2]
            d = {o: float(np.linalg.norm(f - pv.pos(o)[:2])) for o in ('cup', 'clock') if o not in pv.wiped}
            best = min(d, key=d.get)
            return best if d[best] < .14 and (len(d) == 1 or sorted(d.values())[1] - d[best] > .04) else None
        def lifted():
            for o in ('cup', 'clock'):
                if o not in pv.wiped and pv.pos(o)[2] > pv.initial[o][2] + .07: return o
            return None
        target = None
        if pv.mode == 'intent':
            while lifted() is None:
                h = heading()
                if h is not None:
                    target = h; pv.event('forecast_prep', object=h)
                    break
                yield pv.hold(GRIP_CLOSED)
            if target is not None:
                # pre-position the cloth beside the predicted spot at human pace, high enough to stay clear
                prep = np.array([np.mean([s[0] for s in SPOTS[target]]), .02, .76])
                yield from pv.goto(prep, 'approach', GRIP_CLOSED, following=True, until=lambda: lifted() is not None)
        yield from pv.wait(lambda: lifted() is not None, 'wait_lift', GRIP_CLOSED)
        obj = lifted(); pv.event('wipe_start', object=obj)
        wipe_z = SHELF_FLOOR + .028 + .003
        spots = [np.array(s) for s in SPOTS[obj]]
        yield from pv.goto(np.r_[spots[0], .72], 'move', GRIP_CLOSED)
        yield from pv.goto(np.r_[spots[0], wipe_z], 'wipe_lower', GRIP_CLOSED)
        for s in spots + spots[::-1][1:]:
            for cmd in pv.goto(np.r_[s, wipe_z], 'wipe', GRIP_CLOSED, tol=.006):
                wipe_marks(pv, pv.tcp()[:2]); yield cmd
            wipe_marks(pv, pv.tcp()[:2])
        pv.wiped[obj] = True; pv.event('wipe_end', object=obj, marks_left=marks_left(pv))
        yield from pv.goto(np.r_[spots[-1], .74], 'lift', GRIP_CLOSED)
        yield from pv.goto(STANDBY, 'retreat', GRIP_CLOSED)
    # 3. return the cloth: present it at the handover point and release when the human holds it
    yield from pv.wait(lambda: pv.human_phase in ('rest', 'wait_cloth') and pv.pos('clock')[2] < pv.initial['clock'][2] + .02
                       and pv.pos('cup')[2] < pv.initial['cup'][2] + .02, 'wait_objects_back', GRIP_CLOSED)
    yield from pv.goto(HANDOVER + [0, .012, GRASP_DZ + .06], 'move', GRIP_CLOSED)
    yield from pv.goto(HANDOVER + [0, .012, GRASP_DZ], 'present_cloth', GRIP_CLOSED)
    yield from pv.wait(lambda: 'cloth' in pv.held_by_human and pv.human_phase == 'wait_robot_release', 'present_cloth', GRIP_CLOSED)
    yield from pv.open_gripper(); pv.event('return_handover')
    yield from pv.goto(HANDOVER + [0, .012, GRASP_DZ + .10], 'retreat')
    yield from pv.goto(np.array([.25, .05, .82]), 'home')
    pv.phase = 'done'


def run(mode, seed, predictor, out):
    pv = Preview('C2', mode, seed, out, predictor, scene_hook=hook, max_ticks=1400)
    order = list(np.random.default_rng(seed).permutation(['cup', 'clock']))
    pv.event('task_active', order_evaluator_only=order)
    pv.run(human(pv, order), robot(pv), done=lambda: pv.human_phase == 'done')
    back = all(np.linalg.norm(pv.pos(o)[:2] - pv.initial[o][:2]) < .02 and abs(pv.pos(o)[2] - pv.initial[o][2]) < .01 for o in ('cup', 'clock'))
    success = back and marks_left(pv) == 0 and 'cloth' in pv.held_by_human and pv.attached_name() is None
    return pv.finish(success, {'marks_left': marks_left(pv), 'human_order_evaluator_only': order,
                               'limitations': ['wiping is a visual mark-removal proxy, no fabric physics',
                                               'rigid cloth box grasped by its edge', 'scripted human, not HY-Motion',
                                               'control-frame contact checks only', 'expert preview, not trained ACT']})


if __name__ == '__main__':
    main_pair('C2', run, 300100)
