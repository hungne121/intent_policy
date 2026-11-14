"""C2 nominal paired preview: the human hands a cloth to the robot, lifts the cup / clock in a
random order, the robot wipes the exposed marks and returns the cloth. Scripted expert, not ACT.
Wiping is a visual proxy (marks disappear under the cloth); no fabric physics."""
import os
import numpy as np
from v2_preview_common import (Preview, move, hold, main_pair, wipe_marks, marks_left, apply_station, TRAVEL_Z, GRIP_OPEN,
                               GRIP_CLOSED, GRASP_DZ)

HANDOVER = np.array([.28, -.14, .78])        # cloth centre where the human offers / receives it
# 14/11: the towel lies along the upturned hand: its near edge at the palm, most of it beyond the fingertips.
# The robot pinches it ACROSS (fingers closing along world X, see TOOL_YAW) 3 cm past its centre, 6.5 cm
# beyond the human fingertips, so no robot finger comes near the hand.
PALM_TO_CLOTH = np.array([0., .11, .02])       # cloth centre relative to the upturned palm point
RACK_XY, RACK_TOP = (.28, -.32), .68           # small stand the folded towel waits on; the open hand slides under it
ROBOT_GRIP_OFF = np.array([0., .03])           # robot grasp point on the cloth (xy offset from its centre)
TOOL_YAW_DEG = 0.                              # with the station's base yaw this closes the fingers along world X (probe 14/11)
CLOTH_SIZE = (.03, .09, .004)                  # 6 x 18 cm towel (gripper opening 8.5 cm)
SHELF_FLOOR = .623
SPOTS = {'cup': [(.13, -.11), (.18, -.12), (.22, -.11)], 'clock': [(.33, -.11), (.38, -.12), (.43, -.11)]}
STANDBY = np.array([.28, .06, .80])


def hook(scene, positions, xml):
    scene['objects']['cloth']['size'] = list(CLOTH_SIZE)
    positions['cloth'] = (RACK_XY[0], RACK_XY[1], RACK_TOP + CLOTH_SIZE[2] + .002)
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    g = root.find("./worldbody/body[@name='cloth']/geom")
    g.set('size', ' '.join(str(v) for v in CLOTH_SIZE))
    half_h = (RACK_TOP - .62) / 2
    ET.SubElement(root.find('worldbody'), 'geom', name='decor_rack_0', type='box', size=f'.05 .05 {half_h:.3f}',
                  pos=f'{RACK_XY[0]} {RACK_XY[1]} {.62 + half_h:.3f}', rgba='0.55 0.45 0.35 1', contype='1', conaffinity='1')
    return scene, positions, apply_station(scene, ET.tostring(root, encoding='unicode'))


def human(pv, order):
    home = pv.human.hanging_palm(pv.human.active)
    under = pv.pos('cloth') - PALM_TO_CLOTH          # open hand slides under the towel's near edge
    yield from hold(home, 1.0, phase='idle')
    for f in move(home, under, 1.2, lift=.04, shape='offer', shape_from='relaxed', phase='reach_cloth'):
        f['palm_up'] = f['progress']
        yield f
    yield from hold(under, .2, shape='offer', palm_up=1., phase='scoop_cloth')
    pv.human_grasp('cloth')
    raised = under + [0, 0, .08]
    yield from move(under, raised, .6, shape='offer', palm_up=1., phase='lift_cloth')
    offer = HANDOVER - PALM_TO_CLOTH
    yield from move(raised, offer, 1.2, lift=.02, shape='offer', palm_up=1., phase='offer_cloth')
    yield from hold(offer, until=lambda: pv.attached_name() == 'cloth', shape='offer', palm_up=1., phase='wait_robot_grip')
    pv.human_release('cloth'); pv.event('handover_complete')
    for f in move(offer, home, .9, lift=.05, shape='relaxed', shape_from='offer', phase='withdraw'):
        f['palm_up'] = 1. - f['progress']
        yield f
    cur = home
    for obj in order:
        top = pv.pos(obj) + [0, 0, (.037 if obj == 'cup' else .035) + .045]   # hand above the object top, not inside it
        yield from move(cur, top + [0, 0, .06], 1.0, lift=.04, shape='grasp', shape_from='relaxed', phase=f'reach_{obj}')
        yield from move(top + [0, 0, .06], top, .35, shape='grasp', phase=f'grip_{obj}')
        # Let the smoothed reaching hand settle over the object before lifting vertically.
        yield from hold(top, .35, shape='grasp', phase=f'settle_{obj}')
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
    recv = HANDOVER - PALM_TO_CLOTH
    for f in move(home, recv, 1.0, lift=.04, shape='offer', shape_from='relaxed', phase='reach_cloth_back'):
        f['palm_up'] = f['progress']
        yield f
    pv.human_grasp('cloth'); pv.event('human_holds_cloth')
    yield from hold(recv, until=lambda: pv.attached_name() is None, shape='offer', palm_up=1., phase='wait_robot_release')
    for f in move(recv, home + [0, 0, .05], 1.0, lift=.03, shape='grasp', shape_from='offer', phase='take_cloth'):
        f['palm_up'] = 1. - f['progress']
        yield f
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
    yield from pv.goto(lambda: np.r_[cloth()[:2] + ROBOT_GRIP_OFF, cloth()[2] + .06], 'approach')
    yield from pv.goto(lambda: np.r_[cloth()[:2] + ROBOT_GRIP_OFF, cloth()[2] + GRASP_DZ], 'descend')
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
        # Cross the shelf wall while the whole gripper is above it, then lower vertically.
        yield from pv.goto(np.r_[spots[0], .78], 'move', GRIP_CLOSED)
        yield from pv.goto(np.r_[spots[0], wipe_z], 'wipe_lower', GRIP_CLOSED)
        for s in spots + spots[::-1][1:]:
            for cmd in pv.goto(np.r_[s, wipe_z], 'wipe', GRIP_CLOSED, tol=.006):
                wipe_marks(pv, pv.tcp()[:2]); yield cmd
            wipe_marks(pv, pv.tcp()[:2])
        pv.wiped[obj] = True; pv.event('wipe_end', object=obj, marks_left=marks_left(pv))
        yield from pv.goto(np.r_[spots[-1], .78], 'lift', GRIP_CLOSED)
        yield from pv.goto(STANDBY, 'retreat', GRIP_CLOSED)
    # 3. return the cloth: present it at the handover point and release when the human holds it
    yield from pv.wait(lambda: pv.human_phase in ('rest', 'wait_cloth') and pv.pos('clock')[2] < pv.initial['clock'][2] + .02
                       and pv.pos('cup')[2] < pv.initial['cup'][2] + .02, 'wait_objects_back', GRIP_CLOSED)
    yield from pv.goto(HANDOVER + [*ROBOT_GRIP_OFF, GRASP_DZ + .06], 'move', GRIP_CLOSED)
    yield from pv.goto(HANDOVER + [*ROBOT_GRIP_OFF, GRASP_DZ], 'present_cloth', GRIP_CLOSED)
    yield from pv.wait(lambda: 'cloth' in pv.held_by_human and pv.human_phase == 'wait_robot_release', 'present_cloth', GRIP_CLOSED)
    yield from pv.open_gripper(); pv.event('return_handover')
    yield from pv.goto(HANDOVER + [0, .012, GRASP_DZ + .10], 'retreat')
    yield from pv.goto(np.array([.25, .05, .82]), 'home')
    pv.phase = 'done'


class C2MotionStalled(RuntimeError):
    pass


class C2Preview(Preview):
    """End a blocked motion honestly instead of recording a long frozen preview."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._palm_up = 0.
        self.human_cloth_contacts = {}
        self._motion_signature = None
        self._motion_best = float('inf')
        self._motion_improved_at = 0

    def apply_human(self, frame):
        # Palm orientation used to jump from 0 to 1 on the first offer frame and
        # back to 0 on withdrawal, even while palm position itself was filtered.
        frame = dict(frame)
        target = float(frame.get('palm_up', 0.))
        self._palm_up += float(np.clip(target - self._palm_up, -.06, .06))
        frame['palm_up'] = self._palm_up
        return super().apply_human(frame)

    def step(self, command):
        super().step(command)
        # Generic preview auditing excludes held objects; retain actual hand/cloth
        # intersections separately, including while the cloth is welded to a hand.
        for index, distance, _, _ in self.audit.step_violations:
            rule = self.audit.rules[index]
            a = self.m.body(self.m.geom_bodyid[rule.geom_a]).name
            b = self.m.body(self.m.geom_bodyid[rule.geom_b]).name
            if a.startswith('human_seg_') and b == 'cloth':
                key = f'{a}|{b}'
                old = self.human_cloth_contacts.get(key)
                if old is None or -distance > old['depth_m']:
                    self.human_cloth_contacts[key] = {'depth_m': -float(distance),
                        't_s': round(self.t, 3), 'human_phase': self.human_phase,
                        'robot_phase': self.phase, 'held': 'cloth' in self.held_by_human}

    def command_to(self, goal, phase, grip, following=False):
        goal = np.asarray(goal, float)
        signature = (phase, tuple(np.round(goal, 2)))
        error = float(np.linalg.norm(self.tcp() - goal))
        if signature != self._motion_signature:
            self._motion_signature = signature
            self._motion_best = error
            self._motion_improved_at = self.tick
        elif error < self._motion_best - .003:
            self._motion_best = error
            self._motion_improved_at = self.tick
        elif self.tick - self._motion_improved_at >= 40:
            raise C2MotionStalled(f'{phase} target {goal.round(3).tolist()}, TCP {self.tcp().round(3).tolist()}')
        return super().command_to(goal, phase, grip, following=following)


def run(mode, seed, predictor, out):
    pv = C2Preview('C2', mode, seed, out, predictor, scene_hook=hook, max_ticks=int(os.environ.get('C2_PREVIEW_MAX_TICKS', '1400')))
    pv.env.cfg['tool_yaw_deg'] = TOOL_YAW_DEG
    order = list(np.random.default_rng(seed).permutation(['cup', 'clock']))
    pv.event('task_active', order_evaluator_only=order)
    stalled = None
    try:
        pv.run(human(pv, order), robot(pv), done=lambda: pv.human_phase == 'done')
    except C2MotionStalled as exc:
        stalled = str(exc)
        pv.event('motion_stalled', detail=stalled)
    if pv.human_phase != 'done' and pv.tick >= pv.max_ticks and stalled is None:
        stalled = 'preview time limit reached; task remains unfinished'
        pv.event('preview_limit', detail=stalled)
    back = all(np.linalg.norm(pv.pos(o)[:2] - pv.initial[o][:2]) < .02 and abs(pv.pos(o)[2] - pv.initial[o][2]) < .01 for o in ('cup', 'clock'))
    success = stalled is None and back and marks_left(pv) == 0 and 'cloth' in pv.held_by_human and pv.attached_name() is None
    return pv.finish(success, {'human_cloth_contacts': pv.human_cloth_contacts, 'marks_left': marks_left(pv), 'human_order_evaluator_only': order, 'stalled': stalled,
                               'limitations': ['wiping is a visual mark-removal proxy, no fabric physics',
                                               'rigid cloth box on open palm, grasped at far edge', 'scripted human, not HY-Motion',
                                               'control-frame contact checks only', 'expert preview, not trained ACT']})


if __name__ == '__main__':
    main_pair('C2', run, 300100)
