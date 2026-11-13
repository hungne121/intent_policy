"""C3 nominal paired preview: human and robot lift a two-handled tray together, the human wipes
the table under it with a cloth in the other hand, both lower it. Scripted expert, not ACT.
The tray is one free body with two handle bars; the human handle is held by the hand weld, the
robot handle by the assisted-grasp weld. Wiping is a visual mark-removal proxy."""
import copy
import xml.etree.ElementTree as ET
import numpy as np
from v2_preview_common import (Preview, move, hold, main_pair, wipe_marks, marks_left, GRIP_OPEN, GRIP_CLOSED, GRASP_DZ, DT)
from layout_gallery import base, build_scene_xml

TRAY_XY = np.array([.28, -.05]); TRAY_HALF = (.16, .10); TRAY_T = .004
HANDLE = {'human': np.array([.28, -.05 - .10 - .02]), 'robot': np.array([.28, -.05 + .10 + .02])}
HANDLE_Z = .62 + TRAY_T + .012                      # handle bar centre (tray floor .624, bar raised 12 mm)
LIFT = .10
SPOTS = [(.18, -.09), (.23, -.03), (.28, -.09), (.33, -.03), (.38, -.09)]


def make_c3():
    scene = copy.deepcopy(base)
    scene['zone_markers'] = False
    scene['human'].update({'hand_mesh_mode': 'compiled_variants', 'idle_palm_offset': [.12, .07, .50],
                           'hang_blend_m': .10, 'hang_rest_upright': True, 'idle_clearance_m': .01, 'self_clearance_m': .01,
                           'lean_rate_deg_s': 40., 'reach_margin': .88, 'max_extension': .94,
                           'stand_xy': [.30, -.58]})
    scene['cameras']['high'] = {'pos': [.3, .95, 1.65], 'target': [.3, -.20, .86], 'fovy': 65}
    scene['cameras']['top'] = {'pos': [.30, .04, 1.72], 'target': [.30, -.04, .62], 'fovy': 54}
    scene['objects'] = {'tray': {'shape': 'box', 'size': [TRAY_HALF[0], TRAY_HALF[1], TRAY_T], 'color': 'tray', 'mass': .12},
                        'cloth': {'shape': 'box', 'size': [.03, .035, .004], 'color': 'blue', 'mass': .03}}
    scene['decor'] = []
    positions = {'tray': (TRAY_XY[0], TRAY_XY[1], .62 + TRAY_T), 'cloth': (.10, -.30, .62 + .004 + .002)}
    root = ET.fromstring(build_scene_xml(scene))
    body = root.find("./worldbody/body[@name='tray']")
    for side, xy in HANDLE.items():
        ET.SubElement(body, 'geom', name=f'tray_handle_{side}', type='box', size='.035 .02 .008',
                      pos=f'0 {xy[1] - TRAY_XY[1]:.3f} {.012 - TRAY_T:.3f}', rgba='0.45 0.30 0.15 1', mass='0.02')
    world = root.find('worldbody')
    for i, (x, y) in enumerate(SPOTS):
        ET.SubElement(world, 'geom', name=f'dirt_preview_{i}', type='cylinder', size='.017 .0001', pos=f'{x} {y} .6202',
                      rgba='.27 .16 .08 1', contype='0', conaffinity='0')
    return scene, positions, ET.tostring(root, encoding='unicode')


def human(pv):
    home = pv.human.hanging_palm('r'); idle_home = pv.human.hanging_palm('l')
    handle = np.r_[HANDLE['human'], HANDLE_Z + .035]
    cloth0 = pv.pos('cloth')
    # left hand takes the cloth first (kinematic second hand), right hand goes to the human handle
    yield from hold(home, .6, idle_palm=idle_home, phase='idle')
    for f in move(idle_home, cloth0 + [0, 0, .035], 1.0, lift=.04, phase='left_take_cloth'):
        yield {'palm': home, 'idle_palm': f['palm'], 'idle_shape': 'grasp'}
    pv.kinematic['cloth'] = cloth0.copy()
    idle = cloth0 + [0, 0, .035]
    def with_cloth(p):   # cloth hangs 2.5 cm under the left palm
        pv.kinematic['cloth'] = np.asarray(p) + [0, 0, -.025]
    for f in move(idle, np.array([.12, -.32, .76]), .8, phase='left_hold'):
        with_cloth(f['palm']); yield {'palm': home, 'idle_palm': f['palm'], 'idle_shape': 'grasp'}
    idle = np.array([.12, -.32, .76])
    for f in move(home, handle, 1.3, lift=.05, shape='grasp', shape_from='relaxed', phase='reach_handle'):
        yield {**f, 'idle_palm': idle, 'idle_shape': 'grasp'}
    pv.human_grasp('tray'); pv.event('human_grasp_handle')
    yield from hold(handle, until=lambda: pv.attached_name() == 'tray', shape='grasp', idle_palm=idle, idle_shape='grasp', phase='wait_robot_handle')
    yield from hold(handle, .3, shape='grasp', idle_palm=idle, idle_shape='grasp', phase='wait_robot_handle')
    pv.event('lift_onset')
    for f in move(handle, handle + [0, 0, LIFT], 1.4, shape='grasp', phase='lift'):
        yield {**f, 'idle_palm': idle, 'idle_shape': 'grasp'}
    up = handle + [0, 0, LIFT]
    # wipe under the raised tray with the left hand: sweep along the spots at table height
    path = [np.array([x, y, .62 + .004 + .025]) for x, y in SPOTS]
    cur = idle
    for p in [np.array([.12, -.20, .70])] + path + [np.array([.12, -.20, .70])]:
        for f in move(cur, p, .55, phase='wipe'):
            with_cloth(f['palm']); wipe_marks(pv, f['palm'][:2], .035)
            yield {'palm': up, 'shape': 'grasp', 'idle_palm': f['palm'], 'idle_shape': 'grasp'}
        cur = p
    pv.event('wipe_end', marks_left=marks_left(pv))
    for f in move(cur, idle, .6, phase='wipe_exit'):
        with_cloth(f['palm']); yield {'palm': up, 'shape': 'grasp', 'idle_palm': f['palm'], 'idle_shape': 'grasp'}
    yield from hold(up, .3, shape='grasp', idle_palm=idle, idle_shape='grasp', phase='pre_lower')
    pv.event('lower_onset')
    for f in move(up, handle, 1.4, shape='grasp', phase='lower'):
        yield {**f, 'idle_palm': idle, 'idle_shape': 'grasp'}
    yield from hold(handle, .4, shape='grasp', idle_palm=idle, idle_shape='grasp', phase='settle')
    pv.human_release('tray'); pv.event('human_release_handle')
    for f in move(handle, home, .9, lift=.05, shape='relaxed', shape_from='grasp', phase='withdraw'):
        yield {**f, 'idle_palm': idle, 'idle_shape': 'grasp'}
    yield from hold(home, 1.0, idle_palm=idle, idle_shape='grasp', phase='done')


def robot(pv):
    hxy = HANDLE['robot']
    handle_tcp = lambda: np.r_[pv.pos('tray')[:2] + (hxy - TRAY_XY), pv.pos('tray')[2] + .012 - .004 + GRASP_DZ]
    above = lambda: handle_tcp() + [0, 0, .10]
    near_handle = lambda: np.linalg.norm(pv.wrist()[:2] - HANDLE['human']) < .06 and pv.hand_speed < .03
    if pv.mode == 'intent':
        # forecast of the hand heading for its handle -> approach the robot handle early at human pace
        def heading():
            return pv.hand_speed > .03 and pv.forecast is not None and np.linalg.norm(pv.wrist_future()[-1][:2] - HANDLE['human']) < .10
        yield from pv.wait(lambda: heading() or near_handle(), 'observe')
        pv.event('forecast_prep')
        yield from pv.goto(above, 'approach', following=True, until=near_handle)
    yield from pv.wait(lambda: near_handle() and 'tray' in pv.held_by_human, 'wait_human_grasp')
    pv.event('human_grasp_observed')
    yield from pv.goto(above, 'approach')
    yield from pv.goto(handle_tcp, 'descend')
    ok = yield from pv.close_on('tray')
    z0 = pv.tcp()[2]; palm0 = pv.wrist()[2]
    # co-lift: track the human handle height. Observed follows the measured hand; Intent leads with the
    # 0.2 s forecast of the wrist so the robot side rises with, not after, the human side.
    def target():
        if pv.mode == 'intent' and pv.forecast is not None:
            dz = pv.wrist_future()[1][2] - palm0
        else:
            dz = pv.wrist()[2] - palm0
        return np.r_[hxy, z0 + float(np.clip(dz, -.01, LIFT + .01))]
    while pv.human_phase != 'settle':
        yield pv.command_to(target(), 'colift', GRIP_CLOSED)
    yield from pv.wait(lambda: 'tray' not in pv.held_by_human, 'settle', GRIP_CLOSED)
    yield from pv.open_gripper(); pv.event('robot_release_handle')
    yield from pv.goto(np.r_[hxy, z0 + .12], 'retreat')
    yield from pv.goto(np.array([.25, .15, .82]), 'home')
    pv.phase = 'done'


def run(mode, seed, predictor, out):
    pv = Preview('C3', mode, seed, out, predictor, scene_hook=lambda s, p, x: make_c3(), max_ticks=900)
    pv.event('task_active')
    pv.run(human(pv), robot(pv), done=lambda: pv.human_phase == 'done')
    tray = pv.pos('tray')
    success = (np.linalg.norm(tray[:2] - TRAY_XY) < .03 and abs(tray[2] - pv.initial['tray'][2]) < .01
               and marks_left(pv) == 0 and pv.attached_name() is None)
    return pv.finish(success, {'marks_left': marks_left(pv),
                               'limitations': ['two-weld co-lift (hand anchor + assisted grasp), not force control',
                                               'second hand moves the cloth kinematically', 'wiping is a visual mark proxy',
                                               'scripted human, not HY-Motion', 'expert preview, not trained ACT']})


if __name__ == '__main__':
    main_pair('C3', run, 300100)
