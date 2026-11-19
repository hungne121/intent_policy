"""C3 — person and robot lift a two-handled tray together; the person's left hand wipes the table under it
with a dusting brush (55/QĐ-DL; visual mark proxy, kinematic second hand); both lower it (Collaborator).
The brush body keeps the builder name `cloth`."""
import copy
import xml.etree.ElementTree as ET
import numpy as np

from intent_policy.shared.events import detector
from intent_policy.scenarios.layout import base, build_scene_xml, align_robot_station
from intent_policy.shared.sim_env import STATION, connected_human
from intent_policy.shared.episode import hold, move, wipe_marks, marks_left

TASK, MAX_TICKS = 'C3', 900
TRAY_XY = np.array([.28, -.05]); TRAY_HALF = (.16, .10); TRAY_T = .004
HANDLE = {'human': np.array([.28, -.05 - .10 - .02]), 'robot': np.array([.28, -.05 + .10 + .02])}
HANDLE_Z = .62 + TRAY_T + .025              # 55/QĐ-DL: raised grip bar, centre 2.5 cm above the tray floor
HANDLE_HALF = (.05, .012, .02)               # 10 cm long, 2.4 cm thick, 4 cm tall: easy for the hand and the gripper
BRUSH_HALF = (.02, .025, .012)
PALM_TO_BRUSH = np.array([0., -.15, .085])   # 56/QĐ-DL: palm on the raised end of a 15 cm handle, behind the bristles
LIFT = .14                                   # 55/QĐ-DL: room for the brushing forearm under the tray (was 0.10)
# 56/QĐ-DL: dust under the left half of the tray, where the long-handled brush reaches with a modest lean
SPOTS = [(.14, -.10), (.20, -.10), (.25, -.07), (.20, -.04), (.14, -.04)]
HOLD_PALM = np.array([.10, -.36, .76])        # left hand holding the brush, bristles over the table edge


def build(seed):
    import os
    scene = copy.deepcopy(base)
    scene['zone_markers'] = False
    scene['human'].update({'hand_mesh_mode': 'compiled_variants', 'idle_palm_offset': [.12, .07, .50],
                           'hang_blend_m': .10, 'hang_rest_upright': True, 'idle_clearance_m': .01, 'self_clearance_m': .01,
                           'lean_rate_deg_s': 40., 'reach_margin': .88, 'max_extension': .94,
                           'stand_xy': [.30, -.58]})
    scene['human']['held_object_frame'] = STATION.get('human_held_object_frame', 'world')   # 61/QĐ-DL
    model = os.environ.get('HUMAN_MODEL', STATION.get('human_model'))
    if model in ('v3', 'v4'):
        scene['human']['model_version'] = model
    scene['cameras']['high'] = {'pos': [.3, .95, 1.65], 'target': [.3, -.20, .86], 'fovy': 65}
    scene['cameras']['top'] = {'pos': [.30, .04, 1.72], 'target': [.30, -.04, .62], 'fovy': 54}
    scene['objects'] = {'tray': {'shape': 'box', 'size': [TRAY_HALF[0], TRAY_HALF[1], TRAY_T], 'color': 'tray', 'mass': .12},
                        'cloth': {'shape': 'box', 'size': list(BRUSH_HALF), 'color': 'blue', 'mass': .03}}
    scene['decor'] = []
    positions = {'tray': (TRAY_XY[0], TRAY_XY[1], .62 + TRAY_T), 'cloth': (.14, -.27, .62 + BRUSH_HALF[2] + .002)}   # 56/QĐ-DL: handle end within the left hand's measured reach
    root = ET.fromstring(build_scene_xml(scene))
    align_robot_station(root)
    body = root.find("./worldbody/body[@name='tray']")
    g = body.find('geom'); rgba = (g.get('rgba') or '0.75 0.88 0.98 1').split()
    g.set('rgba', ' '.join(rgba[:3] + ['0.45']))      # 56/QĐ-DL: see-through tray, so the brushing under it is visible from top
    for side, xy in HANDLE.items():
        ET.SubElement(body, 'geom', name=f'tray_handle_{side}', type='box', size=' '.join(str(v) for v in HANDLE_HALF),
                      pos=f'0 {xy[1] - TRAY_XY[1]:.3f} {HANDLE_Z - (.62 + TRAY_T):.3f}', rgba='0.45 0.30 0.15 1', mass='0.02')
    brush = root.find("./worldbody/body[@name='cloth']")
    brush.find('geom').set('rgba', '0.25 0.25 0.28 1')
    ET.SubElement(brush, 'geom', name='cloth_handle', type='box', size='.008 .07 .008', pos='0 -.085 .036', quat='.985 -.174 0 0',
                  rgba='0.80 0.55 0.25 1', contype='0', conaffinity='0', mass='0')
    world = root.find('worldbody')
    for i, (x, y) in enumerate(SPOTS):
        ET.SubElement(world, 'geom', name=f'dirt_preview_{i}', type='cylinder', size='.017 .0001', pos=f'{x} {y} .6202',
                      rgba='.27 .16 .08 1', contype='0', conaffinity='0')
    return scene, positions, connected_human(scene, ET.tostring(root, encoding='unicode')), {}



def human(ep, info):
    home = ep.human.hanging_palm('r'); idle_home = ep.human.hanging_palm('l')
    handle = np.r_[HANDLE['human'], HANDLE_Z + HANDLE_HALF[2] + .03]   # palm over the grip bar
    cloth0 = ep.pos('cloth')
    # left hand takes the cloth first (kinematic second hand), right hand goes to the human handle
    yield from hold(home, .6, idle_palm=idle_home, phase='idle')
    for f in move(idle_home, cloth0 + PALM_TO_BRUSH, 1.0, lift=.04, phase='left_take_cloth'):
        yield {'palm': home, 'idle_palm': f['palm'], 'idle_shape': 'grasp'}
    reached = lambda: np.linalg.norm(np.asarray(ep.pose['idle_palm']) - (cloth0 + PALM_TO_BRUSH)) < .01
    yield from hold(home, 1.0, until=reached, idle_palm=cloth0 + PALM_TO_BRUSH, idle_shape='grasp', phase='left_take_cloth')
    ep.idle_hand_objects['cloth'] = -PALM_TO_BRUSH
    ep.event('human_grasp', object='cloth', hand='idle')     # 65/QĐ-DL: the idle-hand grasp is an intended contact
    idle = cloth0 + PALM_TO_BRUSH
    for f in move(idle, HOLD_PALM, .8, phase='left_hold'):
        yield {'palm': home, 'idle_palm': f['palm'], 'idle_shape': 'grasp'}
    idle = HOLD_PALM.copy()
    # 55/QĐ-DL: come over the grip bar first, then lower onto it (a diagonal reach pushed the tray 10 cm)
    for f in move(home, handle + [0, 0, .08], 1.1, lift=.04, shape='grasp', shape_from='relaxed', phase='reach_handle'):
        yield {**f, 'idle_palm': idle, 'idle_shape': 'grasp'}
    for f in move(handle + [0, 0, .08], handle, .4, shape='grasp', phase='reach_handle'):
        yield {**f, 'idle_palm': idle, 'idle_shape': 'grasp'}
    yield from ep.settle_hand({'palm': handle, 'progress': 1., 'shape': 'grasp', 'phase': 'reach_handle',   # 68/QĐ-DL B.1
                               'idle_palm': idle, 'idle_shape': 'grasp'})
    ep.human_grasp('tray'); ep.event('human_grasp_handle')
    yield from hold(handle, until=detector(ep).until('robot_holds_tray', detector(ep).c3_robot_holds_tray), shape='grasp', idle_palm=idle, idle_shape='grasp', phase='wait_robot_handle')
    yield from hold(handle, .3, shape='grasp', idle_palm=idle, idle_shape='grasp', phase='wait_robot_handle')
    ep.event('lift_onset')
    for f in move(handle, handle + [0, 0, LIFT], 1.4, shape='grasp', phase='lift'):
        yield {**f, 'idle_palm': idle, 'idle_shape': 'grasp'}
    up = handle + [0, 0, LIFT]
    # wipe under the raised tray with the left hand: sweep along the spots at table height
    path = [np.r_[x, y, .62 + BRUSH_HALF[2] + .002] + PALM_TO_BRUSH for x, y in SPOTS]   # bristles on the table
    cur = idle
    entry = np.array([.12, -.30, .72])
    for p in [entry] + path + [entry]:
        for f in move(cur, p, .55, phase='wipe'):
            brush = ep.pos('cloth')
            if abs(brush[2] - BRUSH_HALF[2] - .62) < .02:
                wipe_marks(ep, brush[:2], .035)
            yield {'palm': up, 'shape': 'grasp', 'idle_palm': f['palm'], 'idle_shape': 'grasp', 'phase': 'wipe'}
        cur = p
    ep.event('wipe_end', marks_left=marks_left(ep))
    for f in move(cur, idle, .6, phase='wipe_exit'):
        yield {'palm': up, 'shape': 'grasp', 'idle_palm': f['palm'], 'idle_shape': 'grasp'}
    yield from hold(up, .3, shape='grasp', idle_palm=idle, idle_shape='grasp', phase='pre_lower')
    ep.event('lower_onset')
    for f in move(up, handle, 1.4, shape='grasp', phase='lower'):
        yield {**f, 'idle_palm': idle, 'idle_shape': 'grasp'}
    yield from hold(handle, .4, shape='grasp', idle_palm=idle, idle_shape='grasp', phase='settle')
    ep.human_release('tray'); ep.event('human_release_handle')
    for f in move(handle, home, .9, lift=.05, shape='relaxed', shape_from='grasp', phase='withdraw'):
        yield {**f, 'idle_palm': idle, 'idle_shape': 'grasp'}
    yield from hold(home, 1.0, idle_palm=idle, idle_shape='grasp', phase='done')



def human_done(ep):
    return ep.human_phase == 'done'
