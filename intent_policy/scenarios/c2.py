"""C2 — the person hands a dusting brush to the robot, lifts the cup and the clock (random order) so the robot
can sweep the dust under each, then takes the brush back (Collaborator). Dust = grids of small marks that vanish
under the bristle footprint (visual proxy). The brush body keeps the builder name `cloth` (52/QĐ-DL)."""
import xml.etree.ElementTree as ET
import numpy as np

from intent_policy.scenarios.layout import make
from intent_policy.shared.sim_env import apply_station, GRASP_DZ, STATION
from intent_policy.shared.episode import hold, move, hy_motion, dirt_geoms, marks_left

TASK, MAX_TICKS = 'C2', 1400
HANDOVER = np.array([.28, -.14, STATION['task_tuning']['C2']['handover_z_m']])        # cloth centre where the human offers / receives it
# 14/11: the towel lies along the upturned hand: its near edge at the palm, most of it beyond the fingertips.
# The robot pinches it ACROSS (fingers closing along world X, see TOOL_YAW) 3 cm past its centre, 6.5 cm
# beyond the human fingertips, so no robot finger comes near the hand.
# Brush: bristle block (body origin) + handle along +Y (towards the robot). The human offers it on the open
# palm with the bristles on the fingers and the handle beyond the fingertips; the robot pinches the handle.
BRISTLE_HALF = (.02, .025, .012)               # 4 x 5 cm bristle block, 2.4 cm tall
HANDLE_HALF = (.008, .05, .012)                # 1.6 cm wide handle (gripper probe 06/11), 10 cm long (55/QĐ-DL)
HANDLE_Y = .075                                # handle centre relative to the bristle centre
PALM_TO_CLOTH = np.array([0., .02, -.047])      # 55/QĐ-DL: the hand holds the bristle block FROM ABOVE (palm 4.7 cm over its centre);
                                               # the palm-up scoop hit the stand (26 mm) and sank the fingers into the bristles (12 mm)
RACK_XY, RACK_TOP = (.28, -.27), .70           # brush stand (16/11: 3 cm further from the shelf than the towel rack)
ROBOT_GRIP_OFF = np.array([0., .11])           # pinch the handle 11 cm from the bristle centre, clear of the holding fingers
TOOL_YAW_DEG = 0.                              # with the station's base yaw this closes the fingers along world X (probe 14/11)
CLOTH_SIZE = BRISTLE_HALF                      # collider of the scene body = bristle block
DUST_HALF = .004                               # dust marks: 4 x 4 grid per spot, 8 mm squares, 1.2 cm pitch
DUST_PITCH = .012
SHELF_FLOOR = .623
SPOTS = {'cup': [(.13, -.11), (.18, -.12), (.22, -.11)], 'clock': [(.33, -.11), (.38, -.12), (.43, -.11)]}
STANDBY = np.array([.28, .14, .80])            # 55/QĐ-DL: 8 cm further back, the held brush stayed in the person's reach path



def build(seed):
    scene, positions, xml = make(TASK)
    scene['objects']['cloth']['size'] = list(CLOTH_SIZE)
    positions['cloth'] = (RACK_XY[0], RACK_XY[1], RACK_TOP + BRISTLE_HALF[2] + .002)
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    body = root.find("./worldbody/body[@name='cloth']")
    g = body.find('geom')
    g.set('size', ' '.join(str(v) for v in BRISTLE_HALF)); g.set('rgba', '0.25 0.25 0.28 1')
    for extra in [e for e in body.findall('geom') if e.get('name', '').endswith('_visual')]:
        body.remove(extra)                      # drop the scanned towel mesh
    ET.SubElement(body, 'geom', name='cloth_handle', type='box', size=' '.join(str(v) for v in HANDLE_HALF),
                  pos=f'0 {HANDLE_Y} 0', rgba='0.80 0.55 0.25 1', contype='1', conaffinity='1')
    half_h = (RACK_TOP - .62) / 2
    ET.SubElement(root.find('worldbody'), 'geom', name='decor_rack_0', type='box', size=f'.05 .05 {half_h:.3f}',
                  pos=f'{RACK_XY[0]} {RACK_XY[1]} {.62 + half_h:.3f}', rgba='0.55 0.45 0.35 1', contype='1', conaffinity='1')
    world = root.find('worldbody')
    for old in [e for e in world.findall('geom') if e.get('name', '').startswith('dirt_preview_')]:
        world.remove(old)                       # the gallery's 6 round marks -> dust grids
    i = 0
    for spots in SPOTS.values():
        for (x, y) in spots:
            for a in range(4):
                for b in range(4):
                    ET.SubElement(world, 'geom', name=f'dirt_preview_{i}', type='box', size=f'{DUST_HALF} {DUST_HALF} .0001',
                                  pos=f'{x + (a - 1.5) * DUST_PITCH:.4f} {y + (b - 1.5) * DUST_PITCH:.4f} {SHELF_FLOOR + .0002:.4f}',
                                  rgba='.45 .38 .30 1', contype='0', conaffinity='0'); i += 1
    order = [str(o) for o in np.random.default_rng(seed).permutation(['cup', 'clock'])]
    return scene, positions, apply_station(scene, ET.tostring(root, encoding='unicode')), {'order': order}



def sweep_dust(ep):
    """Hide dust marks under the bristle footprint while the bristles touch the shelf floor."""
    c = ep.pos('cloth'); n = 0
    if c[2] - BRISTLE_HALF[2] > SHELF_FLOOR + .006: return 0
    for g in dirt_geoms(ep.m):
        if ep.m.geom_rgba[g, 3] > 0 and abs(ep.m.geom_pos[g, 0] - c[0]) < BRISTLE_HALF[0] and abs(ep.m.geom_pos[g, 1] - c[1]) < BRISTLE_HALF[1]:
            ep.m.geom_rgba[g, 3] = 0.; n += 1
    return n



def human(ep, info):
    order = info['order']
    home = ep.human.hanging_palm(ep.human.active)
    # 16/11: shape the reaching paths with a HY-Motion clip (as I1 does) instead of pure straight moves
    try:
        segs, _ = hy_motion(ep, 'c2_rack_lift', sample=int(np.random.default_rng(ep.seed).integers(8)))
        reach_shape = segs[0] if segs else None
    except Exception:
        reach_shape = None
    above = ep.pos('cloth') - PALM_TO_CLOTH          # palm over the bristle block on the stand
    yield from hold(home, 1.0, phase='idle')
    yield from move(home, above + [0, 0, .06], 1.2, lift=.04, motion=reach_shape, motion_scale=.2, shape='grasp',
                    shape_from='relaxed', phase='reach_cloth')
    yield from move(above + [0, 0, .06], above, .4, shape='grasp', phase='grip_cloth')
    yield from hold(above, .2, shape='grasp', phase='grip_cloth')
    yield from ep.settle_hand({'palm': above, 'shape': 'grasp', 'phase': 'grip_cloth'})   # 68/QĐ-DL B.1
    ep.human_grasp('cloth')
    raised = above + [0, 0, .04]
    yield from move(above, raised, .6, shape='grasp', phase='lift_cloth')
    offer = HANDOVER - PALM_TO_CLOTH                  # handle towards the robot
    yield from move(raised, offer, 1.2, shape='grasp', phase='offer_cloth')
    yield from hold(offer, until=lambda: ep.attached_name() == 'cloth', shape='grasp', phase='wait_robot_grip')
    ep.human_release('cloth'); ep.event('handover_complete')
    back = offer + [0, -.08, .02]                     # draw the hand back, away from the gripper, before rising
    yield from move(offer, back, .4, shape='relaxed', shape_from='grasp', phase='withdraw')
    yield from move(back, home, .9, lift=.04, shape='relaxed', phase='withdraw')
    cur = home
    for obj in order:
        top = ep.pos(obj) + [0, 0, (.037 if obj == 'cup' else .035) + .055]   # 16/11: +1 cm, the wrist capsule sank 23 mm into the cup
        # 16/11: 1.6 s instead of 1.0 s (peak palm speed was 1.0 m/s), HY-Motion curvature at 0.2
        yield from move(cur, top + [0, 0, .06], 1.6, lift=.04, motion=reach_shape, motion_scale=.2, shape='grasp', shape_from='relaxed', phase=f'reach_{obj}')
        yield from move(top + [0, 0, .06], top, .35, shape='grasp', phase=f'grip_{obj}')
        # Let the smoothed reaching hand settle over the object before lifting vertically.
        yield from hold(top, .35, shape='grasp', phase=f'settle_{obj}')
        yield from ep.settle_hand({'palm': top, 'shape': 'grasp', 'phase': f'settle_{obj}'})   # 68/QĐ-DL B.1
        ep.human_grasp(obj)
        # lift the object well above the shelf and draw it back towards the body, so the robot can wipe
        # underneath without coming within the safety distance of the hand
        up = top + [0, -.20, .28]   # kept: lower holds (14/16 cm, 18/24 cm tried 16/11) bring the robot elbow inside the keep-clear distance of the hand
        yield from move(top, up, 1.2, shape='grasp', phase=f'lift_{obj}'); ep.event('object_lifted', object=obj)
        # Hold the object up until the robot has wiped and withdrawn from the shelf region.
        yield from hold(up, until=lambda: ep.wiped.get(obj) and ep.tcp()[1] > -.02 and ep.tcp()[2] > .74,
                        shape='grasp', phase=f'hold_{obj}')
        yield from move(up, top, 1.2, shape='grasp', phase=f'replace_{obj}')
        # closed-loop set-down: the waist lean is rate-limited, so the reached palm differs between pick-up and
        # return; nudge the hand until the object is back on its mark (observable: the object's own position)
        cmd = np.array(top, float)
        for _ in range(12):
            err = ep.initial[obj] - ep.pos(obj)
            if np.linalg.norm(err[:2]) < .004 and abs(err[2]) < .004: break
            cmd = cmd + np.clip(err, -.01, .01)
            yield {'palm': cmd, 'shape': 'grasp', 'phase': f'replace_{obj}'}
        top = cmd
        ep.human_release(obj); ep.event('object_replaced', object=obj, place_error_m=float(np.linalg.norm(ep.pos(obj) - ep.initial[obj])))
        yield from move(top, top + [0, 0, .06], .3, shape='relaxed', shape_from='grasp', phase=f'release_{obj}')
        cur = top + [0, 0, .06]
    yield from move(cur, home, .8, lift=.04, shape='relaxed', phase='rest')
    # Receive the cloth back once the robot presents it.
    # reach once the robot has stopped presenting the brush (observable: present phase, TCP still)
    still = [0]
    def presented():
        still[0] = still[0] + 1 if ep.phase == 'present_cloth' and ep.controller.speed < .005 else 0
        return still[0] >= 4
    yield from hold(home, until=presented, phase='wait_cloth')
    recv = ep.pos('cloth') - PALM_TO_CLOTH            # over the brush where the robot actually presents it
    yield from move(home, recv + [0, -.06, .06], .7, lift=.04, shape='grasp', shape_from='relaxed', phase='reach_cloth_back')
    yield from move(recv + [0, -.06, .06], recv + [0, 0, .06], .3, lift=.04, shape='grasp', shape_from='relaxed', phase='reach_cloth_back')
    yield from move(recv + [0, 0, .06], recv, .4, shape='grasp', phase='reach_cloth_back')
    yield from ep.settle_hand({'palm': recv, 'progress': 1., 'shape': 'grasp', 'phase': 'reach_cloth_back'})   # 68/QĐ-DL B.1
    ep.human_grasp('cloth'); ep.event('human_holds_cloth')
    yield from hold(recv, until=lambda: ep.attached_name() is None, shape='grasp', phase='wait_robot_release')
    # put the brush back on its stand, then rest (55/QĐ-DL: carrying it down to the hanging hand bent the wrist into it)
    stand = ep.initial['cloth'] - PALM_TO_CLOTH
    yield from move(recv, stand + [0, 0, .05], 1.0, lift=.03, shape='grasp', phase='take_cloth')
    yield from move(stand + [0, 0, .05], stand + [0, 0, .004], .5, shape='grasp', phase='take_cloth')
    ep.human_release('cloth'); ep.event('brush_on_stand')
    yield from move(stand + [0, 0, .004], stand + [0, -.06, .06], .4, shape='relaxed', shape_from='grasp', phase='take_cloth')
    yield from move(stand + [0, -.06, .06], home, .8, lift=.03, shape='relaxed', phase='take_cloth')
    yield from hold(home, 1.0, phase='done')


def human_done(ep):
    return ep.human_phase == 'done'
