"""I1 — cup on coaster (Instructor). Three identical cups clustered on a plate in front of the robot; three
coasters stacked at the person's right. Each round the person takes the top coaster, carries it to a random free
spot (left / middle / right), sets it down and withdraws. The person starts each round once the robot is prepared
(observable). Human paths: minimum-jerk moves shaped by the HY-Motion clip `i1_coaster_place`."""
import xml.etree.ElementTree as ET
import numpy as np

from intent_policy.scenarios.layout import make
from intent_policy.shared.sim_env import apply_station
from intent_policy.shared.episode import hold, move, hy_motion

TASK, MAX_TICKS = 'I1', 1400
CUPS = {'cup0': (.24, .08), 'cup1': (.30, .16), 'cup2': (.36, .08)}
SPOTS = [np.array([.155, -.08]), np.array([.305, -.08]), np.array([.455, -.08])]
CUP_Z = .665
COASTER_T = .006
SOURCE = np.array([.52, -.28])     # coaster stack on the table
HOLD_DZ = .045                     # palm centre above the held coaster


def build(seed):
    scene, positions, xml = make(TASK)
    for k, xy in CUPS.items():
        positions[k] = (xy[0], xy[1], CUP_Z)
    for i in range(3):              # stack at the person's right; coaster2 on top is taken first
        positions[f'coaster{i}'] = (SOURCE[0], SOURCE[1], .623 + i * COASTER_T)
    root = ET.fromstring(apply_station(scene, xml))
    ET.SubElement(root.find('worldbody'), 'geom', name='decor_plate_visual', type='box', size='.11 .09 .001',
                  pos='.30 .12 .621', rgba='0.70 0.82 0.95 1', contype='0', conaffinity='0')
    order = [int(i) for i in np.random.default_rng(seed).permutation(3)]
    return scene, positions, ET.tostring(root, encoding='unicode'), {'spot_order': order}


def human(ep, info):
    rng = np.random.default_rng(ep.seed); rng.permutation(3)        # same stream as the plan
    order = [SPOTS[i] for i in info['spot_order']]
    hang = ep.human.hanging_palm('r')
    rest = hang + np.array([0., .08, .08])
    segs, meta = hy_motion(ep, 'i1_coaster_place', sample=int(rng.integers(8)))
    ep.human_motion_source = meta
    reach_shape = segs[0] if segs else None
    carry_shape = segs[1] if len(segs) > 1 else reach_shape
    back_shape = segs[-1] if len(segs) > 2 else None
    cur = rest
    yield from hold(rest, 1.0, phase='idle')
    for r, spot in enumerate(order):
        co = f'coaster{2 - r}'
        yield from hold(cur, until=lambda: ep.robot_prepared, phase=f'wait_robot_{r}')
        yield from hold(cur, float(rng.uniform(.3, 1.2)), phase=f'free_pause_{r}')
        ep.event('human_motion_onset', round=r)
        top = ep.pos(co) + [0, 0, HOLD_DZ]
        yield from move(cur, top + [0, 0, .04], .9, lift=.03, motion=reach_shape, motion_scale=.2,
                        shape='grasp', shape_from='relaxed', phase=f'reach_{r}')
        yield from move(top + [0, 0, .04], top, .3, shape='grasp', phase=f'grip_{r}')
        yield from ep.settle_hand({'palm': top, 'progress': 1., 'shape': 'grasp', 'phase': f'grip_{r}'})   # 68/QĐ-DL B.1
        ep.human_grasp(co); ep.event('human_carry_onset', round=r)
        # 64/QĐ-DL: place with the REAL palm-to-coaster offset at the grasp (as episode.carry, 55/QĐ-DL); the nominal
        # HOLD_DZ put the coaster 9-12 mm into the table because the reached palm sits higher than its command
        off = np.asarray(ep.pose['palm'], float) - ep.pos(co)
        down = np.array([spot[0], spot[1], .623 + .002]) + off
        up = down + [0, 0, .10]
        yield from move(top, up, 1.4, lift=.06, motion=carry_shape, motion_scale=.25, shape='grasp', phase=f'carry_{r}')
        yield from move(up, down, .6, shape='grasp', phase=f'lower_{r}')
        ep.human_release(co); ep.coaster_down[r] = True
        ep.event('coaster_placed', round=r, spot=spot.round(3).tolist())
        yield from move(down, rest, 1.2, lift=.08, motion=back_shape, motion_scale=.25, shape='relaxed',
                        shape_from='grasp', phase=f'withdraw_{r}')
        cur = rest
    yield from hold(rest, 1.0, phase='done')


def human_done(ep):
    return ep.human_phase == 'done'
