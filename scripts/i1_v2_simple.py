"""I1 nominal paired preview on the plain V2 scaffold (user, 13/11).

Scene: three identical cups clustered on one plate in front of the robot. The human holds a stack of
three coasters in the right hand. Round i: the robot parks above the nearest remaining cup (observable
preparation); the human then carries the bottom coaster to a random spot (left / middle / right); as
soon as the hand starts moving the robot grasps and lifts the cup (both experts). The human sets the
coaster down and withdraws.
Observed: the robot waits until the coaster is down and the hand has left, then places the cup.
Intent: once ManiCast forecasts the coaster-holding hand reaching out over the table, the robot drifts
along the predicted hand path (forecast +0.5 s plus the hand-coaster offset) at a pace bounded by the
observed hand speed, keeping HAND_CLEAR_M from the current hand; after the coaster is down and the hand
leaves it places at work speed. The human never waits for the robot. Scripted expert, not ACT."""
import xml.etree.ElementTree as ET
import numpy as np
from v2_preview_common import Preview, move, hold, main_pair, hy_motion, TRAVEL_Z, GRIP_CLOSED, GRASP_DZ, DT

CUPS = {'cup0': (.24, .04), 'cup1': (.30, .15), 'cup2': (.36, .04)}
SPOTS = [np.array([.155, -.13]), np.array([.305, -.13]), np.array([.455, -.13])]
CUP_Z = .665
COASTER_T = .006
HOLD_DZ = .035                # right palm above the coaster it holds
STACK_DZ = .03                # stack top below the left palm
FOLLOW_Z = .86
HAND_CLEAR_M = .16


def hook(scene, positions, xml):
    scene['human']['stand_xy'] = [.30, -.58]
    for k, xy in CUPS.items():
        positions[k] = (xy[0], xy[1], CUP_Z)
    for i in range(3):
        positions[f'coaster{i}'] = (.75, -.45, .623 + i * COASTER_T)
    root = ET.fromstring(xml)
    ET.SubElement(root.find('worldbody'), 'geom', name='decor_plate_visual', type='box', size='.11 .10 .001',
                  pos='.30 .09 .621', rgba='0.70 0.82 0.95 1', contype='0', conaffinity='0')
    return scene, positions, ET.tostring(root, encoding='unicode')


def human(pv, order, rng):
    hang = pv.human.hanging_palm('r')
    rest = hang + np.array([0., .10, .10])                 # right hand holds the stack in front of the hip
    segs, meta = hy_motion(pv, 'i1_coaster_place', sample=int(rng.integers(8)))
    pv.human_motion_source = meta
    carry_shape = segs[0] if segs else None
    back_shape = segs[-1] if len(segs) > 1 else None
    for i in range(3):      # the stack hangs under the palm; coaster0 is the lowest and is set down first
        pv.place_kinematic(f'coaster{i}', rest + [0, .02, -HOLD_DZ + i * COASTER_T]); pv.kinematic.pop(f'coaster{i}')
    pv.apply_human({'palm': rest, 'shape': 'grasp'})
    for i in range(3):
        pv.human_grasp(f'coaster{i}')
    cur = rest
    yield from hold(rest, 1.0, shape='grasp', phase='idle')
    for r, spot in enumerate(order):
        co = f'coaster{r}'
        yield from hold(cur, until=lambda: pv.robot_prepared, shape='grasp', phase=f'wait_robot_{r}')
        yield from hold(cur, float(rng.uniform(.3, 1.2)), shape='grasp', phase=f'free_pause_{r}')
        pv.event('human_carry_onset', round=r)
        up = np.array([spot[0], spot[1], .623 + HOLD_DZ + .10])
        yield from move(cur, up, 1.4, lift=.04, motion=carry_shape, motion_scale=.25, shape='grasp', phase=f'carry_{r}')
        down = np.array([spot[0], spot[1], .623 + HOLD_DZ + .002])
        yield from move(up, down, .6, shape='grasp', phase=f'lower_{r}')
        pv.human_release(co); pv.coaster_down[r] = True; pv.event('coaster_placed', round=r, spot=spot.round(3).tolist())
        yield from move(down, rest, 1.2, lift=.08, motion=back_shape, motion_scale=.25, shape='grasp', phase=f'withdraw_{r}')
        cur = rest
    yield from hold(rest, 1.0, shape='grasp', phase='done')


def robot(pv):
    pv.robot_prepared = False; pv.coaster_down = [False] * 3
    remaining = list(CUPS)
    for r in range(3):
        cup = min(remaining, key=lambda c: np.linalg.norm(pv.pos(c)[:2] - pv.tcp()[:2])); remaining.remove(cup)
        co = f'coaster{r}'
        above = np.r_[pv.pos(cup)[:2], TRAVEL_Z]
        yield from pv.goto(above, 'approach')
        pv.robot_prepared = True; pv.event('robot_prepared', cup=cup)
        # observable cue: the free hand starts moving (towards the stack) -> take the cup now
        yield from pv.wait(lambda: pv.hand_speed > .05, 'wait_hand_motion')
        pv.robot_prepared = False; pv.event('hand_motion_observed', round=r)
        ok = yield from pv.pick(cup)
        yield from pv.goto(np.r_[pv.tcp()[:2], FOLLOW_Z], 'lift', GRIP_CLOSED)
        pv.event('robot_holds_cup', cup=cup)
        while not pv.coaster_down[r]:
            if pv.mode == 'intent' and pv.forecast is not None and pv.hand_speed > .03:
                palm = np.asarray(pv.pose['palm'])[:2]
                goal_xy = pv.wrist_future()[4][:2] + (palm - pv.wrist()[:2])
                away = goal_xy - palm; dist = float(np.linalg.norm(away))
                if dist < HAND_CLEAR_M:
                    goal_xy = palm + HAND_CLEAR_M * (away / dist if dist > 1e-6 else (pv.tcp()[:2] - palm) / max(np.linalg.norm(pv.tcp()[:2] - palm), 1e-6))
                if np.linalg.norm(pv.tcp()[:2] - palm) < HAND_CLEAR_M and np.dot(goal_xy - pv.tcp()[:2], palm - pv.tcp()[:2]) > 0:
                    yield pv.hold(GRIP_CLOSED)
                else:
                    yield pv.command_to(np.r_[goal_xy, FOLLOW_Z], 'follow', GRIP_CLOSED, following=True)
            else:
                yield pv.hold(GRIP_CLOSED)
        target = lambda: pv.pos(co)
        yield from pv.wait(lambda: np.linalg.norm(np.asarray(pv.pose['palm'])[:2] - target()[:2]) > .15, 'wait_hand_clear', GRIP_CLOSED)
        pv.event('hand_clear', round=r)
        place_z = target()[2] + COASTER_T / 2 + .037 + .008 + GRASP_DZ
        yield from pv.goto(lambda: np.r_[target()[:2], TRAVEL_Z], 'move', GRIP_CLOSED)
        yield from pv.goto(lambda: np.r_[target()[:2], place_z], 'lower', GRIP_CLOSED)
        yield from pv.open_gripper(); pv.event('cup_on_coaster', round=r, cup=cup)
        yield from pv.goto(lambda: np.r_[target()[:2], TRAVEL_Z], 'retreat')
    yield from pv.goto(np.array([.15, 0., .85]), 'home')
    pv.phase = 'done'


def run(mode, seed, predictor, out):
    pv = Preview('I1', mode, seed, out, predictor, scene_hook=hook, max_ticks=1400)
    rng = np.random.default_rng(seed)
    order = [SPOTS[int(i)] for i in rng.permutation(3)]
    pv.event('task_active', spots_evaluator_only=[o.round(3).tolist() for o in order])
    pv.run(human(pv, order, rng), robot(pv), done=lambda: pv.human_phase == 'done')
    cups_on = 0
    for r in range(3):
        co = pv.pos(f'coaster{r}')
        if any(np.linalg.norm(pv.pos(c)[:2] - co[:2]) < .02 and abs(pv.pos(c)[2] - (co[2] + COASTER_T / 2 + .037 + .008)) < .01 for c in CUPS):
            cups_on += 1
    return pv.finish(cups_on == 3 and pv.attached_name() is None,
                     {'cups_on_coasters': cups_on, 'human_motion_source': getattr(pv, 'human_motion_source', None),
                      'limitations': ['straight moves shaped by HY-Motion palm-path curvature, not full-body replay',
                                      'coaster stack welded to the right hand', 'expert preview, not trained ACT']})


if __name__ == '__main__':
    main_pair('I1', run, 300100)
