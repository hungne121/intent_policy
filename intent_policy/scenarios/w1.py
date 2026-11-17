"""W1 — the person moves four cakes from the source tray into two receiving trays (random plan) while the robot
puts one can in each receiving tray (Cowork). The person waits while the robot works inside a tray (observable)."""
import numpy as np

from intent_policy.scenarios.layout import make
from intent_policy.shared.sim_env import shift_decor
from intent_policy.shared.episode import hold, carry, move

TASK, MAX_TICKS = 'W1', 1200
TRAY_X = {'source': .10, 'L': .235, 'R': .37, 'cans': .505}
TRAY_Y = -.12
CAKE_SLOTS = {'L': [(-.03, -.03), (.03, -.03), (-.03, .03), (.03, .03)], 'R': [(-.03, -.03), (.03, -.03), (-.03, .03), (.03, .03)]}
CAN_SPOT = {'L': np.array([TRAY_X['L'], TRAY_Y + .045]), 'R': np.array([TRAY_X['R'], TRAY_Y + .045])}
CAKE_Z, CAN_Z = .641, .66
VARIANTS = ('nominal', 'linger_in_tray')   # 57/QĐ-DL preview cases: the person tidies in the tray it just used
LINGER_S = 1.6
CANS_Y = .10                                  # 55/QĐ-DL: can tray on the robot side (at y -0.12 the cans were 0.53 m from the base)


def build(seed, variant='nominal'):
    scene, positions, xml = make(TASK)
    scene['human']['stand_xy'] = [.28, -.56]
    old = [.12, .28, .44, .60]; new = [TRAY_X['source'], TRAY_X['L'], TRAY_X['R'], TRAY_X['cans']]
    for i in range(4):
        xml = shift_decor(xml, f'tray{i}', dx=new[i] - old[i], dy=(CANS_Y - TRAY_Y) if i == 3 else 0.)
    for i, (dx, dy) in enumerate([(-.028, -.04), (.028, -.04), (-.028, .04), (.028, .04)]):
        positions[f'cake{i}'] = (TRAY_X['source'] + dx, TRAY_Y + dy, CAKE_Z)
    positions['can0'] = (TRAY_X['cans'], CANS_Y - .04, CAN_Z); positions['can1'] = (TRAY_X['cans'], CANS_Y + .04, CAN_Z)
    rng = np.random.default_rng(seed)
    plan = [str(t) for t in rng.choice(['L', 'R'], 4)]
    while plan.count('L') > 3 or plan.count('R') > 3:
        plan = [str(t) for t in rng.choice(['L', 'R'], 4)]
    return scene, positions, xml, {'plan': plan, 'variant': variant}


def in_tray(ep, obj, tray):
    return np.all(np.abs(ep.pos(obj)[:2] - [TRAY_X[tray], TRAY_Y]) < [.07, .09]) and ep.pos(obj)[2] < .70


def human(ep, info):
    home = ep.human.hanging_palm(ep.human.active)
    cur = home
    yield from hold(home, .6, phase='idle')
    used = {'L': 0, 'R': 0}
    for i, tray in enumerate(info['plan']):
        dx, dy = CAKE_SLOTS[tray][used[tray]]; used[tray] += 1
        dest = np.array([TRAY_X[tray] + dx, TRAY_Y - .035 + dy, CAKE_Z])
        yield from hold(cur, until=lambda: not (np.linalg.norm(ep.tcp()[:2] - [TRAY_X[tray], TRAY_Y]) < .11 and ep.tcp()[2] < .76), phase=f'yield_{i}')
        cur = yield from carry(ep, f'cake{i}', dest, cur, dz=.035, lift=.13, reach_s=.9, carry_s=1.0, lower_s=.45, phase=f'cake{i}')   # 55/QĐ-DL: carry over the 7 cm cans
        ep.event('cake_placed', cake=i, tray=tray)
        if info.get('variant') == 'linger_in_tray':     # straighten the cakes: the hand stays over this tray
            a, b = np.array([TRAY_X[tray] - .03, TRAY_Y - .02, cur[2]]), np.array([TRAY_X[tray] + .03, TRAY_Y + .02, cur[2]])
            yield from move(cur, a, LINGER_S / 3, phase=f'tidy_{i}')
            yield from move(a, b, LINGER_S / 3, phase=f'tidy_{i}')
            yield from move(b, cur, LINGER_S / 3, phase=f'tidy_{i}')
    yield from move(cur, home, .9, lift=.04, phase='rest')
    yield from hold(home, 1.0, phase='done')


def human_done(ep):
    return ep.human_phase == 'done'
