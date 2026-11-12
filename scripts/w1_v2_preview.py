"""W1 nominal paired preview: the human moves four cakes into the two receiving trays at random
while the robot puts one can in each. Scripted expert, not ACT."""
import numpy as np
from v2_preview_common import Preview, move, hold, carry, main_pair, shift_decor, TRAVEL_Z, GRIP_OPEN, GRIP_CLOSED

TRAY_X = {'source': .10, 'L': .235, 'R': .37, 'cans': .505}
TRAY_Y = -.12
CAKE_SLOTS = {'L': [(-.03, -.03), (.03, -.03), (-.03, .03), (.03, .03)], 'R': [(-.03, -.03), (.03, -.03), (-.03, .03), (.03, .03)]}
CAN_SPOT = {'L': np.array([TRAY_X['L'], TRAY_Y + .045]), 'R': np.array([TRAY_X['R'], TRAY_Y + .045])}
CAKE_Z, CAN_Z = .641, .66


def hook(scene, positions, xml):
    scene['human']['stand_xy'] = [.28, -.56]
    old = [.12, .28, .44, .60]; new = [TRAY_X['source'], TRAY_X['L'], TRAY_X['R'], TRAY_X['cans']]
    for i in range(4):
        xml = shift_decor(xml, f'tray{i}', dx=new[i] - old[i])
    for i, (dx, dy) in enumerate([(-.028, -.04), (.028, -.04), (-.028, .04), (.028, .04)]):
        positions[f'cake{i}'] = (TRAY_X['source'] + dx, TRAY_Y + dy, CAKE_Z)
    positions['can0'] = (TRAY_X['cans'], TRAY_Y - .04, CAN_Z); positions['can1'] = (TRAY_X['cans'], TRAY_Y + .04, CAN_Z)
    return scene, positions, xml


def human(pv, plan):
    home = pv.human.hanging_palm(pv.human.active)
    cur = home
    yield from hold(home, .6, phase='idle')
    used = {'L': 0, 'R': 0}
    for i, tray in enumerate(plan):
        dx, dy = CAKE_SLOTS[tray][used[tray]]; used[tray] += 1
        dest = np.array([TRAY_X[tray] + dx, TRAY_Y - .035 + dy, CAKE_Z])
        # observable physical coordination only: wait if the robot is working inside this tray
        yield from hold(cur, until=lambda: not (np.linalg.norm(pv.tcp()[:2] - [TRAY_X[tray], TRAY_Y]) < .11 and pv.tcp()[2] < .76), phase=f'yield_{i}')
        cur = yield from carry(pv, f'cake{i}', dest, cur, dz=.035, lift=.07, reach_s=.9, carry_s=1.0, lower_s=.45, phase=f'cake{i}')
        pv.event('cake_placed', cake=i, tray=tray)
    yield from move(cur, home, .9, lift=.04, phase='rest')
    yield from hold(home, 1.0, phase='done')


def robot(pv):
    def hand_in(tray, horizon=False):
        c = np.array([TRAY_X[tray], TRAY_Y])
        if np.linalg.norm(pv.wrist()[:2] - c) < .13: return True
        if horizon and pv.forecast is not None:
            return bool(np.any(np.linalg.norm(pv.wrist_future()[:, :2] - c, axis=1) < .13))
        return False
    order = ['L', 'R']
    if pv.mode == 'intent':
        # start with the tray the forecast says the human is not heading to
        yield from pv.goto(np.array([.40, .02, TRAVEL_Z]), 'prepare')
        for _ in range(30):
            if pv.forecast is not None and pv.hand_speed > .03:
                f = pv.wrist_future()[-1][:2]
                order = ['R', 'L'] if np.linalg.norm(f - [TRAY_X['L'], TRAY_Y]) < np.linalg.norm(f - [TRAY_X['R'], TRAY_Y]) else ['L', 'R']
                pv.event('forecast_order', order=order); break
            yield pv.hold()
    for k, tray in enumerate(order):
        can = f'can{k}'
        ok = yield from pv.pick(can)
        spot = np.r_[CAN_SPOT[tray], TRAVEL_Z]
        yield from pv.goto(spot, 'move', GRIP_CLOSED)
        # do not lower into a tray the human's hand occupies (observed) or is forecast to enter (Intent)
        yield from pv.wait(lambda: not hand_in(tray, horizon=pv.mode == 'intent'), 'wait_tray_clear', GRIP_CLOSED)
        yield from pv.goto(np.r_[CAN_SPOT[tray], CAN_Z + .024], 'lower', GRIP_CLOSED)
        yield from pv.open_gripper(); pv.event('can_placed', can=can, tray=tray)
        yield from pv.goto(spot, 'retreat')
    yield from pv.goto(np.array([.40, .05, .82]), 'home')
    pv.phase = 'done'


def run(mode, seed, predictor, out):
    pv = Preview('W1', mode, seed, out, predictor, scene_hook=hook, max_ticks=1200)
    rng = np.random.default_rng(seed)
    plan = [str(t) for t in rng.choice(['L', 'R'], 4)]
    while plan.count('L') > 3 or plan.count('R') > 3: plan = [str(t) for t in rng.choice(['L', 'R'], 4)]
    pv.event('task_active', plan_evaluator_only=plan)
    pv.run(human(pv, plan), robot(pv), done=lambda: pv.human_phase == 'done')
    def in_tray(obj, tray): return np.all(np.abs(pv.pos(obj)[:2] - [TRAY_X[tray], TRAY_Y]) < [.07, .09]) and pv.pos(obj)[2] < .70
    cans = sorted(t for t in ('L', 'R') if sum(in_tray(f'can{k}', t) for k in range(2)) == 1)
    cakes = all(any(in_tray(f'cake{i}', t) for t in ('L', 'R')) for i in range(4))
    return pv.finish(cans == ['L', 'R'] and cakes and pv.attached_name() is None,
                     {'human_plan_evaluator_only': plan,
                      'limitations': ['human carries cakes by hand weld (no tongs asset)', 'scripted human, not HY-Motion',
                                      'expert preview, not trained ACT']})


if __name__ == '__main__':
    main_pair('W1', run, 300100)
