"""W2 nominal paired preview: six colour blocks sorted into three colour trays; the human picks
at random, the robot goes left to right and skips blocks the human holds or approaches.
Scripted expert, not ACT."""
import numpy as np
from v2_preview_common import Preview, move, hold, carry, main_pair, shift_decor, TRAVEL_Z, GRIP_CLOSED

COLORS = ['red', 'blue', 'green']
TRAY = {c: np.array([x, -.08]) for c, x in zip(COLORS, (.10, .28, .46))}
BLOCK_Z = .649


def hook(scene, positions, xml):
    scene['human']['stand_xy'] = [.28, -.62]
    for i in range(3):
        xml = shift_decor(xml, f'tray{i}', dy=-.08 - (-.14))
    rng = np.random.default_rng(1)
    spots = [(.11, -.30), (.26, -.31), (.41, -.30), (.14, -.22), (.29, -.23), (.44, -.22)]
    for i, (x, y) in enumerate(spots):
        positions[f'block{i}'] = (x + float(rng.uniform(-.01, .01)), y + float(rng.uniform(-.01, .01)), BLOCK_Z)
    return scene, positions, xml


def color(i): return COLORS[i % 3]


def human(pv, order):
    home = pv.human.hanging_palm(pv.human.active)
    cur = home
    yield from hold(home, .8, phase='idle')
    for i in order:
        b = f'block{i}'
        # the robot already holds / is over it: take another block later instead
        if pv.attached_name() == b or (pv.claimed == b):
            pv.event('human_skips_claimed', block=b); continue
        if pv.pos(b)[2] > .70 or np.linalg.norm(pv.pos(b)[:2] - TRAY[color(i)]) < .07: continue
        dest = np.r_[TRAY[color(i)] + [-.03, 0.], BLOCK_Z]
        yield from hold(cur, until=lambda: np.linalg.norm(pv.tcp()[:2] - dest[:2]) > .10 or pv.tcp()[2] > .78, phase='yield')
        cur = yield from carry(pv, b, dest, cur, dz=.03, lift=.07, reach_s=.9, carry_s=.9, lower_s=.4, phase=f'block{i}')
    yield from move(cur, home, .9, lift=.04, phase='rest')
    yield from hold(home, 1.0, phase='done')


def robot(pv):
    pv.claimed = None
    def sorted_block(i):
        return np.linalg.norm(pv.pos(f'block{i}')[:2] - TRAY[color(i)]) < .07
    def human_near(i):
        p = pv.pos(f'block{i}')[:2]
        if f'block{i}' in pv.held_by_human or np.linalg.norm(pv.wrist()[:2] - p) < .08: return True
        if pv.mode == 'intent' and pv.forecast is not None and pv.hand_speed > .03:
            return bool(np.any(np.linalg.norm(pv.wrist_future()[:, :2] - p, axis=1) < .08))
        return False
    yield from pv.goto(np.array([.28, .05, TRAVEL_Z]), 'prepare')
    while True:
        remaining = [i for i in range(6) if not sorted_block(i) and f'block{i}' not in pv.held_by_human]
        if not remaining: break
        free = sorted([i for i in remaining if not human_near(i)], key=lambda i: pv.pos(f'block{i}')[0])
        if not free:
            yield pv.hold(); continue
        i = free[0]; b = f'block{i}'; pv.claimed = b; pv.event('robot_claims', block=b)
        ok = yield from pv.pick(b)
        if not ok:
            pv.claimed = None; yield from pv.goto(np.array([.28, .05, TRAVEL_Z]), 'prepare'); continue
        dest = np.r_[TRAY[color(i)] + [.03, 0.], BLOCK_Z + .024]
        yield from pv.goto(np.r_[dest[:2], TRAVEL_Z], 'move', GRIP_CLOSED)
        yield from pv.wait(lambda: np.linalg.norm(pv.wrist()[:2] - dest[:2]) > .10, 'wait_tray_clear', GRIP_CLOSED)
        yield from pv.goto(dest, 'lower', GRIP_CLOSED)
        yield from pv.open_gripper(); pv.event('robot_placed', block=b)
        pv.claimed = None
        yield from pv.goto(np.r_[dest[:2], TRAVEL_Z], 'retreat')
    yield from pv.goto(np.array([.28, .05, .82]), 'home')
    pv.phase = 'done'


def run(mode, seed, predictor, out):
    pv = Preview('W2', mode, seed, out, predictor, scene_hook=hook, max_ticks=1600)
    order = [int(i) for i in np.random.default_rng(seed).permutation(6)]
    pv.event('task_active', order_evaluator_only=order)
    pv.run(human(pv, order), robot(pv), done=lambda: pv.human_phase == 'done')
    ok = all(np.linalg.norm(pv.pos(f'block{i}')[:2] - TRAY[color(i)]) < .07 and pv.pos(f'block{i}')[2] < .68 for i in range(6))
    return pv.finish(ok and pv.attached_name() is None, {'human_order_evaluator_only': order,
                     'limitations': ['scripted human, not HY-Motion', 'expert preview, not trained ACT']})


if __name__ == '__main__':
    main_pair('W2', run, 300100)
