"""W3 nominal paired preview: six items packed into two boxes; the human packs random items
into their box, the robot packs the nearest unclaimed items into its box. Scripted expert, not ACT."""
import numpy as np
from v2_preview_common import Preview, move, hold, carry, main_pair, shift_decor, TRAVEL_Z, GRIP_CLOSED

BOX = {'human': np.array([.10, -.16]), 'robot': np.array([.10, .12])}
SLOTS = [(-.03, -.045), (.03, -.045), (-.03, .0), (.03, .0), (-.03, .045), (.03, .045)]
ITEM_Z = .647


def hook(scene, positions, xml):
    scene['human']['stand_xy'] = [.32, -.60]
    xml = shift_decor(xml, 'box0', dy=-.16 - (-.14)); xml = shift_decor(xml, 'box1', dy=.12 - .13)
    spots = [(.26, -.30), (.38, -.31), (.48, -.26), (.27, -.19), (.39, -.20), (.49, -.14)]
    rng = np.random.default_rng(2)
    for i, (x, y) in enumerate(spots):
        positions[f'item{i}'] = (x + float(rng.uniform(-.01, .01)), y + float(rng.uniform(-.01, .01)), ITEM_Z)
    return scene, positions, xml


def packed(pv, i):
    p = pv.pos(f'item{i}')
    return any(np.all(np.abs(p[:2] - b) < [.07, .09]) for b in BOX.values()) and p[2] < .70


def human(pv, order):
    home = pv.human.hanging_palm(pv.human.active)
    cur = home; n = 0
    yield from hold(home, .8, phase='idle')
    for i in order:
        it = f'item{i}'
        if packed(pv, i) or pv.attached_name() == it or pv.claimed == it:
            continue
        dest = np.r_[BOX['human'] + SLOTS[n], ITEM_Z]; n += 1
        cur = yield from carry(pv, it, dest, cur, dz=.03, lift=.08, reach_s=.9, carry_s=1.0, lower_s=.45, phase=it)
    yield from move(cur, home, .9, lift=.04, phase='rest')
    yield from hold(home, 1.0, phase='done')


def robot(pv):
    pv.claimed = None; n = 0
    def human_near(i):
        p = pv.pos(f'item{i}')[:2]
        if f'item{i}' in pv.held_by_human or np.linalg.norm(pv.wrist()[:2] - p) < .08: return True
        if pv.mode == 'intent' and pv.forecast is not None and pv.hand_speed > .03:
            return bool(np.any(np.linalg.norm(pv.wrist_future()[:, :2] - p, axis=1) < .08))
        return False
    yield from pv.goto(np.array([.30, .02, TRAVEL_Z]), 'prepare')
    while True:
        remaining = [i for i in range(6) if not packed(pv, i) and f'item{i}' not in pv.held_by_human]
        if not remaining: break
        free = sorted([i for i in remaining if not human_near(i)], key=lambda i: np.linalg.norm(pv.pos(f'item{i}')[:2] - pv.tcp()[:2]))
        if not free:
            yield pv.hold(); continue
        i = free[0]; it = f'item{i}'; pv.claimed = it; pv.event('robot_claims', item=it)
        ok = yield from pv.pick(it)
        if not ok:
            pv.claimed = None; yield from pv.goto(np.array([.30, .02, TRAVEL_Z]), 'prepare'); continue
        dest = np.r_[BOX['robot'] + SLOTS[n], ITEM_Z + .024]; n += 1
        yield from pv.place(dest)
        pv.event('robot_packed', item=it); pv.claimed = None
    yield from pv.goto(np.array([.30, .02, .82]), 'home')
    pv.phase = 'done'


def run(mode, seed, predictor, out):
    pv = Preview('W3', mode, seed, out, predictor, scene_hook=hook, max_ticks=1600)
    order = [int(i) for i in np.random.default_rng(seed).permutation(6)]
    pv.event('task_active', order_evaluator_only=order)
    pv.run(human(pv, order), robot(pv), done=lambda: pv.human_phase == 'done')
    ok = all(packed(pv, i) for i in range(6))
    return pv.finish(ok and pv.attached_name() is None, {'human_order_evaluator_only': order,
                     'limitations': ['scripted human, not HY-Motion', 'expert preview, not trained ACT']})


if __name__ == '__main__':
    main_pair('W3', run, 300100)
