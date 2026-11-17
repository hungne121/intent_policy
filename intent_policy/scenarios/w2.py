"""W2 — sort six colour blocks into three colour trays (Cowork). The person picks blocks in a random order and
puts each into its colour tray; a block the robot holds or is over (observable claim) is skipped."""
import numpy as np

from intent_policy.scenarios.layout import make
from intent_policy.shared.sim_env import shift_decor
from intent_policy.shared.episode import hold, carry, move

TASK, MAX_TICKS = 'W2', 1600
COLORS = ['red', 'blue', 'green']
TRAY = {c: np.array([x, -.02]) for c, x in zip(COLORS, (.10, .28, .46))}   # 55/QĐ-DL: trays reachable by both
BLOCK_Z = .649
PERSON_REACH_Y = 0.                           # the person takes blocks on its side of the trays only
TRAY_SLOTS = (-.03, .03)                      # 58/QĐ-DL: two places per tray (tray half 0.07, block half 0.02)


def color(i):
    return COLORS[i % 3]


def build(seed):
    scene, positions, xml = make(TASK)
    scene['human']['stand_xy'] = [.28, -.56]
    for i in range(3):
        xml = shift_decor(xml, f'tray{i}', dy=TRAY['red'][1] - (-.14))
    # 55/QĐ-DL: two blocks on the person's side, two in the shared middle, two on the robot's side
    rng = np.random.default_rng(1)
    spots = [(.14, -.25), (.42, -.26), (.22, -.13), (.38, -.13), (.16, .10), (.42, .11)]
    for i, (x, y) in enumerate(spots):
        positions[f'block{i}'] = (x + float(rng.uniform(-.01, .01)), y + float(rng.uniform(-.01, .01)), BLOCK_Z)
    order = [int(i) for i in np.random.default_rng(seed).permutation(6)]
    return scene, positions, xml, {'order': order}


def sorted_block(ep, i):
    return np.linalg.norm(ep.pos(f'block{i}')[:2] - TRAY[color(i)]) < .07 and ep.pos(f'block{i}')[2] < .68


def free_slot(ep, c, prefer, moving=None):
    """58/QĐ-DL: first free place in the colour tray, in the order `prefer` (the person's side first for the person,
    the robot's side first for the robot); a second block used to be stacked on the first."""
    for dx in prefer:
        xy = TRAY[c] + [dx, 0.]
        if all(np.linalg.norm(ep.pos(f'block{j}')[:2] - xy) > .025 for j in range(6) if f'block{j}' != moving):
            return xy
    return TRAY[c] + [prefer[0], 0.]


def human(ep, info):
    home = ep.human.hanging_palm(ep.human.active)
    cur = home
    yield from hold(home, .8, phase='idle')
    for i in info['order']:
        b = f'block{i}'
        if ep.attached_name() == b or getattr(ep, 'claimed', None) == b:
            ep.event('human_skips_claimed', block=b); continue
        if ep.pos(b)[2] > .70 or np.linalg.norm(ep.pos(b)[:2] - TRAY[color(i)]) < .07 or ep.pos(b)[1] > PERSON_REACH_Y:
            continue                                    # sorted already, or on the robot's side of the table
        dest = np.r_[free_slot(ep, color(i), TRAY_SLOTS, b), BLOCK_Z]
        yield from hold(cur, until=lambda: np.linalg.norm(ep.tcp()[:2] - dest[:2]) > .10 or ep.tcp()[2] > .78, phase='yield')
        cur = yield from carry(ep, b, dest, cur, dz=.03, lift=.07, reach_s=.9, carry_s=.9, lower_s=.4, phase=f'block{i}')
    yield from move(cur, home, .9, lift=.04, phase='rest')
    yield from hold(home, 1.0, phase='done')


def human_done(ep):
    return ep.human_phase == 'done'
