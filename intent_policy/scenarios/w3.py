"""W3 — pack six items into two boxes (Cowork). The person packs random items into the person's box and skips
items the robot has claimed; the robot packs into its own box."""
import numpy as np

from intent_policy.scenarios.layout import make
from intent_policy.shared.sim_env import shift_decor
from intent_policy.shared.episode import hold, carry, move

TASK, MAX_TICKS = 'W3', 1600
BOX = {'human': np.array([.10, -.16]), 'robot': np.array([.10, .12])}
SLOTS = [(-.03, -.045), (.03, -.045), (-.03, .0), (.03, .0), (-.03, .045), (.03, .045)]
ITEM_Z = .647
VARIANTS = ('nominal', 'contest_middle')   # 57/QĐ-DL preview cases: the person goes for the shared middle items
# 58/QĐ-DL: item places drawn per seed from a grid over the whole shared table (the person used to take only the
# fixed items in front of it); the person also reaches the middle rows, where the robot works and passes.
GRID_X = (.27, .37, .47)
GRID_Y = (-.27, -.17, -.07, .03, .12)
PERSON_REACH_Y = .05                       # the person takes items up to the y 0.03 row (robot side row excluded)


def draw_spots(seed):
    """6 of the 15 grid places: >= 2 on the person's rows (y <= -0.17), >= 2 on the middle rows (-0.07, 0.03) and
    >= 1 on the robot's row (y 0.12); +-1 cm jitter."""
    rng = np.random.default_rng(seed + 7)
    cells = [(x, y) for y in GRID_Y for x in GRID_X]
    while True:
        pick = [cells[k] for k in rng.choice(len(cells), 6, replace=False)]
        ys = np.array([c[1] for c in pick])
        if (ys <= -.17).sum() >= 2 and ((ys > -.17) & (ys < .1)).sum() >= 2 and (ys > .1).sum() >= 1:
            return [(x + float(rng.uniform(-.01, .01)), y + float(rng.uniform(-.01, .01))) for x, y in pick]


def build(seed, variant='nominal'):
    scene, positions, xml = make(TASK)
    scene['human']['stand_xy'] = [.32, -.60]
    xml = shift_decor(xml, 'box0', dy=-.16 - (-.14)); xml = shift_decor(xml, 'box1', dy=.12 - .13)
    spots = draw_spots(seed)
    for i, (x, y) in enumerate(spots):
        positions[f'item{i}'] = (x, y, ITEM_Z)
    order = [int(i) for i in np.random.default_rng(seed).permutation(6)]
    if variant == 'contest_middle':
        # the middle-row items first (nearest the robot first), then the rest in the random order
        mid = sorted([i for i in range(6) if -.17 < spots[i][1] < .1], key=lambda i: -spots[i][1])
        order = mid + [i for i in order if i not in mid]
    return scene, positions, xml, {'order': order, 'variant': variant, 'spots': [list(map(float, p)) for p in spots]}


def packed(ep, i):
    p = ep.pos(f'item{i}')
    return any(np.all(np.abs(p[:2] - b) < [.07, .09]) for b in BOX.values()) and p[2] < .70


def human(ep, info):
    home = ep.human.hanging_palm(ep.human.active)
    cur = home; n = 0
    yield from hold(home, .8, phase='idle')
    for i in info['order']:
        it = f'item{i}'
        if packed(ep, i) or ep.attached_name() == it or getattr(ep, 'claimed', None) == it or ep.pos(it)[1] > PERSON_REACH_Y:
            continue                                    # done, claimed by the robot, or on the robot's side
        dest = np.r_[BOX['human'] + SLOTS[n], ITEM_Z]; n += 1
        cur = yield from carry(ep, it, dest, cur, dz=.03, lift=.08, reach_s=.9, carry_s=1.0, lower_s=.45, phase=it,
                         lift_first=.06)   # 58/QĐ-DL: items now lie on the carry path (a low carry dragged one off)
    yield from move(cur, home, .9, lift=.04, phase='rest')
    yield from hold(home, 1.0, phase='done')


def human_done(ep):
    return ep.human_phase == 'done'
