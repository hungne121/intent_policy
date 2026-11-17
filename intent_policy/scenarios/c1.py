"""C1 — lay a cloth under each cup (Collaborator; amendment 14/11, cruise fix 16/11). Two trays far apart, a cup
on each and a folded cloth in front of each. The person stands between the trays; in a random order, once the
robot waits at its neutral point between the trays (observable), the person takes the cloth in front of a tray,
lifts it, waits for the cup to be lifted clear, spreads the cloth on the tray and withdraws to rest; the next
round starts after the cup is back on the cloth and the gripper has left."""
import xml.etree.ElementTree as ET
import numpy as np

from intent_policy.scenarios.layout import make
from intent_policy.shared.sim_env import DT, smooth, apply_station, connected_human

TASK, MAX_TICKS = 'C1', 1600
TRAY_X = {'L': .12, 'R': .48}
CRUISE_Z = .78            # 16/11: 0.84 put the UR3e at full elbow extension (docs/reports/C1_GIAT_NGUYEN_NHAN_2026-11-16.md)
NEUTRAL = np.array([.30, -.10, CRUISE_Z])


def build(seed):
    scene, positions, xml = make(TASK)
    scene['human']['stand_xy'] = [.28, -.47]
    scene['human']['max_lean_deg'] = 45.
    root = ET.fromstring(xml)
    for suffix, x in TRAY_X.items():
        old_x = positions['cup' + suffix][0]
        for geom in root.findall('./worldbody/geom'):
            if geom.get('name', '').startswith('decor_tray' + suffix + '_'):
                gpos = np.fromstring(geom.get('pos'), sep=' '); gpos[0] += x - old_x
                geom.set('pos', ' '.join(map(str, gpos)))
        for name in ('cup' + suffix, 'cloth' + suffix):
            xyz = positions[name]; positions[name] = (x, -.29 if name.startswith('cloth') else xyz[1], xyz[2])
    xml = connected_human(scene, ET.tostring(root, encoding='unicode'))
    xml = apply_station(scene, xml, stand_xy=[.28, -.47], cam={'pos': [-1.45, .10, 1.65], 'target': [.30, -.30, .78], 'fovy': 65})
    order = [str(o) for o in np.random.default_rng(seed).permutation(['L', 'R'])]
    return scene, positions, xml, {'order': order}


def human(ep, info):
    for r, side in enumerate(info['order']):
        cup, cloth = 'cup' + side, 'cloth' + side
        target = ep.initial[cup]
        home = ep.human.hanging_palm(ep.human.active)
        ep.event('human_wait_neutral', round=r)
        while np.linalg.norm(ep.tcp() - NEUTRAL) >= .03:
            yield {'palm': home, 'shape': 'grasp', 'shape_from': 'relaxed', 'progress': 0., 'phase': 'wait_neutral'}
        # commanded palm positions throughout, as in the reviewed version: the reach-limited real palm differs at
        # the far tray and must not enter the spread offset (16/11 port fix)
        start = home.copy()
        grip = ep.pos(cloth) + [0, 0, .055]
        # 57/QĐ-DL: reach over the cloth, then lower onto it (as `carry`). The straight path from the hanging hand
        # came up from below the table edge, swept the wrist along the table into the cloth (16 mm, 60 N) and
        # stood it on edge before the grasp, so it was carried upright through the hand.
        over_cloth = grip + [0, 0, .06]
        n = int(round(.8 / DT))
        for i in range(1, n + 1):
            u = smooth(i / n)
            yield {'palm': start + (over_cloth - start) * u + [0, 0, .04 * np.sin(np.pi * i / n)], 'shape': 'grasp',
                   'shape_from': 'relaxed', 'progress': 0., 'phase': 'reach_cloth'}
        n = int(round(.3 / DT))
        for i in range(1, n + 1):
            yield {'palm': over_cloth + (grip - over_cloth) * smooth(i / n), 'shape': 'grasp',
                   'shape_from': 'relaxed', 'progress': 0., 'phase': 'reach_cloth'}
        n = int(round(.35 / DT))
        for i in range(1, n + 1):
            yield {'palm': grip, 'shape': 'grasp', 'shape_from': 'relaxed', 'progress': smooth(i / n), 'phase': 'grasp_cloth'}
        ep.human_grasp(cloth)
        held = np.array([target[0], -.27, .77]); n = int(round(.7 / DT))
        for i in range(1, n + 1):
            yield {'palm': grip + (held - grip) * smooth(i / n), 'shape': 'grasp', 'progress': 1., 'phase': 'lift_cloth'}
        while ep.pos(cup)[2] <= .77:
            yield {'palm': held, 'shape': 'grasp', 'progress': 1., 'phase': 'wait_cup_clear'}
        hand_offset = ep.pos(cloth)[:2] - held[:2]
        spread = np.array([target[0] - hand_offset[0], target[1] - hand_offset[1], .680])
        # 55/QĐ-DL: lay the cloth, do not throw it: lower it in front of the tray, glide it in just above the rim,
        # set it down, release, lift the hand, then withdraw
        front = spread + np.array([0., -.08, .02]); over = spread + np.array([0., 0., .02])
        for a, b, dur in ((held, front, .9), (front, over, .9)):
            n = int(round(dur / DT))
            for i in range(1, n + 1):
                yield {'palm': a + (b - a) * smooth(i / n), 'shape': 'grasp', 'progress': 1., 'phase': 'spread_cloth'}
        # set down until the cloth touches the tray (observed cloth height; resting centre z 0.627)
        spread = over - np.array([0., 0., max(0., ep.pos(cloth)[2] - .629)])
        n = int(round(.5 / DT))
        for i in range(1, n + 1):
            yield {'palm': over + (spread - over) * smooth(i / n), 'shape': 'grasp', 'progress': 1., 'phase': 'spread_cloth'}
        for _ in range(4):
            yield {'palm': spread, 'shape': 'grasp', 'progress': 1., 'phase': 'spread_cloth'}
        ep.human_release(cloth)
        up = spread + np.array([0., 0., .06]); n = int(round(.4 / DT))
        for i in range(1, n + 1):
            yield {'palm': spread + (up - spread) * smooth(i / n), 'shape': 'grasp', 'shape_from': 'relaxed',
                   'progress': 1 - smooth(i / n), 'phase': 'withdraw'}
        n = int(round(1.0 / DT))
        for i in range(1, n + 1):
            yield {'palm': up + (home - up) * smooth(i / n), 'shape': 'relaxed', 'phase': 'withdraw'}
        while not (np.linalg.norm(ep.pos(cup)[:2] - target[:2]) < .03 and ep.pos(cup)[2] < .71 and ep.attached_name() is None
                   and np.linalg.norm(ep.tcp() - ep.pos(cup)) > .13):
            yield {'palm': home, 'shape': 'relaxed', 'phase': 'wait_cup_return'}
    while True:
        yield {'palm': ep.human.hanging_palm(ep.human.active), 'shape': 'relaxed', 'phase': 'done'}


def human_done(ep):
    return ep.human_phase == 'done'
