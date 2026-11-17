"""C4 — the person holds an upturned palm at one of three sockets (random order), receives the matching part from
the robot and installs it (Collaborator)."""
import numpy as np

from intent_policy.scenarios.layout import make
from intent_policy.shared.episode import hold, move

TASK, MAX_TICKS = 'C4', 1400
SOCKETS = {i: np.array([x, -.06]) for i, x in enumerate((.16, .28, .40))}   # 55/QĐ-DL: 6 cm nearer the robot (reach)
PART_Y = .12                                       # 55/QĐ-DL: parts away from the robot body (were at y 0.20)
PALM_DZ = .09               # upturned palm above the socket
PART_ON_PALM = .054         # part centre above the palm centre when resting on the palm support
IN_HAND = np.array([0., .03, 0.])
SEATED_Z = .62 + .006 + .023 + .001         # part centre resting on the socket disk
COLLAR_R = .022                              # inner radius of the socket collar (part radius 0.018)
HOLD_EXTRA = .01                             # part held this much lower in the turned hand: its top clears the palm
# 58/QĐ-DL change of mind (01/KH-DL step 3; commit = the robot grips the part). The person holds the palm at socket A
# for h s (its own clock + seed, independent of the robot), then moves it to socket B. Nominal preview 17/11:
# palm stops at A -> forecast choice +0.15-0.30 s, observed choice +0.25-0.50 s, robot grip +1.75-2.65 s,
# part on the palm >= +4.2 s. before: h ~ U(0.6, 1.2) (both modes chose A, nobody gripped);
# after: h ~ U(2.95, 3.6) (latest grip + 0.3 s, before the earliest handover).
VARIANTS = ('nominal', 'switch_before_commit', 'switch_after_commit')
SWITCH_HOLD_S = {'switch_before_commit': (.6, 1.2), 'switch_after_commit': (2.95, 3.6)}


def build(seed, variant='nominal'):
    scene, positions, xml = make(TASK)
    scene['human']['stand_xy'] = [.30, -.56]
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    for i, s in SOCKETS.items():
        positions[f'part{i}'] = (s[0], PART_Y, positions[f'part{i}'][2])
        # sockets are fixed to the table (55/QĐ-DL: the free disks sank and let the parts slide off)
        body = root.find(f"./worldbody/body[@name='socket{i}']")
        geom = body.find('geom'); color = geom.get('rgba')
        root.find('worldbody').remove(body); positions.pop(f'socket{i}', None)
        ET.SubElement(root.find('worldbody'), 'geom', name=f'decor_socket{i}', type='cylinder', size='.022 .003',
                      pos=f'{s[0]} {s[1]} .623', rgba=color or '0.5 0.5 0.5 1', contype='1', conaffinity='1')
        # 56/QĐ-DL: a 1.4 cm collar (4 mm clearance) so a seated part stays in its socket (a flat disk let it tip and roll)
        for k in range(8):
            a = np.pi * k / 4; r = COLLAR_R + .003
            ET.SubElement(root.find('worldbody'), 'geom', name=f'decor_socket{i}_collar{k}', type='box', size='.003 .0095 .007',
                          pos=f'{s[0] + r * np.cos(a):.4f} {s[1] + r * np.sin(a):.4f} .633', euler=f'0 0 {a:.4f}',
                          rgba=color or '0.5 0.5 0.5 1', contype='1', conaffinity='1')
    # 57/QĐ-DL: tray, socket disks and collars form one fixed block centred on the socket row (the tray stayed at
    # y -0.12 when 55/QĐ-DL moved the sockets to y -0.06, so they sat on its edge)
    world = root.find('worldbody')
    block = ET.SubElement(world, 'body', name='assembly_block', pos=f'{SOCKETS[1][0]} {SOCKETS[1][1]} 0')
    tray_y = -.12
    for g in list(world.findall('geom')):
        name = g.get('name', '')
        if name.startswith(('decor_assembly_base_', 'decor_socket')):
            p = np.fromstring(g.get('pos'), sep=' ')
            if name.startswith('decor_assembly_base_'):
                p[1] += SOCKETS[1][1] - tray_y
            p[:2] -= SOCKETS[1]
            g.set('pos', ' '.join(f'{v:.5f}' for v in p))
            world.remove(g); block.append(g)
    for eq in list(root.find('equality') if root.find('equality') is not None else []):
        if 'socket' in (eq.get('name', '') + eq.get('body1', '') + eq.get('body2', '')):
            root.find('equality').remove(eq)
    xml = ET.tostring(root, encoding='unicode')
    rng = np.random.default_rng(seed)
    order = [int(i) for i in rng.permutation(3)]
    info = {'order': order, 'variant': variant}
    if variant != 'nominal':
        turn = int(rng.integers(0, 2))                       # the switch happens at turn 1 or 2 (seed)
        b = int(rng.choice([i for i in order[turn + 1:]]))
        lo, hi = SWITCH_HOLD_S[variant]
        info.update({'switch_turn': turn, 'A': order[turn], 'B': b, 'hold_s': float(rng.uniform(lo, hi))})
        # after the switch: B now, then A, then the rest
        info['order'] = order[:turn] + [b, order[turn]] + [i for i in order[turn + 1:] if i != b]
    return scene, positions, xml, info


def human(ep, info):
    home = ep.human.hanging_palm(ep.human.active)
    cur = home
    recv_at = lambda k: np.array([SOCKETS[k][0], SOCKETS[k][1] - .02, .62 + PALM_DZ])
    for turn, i in enumerate(info['order']):
        s = SOCKETS[i]
        recv = recv_at(i)
        if info.get('switch_turn') == turn:
            # change of mind: palm up at A first, hold h s, then move the open palm to B (= i)
            a = recv_at(info['A'])
            for f in move(cur, a, 1.4, lift=.05, shape='offer', shape_from='relaxed', phase=f'reach_{info["A"]}'):
                yield {**f, 'palm_up': f['progress']}
            ep.event('palm_up', socket=info['A'])
            yield from hold(a, info['hold_s'], shape='offer', palm_up=1., phase=f'wait_part_{info["A"]}')
            ep.event('human_intention_change', socket_from=info['A'], socket_to=i, robot_holds=ep.attached_name())
            for f in move(a, recv, .9, lift=.03, shape='offer', phase=f'reach_{i}'):
                yield {**f, 'palm_up': 1.}
        else:
            for f in move(cur, recv, 1.4, lift=.05, shape='offer', shape_from='relaxed', phase=f'reach_{i}'):
                yield {**f, 'palm_up': f['progress']}
        ep.event('palm_up', socket=i)
        yield from hold(recv, until=lambda: getattr(ep, 'delivered', None) == i, shape='offer', palm_up=1., phase=f'wait_part_{i}')
        part = f'part{i}'
        for f in move(recv, recv, .4, shape='grasp', shape_from='offer', phase=f'grip_{i}'):
            yield {**f, 'palm_up': 1.}
        # Keep the actual handover transform; teleporting into the fingers caused an impulse.
        ep.event('human_holds_part', part=part)
        yield from hold(recv, until=lambda: ep.attached_name() is None and ep.tcp()[2] > recv[2] + .12, shape='grasp', palm_up=1., phase=f'wait_robot_clear_{i}')
        # 56/QĐ-DL: the wrist turns palm-down carrying the part round the thumb side (pronation), so it ends under
        # the palm; it used to stay above the turned hand, which then had to go into the table to seat it and
        # knocked it off when lifting.
        offset0 = ep.pos(part) - np.asarray(ep.pose['palm'])
        turned = lambda p: (1. + HOLD_EXTRA * p / np.linalg.norm(offset0)) * np.array([
            offset0[0] * np.cos(np.pi * p) - offset0[2] * np.sin(np.pi * p), offset0[1],
            offset0[0] * np.sin(np.pi * p) + offset0[2] * np.cos(np.pi * p)])
        offset = turned(1.)
        down = np.array([s[0], s[1], SEATED_Z + .002]) - offset
        above = down + [0, 0, .08]
        for f in move(recv, above, .9, shape='grasp', phase=f'to_socket_{i}'):
            ep.set_human_hold_offset(part, turned(f['progress']))
            yield {**f, 'palm_up': 1. - f['progress']}
        yield from move(above, down, .6, shape='grasp', phase=f'insert_{i}')
        # closed-loop set-down onto the socket (observable part position), then release
        goal = np.array([s[0], s[1], SEATED_Z]); cmd = np.array(down, float)
        for _ in range(16):
            err = goal - ep.pos(part)
            if np.linalg.norm(err) < .004: break
            cmd = cmd + np.clip(err, -.008, .008)
            yield {'palm': cmd, 'shape': 'grasp', 'phase': f'insert_{i}'}
        down = cmd
        ep.human_release(part); ep.event('install_complete', socket=i)
        # open the hand flat in place over the seated part (relaxed fingers droop into it and dragged it off),
        # lift straight up, then relax
        yield from move(down, down, .3, shape='offer', shape_from='grasp', phase=f'release_{i}')
        yield from move(down, down + [0, 0, .06], .4, shape='offer', phase=f'release_{i}')
        yield from move(down + [0, 0, .06], above + [0, 0, .04], .3, shape='relaxed', shape_from='offer', phase=f'release_{i}')
        cur = above + [0, 0, .04]
    yield from move(cur, home, .9, lift=.04, phase='rest')
    yield from hold(home, 1.0, phase='done')


def human_done(ep):
    return ep.human_phase == 'done'
