"""I3 — give the cake the person points at, with change of mind (Instructor). Four distinct cakes in two trays in
front of the person, a receive tray between the robot and the cake trays. Once the robot stands at the shared
neutral point between the trays, the person points at cake A (HY-Motion clip) and keeps pointing until the robot
visibly heads for A with an empty gripper, then withdraws. Variants (14/11): `nominal`; `switch_before_commit`
(short partial withdraw, second reach to B while the robot is still above A, before the grasp);
`switch_after_commit` (full withdraw, 1.2 s pause, second reach to B after the robot grasped A)."""
import xml.etree.ElementTree as ET
import numpy as np

from intent_policy.scenarios.layout import make
from intent_policy.shared.sim_env import ROOT, DT, STATION, apply_station, smooth, smooth_motion_path, motion_deviation
from intent_policy.sim.motion_bank_v2 import to_scene, palm_path, segment_ranges, clip_id

TASK, MAX_TICKS = 'I3', 900
VARIANTS = ('nominal', 'switch_before_commit', 'switch_after_commit')
CAKES = ('Lx', 'Ly', 'Rx', 'Ry')
FORCE_LIMIT_N = 10.0                      # pilot grasp force (PLAN_AUDIT, 10/11)
PLATE_XY = np.array([.28, .08]); PLATE_HALF = np.array([.07, .06]); CAKE_R = .025
RECEIVE_TOL = PLATE_HALF - CAKE_R
STAY_TOL, RETURN_TOL = .01, .02
MIDPOINT = np.array([.28, -.10, .82])     # shared neutral point between the cake trays
SWITCH_WITHDRAW_M = float(STATION['switch_withdraw_m'])
BANK = ROOT / 'outputs/hymotion_v2_candidates/hymotion_out'


def build(seed, variant='nominal'):
    rng = np.random.default_rng(seed)
    scene, positions, xml = make(TASK)
    scene['gripper_force_limit_n'] = FORCE_LIMIT_N
    scene['human']['stand_xy'][0] += float(rng.uniform(-.02, .02))
    jitter = {k: rng.uniform(-.008, .008, 2) for k in positions}
    positions = {k: (v[0] + float(jitter[k][0]), v[1] + float(jitter[k][1]), v[2]) for k, v in positions.items()}
    root = ET.fromstring(xml)
    for actuator in root.find('actuator'):
        if actuator.get('name') == 'gripper': actuator.set('forcerange', f'-{FORCE_LIMIT_N:g} {FORCE_LIMIT_N:g}')
    xml = apply_station(scene, ET.tostring(root, encoding='unicode'), stand_xy=[scene['human']['stand_xy'][0], -.54])
    a = str(rng.choice(CAKES))
    b = str(rng.choice([c for c in CAKES if c != a])) if variant != 'nominal' else None
    onset = .25 + float(rng.uniform(0., .4))
    return scene, positions, xml, {'variant': variant, 'A': a, 'B': b, 'onset_s': onset}


def _clip_commands(ep, info):
    """Palm / pointing commands of the whole gesture (one row per tick) and the hold ticks."""
    variant = info['variant']
    withdraw_s = {'nominal': 1.0, 'switch_before_commit': .5, 'switch_after_commit': 1.0}[variant]
    pause_s = {'nominal': 0., 'switch_before_commit': .1, 'switch_after_commit': 1.2}[variant]
    src = BANK / ('i3_point_cake.npz' if variant == 'nominal' else 'i3_switch.npz')
    with np.load(src, allow_pickle=False) as raw:
        k3d = raw['keypoints3d'][0].copy()
    w = to_scene(k3d, np.array([*ep.scene['human']['stand_xy'], 0]), ep.scene['human']['facing_deg'], ep.scene['floor_z'])
    generated = palm_path(w); parts = segment_ranges(generated, tails=True)
    first = 1 if variant != 'nominal' and len(parts) > 2 else 0
    a0, z0 = parts[first]; path_a = smooth_motion_path(generated[a0:z0])
    a1, z1 = parts[first + 1] if len(parts) > first + 1 else parts[first]; path_b = smooth_motion_path(generated[a1:z1])
    ep.human_motion_source = {'clip_id': clip_id(k3d), 'source': str(src.relative_to(ROOT)), 'frames': [int(a0), int(z0)]}
    home = ep.human.hanging_palm(ep.human.active)
    clear = home + np.array([.08, .14, .12])
    A = ep.initial[info['A']]; B = ep.initial[info['B']] if info['B'] else A
    arm_a = np.array([A[0], A[1] - .27, .84]); arm_b = np.array([B[0], B[1] - .27, .84])
    cmds, holds = [], {}

    def still(palm, dur):
        for _ in range(int(round(dur / DT))): cmds.append((palm.copy(), None, 0.))

    def go(p0, p1, dur, point, s0, s1, dev=None, arc=0.):
        k = int(round(dur / DT))
        for i in range(1, k + 1):
            u = smooth(i / k)
            palm = p0 + (p1 - p0) * u + np.array([0., 0., arc * np.sin(np.pi * i / k)])
            if dev is not None: palm = palm + motion_deviation(dev, u)
            st = s0 + (s1 - s0) * u
            cmds.append((palm, point if st > .001 else None, st))
    still(home, info['onset_s'])
    n = int(round(2.10 / DT))
    for i in range(1, n + 1):                       # reach A via the clear point, HY-Motion shape
        t = i * DT; reach = smooth(t / 2.10); pre = smooth(t / .75)
        cmds.append((home + (clear - home) * pre + (arm_a - clear) * reach + motion_deviation(path_a, reach), A, reach))
    holds[len(cmds) - 1] = info['A']
    if variant == 'nominal':
        go(arm_a, clear, withdraw_s * .7, A, 1., 0.); go(clear, home, withdraw_s * .5, None, 0., 0.)
    else:
        if variant == 'switch_before_commit':
            back_to = arm_a + (home - arm_a) * .6
            go(arm_a, back_to, withdraw_s, A, 1., 0., arc=.03)
            still(back_to, pause_s)
            go(back_to, arm_b, 1.2, B, 0., 1., dev=path_b, arc=.03)
        else:
            go(arm_a, clear, withdraw_s * .7, A, 1., 0.); go(clear, home, withdraw_s * .5, None, 0., 0.)
            still(home, pause_s)
            go(home, clear, .4, None, 0., 0.)
            go(clear, arm_b, 1.5, B, 0., 1., dev=path_b)
        holds[len(cmds) - 1] = info['B']
        go(arm_b, clear, .7, B, 1., 0.); go(clear, home, .5, None, 0., 0.)
    still(home, .5)
    return cmds, holds


def human(ep, info):
    ep.human.set_active('r')
    cmds, holds = _clip_commands(ep, info)
    clock, started, released, hold_start = 0, False, set(), {}
    tick = 0
    while True:
        # the person starts once the robot stands at the neutral point (observable)
        if not started and np.linalg.norm(ep.tcp() - MIDPOINT) < .025:
            started = True; ep.event('robot_neutral_ready')
        if started:
            tgt = holds.get(clock)
            if tgt is not None and clock not in released:
                # keep pointing until the robot visibly heads for THIS cake with an empty gripper (cap 8 s)
                dist = float(np.linalg.norm(ep.tcp()[:2] - ep.initial[tgt][:2]))
                hold_start.setdefault(clock, (dist, tick))
                d0, t0 = hold_start[clock]
                if (dist < max(.03, d0 - .025) and ep.attached_name() is None) or tick - t0 > 160:
                    released.add(clock); ep.event('human_withdraw_after_robot_toward_' + tgt, timed_out=tick - t0 > 160)
            if tgt is None or clock in released:
                clock += 1
        palm, point, strength = cmds[min(clock, len(cmds) - 1)]
        phase = 'done' if clock >= len(cmds) - 1 else ('idle' if not started else 'gesture')
        yield {'palm': palm, 'point_at': point if strength > .001 else None, 'shape': 'point', 'shape_from': 'relaxed',
               'progress': strength, 'aim_blend': strength, 'phase': phase}
        tick += 1


def human_done(ep):
    return ep.human_phase == 'done'


def check(ep, info):
    """Evaluator-only (I3 quality gate `task_success`): the last pointed cake on the receive tray, the others
    unmoved, and the grasp sequence matching the variant."""
    a, b = info['A'], info['B']; target = b or a
    reasons = []
    dxy = np.abs(ep.pos(target)[:2] - PLATE_XY)
    if not (np.all(dxy < RECEIVE_TOL) and .63 < ep.pos(target)[2] < .66):
        reasons.append(f'{target} not in receive tray')
    for k in CAKES:
        if k != target and np.linalg.norm(ep.pos(k)[:2] - ep.initial[k][:2]) > STAY_TOL:
            reasons.append(f'{k} moved')
    grasps = [e['object'] for e in ep.events if e['event'] == 'robot_grasp']
    if info['variant'] == 'nominal' and grasps != [target]:
        reasons.append(f'grasp sequence {grasps}')
    if info['variant'] == 'switch_before_commit' and grasps != [target]:
        reasons.append(f'grasp sequence {grasps} (A must not be grasped)')
    # 66/QĐ-DL: the switch time is on the person's clock (seed); a robot that had not yet grasped A when the person
    # switched (the slower observed expert) correctly takes only B. Outcome-based: [A, B] (A taken, returned) or [B].
    if info['variant'] == 'switch_after_commit' and grasps not in ([a, target], [target]):
        reasons.append(f'grasp sequence {grasps} (A then B, or only B, expected)')
    return not reasons, '; '.join(reasons) or None
