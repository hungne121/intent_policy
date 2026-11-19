"""I2 — put the cake on the plate the person points at (Instructor). One cake on its central source; three
destination plates. The person points at one plate (left hand for the left plate, else right) with the HY-Motion
clip `i2_point_plate` (reach 2.2 s), keeps pointing until the robot visibly carries the cake towards the chosen
plate, then withdraws (contingent, logged). 12/11 user update; seed 300106 is the middle-plate review seed."""
import numpy as np

from intent_policy.shared.events import detector
from intent_policy.scenarios.layout import make
from intent_policy.shared.sim_env import ROOT, DT, apply_station, smooth, smooth_motion_path, motion_deviation
from intent_policy.sim.motion_bank_v2 import to_scene, palm_path, segment_ranges, clip_id

TASK, MAX_TICKS = 'I2', 700
PLATES = {f'plate{i}': np.array([x, -.11, .64]) for i, x in enumerate((.105, .30, .495))}
CLIP = ROOT / 'outputs/hymotion_v2_candidates/hymotion_out/i2_point_plate.npz'


def build(seed):
    scene, positions, xml = make(TASK)
    xml = apply_station(scene, xml)
    rng = np.random.default_rng(seed)
    chosen = str(rng.choice(list(PLATES)))
    onset = float(rng.uniform(2.3, 2.8))
    return scene, positions, xml, {'plate': chosen, 'onset_s': onset, 'arm': 'l' if chosen == 'plate0' else 'r'}


def human(ep, info):
    with np.load(CLIP, allow_pickle=False) as raw:
        keypoints = raw['keypoints3d'][0].copy()
    world = to_scene(keypoints, np.array([*ep.scene['human']['stand_xy'], 0]), ep.scene['human']['facing_deg'], ep.scene['floor_z'])
    path = palm_path(world)
    parts = segment_ranges(path, tails=True)
    a, b = parts[0] if parts else (0, len(path))
    motion = smooth_motion_path(path[a:b])
    ep.human.set_active(info['arm'])
    ep.human_motion_source = {'clip_id': clip_id(keypoints), 'source': str(CLIP.relative_to(ROOT)), 'arm': info['arm']}
    home = ep.human.hanging_palm(ep.human.active)
    plate = PLATES[info['plate']]
    forward = np.array([plate[0], -.29, .86])
    withdraw_tick, prev_tcp = None, None
    for tick in range(10**6):
        t = tick * DT
        out = smooth((t - info['onset_s']) / 2.2)
        palm = home + (forward - home) * out + motion_deviation(motion, out) * .15
        strength = out
        # contingent withdrawal: the cake is carried towards the chosen plate (observable robot progress)
        tcp = ep.tcp()
        if withdraw_tick is None and strength > .95 and detector(ep).i2_cake_moves_to(plate, prev_tcp):
            withdraw_tick = tick; ep.event('human_withdraw_after_robot_direction')
            ep.event('human_cue', cue='cake_moves_to_plate')
        prev_tcp = tcp
        phase = 'point'
        if withdraw_tick is not None:
            fade = smooth((tick - withdraw_tick) * DT / .8)
            palm = palm * (1 - fade) + home * fade
            strength *= 1 - fade
            phase = 'withdraw' if fade < 1 else 'done'
        elif out <= .001:
            phase = 'idle'
        yield {'palm': palm, 'point_at': plate if strength > .001 else None, 'shape': 'point', 'shape_from': 'relaxed',
               'progress': strength, 'aim_blend': strength, 'phase': phase}


def human_done(ep):
    return ep.human_phase == 'done'


def check(ep, info):
    """Evaluator-only: the cake ended on the plate the person chose."""
    cake = ep.pos('cake')
    ok = bool(np.all(np.abs(cake[:2] - PLATES[info['plate']][:2]) < [.035, .03]))
    return ok, None if ok else 'wrong_plate'
