"""Collect paired Observed/Intent I3 episodes through the V2 quality gate.

One seed produces one human clip (shared by both experts), two rollouts, an
interpolated swept audit, an eleven-check quality report with measured values,
metadata separated from policy frames, and a gated NPZ staging record via
``dataset_records.write_episode``. Rejected episodes are logged with reasons,
never written as data. The expert uses a shared neutral preparation, continuous human-paced ManiCast
approach, and a human who withdraws after seeing correct robot motion. Each
realized human trajectory and its interaction events are saved for paired audits.

Evaluator-only knowledge (targets, switch schedule, case) never reaches the
controller: it reads keypoints, current object poses and, for Intent, the real
ManiCast forecast of the past 0.4 s.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from layout_gallery import make
from intent_policy.sim.base_env import ManipulationEnv
from intent_policy.sim.human_body_v2 import HumanBody
from intent_policy.sim.scene_builder_v2 import robot_weld_name
from intent_policy.sim.demo_contact_audit import DemoContactAudit, PairRule
from i3_self_check import I3SelfCheck as HumanSelfCheck
from intent_policy.sim.motion_bank_v2 import to_scene, palm_path, segment_ranges, warp, sample, clip_id
from intent_policy.experts.human_paced import (observed_hand_speed, forecast_hand_speed,
                                                HumanPace, CartesianFollower, work_speed, WORK_ACCEL)
from intent_policy.benchmark.video import VideoWriter
from i3_observed import select_pointed
from dataset_records import validate_frames, write_episode
from quality_gate import evaluate as quality_errors
from preview_metrics import compare_pair
from contact_context import overlap_tolerance, record_overlap

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
CAKES = ('Lx', 'Ly', 'Rx', 'Ry')
JOINTS = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist']
HORIZONS = (np.arange(1, 11) / 10).tolist()
DT = .05
SHIFT = 0.0                       # 14/11: no shift; the shared station puts the base at (0.30, 0.33)
FORCE_LIMIT_N = 10.0              # pilot grasp force (PLAN_AUDIT, 10/11)
TOL = {                           # metres; see docs/PLAN_AUDIT_I3_PIPELINE.md
    'human': 0.0,                 # hand/forearm vs table, objects, robot: no sampled overlap
    'robot_table': 1e-4,          # numeric tolerance; the welded base is excluded
    'robot_other': 1e-4,          # robot vs any object it is not grasping
    'finger_grasp': 2.0e-3,       # provisional diagnostic tolerance; not an approved acceptance threshold
}
PLATE_XY = np.array([.28 + SHIFT, .08]); PLATE_HALF = np.array([.07, .06]); CAKE_R = .025
RECEIVE_TOL = PLATE_HALF - CAKE_R
STAY_TOL = .01                    # non-target objects must stay within 1 cm of their start
RETURN_TOL = .02                  # a returned object must be within 2 cm of its start before B is grasped


def split_for(seed):
    # All current variants reuse two source motions: never split jittered siblings.
    return 'train'


def smooth(x):
    x = min(1., max(0., x))
    return x * x * (3 - 2 * x)


def smooth_motion_path(path, radius=3):
    """Suppress frame-scale HY-Motion wrist noise before retargeting to MuJoCo."""
    path = np.asarray(path, float)
    if len(path) < 3:
        return path.copy()
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-.5 * (offsets / max(radius / 1.5, 1)) ** 2)
    weights /= weights.sum()
    padded = np.pad(path, ((radius, radius), (0, 0)), mode='edge')
    return sum(weights[i] * padded[i:i + len(path)] for i in range(len(weights)))


def motion_deviation(path, progress):
    """Keep generated curvature while entering and leaving the gesture at rest."""
    u = min(1., max(0., float(progress)))
    envelope = smooth(u / .18) * smooth((1. - u) / .18)
    return sample(warp(path, np.zeros(3), np.zeros(3)), u) * envelope


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_versions():
    out = {}
    try:
        out['repo_commit'] = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        out['repo_dirty'] = bool(subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip())
    except Exception as exc:   # noqa: BLE001
        out['repo_commit'] = f'unavailable: {exc}'
    sim = ROOT / 'intent_policy'
    digest = hashlib.sha256()
    for file in sorted(list((sim / 'sim').glob('*.py')) + list((sim / 'intent').glob('*.py')) + list((sim / 'benchmark').glob('*.py'))):
        digest.update(file.name.encode()); digest.update(file.read_bytes())
    out['runtime_intent_policy_sha256'] = digest.hexdigest()
    out['collector'] = sha256_file(__file__)
    out['local_source_sha256'] = {str(p.relative_to(ROOT)): sha256_file(p)
                                  for p in sorted(list(ROOT.glob('*.py')) + list((ROOT / 'scripts').glob('*v2*.py')) + list((ROOT / 'scripts').glob('i3_collect.py')))}
    return out


def build_scene(rng):
    scene, positions, xml = make('I3')
    scene['gripper_force_limit_n'] = FORCE_LIMIT_N
    scene['human']['stand_xy'][0] += SHIFT + float(rng.uniform(-.02, .02))
    jitter = {k: rng.uniform(-.008, .008, 2) for k in positions}
    positions = {k: (v[0] + SHIFT + float(jitter[k][0]), v[1] + float(jitter[k][1]), v[2]) for k, v in positions.items()}
    root = ET.fromstring(xml)
    for actuator in root.find('actuator'):
        if actuator.get('name') == 'gripper': actuator.set('forcerange', f'-{FORCE_LIMIT_N:g} {FORCE_LIMIT_N:g}')
    for g in root.findall('./worldbody/geom'):
        if g.get('name', '').startswith('decor_'):
            p = np.fromstring(g.get('pos', '0 0 0'), sep=' '); p[0] += SHIFT; g.set('pos', ' '.join(map(str, p)))
    scene_xml = ET.tostring(root, encoding='unicode')
    from v2_preview_common import apply_station
    stand = [scene['human']['stand_xy'][0], -.54]
    scene_xml = apply_station(scene, scene_xml, stand_xy=stand)
    return scene, positions, scene_xml


def make_env(scene, positions, scene_xml):
    env = ManipulationEnv(model_xml=scene_xml)
    m, d = env.model, env.data
    for name, pos in positions.items():
        q = int(m.joint(name + '_free').qposadr[0]); d.qpos[q:q + 7] = [*pos, 1, 0, 0, 0]
        env.register_graspable(m.body(name).id, m.equality(robot_weld_name(name)).id)
    env.attach_substep = True
    env.hold_squeeze = 0.0
    human = HumanBody({**scene['human'], 'floor_z': scene['floor_z']}); human.bind(m)
    mujoco.mj_forward(m, d)
    return env, human


def geom_sets(m, positions):
    robot_root = m.body('robot_mount').id

    def is_robot(g):
        body = int(m.geom_bodyid[g])
        while body and body != robot_root: body = int(m.body_parentid[body])
        return body == robot_root
    robot = [g for g in range(m.ngeom) if is_robot(g)]
    human = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith('human_seg_')]
    # Include visible cake decoration and table legs as well as collision shapes.
    # The floor is excluded here: I3 has a stationary human and no foot interaction.
    world = [g for g in range(m.ngeom) if not is_robot(g) and g not in human
             and m.geom(g).name != 'floor'
             and (m.geom_rgba[g, 3] > 0 or m.geom_contype[g] or m.geom_conaffinity[g])]
    return is_robot, robot, human, world


def geometry_coverage(m, positions):
    _, robot, human, world = geom_sets(m, positions)
    covered = set(robot + human + world)
    excluded = [{'id': g, 'name': m.geom(g).name,
                 'reason': 'floor: stationary-human I3; foot interaction outside task' if m.geom(g).name == 'floor'
                           else 'invisible non-colliding legacy helper'}
                for g in range(m.ngeom) if g not in covered]
    unexpected = [x for x in excluded if x['name'] != 'floor' and
                  (m.geom_rgba[x['id'], 3] > 0 or m.geom_contype[x['id']] or m.geom_conaffinity[x['id']])]
    return {'review': 'docs/I3_GEOMETRY_REVIEW.md', 'reviewed': not unexpected,
            'robot_geoms': robot, 'human_geoms': human, 'world_geoms': world,
            'excluded': excluded, 'unexpected_exclusions': unexpected,
            'geometry_names': {str(g): m.geom(g).name or m.body(m.geom_bodyid[g]).name for g in range(m.ngeom)}}



# ----------------------------------------------------------------------------- human clip
def generate_clip(seed, case, scene, positions, scene_xml, rng):
    """Deterministic I3 cue for this seed: the generator knows the targets; the policy never does."""
    a_name = str(rng.choice(CAKES))
    b_name = str(rng.choice([c for c in CAKES if c != a_name])) if case != 'nominal' else None
    onset = .25 + float(rng.uniform(0., .4))
    switch_at = {'nominal': None, 'switch_before_commit': 2.5 + float(rng.uniform(-.3, .3)) + onset - .25,
                 'switch_after_commit': float(rng.uniform(6.4, 6.6))}[case]
    source = ROOT / 'outputs/hymotion_v2_candidates/hymotion_out' / ('i3_point_cake.npz' if case == 'nominal' else 'i3_switch.npz')
    with np.load(source, allow_pickle=False) as raw:
        k3d = raw['keypoints3d'][0].copy(); hy_seed = int(raw['seeds'][0]); prompt = str(raw['text'])
    w = to_scene(k3d, np.array([*scene['human']['stand_xy'], 0]), scene['human']['facing_deg'], scene['floor_z'])
    generated = palm_path(w); parts = segment_ranges(generated, tails=True)
    if not parts: raise ValueError('HY-Motion clip lacks usable hand motion')
    first = 1 if case != 'nominal' and len(parts) > 2 else 0
    a, z = parts[first]; motion_path = smooth_motion_path(generated[a:z])
    a2, z2 = parts[first + 1] if len(parts) > first + 1 else parts[first]; motion_path_b = smooth_motion_path(generated[a2:z2])
    source_info = {'model': 'HY-Motion-1.0-Lite', 'seed': hy_seed, 'prompt': prompt, 'clip_id': clip_id(k3d),
                   'frames': [a, z], 'second_frames': [a2, z2], 'file_sha256': sha256_file(source),
                   'retarget': 'HY-Motion palm path smoothed, endpoint-warped and eased to scene; body reconstructed by IK'}
    env, human = make_env(scene, positions, scene_xml)
    try:
        m, d = env.model, env.data
        home = human.hanging_palm(human.active)
        is_robot, robot_g, arms, world_g = geom_sets(m, positions)
        audit = DemoContactAudit(m, [PairRule(x, y, 0., 0.) for x in arms for y in world_g + robot_g])
        check = HumanSelfCheck(m)
        A = np.array(positions[a_name]); B = np.array(positions[b_name]) if b_name else A
        arm_a = np.array([A[0], A[1] - .27, .84]); arm_b = np.array([B[0], B[1] - .27, .84])
        rows = []; first_self = None; first_pair = None
        for tick in range(360):
            t = tick * DT
            reach = smooth((t - onset) / 2.10)
            clear = home + np.array([.12, 0., .03]); pre = smooth((t - onset) / .75)
            switch = 0. if switch_at is None else smooth((t - switch_at) / .8)
            end = (onset + 3.15) if case == 'nominal' else (switch_at + 1.8)
            back = smooth((t - end) / 1.)
            current = home + (clear - home) * pre + (arm_a - clear) * reach + switch * (arm_b - arm_a)
            deviation_a = motion_deviation(motion_path, reach)
            deviation_b = motion_deviation(motion_path_b, reach)
            current = current + (1 - switch) * deviation_a + switch * deviation_b
            palm = (1. - back)**2 * current + 2 * (1. - back) * back * clear + back**2 * home
            direction = A + (B - A) * switch
            active = reach * (1 - back)
            pose = human.apply(d, palm=palm, point_at=direction if active > .001 else None,
                               hand_shape='point', shape_from='relaxed', shape_progress=active, aim_blend=active)
            mujoco.mj_forward(m, d); audit.observe(m, d); hits = check.measure(d)
            if hits and first_self is None: first_self = {'tick': tick, 'pairs': {str(k): float(v) for k, v in hits.items()}}
            if audit.first_violation and first_pair is None: first_pair = {'tick': tick, **audit.first_violation}
            rows.append({'t_s': t, 'palm_command': palm.tolist(), 'point_at_command': direction.tolist() if active > .001 else None,
                         'point_strength': float(active), 'keypoints': human.keypoints(pose, JOINTS).tolist(),
                         'index_ray': human.keypoints(pose, ['r_index_base', 'r_index_tip']).tolist(), 'reach_error_m': pose['reach_error']})
    finally:
        env.close()
    clip = {'case': case, 'seed': seed, 'target_A_evaluator_only': a_name, 'target_B_evaluator_only': b_name,
            'human_switch_schedule_s': switch_at, 'gesture_onset_s': onset, 'source_motion': source_info,
            'first_self': first_self, 'first_pair': first_pair, 'trace': rows}
    clip['clip_sha256'] = hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()
    return clip


# ----------------------------------------------------------------------------- rollout
def rollout(mode, clip, scene, positions, scene_xml, out_dir, predictor, prep_threshold=.02, video=True):
    """Faithful copy of i3_cases_pilot.py (--hymotion --predictive-prep-experiment) with frame capture."""
    env, human = make_env(scene, positions, scene_xml)
    frames = []; trace = []; phases = []; violations = {}; self_violations = {}; substep_contacts = {}
    forecast_frames = []; forecast_runtime = []
    try:
        m, d = env.model, env.data
        is_robot, robot_geoms, human_geoms, world_geoms = geom_sets(m, positions)
        pairs = [PairRule(a, b, 0., 0.) for a in human_geoms for b in robot_geoms + world_geoms]
        pairs += [PairRule(a, b, 0., 0.) for a in robot_geoms for b in world_geoms]
        geometry = DemoContactAudit(m, pairs); self_check = HumanSelfCheck(m)

        def audit_substep(model, data):
            for c in data.contact:
                a, b = int(c.geom1), int(c.geom2)
                if is_robot(a) == is_robot(b): continue
                rg = a if is_robot(a) else b; other = b if is_robot(a) else a
                name = model.geom(other).name; body = model.body(model.geom_bodyid[other]).name
                if not (name == 'table' or name.startswith('decor_') or body in positions): continue
                if name == 'table' and model.body_weldid[model.geom_bodyid[rg]] == 0: continue
                if c.dist >= 0: continue
                key = f'{model.body(model.geom_bodyid[rg]).name}/{model.geom(rg).name or rg}|{body}/{name or other}'
                v = substep_contacts.setdefault(key, {'first_sim_s': float(data.time), 'max_depth_m': 0., 'samples': 0})
                tolerance = overlap_tolerance(
                    robot_body=model.body(model.geom_bodyid[rg]).name, other_body=body,
                    other_geom=name, attached=None if env.attached is None else model.body(env.attached).name,
                    selection=selected, phase=phase, static_robot=False, human_pair=False, tolerances=TOL)
                record_overlap(v, -float(c.dist), tolerance, float(data.time))
        env.physics_observers.append(audit_substep)
        selected = None; pending = None; source_pos = None; phase = 'observe'; stable = []; clear_samples = 0; settled_samples = 0
        stage_start = 0; trajectory_start = None; forecast_prep = None; forecast_prep_target = None
        human_history = []; prep_started = None; prep_from = None
        observed_history = []; observed_prep_started = None; observed_prep_from = None
        early_candidate = None; candidate_history = []
        pace = HumanPace(DT); follower = CartesianFollower(DT, acceleration=WORK_ACCEL)
        continuous_prep_goal = None; prep_follow_started = False
        switched = False   # after an observed change of mind both experts react identically
        visible_still = 0; last_confirmation = None
        midpoint = np.array([.28 + SHIFT, -.10, .82])
        human_clock = 0; human_started = False; human_released = False
        executed_human = []; human_events = []
        hold_tick = int(np.ceil((clip['gesture_onset_s'] + 2.10) / DT))
        return_tick = int(np.ceil((clip['gesture_onset_s'] + 3.15) / DT))
        human_target = np.asarray(positions[clip['target_A_evaluator_only']])
        initial_target_distance = np.linalg.norm(midpoint[:2] - human_target[:2])
        speed_history = []; peak_hand_speed = 0.; speed = None
        prediction_times = np.arange(1, 11) / 10
        attach_log = []; last_attached = None
        writer = VideoWriter(out_dir / 'top.mp4', width=640, height=480) if video else None
        high_writer = VideoWriter(out_dir / 'high.mp4', width=640, height=480) if video else None
        try:
            for tick in range(900):
                # Human responds to visible robot geometry, not a timed withdrawal.
                if not human_started and np.linalg.norm(d.site_xpos[env.tcp] - midpoint) < .025:
                    human_started = True
                    human_events.append({'event': 'robot_neutral_ready', 't_s': tick * DT})
                if human_started:
                    if clip['case'] == 'nominal' and human_clock >= hold_tick and not human_released:
                        distance = np.linalg.norm(d.site_xpos[env.tcp, :2] - human_target[:2])
                        if distance < max(.025, initial_target_distance - .025):
                            human_released = True; human_clock = return_tick
                            human_events.append({'event': 'human_withdraw_after_correct_robot_direction', 't_s': tick * DT})
                    if clip['case'] != 'nominal' or human_clock < hold_tick or human_released:
                        human_clock += 1
                clip_frame = dict(clip['trace'][min(human_clock, len(clip['trace']) - 1)])
                clip_frame['t_s'] = tick * DT
                executed_human.append(clip_frame)
                strength = clip_frame['point_strength']
                pose = human.apply(d, palm=clip_frame['palm_command'], point_at=clip_frame['point_at_command'],
                                   hand_shape='point', shape_from='relaxed', shape_progress=strength, aim_blend=strength)
                mujoco.mj_forward(m, d)
                keypoints = human.keypoints(pose, ['r_index_base', 'r_index_tip'])
                keypoint6 = human.keypoints(pose, JOINTS)
                speed_history = (speed_history + [keypoint6.copy()])[-9:]
                measured_hand_speed = observed_hand_speed(speed_history, 5, DT)
                predicted_hand_speed = 0.
                if mode == 'observed':
                    observed_history = (observed_history + [keypoint6.copy()])[-9:]
                    observed_activity = float(np.mean(np.linalg.norm(keypoint6 - observed_history[0], axis=1)))
                    if observed_prep_started is None and observed_activity > prep_threshold and phase == 'observe':
                        observed_prep_started = tick; observed_prep_from = d.site_xpos[env.tcp].copy()
                        phases.append({'phase': 'observed_neutral_prep', 't': tick * DT})
                if mode == 'intent':
                    from intent_policy.intent.manicast_intent import forecast_xi
                    human_history = (human_history + [keypoint6])[-9:]
                    tic = time.perf_counter(); forecast = forecast_xi(np.stack(human_history), predictor, 20., prediction_times)[-1]
                    forecast_runtime.append(time.perf_counter() - tic); forecast_frames.append(forecast.tolist())
                    predicted_hand_speed = forecast_hand_speed(forecast, 5, prediction_times)
                    signal = float(np.mean(np.linalg.norm(forecast, axis=2)))
                    if prep_started is None and human_started and measured_hand_speed > .015 and phase == 'observe':
                        prep_started = tick; prep_from = d.site_xpos[env.tcp].copy()
                        phases.append({'phase': 'forecast_neutral_prep', 't': tick * DT})
                current = {k: d.xpos[m.body(k).id].copy() for k in positions}
                pace.update(measured_hand_speed, predicted_hand_speed)
                peak_hand_speed = max(peak_hand_speed, measured_hand_speed, predicted_hand_speed)
                if mode == 'intent' and phase == 'observe' and prep_started is not None:
                    # Start continuous reversible motion from the full predicted wrist path.
                    # Candidate confidence only gates the descent, not the first useful motion.
                    wrist_future = keypoint6[5][None, :] + forecast[:, 5, :]
                    weights = prediction_times / prediction_times.sum()
                    projected = wrist_future[:, :2] + np.array([0., .27])
                    scores = {k: float(np.sum(weights * np.linalg.norm(projected - v[:2], axis=1)))
                              for k, v in current.items()}
                    projected_goal = np.sum(weights[:, None] * projected, axis=0)
                    bounds = np.asarray([v[:2] for v in current.values()])
                    projected_goal = np.clip(projected_goal, bounds.min(axis=0), bounds.max(axis=0))
                    proposed = np.r_[projected_goal, .82]
                    if continuous_prep_goal is None: continuous_prep_goal = d.site_xpos[env.tcp].copy()
                    continuous_prep_goal += (1 - np.exp(-DT / .18)) * (proposed - continuous_prep_goal)
                    if not prep_follow_started:
                        prep_follow_started = True
                        phases.append({'phase': 'forecast_continuous_follow', 't': tick * DT})
                    ranked = sorted(scores, key=scores.get)
                    candidate = ranked[0] if len(ranked) > 1 and scores[ranked[1]] - scores[ranked[0]] > .012 else None
                    candidate_history = (candidate_history + [candidate])[-3:]
                    if candidate != early_candidate:
                        early_candidate = candidate
                        phases.append({'phase': 'forecast_route_update', 't': tick * DT, 'item': candidate})
                selection = select_pointed(keypoints[0], keypoints[1], current, max_distance_m=.065, margin_m=.01)
                stable = (stable + [selection])[-5:]
                decided = stable[0] if len(stable) == 5 and stable[0] is not None and all(x == stable[0] for x in stable) else None
                visible_still = visible_still + 1 if measured_hand_speed < .045 and decided is not None else 0
                if decided is not None and decided != last_confirmation:
                    phases.append({'phase': 'pointing_confirmed', 't': tick * DT, 'item': decided})
                    last_confirmation = decided
                if (phase == 'observe' and mode == 'intent' and human_started and
                        len(candidate_history) == 3 and early_candidate is not None and
                        all(c == early_candidate for c in candidate_history) and
                        measured_hand_speed > .025):
                    selected = early_candidate; source_pos = current[selected].copy()
                    phase = 'approach'; stage_start = tick; trajectory_start = d.site_xpos[env.tcp].copy()
                    phases.append({'phase': 'forecast_pick_approach', 't': tick * DT, 'item': selected})
                if phase == 'observe' and decided is not None and (mode == 'intent' or visible_still >= 3):
                    selected = decided; source_pos = current[selected].copy(); phase = 'wait_hand_clear'
                    phases.append({'phase': phase, 't': tick * DT, 'item': selected})
                elif decided is not None and selected is not None and decided != selected and phase in ('wait_hand_clear', 'approach', 'descend', 'close', 'lift', 'move_plate', 'lower_plate', 'open'):
                    switched = True
                    if env.attached is not None:
                        pending = decided; phase = 'return_above'; stage_start = tick; trajectory_start = d.site_xpos[env.tcp].copy()
                        phases.append({'phase': phase, 't': tick * DT, 'item': selected, 'new': pending})
                    else:
                        selected = decided; source_pos = current[selected].copy(); phase = 'wait_hand_clear'; clear_samples = 0
                        phases.append({'phase': phase, 't': tick * DT, 'item': selected, 'switched_from': stable[0]})
                if phase == 'wait_hand_clear':
                    clear_samples = clear_samples + 1 if (np.linalg.norm(keypoint6[5] - current[selected]) > .22) else 0
                    if clear_samples >= 3:
                        phase = 'approach'; stage_start = tick; trajectory_start = d.site_xpos[env.tcp].copy()
                        phases.append({'phase': phase, 't': tick * DT, 'item': selected})
                    elif mode == 'intent' and not switched:
                        predicted = keypoint6[None, :, :] + forecast
                        predicted_distance = np.linalg.norm(predicted[:, 5, :] - current[selected], axis=1)
                        present_distance = np.linalg.norm(keypoint6[5] - current[selected])
                        activity = float(np.mean(np.linalg.norm(forecast, axis=2)))
                        retreat = (np.mean(predicted_distance[-3:]) > present_distance + .03 and activity > .01)
                        if retreat and (forecast_prep_target != selected):
                            forecast_prep = {'start': tick, 'from': d.site_xpos[env.tcp].copy()}
                            forecast_prep_target = selected
                            phases.append({'phase': 'forecast_high_prep', 't': tick * DT, 'item': selected})
                advance_phase = False
                if phase in ('approach', 'descend', 'close', 'lift', 'move_plate', 'lower_plate', 'open', 'retreat', 'return_above', 'return_lower', 'return_open', 'return_retreat'):
                    xyz = current[selected].copy(); xy = xyz[:2]
                    targets = {'approach': ([*xy, .78], 32, .085), 'descend': ([*xy, .667], 27, .085),
                               'close': ([*xy, .667], 32, 0.), 'lift': ([*xy, .78], 29, 0.),
                               'move_plate': ([*PLATE_XY, .78], 38, 0.), 'lower_plate': ([*PLATE_XY, .668], 27, 0.),
                               'open': ([*PLATE_XY, .668], 10, .085), 'retreat': ([*PLATE_XY, .78], 25, .085),
                               'return_above': ([*source_pos[:2], .78], 33, 0.), 'return_lower': ([*source_pos[:2], .667], 27, 0.),
                               'return_open': ([*source_pos[:2], .667], 10, .085), 'return_retreat': ([*source_pos[:2], .78], 25, .085)}
                    goal, count, grip = targets[phase]
                    u = min(1., (tick - stage_start + 1) / count)
                    # Human-paced following only while Intent approaches on a forecast that the
                    # observed cue has not confirmed yet; every other phase is independent work.
                    following = (mode == 'intent' and phase == 'approach' and decided is None and not switched)
                    speed = pace.speed(.24, following=True) if following else work_speed(phase)
                    xyz_cmd = follower.step(d.site_xpos[env.tcp], goal, speed)
                    q = env.solve_ik(xyz_cmd); command = np.r_[q, grip]
                    advance_phase = (u >= 1 if phase in ('close', 'open', 'return_open') else
                                     np.linalg.norm(d.site_xpos[env.tcp] - goal) < .004)
                elif mode == 'intent' and phase == 'wait_hand_clear' and forecast_prep is not None and forecast_prep_target == selected:
                    u = min(1., (tick - forecast_prep['start'] + 1) / 35.)
                    prep_goal = np.r_[current[selected][:2], .91]
                    command = np.r_[env.solve_ik(forecast_prep['from'] + (prep_goal - forecast_prep['from']) * u), .085]
                elif phase == 'observe':
                    # Both experts first stage between the two cake trays.
                    # Intent follows the current ManiCast candidate at high Z;
                    # Observed stays at the shared midpoint until the ray is clear.
                    prep_goal = (continuous_prep_goal if mode == 'intent' and human_started and continuous_prep_goal is not None else midpoint)
                    following = mode == 'intent' and human_started and continuous_prep_goal is not None
                    speed = pace.speed(.24, following=True) if following else work_speed('observe')
                    command = np.r_[env.solve_ik(follower.step(d.site_xpos[env.tcp], prep_goal, speed)), .085]
                else:
                    command = env.hold()
                snapshot = {'timestamp': float(d.time), 'robot_qpos': d.qpos[env.qids].copy(), 'robot_qvel': d.qvel[env.dids].copy(),
                            'gripper': float(d.qpos[env.gid]), 'human_joints': keypoint6.copy(), 'action_target': np.asarray(command, float).copy(),
                            'control_dt': env.dt}
                for camera in ('high', 'top', 'wrist'):
                    snapshot['image_' + camera] = env.render(camera, 256, 256)
                if mode == 'intent':
                    snapshot.update(forecast=forecast.copy(), forecast_valid=np.ones(10, dtype=bool),
                                    forecast_generated_at=float(d.time), forecast_observation_end=float(d.time))
                command_phase, command_selection = phase, selected
                env.step(command)
                if phase == 'close':
                    advance_phase = env.attached is not None
                    if tick - stage_start > 60:
                        phases.append({'phase': 'grasp_failed', 't': tick * DT, 'item': selected})
                        break
                if advance_phase:
                    normal = ['approach', 'descend', 'close', 'lift', 'move_plate', 'lower_plate', 'open', 'retreat', 'settle']
                    undo = ['return_above', 'return_lower', 'return_open', 'return_retreat']
                    if phase == 'return_retreat':
                        selected = pending; pending = None; source_pos = current[selected].copy(); phase = 'wait_hand_clear'; clear_samples = 0
                    elif phase in undo: phase = undo[undo.index(phase) + 1]
                    else: phase = normal[normal.index(phase) + 1]
                    stage_start = tick + 1; trajectory_start = d.site_xpos[env.tcp].copy()
                    phases.append({'phase': phase, 't': (tick + 1) * DT, 'item': selected})
                snapshot['action_executed'] = d.ctrl.copy(); frames.append(snapshot)
                mujoco.mj_forward(m, d); geometry.observe(m, d)
                for index, distance, boundary, allowed in geometry.step_violations:
                    rule = geometry.rules[index]
                    key = f'{m.body(m.geom_bodyid[rule.geom_a]).name}/{m.geom(rule.geom_a).name or rule.geom_a}|{m.body(m.geom_bodyid[rule.geom_b]).name}/{m.geom(rule.geom_b).name or rule.geom_b}'
                    entry = violations.setdefault(key, {'first_t_s': tick * DT, 'max_depth_m': 0., 'samples': 0, 'phase': phase})
                    ga, gb = rule.geom_a, rule.geom_b
                    tolerance = overlap_tolerance(
                        robot_body=m.body(m.geom_bodyid[ga]).name, other_body=m.body(m.geom_bodyid[gb]).name,
                        other_geom=m.geom(gb).name,
                        attached=None if env.attached is None else m.body(env.attached).name,
                        selection=command_selection, phase=command_phase,
                        static_robot=bool(m.body_weldid[m.geom_bodyid[ga]] == 0),
                        human_pair=m.body(m.geom_bodyid[ga]).name.startswith('human_seg_'), tolerances=TOL)
                    record_overlap(entry, -float(distance), tolerance, float(d.time))
                for pair, depth in self_check.measure(d).items():
                    key = str(pair); entry = self_violations.setdefault(key, {'first_t_s': tick * DT, 'max_depth_m': 0., 'samples': 0})
                    entry['max_depth_m'] = max(entry['max_depth_m'], float(depth)); entry['samples'] += 1
                attached = None if env.attached is None else m.body(env.attached).name
                if attached != last_attached:
                    attach_log.append({'t_s': (tick + 1) * DT, 'attached': attached}); last_attached = attached
                if phase == 'settle' and selected is not None:
                    dof = int(m.joint(selected + '_free').dofadr[0]); vel = d.qvel[dof:dof + 6]
                    settled_samples = settled_samples + 1 if np.linalg.norm(vel[:3]) < .01 and np.linalg.norm(vel[3:]) < .1 else 0
                    if settled_samples >= 10:
                        phase = 'done'; phases.append({'phase': 'done', 't': (tick + 1) * DT})
                trace.append({'t_s': tick * DT, 'qpos_for_audit': d.qpos.tolist(), 'observed_selection': selection, 'phase': phase,
                              'observed_hand_speed_m_s': measured_hand_speed, 'forecast_hand_speed_m_s': predicted_hand_speed if mode == 'intent' else None,
                              'robot_command_speed_limit_m_s': speed,
                              'command_speed_m_s': float(np.linalg.norm(follower.velocity)),
                              'human_filtered_speed_m_s': pace.filtered,
                              'selection': selected, 'tcp': d.site_xpos[env.tcp].tolist(),
                              'objects': {k: d.xpos[m.body(k).id].tolist() for k in positions}, 'attached': attached})
                if writer is not None:
                    writer.add(env.render('top', 640, 480)); high_writer.add(env.render('high', 640, 480))
                if phase == 'done': break
        finally:
            if writer is not None: writer.close(); high_writer.close()
        # --- interpolated swept audit on the finished rollout (same env, same geometry rules)
        executed_clip = {**clip, 'trace': executed_human}
        swept = swept_audit(env, human, executed_clip, trace, geometry, self_check)
        # --- camera visibility of the cue and the final target at the strongest gesture sample
        visibility = camera_visibility(env, human, executed_clip, trace, positions)
        result = {'_executed_clip': executed_clip, 'human_events': human_events, 'mode': mode, 'phases': phases, 'last_phase': phase, 'expert_selected': selected, 'attach_log': attach_log,
                  'attached_at_end': env.attached is not None, 'gripper_m': float(d.qpos[env.gid]),
                  'geometry_violations': violations, 'substep_contacts': substep_contacts, 'human_self_violations': self_violations,
                  'swept_audit': swept, 'camera_visibility': visibility, 'geometry_coverage': geometry_coverage(m, positions),
                  'prep_started_s': prep_started * DT if prep_started is not None else None,
                  'forecast_runtime_s': {'mean': float(np.mean(forecast_runtime)), 'max': float(np.max(forecast_runtime))} if forecast_runtime else None,
                  'final_positions': trace[-1]['objects'], 'ticks': len(trace)}
        return result, frames, trace
    finally:
        env.close()


def swept_audit(env, human, clip, trace, geometry, self_check, subdivisions=5):
    m, d = env.model, env.data
    qvel = np.zeros(m.nv); samples = 0; worst_pair = None; worst_self = None; first_pair = None; first_self = None
    for i in range(len(trace) - 1):
        ca, cb = clip['trace'][min(i, len(clip['trace']) - 1)], clip['trace'][min(i + 1, len(clip['trace']) - 1)]
        qa, qb = np.asarray(trace[i]['qpos_for_audit']), np.asarray(trace[i + 1]['qpos_for_audit'])
        mujoco.mj_differentiatePos(m, qvel, 1., qa, qb)
        for j in range(subdivisions):
            u = j / subdivisions
            d.qpos[:] = qa; mujoco.mj_integratePos(m, d.qpos, qvel, u)
            palm = (1 - u) * np.asarray(ca['palm_command']) + u * np.asarray(cb['palm_command'])
            strength = (1 - u) * ca['point_strength'] + u * cb['point_strength']
            da, db = ca['point_at_command'], cb['point_at_command']
            if da is None: da = db
            if db is None: db = da
            direction = None if da is None else (1 - u) * np.asarray(da) + u * np.asarray(db)
            human.apply(d, palm=palm, point_at=direction if strength > .001 else None,
                        hand_shape='point', shape_from='relaxed', shape_progress=strength, aim_blend=strength)
            mujoco.mj_forward(m, d); geometry.observe(m, d); hits = self_check.measure(d)
            t = (i + u) * DT
            for rule_i, distance, _, _ in geometry.step_violations:
                rule = geometry.rules[rule_i]
                entry = {'time_s': t, 'geom_a': m.geom(rule.geom_a).name, 'geom_b': m.geom(rule.geom_b).name, 'signed_distance_m': float(distance),
                         'body_a': m.body(m.geom_bodyid[rule.geom_a]).name, 'body_b': m.body(m.geom_bodyid[rule.geom_b]).name}
                context = trace[i]
                tolerance = overlap_tolerance(
                    robot_body=entry['body_a'], other_body=entry['body_b'],
                    other_geom=entry['geom_b'], attached=context['attached'],
                    selection=context['selection'], phase=context['phase'],
                    static_robot=bool(m.body_weldid[m.geom_bodyid[rule.geom_a]] == 0),
                    human_pair=entry['body_a'].startswith('human_seg_'), tolerances=TOL)
                # Evaluate EACH pair in its own time context; never clear unrelated hits.
                if tolerance is None or -float(distance) <= tolerance:
                    continue
                entry['tolerance_m'] = tolerance
                if first_pair is None: first_pair = entry
                if worst_pair is None or distance < worst_pair['signed_distance_m']: worst_pair = entry
            if hits:
                entry = {'time_s': t, 'pairs': {str(k): float(v) for k, v in hits.items()}}
                if first_self is None: first_self = entry
                worst_self = entry if worst_self is None or max(entry['pairs'].values()) > max(worst_self['pairs'].values()) else worst_self
            samples += 1
    return {'samples': samples, 'sample_period_s': DT / subdivisions, 'first_pair': first_pair, 'worst_pair': worst_pair,
            'first_self': first_self, 'worst_self': worst_self,
            'limitations': ['interpolated configurations, not recorded solver substeps', 'finite sampling is not continuous collision detection',
                            'all human_seg geoms included; self-check has its own coverage']}


def camera_visibility(env, human, clip, trace, positions, size=256):
    """Render the gesture frame and check visible hand/target pixels plus cue projection."""
    m, d = env.model, env.data
    checks = []
    strengths = np.array([f['point_strength'] for f in clip['trace']])
    for target in [clip['target_A_evaluator_only']] + ([clip['target_B_evaluator_only']] if clip['target_B_evaluator_only'] else []):
        # strongest gesture sample while the generator points at this target
        pointing = np.array([f['point_at_command'] is not None and np.allclose(f['point_at_command'][:2], positions[target][:2], atol=.02) for f in clip['trace']])
        if not pointing.any(): checks.append({'target': target, 'visible': False, 'reason': 'no gesture sample at this target'}); continue
        tick = int(np.argmax(np.where(pointing, strengths, -1)))
        tip = np.asarray(clip['trace'][tick]['index_ray'][1])
        obj = np.asarray(trace[min(tick, len(trace) - 1)]['objects'][target])
        frame = clip['trace'][tick]
        d.qpos[:] = trace[min(tick, len(trace) - 1)]['qpos_for_audit']
        human.apply(d, palm=frame['palm_command'], point_at=frame['point_at_command'],
                    hand_shape='point' if frame['point_strength'] > .001 else 'relaxed',
                    aim_blend=frame['point_strength'])
        mujoco.mj_forward(m, d)
        target_geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == m.body(target).id]
        per_camera = {}
        for cam in ('high', 'top'):
            cid = m.camera(cam).id; c = d.cam_xpos[cid]; R = d.cam_xmat[cid].reshape(3, 3)
            half = np.tan(np.deg2rad(m.cam_fovy[cid]) / 2)
            def inside(p):
                q = R.T @ (p - c)
                return bool(q[2] < 0 and abs(q[0] / -q[2]) < half and abs(q[1] / -q[2]) < half)
            env.render(cam, size, size)
            renderer = env.renderers[(size, size)]
            renderer.enable_segmentation_rendering()
            try:
                segmentation = renderer.render().copy()
            finally:
                renderer.disable_segmentation_rendering()
            geom_pixels = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            hand_pixels = int(np.count_nonzero(geom_pixels & (segmentation[..., 0] == human.hand_geom)))
            target_pixels = int(np.count_nonzero(geom_pixels & np.isin(segmentation[..., 0], target_geoms)))
            per_camera[cam] = {'index_tip': inside(tip), 'target': inside(obj),
                               'hand_pixels': hand_pixels, 'target_pixels': target_pixels}
        visible = any(v['index_tip'] and v['target'] and v['hand_pixels'] > 0 and v['target_pixels'] > 0 for v in per_camera.values())
        checks.append({'target': target, 'tick': tick, 't_s': tick * DT, 'cameras': per_camera, 'visible': visible})
    return {'checks': checks, 'passed': all(c['visible'] for c in checks),
            'limitation': 'visible pixels at one strongest frame per target; does not certify cue readability throughout motion'}


# ----------------------------------------------------------------------------- evaluation
def events_from(result, clip):
    phases = result['phases']; ev = {'task_active': 0.0}
    strengths = [f['point_strength'] for f in clip['trace']]
    onset = next((i for i, s in enumerate(strengths) if s > .001), None)
    if onset is not None: ev['gesture_onset'] = onset * DT
    def first(name, **match):
        for p in phases:
            if p['phase'] == name and all(p.get(k) == v for k, v in match.items()): return p['t']
        return None
    final_target = clip['target_B_evaluator_only'] or clip['target_A_evaluator_only']
    ev['target_observed'] = first('pointing_confirmed', item=final_target)
    final_cue = next((f['point_at_command'] for f in reversed(clip['trace']) if f['point_strength'] >= .95), None)
    ev['human_ready'] = next((i * DT for i, f in enumerate(clip['trace']) if f['point_strength'] >= .95 and final_cue is not None and np.linalg.norm(np.asarray(f['point_at_command']) - final_cue) < .02), None)
    ev['robot_ready'] = first('descend', item=final_target)           # TCP above the final target, ready to grasp
    attaches = [a for a in result['attach_log'] if a['attached'] is not None]
    if attaches: ev['robot_commit'] = attaches[0]['t_s']
    for a in attaches:
        if a['attached'] == final_target: ev['robot_commit_final'] = a['t_s']; break
    sw = next((p for p in phases if p['phase'] == 'return_above' or 'switched_from' in p), None)
    if sw is not None: ev['switch_observed'] = sw['t']
    # A visible A-to-B cue is observable even if Observed has not picked anything.
    confirmations = [p for p in phases if p['phase'] == 'pointing_confirmed']
    if clip['target_B_evaluator_only'] is not None:
        seen_a = False
        for cue in confirmations:
            seen_a = seen_a or cue.get('item') == clip['target_A_evaluator_only']
            if seen_a and cue.get('item') == clip['target_B_evaluator_only']:
                ev['switch_observed'] = cue['t']
                break
    ret = first('return_above')
    if ret is not None:
        ev['correction'] = ret
        after = [p['t'] for p in phases if p['phase'] == 'approach' and p['t'] > ret]
        if after: ev['recovery_complete'] = after[0]
    ev['transfer_complete'] = first('retreat')
    if result['last_phase'] == 'done': ev['completed'] = phases[-1]['t']
    for k in ('prep_onset',):
        pass
    prep = next((p for p in phases if p['phase'] in ('forecast_continuous_follow', 'observed_neutral_prep', 'forecast_neutral_prep', 'forecast_high_prep')), None)
    if prep is not None: ev['prep_onset'] = prep['t']
    return {k: v for k, v in ev.items() if v is not None}


def task_success(result, clip, initial, events):
    reasons = []
    a, b = clip['target_A_evaluator_only'], clip['target_B_evaluator_only']; target = b or a
    final = {k: np.asarray(v) for k, v in result['final_positions'].items()}
    if result['last_phase'] != 'done': reasons.append(f"last phase {result['last_phase']}")
    if result['attached_at_end']: reasons.append('object still attached at end')
    if result['expert_selected'] != target: reasons.append(f"expert selected {result['expert_selected']}, evaluator target {target}")
    dxy = np.abs(final[target][:2] - PLATE_XY)
    if not (np.all(dxy < RECEIVE_TOL) and .63 < final[target][2] < .66): reasons.append(f'{target} not in receive tray: {final[target].round(4).tolist()}')
    for k in CAKES:
        if k == target: continue
        if np.linalg.norm(final[k][:2] - np.asarray(initial[k][:2])) > STAY_TOL:
            reasons.append(f'{k} moved {np.linalg.norm(final[k][:2] - np.asarray(initial[k][:2])):.4f} m')
    attached_names = [x['attached'] for x in result['attach_log'] if x['attached'] is not None]
    semantics = []
    if clip['case'] == 'nominal':
        if attached_names != [target]: semantics.append(f'nominal attach sequence {attached_names}')
    elif clip['case'] == 'switch_before_commit':
        if a in attached_names: semantics.append('A was grasped in switch_before_commit')
        if attached_names != [target]: semantics.append(f'SB attach sequence {attached_names}')
        if 'switch_observed' not in events: semantics.append('no switch observed')
    elif clip['case'] == 'switch_after_commit':
        if attached_names[:1] != [a]: semantics.append(f'SA first grasp {attached_names[:1]} is not A={a}')
        elif attached_names != [a, target]: semantics.append(f'SA attach sequence {attached_names}')
        sw = events.get('switch_observed'); commit = events.get('robot_commit')
        if sw is None or commit is None or not (commit < sw): semantics.append(f'switch {sw} not after commit {commit}')
        # A must be back home before B is grasped
        t_b = events.get('robot_commit_final')
        if t_b is not None:
            tr = result['_trace'][min(int(round(t_b / DT)), len(result['_trace']) - 1)]
            back = np.linalg.norm(np.asarray(tr['objects'][a][:2]) - np.asarray(initial[a][:2]))
            if back > RETURN_TOL: semantics.append(f'A {back:.4f} m from its start when B was grasped')
        if 'correction' not in events: semantics.append('no correction phase')
    return {'passed': not reasons and not semantics, 'reasons': reasons, 'semantic_mismatch': semantics}


def geometry_checks(result, clip):
    """Classify every sampled overlap (20 Hz geometry audit + 500 Hz solver contacts) against the tolerances."""
    def is_human(k): return 'human_seg_' in k
    def robot_side(k): return k.split('|')[0]
    def other_side(k): return k.split('|')[1]
    out = {n: {'max_depth_m': 0., 'samples': 0, 'unresolved': []} for n in ('human_table', 'human_object', 'human_robot', 'robot_table', 'robot_object')}
    def put(name, key, depth, samples, tol, context=None):
        o = out[name]; o['max_depth_m'] = max(o['max_depth_m'], depth); o['samples'] += samples
        if context is not None:
            if 'context_violations' not in context or context['context_violations']:
                o['unresolved'].append({'pair': key, 'detail': context.get('first_context_violation', 'missing time-local audit')})
        elif depth > tol:
            o['unresolved'].append({'pair': key, 'max_depth_m': depth, 'tolerance_m': tol})
    for key, v in result['geometry_violations'].items():
        depth, n = v['max_depth_m'], v['samples']
        a, b = key.split('|')
        if is_human(a) or is_human(b):
            other = b if is_human(a) else a
            if 'table' in other or 'decor_' in other: put('human_table', key, depth, n, TOL['human'])
            elif other.split('/')[0] in CAKES: put('human_object', key, depth, n, TOL['human'])
            else: put('human_robot', key, depth, n, TOL['human'])
        else:
            if other_side(key).endswith('/table') or 'decor_' in other_side(key):
                if 'base_link' in robot_side(key) or 'robot_mount' in robot_side(key): continue   # welded base resting on the table
                put('robot_table', key, depth, n, TOL['robot_table'])
            else:
                obj = other_side(key).split('/')[0]
                put('robot_object', key, depth, n, TOL['robot_other'], v)
    for key, v in result['substep_contacts'].items():
        depth, n = v['max_depth_m'], v['samples']
        if other_side(key).endswith('/table') or 'decor_' in other_side(key): put('robot_table', key + '@substep', depth, n, TOL['robot_table'])
        else:
            obj = other_side(key).split('/')[0]
            put('robot_object', key + '@substep', depth, n, TOL['robot_other'], v)
    return out


def quality_report(episode_id, scene_hash, seed, case, result, clip, frames, initial, events):
    geo = geometry_checks(result, clip)
    ticks = result['ticks']; checks = {}; unresolved = []
    for name in ('human_table', 'human_object', 'human_robot', 'robot_table', 'robot_object'):
        g = geo[name]
        checks[name] = {'passed': not g['unresolved'], 'method': 'DemoContactAudit signed distance at 20 Hz on every control tick' + (' plus solver contacts at every physics substep' if name.startswith('robot') else ''),
                        'evidence': f"max sampled overlap {g['max_depth_m']*1000:.3f} mm over {ticks} ticks; tolerance {'per pair' if name=='robot_object' else TOL['human' if name.startswith('human') else name]} m",
                        'samples': ticks, 'max_depth_m': g['max_depth_m'], 'unresolved': g['unresolved']}
        unresolved += [{'check': name, **u} for u in g['unresolved']]
    hs = result['human_self_violations']
    checks['human_self'] = {'passed': not hs, 'method': 'HumanSelfCheck per control tick', 'evidence': f'{len(hs)} self-overlap pairs' + (f': {hs}' if hs else ''), 'samples': ticks}
    if hs: unresolved.append({'check': 'human_self', 'pairs': hs})
    sw = result['swept_audit']
    swept_ok = sw['first_pair'] is None and sw['first_self'] is None
    checks['swept_motion'] = {'passed': swept_ok, 'method': f"interpolated configurations every {sw['sample_period_s']*1000:.0f} ms between control frames (human segments vs table/objects/robot; self)",
                              'evidence': f"{sw['samples']} samples; first pair {sw['first_pair']}; first self {sw['first_self']}", 'samples': sw['samples'], 'detail': sw}
    if not swept_ok: unresolved.append({'check': 'swept_motion', 'first_pair': sw['first_pair'], 'first_self': sw['first_self']})
    ts = task_success(result, clip, initial, events)
    checks['task_success'] = {'passed': ts['passed'], 'method': 'evaluator predicate: final target in receive tray, other cakes within 1 cm, phase done, gripper released, case attach/switch order',
                              'evidence': json.dumps({'reasons': ts['reasons'], 'semantic_mismatch': ts['semantic_mismatch'], 'final_positions': result['final_positions']})}
    hm_ok = clip['first_self'] is None and clip['first_pair'] is None and not ts['semantic_mismatch']
    checks['human_motion'] = {'passed': hm_ok, 'method': 'clip screening (arm vs table/objects/robot, self) at 20 Hz; case order from attach log and switch events',
                              'evidence': f"clip first_pair {clip['first_pair']}, first_self {clip['first_self']}, semantic {ts['semantic_mismatch']}", 'samples': len(clip['trace'])}
    cv = result['camera_visibility']
    checks['camera_visibility'] = {'passed': cv['passed'], 'method': 'gesture-frame hand/target segmentation visibility and index-tip frustum projection in high/top cameras',
                                   'evidence': json.dumps(cv['checks'])}
    try:
        arrays = validate_frames(frames, result['mode'])
        uniform = bool(np.allclose(np.diff(arrays['timestamp']), DT, atol=1e-6)) and len(frames) == ticks
        checks['data_integrity'] = {'passed': uniform, 'method': 'dataset_records.validate_frames; uniform 20 Hz timestamps; frame count equals ticks',
                                    'evidence': f"{len(frames)} frames, dt {DT}, uniform {uniform}, images {arrays['image_high'].shape[1:]}"}
    except ValueError as exc:
        checks['data_integrity'] = {'passed': False, 'method': 'dataset_records.validate_frames', 'evidence': str(exc)}
    return {'episode_id': episode_id, 'scene_hash': scene_hash, 'motion_seed': seed, 'case_id': case, 'task_id': 'I3',
            'geometry_coverage_reviewed': result['geometry_coverage']['reviewed'],
            'geometry_coverage': result['geometry_coverage'],
            'tolerances_m': TOL, 'unresolved_violations': unresolved, 'checks': checks}


# ----------------------------------------------------------------------------- main
def collect(seed, case, out_root, predictor, versions, checkpoint_sha, modes, video):
    rng = np.random.default_rng(seed)
    scene, positions, scene_xml = build_scene(rng)
    scene_hash = hashlib.sha256((scene_xml + json.dumps(positions, sort_keys=True) + json.dumps(scene['human']['stand_xy'])).encode()).hexdigest()
    pair_dir = out_root / 'runs' / f'{case}_s{seed:04d}'; pair_dir.mkdir(parents=True, exist_ok=True)
    clip = generate_clip(seed, case, scene, positions, scene_xml, rng)
    (pair_dir / 'clip.json').write_text(json.dumps({k: v for k, v in clip.items() if k != 'trace'}, indent=2) + '\n')
    pair_id = f'I3_{case}_s{seed:04d}'
    family = f"hym{clip['source_motion']['file_sha256']}"
    summary = {'pair_id': pair_id, 'seed': seed, 'case': case, 'split': split_for(seed), 'scene_hash': scene_hash,
               'human_clip_hash': clip['clip_sha256'], 'targets_evaluator_only': [clip['target_A_evaluator_only'], clip['target_B_evaluator_only']],
               'episodes': {}}
    if clip['first_self'] is not None or clip['first_pair'] is not None:
        summary['rejected'] = {'stage': 'clip_screen', 'first_self': clip['first_self'], 'first_pair': clip['first_pair']}
        (out_root / 'rejected').mkdir(exist_ok=True)
        (out_root / 'rejected' / f'{pair_id}.json').write_text(json.dumps(summary, indent=2) + '\n')
        return summary
    records = {}
    for mode in modes:
        episode_id = f'I3_{case}_{mode}_s{seed:04d}'
        mode_dir = pair_dir / mode; mode_dir.mkdir(exist_ok=True)
        existing = out_root / 'episodes' / episode_id
        if existing.exists():
            saved = json.loads((existing / 'record.json').read_text())
            meta, quality = saved['metadata'], saved['quality']
            if sha256_file(existing / 'frames.npz') != saved['frames_sha256']:
                raise ValueError(f'corrupt existing episode: {episode_id}')
            if quality_errors(quality) or meta['scene_hash'] != scene_hash or meta['human_clip_hash'] != clip['clip_sha256']:
                raise ValueError(f'existing episode incompatible with current pair: {episode_id}')
            records[mode] = saved
            summary['episodes'][mode] = {'episode_id': episode_id, 'gate_errors': [],
                'events': meta['events'], 'outcome': meta['outcome'], 'wall_s': 0.,
                'unresolved': [], 'written': str(existing), 'resumed': True}
            continue
        t0 = time.time()
        result, frames, trace = rollout(mode, clip, scene, positions, scene_xml, mode_dir, predictor, video=video)
        result['_trace'] = trace
        executed_clip = result.pop('_executed_clip')
        realized_human_hash = hashlib.sha256(json.dumps(executed_clip['trace'], separators=(',', ':')).encode()).hexdigest()
        (mode_dir / 'human_trace.json').write_text(json.dumps(executed_clip['trace']))
        events = events_from(result, executed_clip)
        quality = quality_report(episode_id, scene_hash, seed, case, result, executed_clip, frames, positions, events)
        errors = quality_errors(quality)
        result.pop('_trace')
        metadata = {'episode_id': episode_id, 'pair_id': pair_id, 'task_id': 'I3', 'case_id': case, 'scene_hash': scene_hash, 'seed': seed,
                    'human_clip_hash': clip['clip_sha256'], 'realized_human_clip_hash': realized_human_hash,
                    'pairing': 'same source clip, seed and interaction rule; realized withdrawal responds to visible robot motion',
                    'human_interaction_events': result['human_events'], 'motion_family_id': family, 'split': split_for(seed),
                    'human_source': clip['source_motion'], 'source_commit': versions, 'expert_mode': mode,
                    'expert_rule': 'interactive I3: shared neutral prep, observed confirmation or real forecast approach, contingent human withdrawal',
                    'forecast_source': 'manicast' if mode == 'intent' else 'none', 'events': events, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    'outcome': 'success' if quality['checks']['task_success']['passed'] else 'failure',
                    'evaluator_only': {'target_A': clip['target_A_evaluator_only'], 'target_B': clip['target_B_evaluator_only'],
                                       'switch_schedule_s': clip['human_switch_schedule_s'], 'gesture_onset_s': clip['gesture_onset_s']},
                    'control_hz': 20, 'cameras': ['high', 'top', 'wrist'], 'image_hw': [256, 256]}
        if mode == 'intent':
            metadata.update(predictor_version='ManiCast loaded from pif.yaml v6_expert.checkpoint_path; checkpoint identified by SHA-256',
                            checkpoint_sha256=checkpoint_sha, forecast_horizons_s=HORIZONS, forecast_joint_order=JOINTS,
                            forecast_runtime_s=result['forecast_runtime_s'])
        (mode_dir / 'rollout.json').write_text(json.dumps({'result': result, 'events': events, 'quality': quality, 'gate_errors': errors, 'trace': trace}, indent=1) + '\n')
        entry = {'episode_id': episode_id, 'gate_errors': errors, 'events': events, 'outcome': metadata['outcome'],
                 'wall_s': round(time.time() - t0, 1), 'ticks': result['ticks'], 'unresolved': quality['unresolved_violations']}
        if errors:
            # Preserve expensive RGB/state/action capture for diagnosis, never in episodes/.
            # This does not allow bypassing a failed gate or training on rejected runs.
            try:
                debug_arrays = validate_frames(frames, mode)
                np.savez_compressed(mode_dir / 'rejected_frames.npz', **debug_arrays)
                entry['diagnostic_frames'] = str(mode_dir / 'rejected_frames.npz')
            except ValueError as exc:
                entry['diagnostic_capture_error'] = str(exc)
            (out_root / 'rejected').mkdir(exist_ok=True)
            (out_root / 'rejected' / f'{episode_id}.json').write_text(json.dumps({'metadata': metadata, 'quality': quality, 'gate_errors': errors}, indent=2) + '\n')
            entry['written'] = None
        else:
            dest = write_episode(out_root / 'episodes', metadata, frames, quality)
            entry['written'] = str(dest.resolve())
        records[mode] = {'metadata': metadata, 'quality': quality}
        summary['episodes'][mode] = entry
        del frames
    if all(m in records and not summary['episodes'][m]['gate_errors'] for m in ('observed', 'intent')):
        pm = {'task_id': 'I3', 'case_id': case, 'scene_hash': scene_hash, 'human_clip_hash': clip['clip_sha256'], 'seed': seed}
        obs = {**pm, 'forecast_source': 'none', 'events': records['observed']['metadata']['events'], 'success': records['observed']['quality']['checks']['task_success']['passed']}
        inte = {**pm, 'forecast_source': 'manicast', 'events': records['intent']['metadata']['events'], 'success': records['intent']['quality']['checks']['task_success']['passed']}
        summary['pair_metrics'] = compare_pair(obs, inte)
    (pair_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', choices=['nominal', 'switch_before_commit', 'switch_after_commit'], required=True)
    p.add_argument('--seeds', type=int, nargs='+', required=True, help='explicit seeds, or two values START END for a range')
    p.add_argument('--range', action='store_true', help='interpret --seeds as START END (END exclusive)')
    p.add_argument('--modes', nargs='+', default=['observed', 'intent'], choices=['observed', 'intent'])
    p.add_argument('--out', type=Path, default=ROOT / 'outputs/v2_staging/I3')
    p.add_argument('--no-video', action='store_true')
    args = p.parse_args()
    args.out = args.out.resolve()
    seeds = list(range(args.seeds[0], args.seeds[1])) if args.range else args.seeds
    args.out.mkdir(parents=True, exist_ok=True)
    versions = source_versions()
    checkpoint_sha = None
    predictor = None
    if 'intent' in args.modes:
        from intent_policy.intent.manicast_intent import load_forecaster
        from pif.config import load_config
        from pif.utils import resolve
        config = load_config('configs/pif/pif.yaml', [])
        checkpoint_sha = sha256_file(resolve(config['v6_expert']['checkpoint_path']))
        predictor = load_forecaster()
    log = (args.out / 'summary.jsonl').open('a')
    for seed in seeds:
        try:
            summary = collect(seed, args.case, args.out, predictor, versions, checkpoint_sha, args.modes, not args.no_video)
        except Exception as exc:   # noqa: BLE001
            summary = {'pair_id': f'I3_{args.case}_s{seed:04d}', 'seed': seed, 'case': args.case, 'error': repr(exc)}
            print(f'seed {seed} error: {exc!r}', file=sys.stderr, flush=True)
        log.write(json.dumps(summary, default=str) + '\n'); log.flush()
        short = {k: v for k, v in summary.items() if k in ('pair_id', 'rejected', 'error', 'targets_evaluator_only')}
        short['episodes'] = {m: {'gate_errors': e['gate_errors'], 'wall_s': e['wall_s'], 'outcome': e['outcome'], 'written': bool(e['written'])} for m, e in summary.get('episodes', {}).items()}
        short['pair_metrics'] = summary.get('pair_metrics')
        print(json.dumps(short), flush=True)
    log.close()


if __name__ == '__main__':
    main()
