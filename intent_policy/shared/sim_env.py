"""Simulation set-up shared by every HRC task: station config, MuJoCo env + human, geometry sets, motion
smoothing helpers. Moved from scripts/i3_collect.py and scripts/v2_preview_common.py (QĐ 53/QĐ-DL)."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from intent_policy.sim.base_env import ManipulationEnv
from intent_policy.sim.human_body_v2 import HumanBody
from intent_policy.sim.scene_builder_v2 import robot_weld_name, look_at_xyaxes
from intent_policy.sim.motion_bank_v2 import warp, sample

ROOT = Path(__file__).resolve().parents[2]
STATION = yaml.safe_load((ROOT / 'configs/hrc/station.yaml').read_text())
CONTROL = STATION['controller']
INTENT = STATION['intent']
RECORDED_CAMERAS = STATION['recorded_cameras']                                         # 68/QĐ-DL A
WRIST_CAMERA = yaml.safe_load((ROOT / 'configs/scene/common.yaml').read_text())['cameras']['wrist']

DT = float(CONTROL['dt_s'])
JOINTS = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist']   # ManiCast joints (J = 6)
HORIZONS = np.asarray(INTENT['horizons_s'], float)
BASE_XY = tuple(float(v) for v in STATION['robot_base_xy'])
GRIP_OPEN, GRIP_CLOSED = .085, 0.
TRAVEL_Z = float(CONTROL['travel_z_m'])
GRASP_DZ = float(CONTROL['grasp_dz_m'])
ROBOT_REACH = float(CONTROL['robot_reach_m'])


def robot_reaches(xy):
    """Static workspace geometry: the object lies within the robot's horizontal reach from its base."""
    return float(np.hypot(xy[0] - BASE_XY[0], xy[1] - BASE_XY[1])) <= ROBOT_REACH
PALM_TAU = .08            # s, palm command smoothing (shared human rule)
TOL = {                   # metres, contact audit tolerances (docs/PLAN_AUDIT_I3_PIPELINE.md)
    'human': 0.0,
    'robot_table': 1e-4,
    'robot_other': 1e-4,
    'finger_grasp': 2.0e-3,
}


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


def apply_station(scene, xml, stand_xy=None, cam=None):
    """Shared station: robot base across the table facing the human, human stand, continuous skin,
    side overview (`high`, review only; `cam=` overrides it), back camera and the recorded cameras (68/QĐ-DL, fixed).
    Returns the edited scene XML."""
    from intent_policy.sim.scene_builder_v2 import add_continuous_human_skin, ROOT as SIM_ROOT
    scene['human']['stand_xy'] = list(stand_xy if stand_xy is not None else STATION['human_stand_xy'])
    scene['human']['continuous_skin'] = bool(STATION['continuous_skin'])
    root = ET.fromstring(xml)
    if scene['human']['continuous_skin'] and scene['human'].get('model_version') not in ('v3', 'v4') \
            and root.find("./asset/skin[@name='human_surface']") is None:
        add_continuous_human_skin(root, SIM_ROOT / 'assets/human')
    cam = cam or STATION['camera_high']
    set_camera(root, 'high', cam)
    set_camera(root, 'back', STATION['camera_back'])
    mount = root.find("./worldbody/body[@name='robot_mount']")
    mount.set('pos', f'{BASE_XY[0]} {BASE_XY[1]} 0.62')
    half = np.radians(float(STATION['robot_base_yaw_deg'])) / 2
    mount.set('quat', f'{np.cos(half):.7f} 0 0 {np.sin(half):.7f}')
    fix_recorded_cameras(root)                      # 68/QĐ-DL: last, so `cam=` / scene overrides cannot change them
    return ET.tostring(root, encoding='unicode')


def set_camera(root, name, cam):
    """Create or overwrite a named world camera from {pos, target, fovy}."""
    el = root.find(f"./worldbody/camera[@name='{name}']")
    if el is None:
        el = ET.SubElement(root.find('worldbody'), 'camera', name=name)
    el.set('pos', ' '.join(str(v) for v in cam['pos']))
    el.set('xyaxes', look_at_xyaxes(cam['pos'], cam['target']))
    el.set('fovy', str(cam['fovy']))


def fix_recorded_cameras(root):
    """68/QĐ-DL A: (re)set the recorded cameras from config, identical for every task: the world cameras of
    station.yaml `recorded_cameras` and the gripper `wrist` camera of configs/scene/common.yaml. Called last
    (apply_station, ensure_review_cameras via Episode), so per-task scene / `cam=` overrides cannot change them."""
    for name, cam in RECORDED_CAMERAS.items():
        set_camera(root, name, cam)
    w = WRIST_CAMERA
    el = root.find(".//camera[@name='wrist']")
    body = root.find(f".//body[@name='{w['body']}']")
    if el is not None and body is not None and el not in list(body):
        root.find(".//camera[@name='wrist']/..").remove(el); el = None
    if body is None:
        return
    if el is None:
        el = ET.SubElement(body, 'camera', name='wrist')
    el.set('pos', ' '.join(str(float(v)) for v in w['pos']))
    el.set('xyaxes', look_at_xyaxes(w['pos'], w['target'], w.get('up', (0.0, 0.0, 1.0))))
    el.set('fovy', str(w['fovy']))
    for k in ('quat', 'axisangle', 'euler', 'zaxis', 'mode', 'target'):
        el.attrib.pop(k, None)


def ensure_review_cameras(xml):
    """Every task (also those with their own station) gets the shared back and top review cameras, then the fixed
    recorded cameras (68/QĐ-DL), set last."""
    root = ET.fromstring(xml)
    if root.find("./worldbody/camera[@name='back']") is None:
        set_camera(root, 'back', STATION['camera_back'])
    set_camera(root, 'top', STATION['camera_top'])
    fix_recorded_cameras(root)
    return ET.tostring(root, encoding='unicode')


def shift_decor(xml, name_prefix, dx=0., dy=0.):
    """Move static decor geoms (trays) in the scene XML by (dx, dy)."""
    root = ET.fromstring(xml)
    for g in root.iter('geom'):
        if g.get('name', '').startswith('decor_' + name_prefix + '_'):
            p = np.fromstring(g.get('pos'), sep=' '); p[0] += dx; p[1] += dy
            g.set('pos', ' '.join(f'{v:.5f}' for v in p))
    return ET.tostring(root, encoding='unicode')


def connected_human(scene, xml):
    """Continuous HY-Motion skin for the v2 mannequin (v3/v4 bind their own skin)."""
    from intent_policy.sim.scene_builder_v2 import add_continuous_human_skin
    scene['human']['continuous_skin'] = True
    if scene['human'].get('model_version') in ('v3', 'v4'):
        return xml
    root = ET.fromstring(xml)
    add_continuous_human_skin(root, ROOT / 'assets/human')
    return ET.tostring(root, encoding='unicode')


def make_env(scene, positions, scene_xml):
    env = ManipulationEnv(model_xml=scene_xml)
    m, d = env.model, env.data
    solver = STATION.get('contact_solver')
    if solver:
        solref = np.asarray(solver['solref'], float)
        if solref.shape != (2,) or solref[0] < 2 * m.opt.timestep or solref[1] <= 0:
            raise ValueError('contact solref requires timeconst >= 2*timestep and positive damping')
        m.geom_solref[:] = solref
    for name, pos in positions.items():
        q = int(m.joint(name + '_free').qposadr[0]); d.qpos[q:q + 7] = [*pos, 1, 0, 0, 0]
        env.register_graspable(m.body(name).id, m.equality(robot_weld_name(name)).id)
    env.attach_substep = True
    env.hold_squeeze = 0.0
    if scene['human'].get('model_version') == 'v4':
        from intent_policy.sim.human_physics import PhysicalHuman
        human = PhysicalHuman({**scene['human'], 'floor_z': scene['floor_z']})
    elif scene['human'].get('model_version') == 'v3':
        from intent_policy.sim.human_v3 import HumanV3
        human = HumanV3({**scene['human'], 'floor_z': scene['floor_z']})
    else:
        human = HumanBody({**scene['human'], 'floor_z': scene['floor_z']})
    human.bind(m)
    if scene['human'].get('model_version') == 'v4':
        env.physics_controllers.append(human.control)
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
    world = [g for g in range(m.ngeom) if not is_robot(g) and g not in human
             and m.geom(g).name != 'floor'
             and (m.geom_rgba[g, 3] > 0 or m.geom_contype[g] or m.geom_conaffinity[g])]
    return is_robot, robot, human, world
