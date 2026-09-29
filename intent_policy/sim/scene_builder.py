"""Build an in-memory MJCF scene for one scenario from its (merged) scene configuration.

The UR3e/SusGrip robot, table and physics options come unchanged from `assets/scene.xml`;
this module only adds task objects, target regions, bowls, the scripted human proxy, the
instruction board and cameras. Object poses/colours are set per episode by the scenario.
"""
import xml.etree.ElementTree as ET
import numpy as np
from intent_policy.sim.base_env import ROOT

BOARD_SHAPES = {'box': '0.036 0.036 0.036', 'cylinder': '0.036 0.036', 'sphere': '0.038', 'cross': '0.048 0.014 0.018'}


def _fmt(values) -> str:
    return ' '.join(f'{float(v):.6g}' for v in np.atleast_1d(values))


def _sub(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: v if isinstance(v, str) else _fmt(v) for k, v in attrs.items()})


def look_at_xyaxes(pos, target, up=(0.0, 0.0, 1.0)) -> str:
    """MuJoCo camera looks along -z_cam; return 'x_cam y_cam' axes for a look-at camera."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    if np.linalg.norm(x) < 1e-6:          # looking straight down: image x = world +X
        x = np.cross(f, (0.0, 1.0, 0.0))
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return _fmt(np.r_[x, y])


def resolve_color(spec, palette) -> list[float]:
    return list(palette[spec]) if isinstance(spec, str) else list(spec)


def object_half_height(obj: dict) -> float:
    shape, size = obj['shape'], obj['size']
    return float(size[-1] if shape in ('box', 'cylinder', 'cross') else size[0])


def object_geoms(key: str, obj: dict) -> list[dict]:
    """Geoms of one task object. 'cross' = two crossing boxes (size: arm half-length,
    arm half-width, half-height); the first geom is named `<key>_geom` and carries the colour."""
    if obj['shape'] == 'cross':
        a, w, h = obj['size']
        return [dict(name=f'{key}_geom', type='box', size=[a, w, h]),
                dict(name=f'{key}_geom2', type='box', size=[w, a, h])]
    return [dict(name=f'{key}_geom', type=obj['shape'], size=obj['size'])]


GRIPPER_BODY = 'sus2f_base_link'


def weld_name(body: str) -> str:
    return f'human_grasp_{body}'


def robot_weld_name(body: str) -> str:
    return f'robot_grasp_{body}'


def build_scene_xml(scene: dict) -> str:
    """Return MJCF text. `scene` is the scenario `scene` section merged with its base file."""
    palette = scene['palette']
    tree = ET.parse(ROOT / scene['base_model'])
    root = tree.getroot()
    root.find('compiler').set('meshdir', str(ROOT / 'assets/meshes'))
    world = root.find('worldbody')
    for name in ('cube', 'receiver'):
        body = world.find(f"body[@name='{name}']")
        if body is not None:
            world.remove(body)
    for cam in world.findall('camera'):
        world.remove(cam)
    option = root.find('option')
    servo = scene.get('robot_servo')
    if servo:
        for act in root.find('actuator'):
            if act.get('name') != 'gripper':
                act.set('kp', str(servo['kp']))
                act.set('kv', str(servo['kv']))
    equality = root.find('equality')
    for weld in equality.findall('weld'):
        equality.remove(weld)
    top = float(scene['table_top_z'])
    # Multi-point convex contacts: flat finger pads on curved objects otherwise get a single
    # contact point and objects spin out of the grasp.
    flag = option.find('flag') if option.find('flag') is not None else ET.SubElement(option, 'flag')
    flag.set('multiccd', 'enable')

    # Task objects: free bodies, physical, graspable. Pose set by the scenario at reset.
    for i, (key, obj) in enumerate(scene.get('objects', {}).items()):
        body = _sub(world, 'body', name=key, pos=[0.1 * i, -0.4, top + object_half_height(obj) + 0.002])
        _sub(body, 'freejoint', name=f'{key}_free')
        geoms = object_geoms(key, obj)
        for g in geoms:
            _sub(body, 'geom', **g, mass=str(obj.get('mass', 0.06) / len(geoms)),
                 rgba=resolve_color(obj['color'], palette), contype='2', conaffinity='3',
                 friction=obj.get('friction', '2.0 0.01 0.001'), condim=str(obj.get('condim', 4)))

    # Flat square target regions: visual only (no collision), moved per episode via mocap.
    for key, region in scene.get('regions', {}).items():
        body = _sub(world, 'body', name=key, mocap='true', pos=[0.4, 0, top])
        h = float(region['half_size'])
        _sub(body, 'geom', name=f'{key}_geom', type='box', size=[h, h, 0.0012], pos=[0, 0, 0.0012],
             rgba=resolve_color(region['color'], palette), contype='0', conaffinity='0')

    # Open bowls with a handle pointing to -X (towards the robot) so the top-down gripper,
    # which closes along world Y, can grasp the handle.
    for key, bowl in scene.get('bowls', {}).items():
        r = float(bowl.get('radius', 0.05))
        rgba = resolve_color(bowl['color'], palette)
        body = _sub(world, 'body', name=key, pos=[0.3, 0, top + 0.005])
        _sub(body, 'freejoint', name=f'{key}_free')
        _sub(body, 'geom', name=f'{key}_bottom', type='cylinder', size=[r + 0.004, 0.004], mass='0.06', rgba=rgba,
             contype='2', conaffinity='3', friction='1.5 0.01 0.001')
        for j in range(16):
            a = 2 * np.pi * j / 16
            _sub(body, 'geom', name=f'{key}_wall_{j}', type='box', pos=[r * np.cos(a), r * np.sin(a), 0.022],
                 size=[0.004, 0.011, 0.02], euler=[0, 0, a], mass='0.003', rgba=rgba, contype='2', conaffinity='3')
        _sub(body, 'geom', name=f'{key}_handle', type='box', pos=[-(r + 0.03), 0, 0.02], size=[0.028, 0.012, 0.012],
             mass='0.012', rgba=rgba, contype='2', conaffinity='3', friction='2.0 0.01 0.001', condim='4')

    # Scripted human proxy. All human geoms are visual only; contact/safety are measured with
    # exact geometric distances, so the mocap hand never pushes the robot physically.
    human = scene['human']
    skin, shirt = '0.85 0.63 0.46 1', '0.25 0.36 0.52 1'
    torso = _sub(world, 'body', name='human_torso', pos=human['torso_pos'])
    _sub(torso, 'geom', name='human_torso_geom', type='box', size=[0.08, 0.18, 0.2], rgba=shirt, contype='0', conaffinity='0')
    _sub(torso, 'geom', name='human_head', type='sphere', pos=[0, 0, 0.3], size='0.085', rgba=skin, contype='0', conaffinity='0')
    hand = _sub(world, 'body', name='human_hand', mocap='true', pos=human['hand_rest'])
    _sub(hand, 'geom', name='human_palm', type='sphere', size=_fmt(human['hand_geom_radius']), rgba=skin, contype='0', conaffinity='0')
    forearm = _sub(world, 'body', name='human_forearm', mocap='true', pos=human['hand_rest'])
    _sub(forearm, 'geom', name='human_forearm_geom', type='capsule', size=[0.024, human['forearm_length'] / 2],
         rgba=skin, contype='0', conaffinity='0')
    # Invisible anchor used to weld an object to the human hand (human grasp/hold).
    _sub(world, 'body', name='human_grasp_anchor', mocap='true', pos=human['hand_rest'])
    for key in list(scene.get('objects', {})) + list(scene.get('bowls', {})):
        _sub(equality, 'weld', name=weld_name(key), body1='human_grasp_anchor', body2=key, active='false',
             relpose='0 0 0 1 0 0 0', solref='0.01 1')
        # Assisted robot grasp (see ManipulationEnv): activated only on two-finger contact.
        _sub(equality, 'weld', name=robot_weld_name(key), body1=GRIPPER_BODY, body2=key, active='false',
             relpose='0 0 0 1 0 0 0', solref='0.005 1', solimp='0.95 0.99 0.001')

    # Instruction board: the human's instruction is rendered as public, camera-visible geometry.
    board = scene.get('instruction_board')
    if board:
        b = _sub(world, 'body', name='instruction_board', pos=board['pos'])
        _sub(b, 'geom', name='board_panel', type='box', size=[0.15, 0.006, 0.11], rgba='0.95 0.95 0.95 1',
             contype='0', conaffinity='0')
        for shape, size in BOARD_SHAPES.items():
            if shape == 'cross':    # drawn as a flat plus sign on the panel
                for i, sz in enumerate(('0.044 0.006 0.013', '0.013 0.006 0.044')):
                    _sub(b, 'geom', name=f'board_obj_cross{i}', type='box', size=sz, pos=[-0.065, -0.014, 0.0],
                         rgba='0 0 0 0', contype='0', conaffinity='0')
                continue
            _sub(b, 'geom', name=f'board_obj_{shape}', type=shape, size=size, pos=[-0.065, -0.045, 0.0],
                 rgba='0 0 0 0', contype='0', conaffinity='0')
        _sub(b, 'geom', name='board_target', type='box', size=[0.048, 0.006, 0.048], pos=[0.07, -0.012, 0.0],
             rgba='0 0 0 0', contype='0', conaffinity='0')

    for name, cam in scene['cameras'].items():
        old_parent = root.find(f".//camera[@name='{name}']/..")          # e.g. the asset's wrist camera
        if old_parent is not None:
            old_parent.remove(old_parent.find(f"camera[@name='{name}']"))
        # world camera by default; `body:` mounts it on a robot body (pos/target/up in that body's frame)
        parent = world if 'body' not in cam else root.find(f".//body[@name='{cam['body']}']")
        _sub(parent, 'camera', name=name, pos=cam['pos'],
             xyaxes=look_at_xyaxes(cam['pos'], cam['target'], cam.get('up', (0.0, 0.0, 1.0))), fovy=str(cam['fovy']))
    _sub(world, 'light', name='fill', pos='0.5 -0.6 2.0', dir='0 0.3 -1', directional='true', diffuse='0.35 0.35 0.35')
    ET.indent(tree, space='  ')
    return ET.tostring(root, encoding='unicode')
