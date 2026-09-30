"""Build an in-memory MJCF scene for one scenario from its (merged) scene configuration.

The UR3e/SusGrip robot, table and physics options come unchanged from `assets/scene.xml`;
this module adds the task objects (cubes, cups), the zone tape markers of configs/layout.yaml, the human
mannequin (HY-Motion wooden model segments + invisible distance proxies), cameras, lights, the floor height
and table legs. Object poses are set per episode by the scenario.
"""
import json
import xml.etree.ElementTree as ET
import numpy as np
from intent_policy.sim.base_env import ROOT

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
    if shape == 'cup':
        return float(size[1]) / 2
    return float(size[-1] if shape in ('box', 'cylinder', 'cross') else size[0])


def object_geoms(key: str, obj: dict) -> list[dict]:
    """Geoms of one task object. 'cross' = two crossing boxes (size: arm half-length,
    arm half-width, half-height); the first geom is named `<key>_geom` and carries the colour."""
    if obj['shape'] == 'cross':
        a, w, h = obj['size']
        return [dict(name=f'{key}_geom', type='box', size=[a, w, h]),
                dict(name=f'{key}_geom2', type='box', size=[w, a, h])]
    if obj['shape'] == 'cup':            # open thin-walled cup (size: outer radius, height), body origin at mid-height
        r, h = obj['size']
        t = float(obj.get('wall', 0.004))
        geoms = [dict(name=f'{key}_geom', type='cylinder', size=[r, t / 2], pos=[0, 0, -h / 2 + t / 2])]
        for j in range(16):
            a = 2 * np.pi * j / 16
            geoms.append(dict(name=f'{key}_wall_{j}', type='box', size=[t / 2, np.pi * r / 16 * 1.1, h / 2],
                              pos=[(r - t / 2) * np.cos(a), (r - t / 2) * np.sin(a), 0.0], euler=[0, 0, a]))
        return geoms
    return [dict(name=f'{key}_geom', type=obj['shape'], size=obj['size'])]


GRIPPER_BODY = 'sus2f_base_link'


def weld_name(body: str) -> str:
    return f'human_grasp_{body}'


def robot_weld_name(body: str) -> str:
    return f'robot_grasp_{body}'


MARKER_RGBA = {'place': '0.20 0.45 0.90 1', 'hand': '0.95 0.55 0.10 1', 'support': '0.15 0.70 0.30 1',
               'human_store': '0.55 0.55 0.55 1', 'slot': '0.10 0.10 0.10 1'}


def load_layout(scene: dict) -> dict | None:
    import yaml
    return yaml.safe_load((ROOT / scene['layout']).read_text()) if scene.get('layout') else None


def _add_zone_markers(world, layout: dict, top: float) -> None:
    """Tape outlines for P/H/U zones and a small cross at each S slot (visual only, flush with the table); zones with
    `marker: false` (U_cup, a part of U) have none."""
    t, h = 0.008, 0.0006                       # tape width, half thickness
    for name, z in layout['zones'].items():
        if not z.get('marker', True):
            continue
        x, y = z['xy']
        rgba = MARKER_RGBA[z['kind']]
        if z['kind'] == 'slot':
            for i, size in enumerate(([0.015, 0.003, h], [0.003, 0.015, h])):
                _sub(world, 'geom', name=f'marker_{name}_{i}', type='box', size=size, pos=[x, y, top + h], rgba=rgba,
                     contype='0', conaffinity='0')
            continue
        sx, sy = np.broadcast_to(z.get('size', layout['marker_size']), 2) / 2
        for i, (dx, dy, hx, hy) in enumerate(((0, sy, sx, t / 2), (0, -sy, sx, t / 2), (sx, 0, t / 2, sy), (-sx, 0, t / 2, sy))):
            _sub(world, 'geom', name=f'marker_{name}_{i}', type='box', size=[hx, hy, h], pos=[x + dx, y + dy, top + h],
                 rgba=rgba, contype='0', conaffinity='0')


def human_mesh_name(segment: str, variant: str | None = None) -> str:
    return f'human_{segment}' + (f'_{variant}' if variant else '')


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
        body = _sub(world, 'body', name=key, pos=[0.1 * i, 0.45, top + object_half_height(obj) + 0.002])
        _sub(body, 'freejoint', name=f'{key}_free')
        geoms = object_geoms(key, obj)
        for g in geoms:
            _sub(body, 'geom', **g, mass=str(obj.get('mass', 0.06) / len(geoms)),
                 rgba=resolve_color(obj['color'], palette), contype='2', conaffinity='3',
                 friction=obj.get('friction', '2.0 0.01 0.001'), condim=str(obj.get('condim', 4)))

    # Human: the HY-Motion wooden mannequin as rigid, visual-only mocap segments (posed every tick by
    # HumanBody) plus invisible capsule/sphere proxies for the human-robot distance and contact checks.
    # Mocap bodies never push the robot physically.
    human = scene['human']
    model = ROOT / human.get('model', 'assets/human/skeleton.json')
    skel = json.loads(model.read_text())
    asset = root.find('asset')
    _sub(asset, 'texture', name='human_wood', type='2d', file=str(model.parent / skel['texture']))
    _sub(asset, 'material', name='human_wood', texture='human_wood', specular='0.15', shininess='0.2')
    for seg, info in skel['segments'].items():
        # hands come in shape variants (relaxed / point / grasp); the scenario switches the geom's mesh at runtime
        for variant, f in info.get('variants', {None: info['mesh']}).items():
            _sub(asset, 'mesh', name=human_mesh_name(seg, variant), file=str(model.parent / f), inertia='shell')
        default = next((k for k, f in info.get('variants', {}).items() if f == info['mesh']), None)
        body = _sub(world, 'body', name=f'human_seg_{seg}', mocap='true', pos=[2.0, 0.0, 0.0])
        _sub(body, 'geom', name=f'human_seg_{seg}_mesh', type='mesh', mesh=human_mesh_name(seg, default),
             material='human_wood', contype='0', conaffinity='0')
        if seg == 'head' and human.get('face_markers'):
            # optional eyes + nose so the head orientation (gaze) reads in the images; the mannequin face is blank.
            # Positions in the head segment frame (neck pivot, T-pose axes: face towards -X, left = -Y).
            for name, gtype, size, pos, rgba in (('eye_l', 'sphere', [0.012], [-0.099, -0.032, 0.150], '0.08 0.05 0.03 1'),
                                                 ('eye_r', 'sphere', [0.012], [-0.099, 0.032, 0.150], '0.08 0.05 0.03 1'),
                                                 ('nose', 'sphere', [0.016], [-0.108, 0.0, 0.118], '0.62 0.42 0.28 1')):
                _sub(body, 'geom', name=f'human_face_{name}', type=gtype, size=size, pos=pos, rgba=rgba,
                     contype='0', conaffinity='0')
    J, S = skel['joints'], human.get('active_arm', 'r').upper()
    length = lambda a, b: float(np.linalg.norm(np.subtract(J[b], J[a])))
    radius, hidden = human['proxy_radius'], '0 0 0 0'
    for name, gtype, size in (('hand', 'sphere', [radius['palm']]),
                              ('forearm', 'capsule', [radius['forearm'], length(f'{S}_Elbow', f'{S}_Wrist') / 2]),
                              ('upperarm', 'capsule', [radius['upperarm'], length(f'{S}_Shoulder', f'{S}_Elbow') / 2]),
                              ('torso', 'capsule', [radius['torso'], length('Spine1', 'Neck') / 2]),
                              ('head', 'sphere', [radius['head']])):
        body = _sub(world, 'body', name=f'human_{name}', mocap='true', pos=human['hand_rest'])
        _sub(body, 'geom', name='human_palm' if name == 'hand' else f'human_{name}_geom', type=gtype, size=size,
             rgba=hidden, contype='0', conaffinity='0')
    # Invisible anchor used to weld an object to the human hand (human grasp/hold).
    _sub(world, 'body', name='human_grasp_anchor', mocap='true', pos=human['hand_rest'])
    for key in scene.get('objects', {}):
        _sub(equality, 'weld', name=weld_name(key), body1='human_grasp_anchor', body2=key, active='false',
             relpose='0 0 0 1 0 0 0', solref='0.01 1')
        # Assisted robot grasp (see ManipulationEnv): activated only on two-finger contact.
        _sub(equality, 'weld', name=robot_weld_name(key), body1=GRIPPER_BODY, body2=key, active='false',
             relpose='0 0 0 1 0 0 0', solref='0.005 1', solimp='0.95 0.99 0.001')

    for name, cam in scene['cameras'].items():
        old_parent = root.find(f".//camera[@name='{name}']/..")          # e.g. the asset's wrist camera
        if old_parent is not None:
            old_parent.remove(old_parent.find(f"camera[@name='{name}']"))
        # world camera by default; `body:` mounts it on a robot body (pos/target/up in that body's frame)
        parent = world if 'body' not in cam else root.find(f".//body[@name='{cam['body']}']")
        _sub(parent, 'camera', name=name, pos=cam['pos'],
             xyaxes=look_at_xyaxes(cam['pos'], cam['target'], cam.get('up', (0.0, 0.0, 1.0))), fovy=str(cam['fovy']))
    # Lighting (config): soft lights without shadows, as in the LIBERO scenes used by LeRobot.
    for light in world.findall('light'):
        world.remove(light)
    for name, light in scene.get('lights', {}).items():
        _sub(world, 'light', name=name, **{k: str(v).lower() if isinstance(v, bool) else v for k, v in light.items()})
    if 'headlight' in scene:
        visual = root.find('visual') if root.find('visual') is not None else ET.SubElement(root, 'visual')
        _sub(visual, 'headlight', **scene['headlight'])
    # Table extents from the layout, floor at `floor_z` and visual-only table legs down to it.
    floor_z = float(scene.get('floor_z', 0.0))
    world.find("geom[@name='floor']").set('pos', _fmt([0.0, 0.0, floor_z]))
    table = world.find("geom[@name='table']")
    tpos, tsize = np.fromstring(table.get('pos'), sep=' '), np.fromstring(table.get('size'), sep=' ')
    layout = load_layout(scene)
    if layout:
        (x0, x1), (y0, y1) = layout['table']['x'], layout['table']['y']
        tpos[:2], tsize[:2] = [(x0 + x1) / 2, (y0 + y1) / 2], [(x1 - x0) / 2, (y1 - y0) / 2]
        table.set('pos', _fmt(tpos))
        table.set('size', _fmt(tsize))
        _add_zone_markers(world, layout, top)
    under = tpos[2] - tsize[2]
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        _sub(world, 'geom', name=f'table_leg_{i}', type='box', size=[0.025, 0.025, (under - floor_z) / 2],
             pos=[tpos[0] + sx * (tsize[0] - 0.05), tpos[1] + sy * (tsize[1] - 0.05), (under + floor_z) / 2],
             rgba=table.get('rgba'), contype='0', conaffinity='0')
    ET.indent(tree, space='  ')
    return ET.tostring(root, encoding='unicode')
