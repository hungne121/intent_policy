"""Scene layouts of the 10 HRC tasks (moved from scripts/layout_gallery.py, QĐ 53/QĐ-DL).

`make(task)` returns (scene config, object positions, MJCF XML) on the shared station."""
from pathlib import Path
import copy
import json
import os
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import yaml
from intent_policy.sim.base_env import ROOT
from intent_policy.sim.scene_builder_v2 import build_scene_xml

HERE = Path(__file__).resolve().parents[2]
base = yaml.safe_load((ROOT / 'configs/scene/common.yaml').read_text())



_MESH_CACHE = {}


def _mesh_vertices(path):
    if path not in _MESH_CACHE:
        _MESH_CACHE[path] = np.array([[float(x) for x in l.split()[1:4]] for l in open(path) if l.startswith('v ')])
    return _MESH_CACHE[path]


def fit_visual(body, mesh_file, yaw_deg=0., exclude_handle=False):
    """61/QĐ-DL: scale + place a visual-only scan so it lies INSIDE the body's collision geom (box or cylinder),
    centred on it: a finger / hand touching the collider must not look sunk into the visual. Returns (scale, pos, quat)
    strings. `exclude_handle`: fit the mug body only (points > 50 deg away from the handle direction); the handle is
    turned by `yaw_deg` out of the gripper's closing axis (world X)."""
    V = _mesh_vertices(str(mesh_file))
    g = body.find('geom'); typ = g.get('type', 'sphere'); size = [float(x) for x in g.get('size').split()]
    half = np.array([size[0], size[0], size[1]]) if typ == 'cylinder' else np.array(size[:3])
    z = V[:, 2]
    if exclude_handle:
        H = z.max(); c = np.median(V[(z > .3 * H) & (z < .7 * H)][:, :2], axis=0)
        ang = np.arctan2(V[:, 1] - c[1], V[:, 0] - c[0]); r = np.linalg.norm(V[:, :2] - c, axis=1)
        hdir = np.median(ang[r > np.percentile(r, 99)])
        body_pts = V[np.abs(np.angle(np.exp(1j * (ang - hdir)))) > np.radians(50)]
        rad = float(np.linalg.norm(body_pts[:, :2] - c, axis=1).max())
        sxy = half[0] / rad; sz = 2 * half[2] / (z.max() - z.min())
        sc = np.array([sxy, sxy, sz]); centre = np.r_[c, z.min()]
        yaw = np.radians(yaw_deg) - hdir if yaw_deg is not None else 0.
    else:
        lo, hi = V.min(0), V.max(0)
        sc = np.min(2 * half / (hi - lo)) * np.ones(3)          # uniform: keep the scan's proportions, fit inside
        centre = np.r_[(lo + hi)[:2] / 2, lo[2]]; yaw = 0.
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    pos = -R @ (sc * centre) + np.array([0., 0., -half[2]])
    quat = [np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]
    return (' '.join(f'{v:.5f}' for v in sc), ' '.join(f'{v:.5f}' for v in pos), ' '.join(f'{v:.6f}' for v in quat))

def align_robot_station(root):
    """Use the same robot mount as the I-series preview station."""
    cfg = yaml.safe_load((HERE / "configs/hrc/station.yaml").read_text())
    mount = root.find("./worldbody/body[@name='robot_mount']")
    x, y = cfg['robot_base_xy']
    half = np.radians(float(cfg['robot_base_yaw_deg'])) / 2
    mount.set('pos', f'{x} {y} 0.62')
    mount.set('quat', f'{np.cos(half):.7f} 0 0 {np.sin(half):.7f}')


def make(task, human_model=None):
    scene = copy.deepcopy(base)
    scene['zone_markers'] = False
    scene['human'].update({'hand_mesh_mode':'compiled_variants', 'idle_palm_offset':[.12,.07,.50],
                           'hang_blend_m':.10, 'hang_rest_upright':True, 'idle_clearance_m':.01,
                           'self_clearance_m':.01, 'lean_rate_deg_s':40., 'reach_margin':.88, 'max_extension':.94})
    scene['human']['stand_xy'] = [.30, -.60]
    _common = yaml.safe_load((HERE / "configs/hrc/station.yaml").read_text())
    # A per-run override keeps v4 trials separate from the accepted preview
    # station until a task-level validation succeeds.
    scene['human']['held_object_frame'] = _common.get('human_held_object_frame', 'world')   # 61/QĐ-DL
    model = human_model or os.environ.get('HUMAN_MODEL') or _common.get('human_model')
    if model in ('v3', 'v4'):
        scene['human']['model_version'] = model
    scene['cameras']['high'] = {'pos':[.3,.95,1.65], 'target':[.3,-.20,.86], 'fovy':65}
    scene['cameras']['top'] = {'pos':[.30,.04,1.72], 'target':[.30,-.04,.62], 'fovy':54}
    objects, positions, decor = {}, {}, []
    def obj(name, x, y, shape='box', size=(.022,.022,.022), color='white', z=None):
        objects[name] = {'shape':shape, 'size':list(size), 'color':color, 'mass':.06}
        height = size[1] if shape == 'cylinder' else size[2]
        positions[name] = (x,y,.62+height+.007 if z is None else z)
    def tray(name,x,y,hx=.075,hy=.07,color='white'):
        decor.append({'name':name,'type':'open_box','xy':[x,y], 'half':[hx,hy], 'height':.015,'wall':.003,'color':color})
    if task == 'I1':
        for i,x in enumerate([.10,.24,.38]):
            # v4 uses a collider sized to the scanned mug's visible width;
            # the old 34 mm proxy let the gripper sink through the mug mesh.
            radius = .042 if model == 'v4' else .034
            obj('cup'+str(i),x,.22,'cylinder',(radius,.037),'white')
            obj('coaster'+str(i),x,-.12,'cylinder',(.045,.003),'cork')
    elif task == 'I2':
        for i,x in enumerate([.105,.30,.495]): tray('plate'+str(i),x,-.11,.06,.055)
        # 14/11: source tray between the robot base (0.30, 0.33) and the plates, like the I1 cup plate
        tray('source',.30,.06,.065,.06)
        obj('cake',.30,.06,'ellipsoid',(.025,.025,.018),'tan',.643)
    elif task == 'I3':
        # 13/11 (user): the two cakes of each tray sit in one row across (x), in front of the human
        for x,n in [(.16,'L'),(.40,'R')]:
            tray('source'+n,x,-.10,.09,.06)
            obj(n+'x',x-.05,-.10,'cylinder',(.025,.018),'tan',.643)
            obj(n+'y',x+.05,-.10,'cylinder',(.025,.018),'brown',.643)
        tray('receive',.28,.08,.07,.06)      # 14/11: receive tray between the base and the cake trays
    elif task == 'C1':
        for x,n in [(.16,'L'),(.40,'R')]:
            tray('tray'+n,x,-.10,.09,.09)
            obj('cup'+n,x,-.10,'cylinder',(.034,.037),'white',.665)
        for x,n in [(.16,'L'),(.40,'R')]:
            obj('cloth'+n,x,-.25,'box',(.040,.035,.004),'blue')
        scene['human']['stand_xy'] = [.28,-.75]
        scene['human']['hand_rest'] = [.45,-.68,.68]
        scene['cameras']['high'] = {'pos':[.30,1.15,1.75], 'target':[.30,-.30,.78], 'fovy':75}
    elif task == 'C2':
        tray('shelf',.28,-.07,.22,.07)
        obj('cup',.18,-.07,'cylinder',(.034,.037),'white',.665)
        obj('clock',.38,-.07,'box',(.035,.025,.035),'brown',.658)
        obj('cloth',.28,-.26,'box',(.04,.035,.004),'blue')
    elif task == 'C3':
        tray('shared_tray',.28,-.05,.18,.12)
        obj('cloth',.28,-.26,'box',(.04,.035,.004),'blue')
        # Handles are visible, separate from the fixed placeholder tray.
        for i,y in enumerate([-.20,.10]):
            obj('handle'+str(i),.28,y,'box',(.035,.01,.012),'brown')
    elif task == 'C4':
        tray('assembly_base',.28,-.12,.18,.055)
        for i,x in enumerate([.16,.28,.40]):
            obj('socket'+str(i),x,-.12,'cylinder',(.022,.003),['red','green','blue'][i],.627)
            obj('part'+str(i),x,.20,'cylinder',(.018,.023),['red','green','blue'][i])
    elif task == 'W1':
        for i,(x,y) in enumerate([(.12,-.12),(.28,-.12),(.44,-.12),(.60,-.12)]): tray('tray'+str(i),x,y,.073,.09)
        for i,(x,y) in enumerate([(.095,-.16),(.145,-.16),(.095,-.08),(.145,-.08)]):
            obj('cake'+str(i),x,y,'ellipsoid',(.022,.022,.016),'tan',.641)
        for i,y in enumerate([-.16,-.08]): obj('can'+str(i),.60,y,'cylinder',(.023,.035),'red',.66)
    elif task == 'W2':
        colors=['red','blue','green']
        for i,x in enumerate([.10,.28,.46]): tray('tray'+str(i),x,-.14,.07,.065,colors[i])
        for i in range(6): obj('block'+str(i),.11+(i%3)*.15,.11+(i//3)*.09,'box',(.02,.02,.02),colors[i%3])
    elif task == 'W3':
        for i,y in enumerate([-.14,.13]): tray('box'+str(i),.10,y,.07,.09)
        for i,(x,y) in enumerate([(.30,-.18),(.43,-.15),(.30,-.04),(.44,-.02),(.30,.10),(.43,.12)]):
            obj('item'+str(i),x,y,'box',(.018+.003*(i%2),.025,.02),['red','blue','green','yellow','brown','white'][i])
    scene['objects'], scene['decor'] = objects, decor
    root = ET.fromstring(build_scene_xml(scene))
    align_robot_station(root)
    # Scanned mug: visual only; cylinder is a provisional physics proxy.
    mugs = [name for name in objects if name.startswith('cup')]
    if mugs:
        asset = root.find('asset'); path = HERE/'assets/gso/Room_Essentials_Mug_White_Yellow'
        # 61/QĐ-DL (17/11): the 0.7 visual was off-centre by 12.5 mm and flared to 45 mm radius around the 34 mm cylinder
        # collider (fingers looked sunk 12-14 mm into the mug, no physical contact); it is now fitted inside the
        # collider, handle turned to -Y (the person's side), out of the fingers' closing axis X.
        first = root.find(f"./worldbody/body[@name='{mugs[0]}']")
        if task == 'I1' and model == 'v4':
            scale, vis_pos, vis_quat = '.60 .60 .60', '0 0 -.037', '1 0 0 0'
        else:
            scale, vis_pos, vis_quat = fit_visual(first, path / 'model.obj', yaw_deg=-90., exclude_handle=True)
        ET.SubElement(asset,'mesh',name='mug_scan',file=str(path/'model.obj'),scale=scale)
        ET.SubElement(asset,'texture',name='mug_tex',type='2d',file=str(path/'texture.png'))
        ET.SubElement(asset,'material',name='mug_mat',texture='mug_tex')
        for name in mugs:
            body = root.find(f"./worldbody/body[@name='{name}']")
            body.find('geom').set('rgba','0 0 0 0')
            ET.SubElement(body,'geom',name=name+'_visual',type='mesh',mesh='mug_scan',material='mug_mat',
                          pos=vis_pos,quat=vis_quat,contype='0',conaffinity='0',mass='0')
    asset = root.find('asset')
    if task == 'I3':
        # Decorative frosting helps distinguish cakes; collision is the same
        # cylindrical cake body, not these massless visual decorations.
        for name in ('Lx','Ly','Rx','Ry'):
            body=root.find(f"./worldbody/body[@name='{name}']")
            color='0.97 0.88 0.67 1' if name.endswith('x') else '0.72 0.42 0.30 1'
            ET.SubElement(body,'geom',name=name+'_icing_visual',type='cylinder',size='.024 .002',pos='0 0 .020',rgba=color,contype='0',conaffinity='0',mass='0')
            ET.SubElement(body,'geom',name=name+'_berry_visual',type='sphere',size='.005',pos='0 0 .026',rgba='.75 .05 .07 1',contype='0',conaffinity='0',mass='0')
    if task in ('C1','C2','C3'):
        towel = HERE/'assets/gso/Cole_Hardware_Dishtowel_Blue'
        towel_names = ['clothL','clothR'] if task == 'C1' else ['cloth']
        t_scale, t_pos, _ = fit_visual(root.find(f"./worldbody/body[@name='{towel_names[0]}']"), towel / 'model.obj')   # 61/QĐ-DL
        ET.SubElement(asset,'mesh',name='towel_scan',file=str(towel/'model.obj'),scale=t_scale)
        ET.SubElement(asset,'texture',name='towel_tex',type='2d',file=str(towel/'texture.png'))
        ET.SubElement(asset,'material',name='towel_mat',texture='towel_tex')
        towel_names = ['clothL','clothR'] if task == 'C1' else ['cloth']
        for name in towel_names:
            body = root.find(f"./worldbody/body[@name='{name}']")
            body.find('geom').set('rgba','0 0 0 0')
            ET.SubElement(body,'geom',name=name+'_visual',type='mesh',mesh='towel_scan',material='towel_mat',
                          pos=t_pos,contype='0',conaffinity='0',mass='0')
    if task == 'C2':
        clock = HERE/'assets/gso/Crosley_Alarm_Clock_Vintage_Metal'
        c_scale, c_pos, _ = fit_visual(root.find("./worldbody/body[@name='clock']"), clock / 'model.obj')        # 61/QĐ-DL
        ET.SubElement(asset,'mesh',name='clock_scan',file=str(clock/'model.obj'),scale=c_scale)
        ET.SubElement(asset,'texture',name='clock_tex',type='2d',file=str(clock/'texture.png'))
        ET.SubElement(asset,'material',name='clock_mat',texture='clock_tex')
        body = root.find("./worldbody/body[@name='clock']")
        body.find('geom').set('rgba','0 0 0 0')
        ET.SubElement(body,'geom',name='clock_visual',type='mesh',mesh='clock_scan',material='clock_mat',
                      pos=c_pos,contype='0',conaffinity='0',mass='0')
    if task in ('C2','C3'):
        world = root.find('worldbody')
        # Marks are visual placeholders; this preview does not implement wiping.
        spots = ([(.13,-.11),(.18,-.12),(.22,-.11),(.33,-.11),(.38,-.12),(.43,-.11)] if task=='C2'
                 else [(.18,-.09),(.23,-.03),(.28,-.09),(.33,-.03),(.38,-.09)])
        z = .6232 if task == 'C2' else .6202
        for i,(x,y) in enumerate(spots):
            ET.SubElement(world,'geom',name=f'dirt_preview_{i}',type='cylinder',size='.017 .0001',
                          pos=f'{x} {y} {z}',rgba='.27 .16 .08 1',contype='0',conaffinity='0')
    return scene, positions, ET.tostring(root,encoding='unicode')


