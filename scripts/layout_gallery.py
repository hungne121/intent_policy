"""Render provisional V2 scene layouts from three proposed camera views.

Static design previews only. Geometry, contacts, reach, visibility and dynamics
have not been accepted for dataset collection.
"""
from pathlib import Path
import copy
import html
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw
from intent_policy.sim.base_env import ManipulationEnv, ROOT
from intent_policy.sim.scene_builder_v2 import build_scene_xml
from intent_policy.sim.human_body_v2 import HumanBody

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / 'outputs/layout_review'
OUT.mkdir(parents=True, exist_ok=True)
base = yaml.safe_load((ROOT / 'configs/scene/common.yaml').read_text())


def make(task):
    scene = copy.deepcopy(base)
    scene['zone_markers'] = False
    scene['human'].update({'hand_mesh_mode':'compiled_variants', 'idle_palm_offset':[.12,.07,.50],
                           'hang_blend_m':.10, 'hang_rest_upright':True, 'idle_clearance_m':.01,
                           'self_clearance_m':.01, 'lean_rate_deg_s':40., 'reach_margin':.88, 'max_extension':.94})
    scene['human']['stand_xy'] = [.30, -.60]
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
            obj('cup'+str(i),x,.22,'cylinder',(.034,.037),'white')
            obj('coaster'+str(i),x,-.12,'cylinder',(.045,.003),'cork')
    elif task == 'I2':
        for i,x in enumerate([.105,.30,.495]): tray('plate'+str(i),x,-.11,.06,.055)
        tray('source',.30,.20,.065,.06)
        obj('cake',.30,.20,'ellipsoid',(.025,.025,.018),'tan',.643)
    elif task == 'I3':
        # 13/11 (user): the two cakes of each tray sit in one row across (x), in front of the human
        for x,n in [(.16,'L'),(.40,'R')]:
            tray('source'+n,x,-.10,.09,.06)
            obj(n+'x',x-.05,-.10,'cylinder',(.025,.018),'tan',.643)
            obj(n+'y',x+.05,-.10,'cylinder',(.025,.018),'brown',.643)
        tray('receive',.28,.25,.07,.06)
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
    # Scanned mug: visual only; cylinder is a provisional physics proxy.
    mugs = [name for name in objects if name.startswith('cup')]
    if mugs:
        asset = root.find('asset'); path = HERE/'assets/gso/Room_Essentials_Mug_White_Yellow'
        ET.SubElement(asset,'mesh',name='mug_scan',file=str(path/'model.obj'),scale='.7 .7 .7')
        ET.SubElement(asset,'texture',name='mug_tex',type='2d',file=str(path/'texture.png'))
        ET.SubElement(asset,'material',name='mug_mat',texture='mug_tex')
        for name in mugs:
            body = root.find(f"./worldbody/body[@name='{name}']")
            body.find('geom').set('rgba','0 0 0 0')
            ET.SubElement(body,'geom',name=name+'_visual',type='mesh',mesh='mug_scan',material='mug_mat',
                          pos='0 0 -.037',contype='0',conaffinity='0',mass='0')
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
        ET.SubElement(asset,'mesh',name='towel_scan',file=str(towel/'model.obj'),scale='.38 .38 .38')
        ET.SubElement(asset,'texture',name='towel_tex',type='2d',file=str(towel/'texture.png'))
        ET.SubElement(asset,'material',name='towel_mat',texture='towel_tex')
        towel_names = ['clothL','clothR'] if task == 'C1' else ['cloth']
        for name in towel_names:
            body = root.find(f"./worldbody/body[@name='{name}']")
            body.find('geom').set('rgba','0 0 0 0')
            ET.SubElement(body,'geom',name=name+'_visual',type='mesh',mesh='towel_scan',material='towel_mat',
                          pos='0 0 -.004',contype='0',conaffinity='0',mass='0')
    if task == 'C2':
        clock = HERE/'assets/gso/Crosley_Alarm_Clock_Vintage_Metal'
        ET.SubElement(asset,'mesh',name='clock_scan',file=str(clock/'model.obj'),scale='.65 .65 .65')
        ET.SubElement(asset,'texture',name='clock_tex',type='2d',file=str(clock/'texture.png'))
        ET.SubElement(asset,'material',name='clock_mat',texture='clock_tex')
        body = root.find("./worldbody/body[@name='clock']")
        body.find('geom').set('rgba','0 0 0 0')
        ET.SubElement(body,'geom',name='clock_visual',type='mesh',mesh='clock_scan',material='clock_mat',
                      pos='0 0 -.035',contype='0',conaffinity='0',mass='0')
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


def main():
    tasks = ['I1','I2','I3','C1','C2','C3','C4','W1','W2','W3']
    rows = []
    for task in tasks:
        scene, positions, xml = make(task)
        env = ManipulationEnv(model_xml=xml)
        try:
            for name, xyz in positions.items():
                addr = int(env.model.joint(name+'_free').qposadr[0])
                env.data.qpos[addr:addr+7] = [*xyz,1,0,0,0]
            human = HumanBody({**scene['human'],'floor_z':scene['floor_z']})
            human.bind(env.model)
            human.apply(env.data, palm=human.hanging_palm(human.active))
            mujoco.mj_forward(env.model,env.data)
            # A still image before contact settling: layout, not physics evidence.
            for cam in ('high','top','wrist'):
                Image.fromarray(env.render(cam, 640, 480)).save(OUT/f'{task}_{cam}.png')
            if task == 'C3':
                # Isolated raised-tray snapshot to make the dirt underneath visible.
                # This is a changed model pose, not a two-person lifting rollout.
                for g in range(env.model.ngeom):
                    if env.model.geom(g).name.startswith('decor_shared_tray_'):
                        env.model.geom_rgba[g,3] = 0.0
                mujoco.mj_forward(env.model, env.data)
                Image.fromarray(env.render('top', 640, 480)).save(OUT/'C3_dirt_under_tray.png')
            rows.append({'task':task,'images':{cam:f'{task}_{cam}.png' for cam in ('high','top','wrist')},
                         'status':'unverified_static_layout','seed':0})
            print(task, flush=True)
        finally:
            env.close()
    (OUT/'index.json').write_text(json.dumps(rows,indent=2)+'\n')
    blocks = []
    for row in rows:
        imgs = ''.join(f'<figure><a href="{html.escape(path)}"><img src="{html.escape(path)}"></a><figcaption>{cam}</figcaption></figure>' for cam,path in row['images'].items())
        blocks.append(f'<h2>{row["task"]}</h2><div>{imgs}</div>')
    blocks.append('<h2>C3 — vết bẩn dưới khay</h2><p>Tạm ẩn khay để nhìn thấy vết bẩn bên dưới; chưa mô phỏng thao tác nâng.</p><a href="C3_dirt_under_tray.png"><img style="max-width:640px" src="C3_dirt_under_tray.png"></a>')
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>HRI layout review</title><style>body{font-family:sans-serif;background:#f4f5f7}div{display:flex;gap:10px}figure{margin:0;width:32%}img{width:100%}</style><h1>Bố cục đề xuất — chưa kiểm vật lý</h1><p>High / top / wrist là ba góc camera dự kiến. Vật tạm và các giới hạn xem báo cáo.</p>'+''.join(blocks),encoding='utf-8')
    sheet = Image.new('RGB',(960,10*260),'white'); draw = ImageDraw.Draw(sheet)
    for i,row in enumerate(rows):
        for j,(cam,path) in enumerate(row['images'].items()):
            im = Image.open(OUT/path).resize((320,240)); sheet.paste(im,(j*320,i*260+20))
            draw.text((j*320+5,i*260+3),row['task']+' / '+cam,fill='black')
    sheet.save(OUT/'contact_sheet.jpg',quality=88)


if __name__ == '__main__': main()
