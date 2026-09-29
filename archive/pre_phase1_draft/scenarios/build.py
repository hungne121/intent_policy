"""Build four MJCF variants from the unchanged UR3e/SusGrip base scene."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]

def text(values):return ' '.join(str(float(x)) for x in values)
def geom(parent,**attrs):return ET.SubElement(parent,'geom',{k:str(v) for k,v in attrs.items()})

def build_scenes():
    cfg=yaml.safe_load((ROOT/'configs/scenarios/tasks.yaml').read_text())
    outputs=[]
    for name,spec in cfg['scenarios'].items():
        tree=ET.parse(ROOT/'assets/scene.xml');r=tree.getroot();r.find('compiler').set('meshdir','../meshes')
        world=r.find('worldbody');world.remove(world.find("body[@name='cube']"))
        receiver=world.find("body[@name='receiver']");receiver.find('geom').set('rgba','0 0 0 0')
        eq=r.find('equality');eq.remove(eq.find("weld[@name='receiver_grasp']"))
        for key in spec['objects']:
            obj=cfg['objects'][key];body=obj['body'];b=ET.SubElement(world,'body',name=body,pos=text(obj['position']))
            ET.SubElement(b,'freejoint',name=body+'_free')
            geom(b,name=body+'_geom',type=obj['shape'],size=text(obj['size']),mass='.07',rgba=text(obj['color']),contype='2',conaffinity='3',friction='1.5 .01 .001',condim='4')
            ET.SubElement(eq,'weld',name='receiver_grasp' if body=='cube' else 'receiver_grasp_'+body,body1='receiver',body2=body,active='false',relpose='0 0 0 1 0 0 0',solref='.01 1')
        for key in spec.get('zones',[]):
            zone=cfg['zones'][key]
            geom(world,name='zone_'+key,type='box',pos=text(zone['position']),size=text([zone['half_size'],zone['half_size'],.001]),rgba=text(zone['color']),contype='0',conaffinity='0')
        if name=='bowl_assist':
            for key,info in cfg['bowls'].items():
                b=ET.SubElement(world,'body',name=info['body'],pos=text(info['position']))
                ET.SubElement(b,'freejoint',name=info['body']+'_free')
                geom(b,name=info['body']+'_bottom',type='cylinder',size='.055 .004',mass='.07',rgba=text(info['color']),contype='2',conaffinity='3',friction='1.5 .01 .001')
                # Open bowl: collision walls, not a solid cylinder masquerading as a bowl.
                for j in range(16):
                    angle=2*np.pi*j/16
                    geom(b,name=f'{info["body"]}_wall_{j}',type='box',pos=text([.052*np.cos(angle),.052*np.sin(angle),.023]),size='.005 .011 .023',euler=f'0 0 {angle}',mass='.004',rgba=text(info['color']),contype='2',conaffinity='3')
                # Narrow handle grippable by the existing parallel gripper.
                geom(b,name=info['body']+'_handle',type='box',pos='-.072 0 .023',size='.025 .015 .009',mass='.015',rgba=text(info['color']),contype='2',conaffinity='3',friction='1.5 .01 .001')
                ET.SubElement(eq,'weld',name='receiver_grasp_'+info['body'],body1='receiver',body2=info['body'],active='false',relpose='0 0 0 1 0 0 0',solref='.01 1')
        # A recognizable, scripted human proxy. Arm geoms are physical for Intruder.
        torso=ET.SubElement(world,'body',name='human_torso',pos='.67 0 .95')
        geom(torso,type='box',size='.065 .13 .19',rgba='.25 .36 .52 1',contype='0',conaffinity='0')
        geom(torso,type='sphere',pos='0 0 .29',size='.07',rgba='.78 .57 .42 1',contype='0',conaffinity='0')
        hand=ET.SubElement(world,'body',name='human_hand',mocap='true',pos='.56 0 .9')
        geom(hand,name='human_palm',type='ellipsoid',size='.035 .025 .016',rgba='.85 .63 .46 1',contype='2' if name=='intruder' else '0',conaffinity='1' if name=='intruder' else '0')
        geom(hand,name='human_finger',type='capsule',fromto='-.02 0 0 -.065 0 0',size='.007',rgba='.85 .63 .46 1',contype='0',conaffinity='0')
        forearm=ET.SubElement(world,'body',name='human_forearm',mocap='true',pos='.64 0 .9')
        geom(forearm,name='human_arm_geom',type='capsule',size='.022 .11',quat='.7071068 0 .7071068 0',rgba='.78 .57 .42 1',contype='2' if name=='intruder' else '0',conaffinity='1' if name=='intruder' else '0')
        # Public visual request panel: actual mesh/geom, visible to RGB cameras.
        geom(world,name='request_board',type='box',pos='.57 -.18 .96',size='.008 .09 .075',rgba='.93 .93 .93 1',contype='0',conaffinity='0')
        request=ET.SubElement(world,'body',name='request_object',mocap='true',pos='.55 -.21 .97')
        for shape,size in [('box','.018 .018 .018'),('cylinder','.019 .023'),('sphere','.021')]:
            geom(request,name='request_'+shape,type=shape,size=size,rgba='1 0 0 0',contype='0',conaffinity='0')
        geom(world,name='request_destination',type='box',pos='.55 -.14 .96',size='.008 .022 .022',rgba='.1 .8 .2 1',contype='0',conaffinity='0')
        # Improve framing: include robot, all targets, human and request card.
        for cam in world.findall('camera'):
            if cam.get('name')=='front':
                cam.set('pos','1.2 -1.2 1.45');cam.set('xyaxes','.707 .707 0 -.32 .32 .89');cam.set('fovy','55')
        ET.indent(tree,space='  ')
        dest=ROOT/f'assets/scenarios/{name}.xml';tree.write(dest,encoding='unicode');outputs.append(dest)
    return outputs

if __name__=='__main__':
    for p in build_scenes():print(p.relative_to(ROOT))
