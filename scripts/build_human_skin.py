"""Rebuild a continuous MuJoCo skin from the pinned HY-Motion source asset.
Run: PYTHONPATH=.:scripts:.deps python scripts/build_human_skin.py --source PATH
Keeps original face topology/UVs and summed skin weights for the existing segment rig.
Finger variants use the same LBS as the collision/audit meshes. No physics changes.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
from build_human_asset import load, SMPL_TO_WORLD, SEGMENTS, blend_shapes, pose_fingers


def build(source, out):
    skel = json.loads((out / 'skeleton.json').read_text())
    for f in ('v_template.bin', 'j_template.bin', 'skinWeights.bin', 'skinIndice.bin',
              'kintree.bin', 'faces.bin', 'uvs.bin', 'joint_names.json'):
        if hashlib.sha256((source / f).read_bytes()).hexdigest() != skel['source_sha256'][f]:
            raise ValueError('Source hash mismatch: ' + f)
    d = load(source); jid = {n: i for i, n in enumerate(d['names'])}
    v, j = d['v'] @ SMPL_TO_WORLD.T, d['j'] @ SMPL_TO_WORLD.T
    shift = np.array([-j[jid['Pelvis'], 0], -j[jid['Pelvis'], 1], -v[:, 2].min()])
    v, j = v + shift, j + shift
    fmt = lambda a: ' '.join(format(float(x), '.8g') for x in np.asarray(a).ravel())
    skin = ET.Element('skin', name='human_surface', material='human_wood',
                      vertex=fmt(v), texcoord=fmt(d['uv']),
                      face=' '.join(map(str, d['faces'].ravel())))
    weights = []
    for seg, (pivot, _, joints) in SEGMENTS.items():
        w = np.sum(d['w'] * np.isin(d['idx'], [jid[n] for n in joints]), axis=1)
        ids = np.flatnonzero(w > 0); weights.append(w)
        ET.SubElement(skin, 'bone', body='human_seg_' + seg, bindpos=fmt(j[jid[pivot]]),
                      bindquat='1 0 0 0', vertid=' '.join(map(str, ids)), vertweight=fmt(w[ids]))
    assert np.allclose(np.sum(weights, axis=0), 1, atol=1e-5)
    ET.ElementTree(skin).write(out / 'surface_skin.xml', encoding='unicode')
    arrays = {}
    for side in 'LR':
        finger_ids = [i for n, i in jid.items() if n.startswith(side+'_') and any(f in n for f in ('Index','Middle','Ring','Pinky','Thumb'))]
        arrays[side.lower()+'_ids'] = np.flatnonzero(np.any(np.isin(d['idx'], finger_ids) & (d['w'] > 0), axis=1))
    for name, shape in blend_shapes().items():
        posed = pose_fingers(v, j, d['parents'], d['names'], d['w'], d['idx'].astype(int), shape)
        for side in 'lr': arrays[side+'_'+name] = posed[arrays[side+'_ids']].astype(np.float32)
    np.savez_compressed(out / 'surface_fingers.npz', **arrays)
    print('Continuous skin:', len(v), 'vertices,', len(d['faces']), 'faces')

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('assets/human'));a=p.parse_args();build(a.source,a.out)
