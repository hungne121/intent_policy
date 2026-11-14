"""Build the articulated human asset "v3" from the HY-Motion 1.0 wooden body model (06/TTr-DL, 14/11/2026).

Unlike assets/human (9 rigid segments posed by IK), v3 keeps the SMPL-H skeleton: one MuJoCo mocap body per
body joint (22), a connected MuJoCo `skin` bound to those bodies with the original skinning weights, and the
existing hand mesh variants (assets/human/<s>_hand_<shape>.obj, relative to the wrist joint) attached to the
wrist bodies. Capsule collision geoms are generated at scene build time from the rest skeleton.

Output (assets/human_v3/):
  skeleton_v3.json   52 joint names, parents, rest positions in world axes (facing -X, Z up, soles z=0), provenance
  skin_body.xml      <skin> element: body vertices (hand-dominated vertices excluded), 22 bones, weights

  ./run.sh scripts/build_human_v3.py --source outputs/hymotion_wooden_src
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SMPL_TO_WORLD = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)   # as scripts/build_human_asset.py
SOURCE_FILES = ('v_template.bin', 'j_template.bin', 'skinWeights.bin', 'skinIndice.bin', 'kintree.bin', 'faces.bin',
                'uvs.bin', 'joint_names.json', 'Boy_lambert4_BaseColor.webp')
BODY_JOINTS = 22


def load(src: Path) -> dict:
    read = lambda name, dtype: np.frombuffer((src / name).read_bytes(), dtype=dtype)
    return dict(v=read('v_template.bin', np.float32).reshape(-1, 3).astype(float),
                j=read('j_template.bin', np.float32).reshape(-1, 3).astype(float),
                w=read('skinWeights.bin', np.float32).reshape(-1, 4).astype(float),
                idx=read('skinIndice.bin', np.uint16).reshape(-1, 4).astype(int),
                faces=read('faces.bin', np.uint16).reshape(-1, 3).astype(int),
                uv=read('uvs.bin', np.float32).reshape(-1, 2).astype(float),
                parents=read('kintree.bin', np.int32).astype(int),
                names=json.loads((src / 'joint_names.json').read_text()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='outputs/hymotion_wooden_src')
    p.add_argument('--out', default='assets/human_v3')
    args = p.parse_args()
    src = (ROOT / args.source).resolve(); out = (ROOT / args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    d = load(src)
    names, parents = d['names'], d['parents']
    v = d['v'] @ SMPL_TO_WORLD.T; j = d['j'] @ SMPL_TO_WORLD.T
    shift = np.array([-j[0, 0], -j[0, 1], -v[:, 2].min()])
    v, j = v + shift, j + shift
    # vertices dominated by a wrist or finger joint belong to the hand meshes (assets/human), not to the body skin
    dominant = d['idx'][np.arange(len(v)), np.argmax(d['w'], axis=1)]
    hand_joints = set(range(20, 52))                       # L_Wrist, R_Wrist and all fingers
    body_vertex = np.array([int(k) not in hand_joints for k in dominant])
    faces = d['faces'][body_vertex[d['faces']].all(axis=1)]
    used, inverse = np.unique(faces, return_inverse=True)
    sv, suv, sf = v[used], d['uv'][used], inverse.reshape(-1, 3)
    # fold finger weights of remaining vertices onto their wrist (fingers have no bodies in v3)
    to_body = lambda k: 20 if 22 <= k <= 36 else (21 if k >= 37 else k)
    weights = {b: {} for b in range(BODY_JOINTS)}
    for n, vid in enumerate(used):
        for k in range(4):
            wk = d['w'][vid, k]
            if wk <= 0: continue
            b = to_body(int(d['idx'][vid, k]))
            weights[b][n] = weights[b].get(n, 0.) + float(wk)
    skin = ET.Element('skin', name='human_v3_surface', material='human_wood', inflate='0.003',
                      vertex=' '.join(f'{x:.5f}' for x in sv.ravel()),
                      texcoord=' '.join(f'{x:.5f}' for x in suv.ravel()),
                      face=' '.join(str(i) for i in sf.ravel()))
    for b in range(BODY_JOINTS):
        if not weights[b]: continue
        ids = sorted(weights[b])
        ET.SubElement(skin, 'bone', body=f'human_seg_{names[b]}', bindpos=' '.join(f'{x:.5f}' for x in j[b]),
                      bindquat='1 0 0 0', vertid=' '.join(map(str, ids)),
                      vertweight=' '.join(f'{weights[b][i]:.5f}' for i in ids))
    (out / 'skin_body.xml').write_text(ET.tostring(skin, encoding='unicode'))
    skeleton = dict(
        source='HY-Motion 1.0 wooden body model (Tencent), scripts/gradio/static/assets/dump_wooden',
        license='Tencent HY-Motion 1.0 Community License Agreement; research use within its Territory',
        source_sha256={f: hashlib.sha256((src / f).read_bytes()).hexdigest() for f in SOURCE_FILES},
        frame='world axes: facing -X, Z up, left = -Y; soles at z=0, pelvis above the origin; T-pose',
        height_m=round(float(v[:, 2].max()), 4),
        names=names, parents=parents.tolist(), joints=j.round(5).tolist(),
        body_joints=BODY_JOINTS, skin_vertices=int(len(used)), skin_faces=int(len(sf)),
        hands={'l': {'pivot': 'L_Wrist', 'meshes': 'assets/human/l_hand_<shape>.obj'},
               'r': {'pivot': 'R_Wrist', 'meshes': 'assets/human/r_hand_<shape>.obj'}},
        texture='assets/human/wood.png')
    (out / 'skeleton_v3.json').write_text(json.dumps(skeleton, indent=1))
    print(f'wrote {out}: skin {len(used)} vertices / {len(sf)} faces, {BODY_JOINTS} bones; height {skeleton["height_m"]} m')


if __name__ == '__main__':
    main()
