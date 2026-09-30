"""Build the human mannequin asset from the HY-Motion 1.0 wooden body model (run once).

Reads the skinned wooden model shipped with HY-Motion 1.0 (SMPL-H skeleton, 52 joints, T-pose, Y-up,
facing +Z) and writes rigid body segments for MuJoCo: every vertex goes to the segment of its dominant
skinning joint, faces that straddle segments are kept whole in the segment of their first vertex (no
holes at the joints). Output, in the simulator's world axes (facing -X, Z up, left = -Y, soles at z=0,
pelvis above the origin):

  assets/human/<segment>.obj   mesh with texture coordinates, vertices relative to the segment pivot joint
  assets/human/<s>_hand_<shape>.obj   hand variants (relaxed / point / grasp / offer and 25/50/75 % blends between
                               every pair) posed by linear blend skinning of the finger joints; the runtime steps
                               through the blends when the hand shape changes
  assets/human/wood.png        base-colour texture
  assets/human/skeleton.json   T-pose joint positions, segments (pivot, parent), palm centres, provenance

The license copy (LICENSE_HY-Motion-1.0.txt) and NOTICE.txt required by the HY-Motion license live next to
the generated files.

The runtime pose (lean, arm IK) is computed in `intent_policy/sim/human_body.py`.

  ./run.sh -m scripts.build_human_asset --source ../HY-Motion-1.0/scripts/gradio/static/assets/dump_wooden
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from intent_policy.utils import resolve

# segment -> (pivot joint, parent segment, skinning joints)
FINGERS = ('Index1', 'Index2', 'Index3', 'Middle1', 'Middle2', 'Middle3', 'Pinky1', 'Pinky2', 'Pinky3',
           'Ring1', 'Ring2', 'Ring3', 'Thumb1', 'Thumb2', 'Thumb3')
SEGMENTS = {
    'pelvis': ('Pelvis', None, ['Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'L_Foot', 'R_Foot']),
    'torso': ('Spine1', 'pelvis', ['Spine1', 'Spine2', 'Spine3', 'L_Collar', 'R_Collar']),
    'head': ('Neck', 'torso', ['Neck', 'Head']),
}
for side in 'LR':
    s = side.lower()
    SEGMENTS[f'{s}_upperarm'] = (f'{side}_Shoulder', 'torso', [f'{side}_Shoulder'])
    SEGMENTS[f'{s}_forearm'] = (f'{side}_Elbow', f'{s}_upperarm', [f'{side}_Elbow'])
    SEGMENTS[f'{s}_hand'] = (f'{side}_Wrist', f'{s}_forearm', [f'{side}_Wrist'] + [f'{side}_{f}' for f in FINGERS])
# finger curl [deg] of joints 1/2/3 per hand shape (towards the palm; thumb tucks across it)
HAND_SHAPES = {
    'relaxed': dict(Index=(18, 28, 16), Middle=(22, 32, 18), Ring=(26, 34, 20), Pinky=(30, 36, 22), Thumb=(10, 15, 10)),
    'point': dict(Index=(0, 0, 0), Middle=(80, 95, 60), Ring=(85, 95, 60), Pinky=(85, 95, 60), Thumb=(30, 45, 30)),
    'grasp': dict(Index=(50, 65, 40), Middle=(55, 70, 40), Ring=(55, 70, 40), Pinky=(55, 70, 40), Thumb=(25, 35, 30)),
    'offer': dict(Index=(6, 10, 6), Middle=(8, 12, 6), Ring=(10, 12, 8), Pinky=(12, 14, 8), Thumb=(0, 8, 6)),  # open, palm up
}
BLEND = (25, 50, 75)          # intermediate finger curls [%] between every pair of shapes: `<a>-<b>-<pct>` variants, so
                              # the runtime changes the hand shape gradually instead of switching meshes at once


def blend_shapes() -> dict:
    out = dict(HAND_SHAPES)
    names = list(HAND_SHAPES)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for pct in BLEND:
                f = pct / 100
                out[f'{a}-{b}-{pct}'] = {k: tuple((1 - f) * x + f * y for x, y in zip(HAND_SHAPES[a][k], HAND_SHAPES[b][k]))
                                         for k in HAND_SHAPES[a]}
    return out
PALM_NORMAL = np.array([0.0, 0.0, -1.0])      # palms face down in the T-pose
# SMPL (x left, y up, z forward) -> world (facing -X, Z up, left = -Y)
SMPL_TO_WORLD = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
SOURCE_FILES = ('v_template.bin', 'j_template.bin', 'skinWeights.bin', 'skinIndice.bin', 'kintree.bin', 'faces.bin',
                'uvs.bin', 'joint_names.json', 'Boy_lambert4_BaseColor.webp')


def load(src: Path) -> dict:
    read = lambda name, dtype: np.frombuffer((src / name).read_bytes(), dtype=dtype)
    v = read('v_template.bin', np.float32).reshape(-1, 3).astype(float)
    return dict(v=v, j=read('j_template.bin', np.float32).reshape(-1, 3).astype(float),
                w=read('skinWeights.bin', np.float32).reshape(-1, 4), idx=read('skinIndice.bin', np.uint16).reshape(-1, 4),
                faces=read('faces.bin', np.uint16).reshape(-1, 3).astype(int), uv=read('uvs.bin', np.float32).reshape(-1, 2),
                parents=read('kintree.bin', np.int32),
                names=json.loads((src / 'joint_names.json').read_text()))


def rotation(axis, angle: float) -> np.ndarray:
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    c, s, C = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def pose_fingers(v, j, parents, names, weights, indices, shape: dict) -> np.ndarray:
    """Linear blend skinning with finger-joint rotations only (rest frames are world-aligned)."""
    jid = {n: i for i, n in enumerate(names)}
    local = [np.eye(3) for _ in names]
    for side in 'LR':
        for finger, angles in shape.items():
            for k, deg in enumerate(angles, start=1):
                a, b = jid[f'{side}_{finger}{k}'], jid[f'{side}_{finger}{min(k + 1, 3)}']
                bone = j[b] - j[a] if k < 3 else j[a] - j[jid[f'{side}_{finger}2']]
                local[a] = rotation(np.cross(bone, PALM_NORMAL), np.radians(deg))
    rot, pos = [None] * len(names), [None] * len(names)
    for i, par in enumerate(parents):                      # parents precede children in the kintree
        if par < 0:
            rot[i], pos[i] = local[i], j[i]
        else:
            rot[i], pos[i] = rot[par] @ local[i], pos[par] + rot[par] @ (j[i] - j[par])
    skin = np.stack([np.eye(4)] * len(names))
    for i in range(len(names)):
        skin[i, :3, :3], skin[i, :3, 3] = rot[i], pos[i] - rot[i] @ j[i]
    vh = np.c_[v, np.ones(len(v))]
    out = np.zeros_like(v)
    for k in range(indices.shape[1]):
        out += weights[:, k:k + 1] * np.einsum('nij,nj->ni', skin[indices[:, k]], vh)[:, :3]
    return out


def write_obj(path: Path, verts: np.ndarray, uvs: np.ndarray, faces: np.ndarray) -> None:
    lines = [f'v {x:.5f} {y:.5f} {z:.5f}' for x, y, z in verts]
    lines += [f'vt {u:.5f} {v:.5f}' for u, v in uvs]
    lines += [f'f {a}/{a} {b}/{b} {c}/{c}' for a, b, c in faces + 1]
    path.write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='HY-Motion 1.0 scripts/gradio/static/assets/dump_wooden directory')
    p.add_argument('--out', default='assets/human')
    p.add_argument('--texture-size', type=int, default=1024)
    args = p.parse_args()
    src, out = resolve(args.source), resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load(src)
    jid = {n: i for i, n in enumerate(d['names'])}
    v = d['v'] @ SMPL_TO_WORLD.T
    j = d['j'] @ SMPL_TO_WORLD.T
    shift = np.array([-j[jid['Pelvis'], 0], -j[jid['Pelvis'], 1], -v[:, 2].min()])
    v, j = v + shift, j + shift

    joint_segment = {jid[n]: seg for seg, (_, _, joints) in SEGMENTS.items() for n in joints}
    dominant = d['idx'][np.arange(len(v)), np.argmax(d['w'], axis=1)]
    vert_segment = np.array([joint_segment[int(k)] for k in dominant])
    face_segment = vert_segment[d['faces'][:, 0]]
    posed = {name: pose_fingers(v, j, d['parents'], d['names'], d['w'], d['idx'].astype(int), shape)
             for name, shape in blend_shapes().items()}
    variants = {}
    for seg, (pivot, _, _) in SEGMENTS.items():
        faces = d['faces'][face_segment == seg]
        used, inverse = np.unique(faces, return_inverse=True)
        if seg.endswith('_hand'):
            variants[seg] = {}
            for name, pv in posed.items():
                write_obj(out / f'{seg}_{name}.obj', pv[used] - j[jid[pivot]], d['uv'][used], inverse.reshape(-1, 3))
                variants[seg][name] = f'{seg}_{name}.obj'
        else:
            write_obj(out / f'{seg}.obj', v[used] - j[jid[pivot]], d['uv'][used], inverse.reshape(-1, 3))
    Image.open(src / 'Boy_lambert4_BaseColor.webp').convert('RGB') \
        .resize((args.texture_size, args.texture_size), Image.LANCZOS).save(out / 'wood.png')

    palm = {s: j[[jid[f'{s}_Wrist'], jid[f'{s}_Middle1']]].mean(axis=0) for s in 'LR'}
    # index finger ray of the pointing hand: knuckle (Index1) and fingertip (hand vertex farthest along Index1->Index3)
    index_ray = {}
    for s in 'LR':
        k1, k3 = j[jid[f'{s}_Index1']], j[jid[f'{s}_Index3']]
        hand = np.unique(d['faces'][face_segment == f'{s.lower()}_hand'])
        tip = posed['point'][hand][np.argmax((posed['point'][hand] - k1) @ (k3 - k1))]
        index_ray[s.lower()] = [k1.round(5).tolist(), tip.round(5).tolist()]
    skeleton = dict(
        source='HY-Motion 1.0 wooden body model (Tencent), scripts/gradio/static/assets/dump_wooden',
        license='Tencent HY-Motion 1.0 Community License Agreement (see HY-Motion-1.0/License.txt); '
                'research use within its Territory',
        source_sha256={f: hashlib.sha256((src / f).read_bytes()).hexdigest() for f in SOURCE_FILES},
        frame='world axes: facing -X, Z up, left = -Y; soles at z=0, pelvis above the origin; T-pose',
        height_m=round(float(v[:, 2].max()), 4),
        joints={n: j[jid[n]].round(5).tolist() for n in jid if not any(f in n for f in FINGERS)},
        palm={'l': palm['L'].round(5).tolist(), 'r': palm['R'].round(5).tolist()},
        index_ray=index_ray,
        segments={seg: dict(pivot=pivot, parent=parent, vertices=int(np.unique(d['faces'][face_segment == seg]).size),
                            **(dict(mesh=variants[seg]['relaxed'], variants=variants[seg]) if seg in variants
                               else dict(mesh=f'{seg}.obj')))
                  for seg, (pivot, parent, _) in SEGMENTS.items()},
        texture='wood.png')
    (out / 'skeleton.json').write_text(json.dumps(skeleton, indent=1))
    print(f"wrote {len(SEGMENTS)} segments to {out} (height {skeleton['height_m']} m)")


if __name__ == '__main__':
    main()
