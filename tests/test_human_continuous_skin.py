"""Regression checks for the I1 skin: connected topology and unchanged audit visibility."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from intent_policy.sim.scene_builder_v2 import add_continuous_human_skin

ASSETS = Path(__file__).resolve().parents[1] / 'assets/human'


def test_skin_weights_reconstruct_bind_pose_and_blend_at_joints():
    skin = ET.parse(ASSETS / 'surface_skin.xml').getroot()
    vertices = np.fromstring(skin.get('vertex'), sep=' ').reshape(-1, 3)
    faces = np.fromstring(skin.get('face'), sep=' ', dtype=int).reshape(-1, 3)
    total = np.zeros(len(vertices)); count = np.zeros(len(vertices), int)
    reconstructed = np.zeros_like(vertices)
    for bone in skin.findall('bone'):
        ids = np.fromstring(bone.get('vertid'), sep=' ', dtype=int)
        w = np.fromstring(bone.get('vertweight'), sep=' ')
        pivot = np.fromstring(bone.get('bindpos'), sep=' ')
        total[ids] += w; count[ids] += 1
        reconstructed[ids] += ((vertices[ids] - pivot) + pivot) * w[:, None]
    assert faces.min() == 0 and faces.max() < len(vertices)
    assert np.allclose(total, 1, atol=1e-6)
    assert np.allclose(reconstructed, vertices, atol=1e-6)
    assert np.count_nonzero(count > 1) > 1000  # joints must not regress to rigid dominant-bone cuts


def test_skin_does_not_disable_collision_audit_or_hand_variant_selection():
    root = ET.fromstring('<mujoco><asset/><worldbody><body name="human_seg_r_hand">'
                        '<geom name="human_seg_r_hand_mesh" rgba="1 1 1 1" contype="0"/>'
                        '<geom name="human_seg_r_hand_variant_grasp" rgba="1 1 1 0" contype="0"/>'
                        '</body></worldbody></mujoco>')
    geoms = root.findall('./worldbody/body/geom')
    before = [(g.get('rgba'), g.get('contype')) for g in geoms]
    add_continuous_human_skin(root, ASSETS)
    add_continuous_human_skin(root, ASSETS)
    assert len(root.findall('./asset/skin')) == 1
    assert before == [(g.get('rgba'), g.get('contype')) for g in geoms]
    assert all(g.get('group') == '5' for g in geoms)


def test_finger_variants_change_only_hand_vertices():
    with np.load(ASSETS / 'surface_fingers.npz', allow_pickle=False) as data:
        assert not np.intersect1d(data['l_ids'], data['r_ids']).size
        for side in 'lr':
            delta = data[side+'_grasp'] - data[side+'_relaxed']
            assert np.isfinite(delta).all()
            assert .005 < np.max(np.linalg.norm(delta, axis=1)) < .2
