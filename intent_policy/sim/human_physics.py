"""Force-driven standing human; HY-Motion skin and targets, MuJoCo actual pose.

Pelvis is supported/fixed, all other body joints have finite torque and limits.
This is a tabletop dynamics model, not a validated biomechanics/walking model.
"""
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from intent_policy.sim.base_env import ROOT
from intent_policy.sim.human_v3 import HumanV3, WRIST, ELBOW, SHOULDER, ARM_JOINTS


def physics_config():
    return yaml.safe_load((ROOT / 'configs/previews/human_physics.yaml').read_text())


def articulate_human(root, world, human):
    """Reparent v3 assets into a physical skeleton, preserving names/skin binding."""
    cfg = physics_config()
    # The tabletop and robot must use the same contact compliance as the person.
    # Otherwise a force-limited gripper can still sink centimetres into a surface
    # using the scene's softer default contact parameters.
    for geom in world.findall('.//geom'):
        if geom.get('contype', '1') == '0' and geom.get('conaffinity', '1') == '0':
            continue
        geom.set('priority', '1')
        geom.set('solref', ' '.join(map(str, cfg['contact_solref'])))
        geom.set('solimp', ' '.join(map(str, cfg['contact_solimp'])))
    skel = json.loads((ROOT / 'assets/human_v3/skeleton_v3.json').read_text())
    names, parents, joints = skel['names'][:22], skel['parents'], np.asarray(skel['joints'])
    bodies = [world.find(f"body[@name='human_seg_{name}']") for name in names]
    for i, body in enumerate(bodies):
        world.remove(body)
        body.attrib.pop('mocap', None)
        body.set('pos', ' '.join(map(str, joints[i] if i == 0 else joints[i] - joints[parents[i]])))
        if i:
            limit = cfg['spine_limit_deg'] if names[i].startswith(('Spine', 'Neck', 'Head', 'L_Collar', 'R_Collar')) else cfg['joint_limit_deg']
            if names[i].endswith('Wrist'):
                limit = cfg['wrist_limit_deg']
            # Existing scene compiler uses radians.
            angle = root.find('compiler').get('angle', 'degree')
            value = np.radians(limit) if angle == 'radian' else limit
            damping = (cfg['torso_kd'] if names[i].startswith('Spine') else
                       cfg['wrist_kd'] if names[i].endswith('Wrist') else
                       cfg['arm_kd'] if names[i].endswith(('Shoulder', 'Elbow')) else cfg['kd'])
            # Native joint damping is integrated implicitly by MuJoCo.  Applying
            # the same large velocity term as an external force at 500 Hz made
            # light limbs oscillate and repeatedly saturated the servos.
            ET.SubElement(body, 'joint', name=f'human_joint_{names[i]}', type='ball', limited='true',
                          range=f'0 {value}', damping=str(damping), armature=str(cfg['joint_armature']))
        physical = []
        for geom in body.findall('geom'):
            if names[i].endswith('Elbow') and geom.get('name', '').endswith('_Wrist'):
                # Keep the forearm's rounded end out of the palm's support
                # region.  The wrist and fingers retain their own colliders.
                a = np.fromstring(geom.get('fromto'), sep=' ')
                a[3:] *= .86
                geom.set('fromto', ' '.join(map(str, a)))
            if geom.get('contype', '1') == '0' and geom.get('conaffinity', '1') == '0':
                geom.set('mass', '0')
            else:
                physical.append(geom)
                geom.set('priority', '1')
                if names[i].endswith('Ankle'):
                    geom.set('size', str(cfg['ankle_radius_m']))
                if geom.get('name', '').endswith('_palm'):
                    # A palm is a support surface, not a ball. Asset palm normal
                    # is Z; local Y follows the fingers in the bind pose.
                    geom.set('type', 'box')
                    geom.set('size', ' '.join(map(str, cfg['palm_half_size_m'])))
                geom.set('density', str(cfg['density_kg_m3']))
                geom.set('solref', ' '.join(map(str, cfg['contact_solref'])))
                if geom.get('name', '').endswith('_palm'):
                    geom.set('solref', ' '.join(map(str, cfg['palm_contact_solref'])))
                geom.set('solimp', ' '.join(map(str, cfg['contact_solimp'])))
                geom.set('friction', ' '.join(map(str, cfg['contact_friction'])))
        if not physical:
            # Terminal feet need inertia even though their collision belongs to the ankle.
            ET.SubElement(body, 'inertial', pos='0 0 0', mass='.1', diaginertia='.001 .001 .001')
    world.append(bodies[0])
    for i in range(1, len(bodies)):
        bodies[parents[i]].append(bodies[i])
    contact = root.find('contact')
    if contact is None:
        contact = ET.SubElement(root, 'contact')
    # Geometry overlaps by construction around articulations. Exclude only local
    # neighbours (<=2 skeletal edges), never hand-vs-trunk or hand-vs-task-object.
    def ancestors(i):
        result = {i: 0}
        while i:
            p = parents[i]; result[p] = result[i] + 1; i = p
        return result
    ancestry = [ancestors(i) for i in range(22)]
    for i in range(22):
        for j in range(i):
            distance = min(ancestry[i][k] + ancestry[j][k] for k in ancestry[i].keys() & ancestry[j].keys())
            torso = {'Pelvis', 'Spine1', 'Spine2', 'Spine3', 'L_Collar', 'R_Collar', 'Neck', 'Head'}
            shoulder_attachment = ((names[i].endswith('Shoulder') and names[j] in {'Spine2', 'Spine3', 'Neck'})
                                   or (names[j].endswith('Shoulder') and names[i] in {'Spine2', 'Spine3', 'Neck'}))
            if distance <= 2 or (names[i] in torso and names[j] in torso) or shoulder_attachment:
                ET.SubElement(contact, 'exclude', body1=bodies[i].get('name'), body2=bodies[j].get('name'))


class PhysicalHuman(HumanV3):
    """Same target API as v3, but apply only sets PD targets after initialization."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.physics = physics_config()
        self.initialized = False
        self.target_pose = None
        self.last_torque = None

    def bind(self, model):
        super().bind(model)
        self.body_ids = np.array([model.body(f'human_seg_{n}').id for n in self.names[:22]])
        self.joints = np.array([model.joint(f'human_joint_{n}').id for n in self.names[1:22]])
        self.qadr = model.jnt_qposadr[self.joints]
        self.dadr = model.jnt_dofadr[self.joints]
        self.dofs = np.array([np.arange(a, a + 3) for a in self.dadr])
        root = self.body_ids[0]
        model.body_pos[root] = self.J[0]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.R_yaw.ravel())
        model.body_quat[root] = quat
        self.targets = np.tile([1., 0., 0., 0.], (21, 1))
        # HumanV3 switches hand variants visually.  In a physical model the
        # hidden variants must also stop colliding; otherwise an invisible
        # curled finger and an invisible open palm can push the same object.
        self.hand_collision = {
            side: {variant: (int(model.geom_contype[g]), int(model.geom_conaffinity[g]))
                   for variant, g in variants.items()}
            for side, variants in self.hand_geoms.items()
        }
        wrist = np.array([n.endswith('Wrist') for n in self.names[1:22]])
        self.kp = np.where(wrist, self.physics['wrist_kp'], self.physics['kp'])
        self.kd = np.where(wrist, self.physics['wrist_kd'], self.physics['kd'])
        self.torque_limit = np.where(wrist, self.physics['wrist_max_torque_nm'], self.physics['max_torque_nm'])
        # The target arm has to carry the hand and an object.  Keep this
        # finite, and do not apply it to the torso/legs: otherwise a stronger
        # reach makes the whole standing mannequin unnaturally stiff.
        arm = np.array([n.endswith(('Shoulder', 'Elbow')) for n in self.names[1:22]])
        self.kp[arm] = self.physics['arm_kp']
        self.kd[arm] = self.physics['arm_kd']
        self.torque_limit[arm] = self.physics['arm_max_torque_nm']
        torso = np.array([n.startswith('Spine') for n in self.names[1:22]])
        self.kp[torso] = self.physics['torso_kp']
        self.kd[torso] = self.physics['torso_kd']
        self.torque_limit[torso] = self.physics['torso_max_torque_nm']

    def control(self, model, data):
        """Physics-rate torque servo. No qpos/qvel writes, robot commands untouched."""
        if not self.initialized:
            return
        errors = np.empty((21, 3))
        for j, q in enumerate(self.qadr):
            mujoco.mju_subQuat(errors[j], self.targets[j], data.qpos[q:q + 4])
        torque = self.kp[:, None] * errors
        torque += data.qfrc_bias[self.dofs]
        norms = np.linalg.norm(torque, axis=1)
        torque *= np.minimum(1., self.torque_limit / np.maximum(norms, 1e-12))[:, None]
        data.qfrc_applied[self.dofs] = torque
        self.last_torque = torque.copy()

    def current_pose(self, data):
        k = self.J.copy()
        R = [self.R_yaw.copy() for _ in self.names]
        for i, bid in enumerate(self.body_ids):
            k[i] = data.xpos[bid]
            R[i] = data.xmat[bid].reshape(3, 3).copy()
        for s in 'lr':
            for i in ARM_JOINTS[s][3:]:
                R[i] = R[WRIST[s]]
                k[i] = k[WRIST[s]] + R[i] @ (self.J0[i] - self.J0[WRIST[s]])
        palms = {s: k[WRIST[s]] + R[WRIST[s]] @ (self.palm_T[s] - self.J0[WRIST[s]]) for s in 'lr'}
        target = palms[self.active] if self.target_pose is None else self.target_pose['palm']
        return dict(k=k, R=R, palm=palms[self.active], idle_palm=palms[self.idle],
                    reach_error=float(np.linalg.norm(palms[self.active] - target)),
                    lean=self._lean_prev, arms={}, segments={})

    def apply(self, data, palm, point_at=None, hand_shape='relaxed', palm_up=0., smooth=False, aim_blend=1.,
              shape_from=None, shape_progress=None, idle_palm=None, idle_shape=None, idle_palm_up=0., base=None):
        pose = self.pose(palm, point_at, palm_up, aim_blend, idle_palm, idle_palm_up, base)
        self.target_pose = pose
        self._lean_prev = pose['lean']
        for i in range(1, 22):
            parent_rotation = self.R_yaw if self.parents[i] == 0 else pose['R'][self.parents[i]]
            local = parent_rotation.T @ pose['R'][i]
            mujoco.mju_mat2Quat(self.targets[i - 1], np.ascontiguousarray(local).ravel())
        if not self.initialized:
            # One reset into the initial pose; all subsequent motion goes through dynamics.
            for q, quat in zip(self.qadr, self.targets):
                data.qpos[q:q + 4] = quat
            self.initialized = True
            mujoco.mj_forward(self.model, data)
        for s, shape, frm, prog in ((self.active, hand_shape, shape_from, shape_progress),
                                    (self.idle, idle_shape or 'relaxed', None, None)):
            variant = self._variant(shape, frm, prog)
            for name, g in self.hand_geoms[s].items():
                self.model.geom_rgba[g, 3] = float(name == variant)
                if name == variant:
                    contype, conaffinity = self.hand_collision[s][name]
                    self.model.geom_contype[g] = contype
                    self.model.geom_conaffinity[g] = conaffinity
                else:
                    self.model.geom_contype[g] = 0
                    self.model.geom_conaffinity[g] = 0
            if s == self.active:
                self.hand_geom = self.hand_geoms[s][variant]
        return self.current_pose(data)
