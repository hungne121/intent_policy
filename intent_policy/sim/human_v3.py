"""Articulated human "v3": the HY-Motion SMPL-H skeleton as 22 mocap bodies with real collision geometry.

06/TTr-DL (14/11/2026). Differences to `human_body_v2.HumanBody` (9 rigid segments posed by formulas):
  - one mocap body per SMPL-H body joint (`human_seg_<Joint>`), capsule collision geoms along the bones and a
    connected MuJoCo skin bound to the 22 bodies with the original skinning weights (assets/human_v3, built by
    scripts/build_human_v3.py); the hands keep the mesh variants of assets/human on the wrist bodies;
  - the whole-body pose comes from a 52-joint keypoint frame (HY-Motion clip replay, or the standing rest pose);
    only the working arm is re-solved towards the task target (two-link IK on shoulder / elbow / wrist, elbow
    swivel kept from the base pose), with the waist lean of v2 as a rate-limited fallback when the target is
    out of reach;
  - rotations of base-driven joints are recovered from the children positions (Procrustes with two or more
    children, swing with one child, twist inherited), which reproduces HY-Motion keypoints exactly (14/11 check).

The body collides with the table, the objects and the robot (the scene builder sets contype / conaffinity and
excludes each hand from the task objects so a held object does not fight the hand). Mocap bodies are kinematic:
the robot has to keep clear of the person, the person never yields.
"""
from __future__ import annotations
import json
import mujoco
import numpy as np
from intent_policy.sim.base_env import ROOT
from intent_policy.sim.human_body_v2 import rotation, frame, unit, quat_z_to, SIDE, PALM_DOWN, MAX_WRIST_DEG

SMPL_TO_WORLD = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
FINGER_BASES = {'l': (22, 25, 28, 31, 34), 'r': (37, 40, 43, 46, 49)}      # Index1 Middle1 Pinky1 Ring1 Thumb1
WRIST = {'l': 20, 'r': 21}
ELBOW = {'l': 18, 'r': 19}
SHOULDER = {'l': 16, 'r': 17}
ARM_JOINTS = {s: [SHOULDER[s], ELBOW[s], WRIST[s]] + list(range(22, 37) if s == 'l' else range(37, 52)) for s in 'lr'}
UPPER_BODY = [3, 6, 9, 12, 13, 14, 15] + ARM_JOINTS['l'] + ARM_JOINTS['r']   # rotated by the waist lean (Spine1 pivot)
SELF_PAD_M = 0.05


def _swing(a, b):
    a, b = unit(a), unit(b)
    c = np.cross(a, b); s = np.linalg.norm(c); d = float(np.dot(a, b))
    if s < 1e-9:
        return np.eye(3) if d > 0 else rotation(np.cross(a, [0., 0., 1.]) if abs(a[2]) < .9 else np.cross(a, [1., 0., 0.]), np.pi)
    K = np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - d) / s ** 2)


def _procrustes(A, B):
    U, _, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


class HumanV3:
    def __init__(self, cfg: dict):
        skel = json.loads((ROOT / 'assets/human_v3/skeleton_v3.json').read_text())
        old = json.loads((ROOT / cfg.get('model', 'assets/human/skeleton.json')).read_text())
        self.names = skel['names']; self.parents = np.asarray(skel['parents'], int)
        self.children = {i: [c for c in range(len(self.names)) if self.parents[c] == i] for i in range(len(self.names))}
        offset = np.array([cfg['stand_xy'][0], cfg['stand_xy'][1], cfg['floor_z']], float)
        self.R_yaw = rotation([0., 0., 1.], np.radians(float(cfg.get('facing_deg', 180.)) - 180.))
        self.offset = offset
        self.J0 = np.asarray(skel['joints'], float)                      # T-pose at the origin (asset frame)
        self.J = offset + self.J0 @ self.R_yaw.T                          # placed T-pose (world)
        self.forward = self.R_yaw @ np.array([-1., 0., 0.])
        self.active = cfg.get('active_arm', 'r'); self.default_active = self.active
        self.idle = 'l' if self.active == 'r' else 'r'
        self.max_lean = np.radians(float(cfg.get('max_lean_deg', 60.)))
        self.lean_rate = float(cfg.get('lean_rate_deg_s', 40.)); self.lean_dt = float(cfg.get('lean_dt_s', .05))
        self.reach_margin = float(cfg.get('reach_margin', .9)); self.max_extension = float(cfg.get('max_extension', .95))
        self.hang_deg = float(cfg.get('hang_deg', 78.))                   # rest: arms hanging, this far down from the T-pose
        self._lean_prev = 0.
        self.palm_T = {s: np.asarray(old['palm'][s], float) for s in 'lr'}               # asset-frame palm centres (T-pose)
        self.index_ray = {s: [np.asarray(q, float) for q in old['index_ray'][s]] for s in 'lr'}
        self.shape_order = list(old['segments']['r_hand']['variants'])
        self.shape_state = None
        self.held_object_frame = cfg.get('held_object_frame', 'world')
        for s in 'lr':
            self.__dict__.setdefault('L1', {})[s] = float(np.linalg.norm(self.J0[ELBOW[s]] - self.J0[SHOULDER[s]]))
            self.__dict__.setdefault('L2', {})[s] = float(np.linalg.norm(self.J0[WRIST[s]] - self.J0[ELBOW[s]]))
            self.__dict__.setdefault('Lp', {})[s] = float(np.linalg.norm(self.palm_T[s] - self.J0[WRIST[s]]))
        self.base = self.rest_keypoints()

    # ------------------------------------------------------------------ frames
    def place(self, k_asset):
        """Asset-frame keypoints (T-pose axes, origin at the pelvis) -> world."""
        return self.offset + np.asarray(k_asset, float) @ self.R_yaw.T

    def rest_keypoints(self):
        """Standing rest: the T-pose with both arms rotated down beside the body."""
        k = self.J0.copy()
        for s in 'lr':
            sh = self.J0[SHOULDER[s]]
            R = rotation([1., 0., 0.], np.radians(self.hang_deg) * (1. if s == 'l' else -1.))    # about the asset facing axis
            for i in ARM_JOINTS[s]:
                k[i] = sh + R @ (self.J0[i] - sh)
        return self.place(k)

    def clip_keypoints(self, k3d_frame, height=None):
        """One HY-Motion keypoints3d frame (52, 3; SMPL axes) placed at this person's stand / facing."""
        from intent_policy.sim.motion_bank_v2 import to_scene
        return to_scene(np.asarray(k3d_frame, float)[None], np.array([*self.offset[:2], 0.]),
                        np.degrees(np.arctan2(self.forward[1], self.forward[0])), self.offset[2], height)[0]

    # ------------------------------------------------------------------ MuJoCo binding
    def bind(self, model):
        self.model = model
        self.seg_mocap = {i: int(model.body(f'human_seg_{n}').mocapid[0]) for i, n in enumerate(self.names[:22])}
        self.anchor_mocap = int(model.body('human_grasp_anchor').mocapid[0])
        self.hand_geoms = {}
        for s in 'lr':
            self.hand_geoms[s] = {}
            for v in self.shape_order + [f'{a}-{b}-{p}' for a in self.shape_order for b in self.shape_order for p in (25, 50, 75)]:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'human_hand_{s}_{v}')
                if gid >= 0: self.hand_geoms[s][v] = gid
        self.hand_geom = self.hand_geoms[self.active]['relaxed']
        self.proxy_mocap = {}
        for name in ('hand', 'forearm', 'upperarm', 'torso', 'head'):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'human_{name}')
            if bid >= 0: self.proxy_mocap[name] = int(model.body_mocapid[bid])

    def set_active(self, side):
        if side not in 'lr' or side == self.active: return
        self.active, self.idle = side, self.active
        self.shape_state = None

    def visible_shape(self, side):
        for v, g in self.hand_geoms.get(side, {}).items():
            if self.model.geom_rgba[g, 3] > 0: return v
        return None

    def _variant(self, target, shape_from=None, progress=None):
        if progress is not None and shape_from and shape_from != target and shape_from in self.shape_order and target in self.shape_order:
            pct = int(round(float(np.clip(progress, 0., 1.)) * 4)) * 25
            if pct in (0, 100): return shape_from if pct == 0 else target
            x, y, q = (shape_from, target, pct) if self.shape_order.index(shape_from) < self.shape_order.index(target) else (target, shape_from, 100 - pct)
            return f'{x}-{y}-{q}'
        return target if target in self.shape_order else 'relaxed'

    # ------------------------------------------------------------------ pose
    def global_rotations(self, k):
        """World rotation of every joint body from its children positions (exact for HY-Motion keypoints)."""
        n = len(self.names); R = [np.eye(3)] * n
        for i in range(n):
            ch = self.children[i]
            if not ch:
                R[i] = R[self.parents[i]]; continue
            D = np.array([self.J0[c] - self.J0[i] for c in ch]); E = np.array([k[c] - k[i] for c in ch])   # asset axes -> world
            if len(ch) >= 2 and np.linalg.matrix_rank(D) >= 2:
                R[i] = _procrustes(D, E)
            else:
                Rp = R[self.parents[i]] if self.parents[i] >= 0 else np.eye(3)
                R[i] = _swing(Rp @ D[0], E[0]) @ Rp
        return R

    def _arm(self, s, shoulder, target, palm_normal, fore_normal=None, hand=None, swivel=None):
        """Two-link IK (as human_body_v2._arm); `swivel`: preferred elbow direction (unit, from the base pose)."""
        side = SIDE[s]
        L1, L2 = self.L1[s], self.L2[s] + (self.Lp[s] if hand is None else 0.)
        end = np.asarray(target, float) - (0. if hand is None else self.Lp[s] * np.asarray(hand, float))
        v = end - shoulder
        D = float(np.clip(np.linalg.norm(v), abs(L1 - L2) + 1e-3, min(L1 + L2 - 1e-4, self.max_extension * (L1 + L2))))
        u = unit(v)
        a = np.arccos(np.clip((L1 ** 2 + D ** 2 - L2 ** 2) / (2 * L1 * D), -1., 1.))
        hint = swivel if swivel is not None else unit(self.R_yaw @ [0.35, 0.6 * side, -1.0])
        w = hint - np.dot(hint, u) * u
        w = unit(w) if np.linalg.norm(w) > 1e-6 else np.array([0., 0., -1.])
        elbow = shoulder + L1 * (np.cos(a) * u + np.sin(a) * w)
        fore = unit(shoulder + D * u - elbow)
        wrist = elbow + self.L2[s] * fore
        hand = fore if hand is None else np.asarray(hand, float)
        n = palm_normal if fore_normal is None else fore_normal
        J0, S = self.J0, s.upper()
        rest = frame(J0[WRIST[s]] - J0[ELBOW[s]], PALM_DOWN).T                   # asset axes -> canonical bone frame
        R_up = frame(elbow - shoulder, palm_normal) @ frame(J0[ELBOW[s]] - J0[SHOULDER[s]], PALM_DOWN).T
        return dict(elbow=elbow, wrist=wrist, palm=wrist + self.Lp[s] * hand, R_up=R_up,
                    R_fo=frame(fore, n) @ rest, R_hand=frame(hand, n) @ rest)

    def _level_hand(self, s, shoulder, target, palm_normal, fore_normal, arm, palm_up, swivel):
        for _ in range(3):
            fore = unit(arm['wrist'] - arm['elbow']); flat = np.r_[fore[:2], 0.]
            if np.linalg.norm(flat) < 1e-6: return arm
            flat = unit(flat)
            pitch = min(np.arccos(np.clip(np.dot(fore, flat), -1., 1.)), np.radians(MAX_WRIST_DEG)) * palm_up
            hand = rotation(np.cross(fore, flat), pitch) @ fore
            arm = self._arm(s, shoulder, target, palm_normal, fore_normal, hand, swivel)
        return arm

    def _aim_hand(self, s, arm, target, blend=1.):
        J0 = self.J0; wrist = arm['wrist']; fore = unit(arm['wrist'] - arm['elbow'])
        rest = frame(J0[WRIST[s]] - J0[ELBOW[s]], PALM_DOWN).T
        rot = lambda aim: frame(aim, PALM_DOWN) @ rest
        aim = unit(target - wrist)
        k, tip = (q - J0[WRIST[s]] for q in self.index_ray[s])
        for _ in range(6):
            R = rot(aim); knuckle, finger = wrist + R @ k, unit(R @ (tip - k))
            want = unit(target - knuckle)
            axis, ang = np.cross(finger, want), np.arccos(np.clip(np.dot(finger, want), -1., 1.))
            if ang < 1e-4: break
            aim = unit(rotation(axis, ang) @ aim)
        angle = np.arccos(np.clip(np.dot(fore, aim), -1., 1.))
        if angle > np.radians(MAX_WRIST_DEG):
            aim = rotation(np.cross(fore, aim), np.radians(MAX_WRIST_DEG)) @ fore; angle = np.radians(MAX_WRIST_DEG)
        blend = float(np.clip(blend, 0., 1.))
        if blend < 1. and angle > 1e-6: aim = rotation(np.cross(fore, aim), blend * angle) @ fore
        return rot(aim)

    def lean_needed(self, k, palm, s):
        """Smallest waist lean (about Spine1) bringing `palm` within reach of the shoulder of arm s."""
        spine, sh = k[3], k[SHOULDER[s]]
        reach = (self.L1[s] + self.L2[s] + self.Lp[s]) * self.reach_margin
        dh = np.asarray(palm)[:2] - spine[:2]; dh = unit(dh) if np.linalg.norm(dh) > 1e-6 else self.forward[:2]
        axis = np.cross([0., 0., 1.], [dh[0], dh[1], 0.])
        gap = lambda th: np.linalg.norm(palm - (spine + rotation(axis, th) @ (sh - spine))) - reach
        if gap(0.) <= 0: return 0., axis
        if gap(self.max_lean) > 0: return self.max_lean, axis
        lo, hi = 0., self.max_lean
        for _ in range(30):
            mid = .5 * (lo + hi); lo, hi = (mid, hi) if gap(mid) > 0 else (lo, mid)
        return hi, axis

    def pose(self, palm, point_at=None, palm_up=0., aim_blend=1., idle_palm=None, idle_palm_up=0., base=None):
        """World keypoints (52, 3) and body rotations (52) for the task targets on top of a base body pose."""
        k = np.array(self.base if base is None else base, float)
        palm = np.asarray(palm, float)
        # waist lean fallback (rate limited): rotate the upper body about Spine1 when the palm is out of reach
        theta, axis = self.lean_needed(k, palm, self.active)
        step = np.radians(self.lean_rate) * self.lean_dt
        theta = float(np.clip(theta, self._lean_prev - step, self._lean_prev + step))
        self._theta = theta
        if abs(theta) > 1e-6:
            Rt = rotation(axis, theta); spine = k[3].copy()
            for i in UPPER_BODY: k[i] = spine + Rt @ (k[i] - spine)
        R = self.global_rotations(k)
        arms = {}
        for s, target, up in ((self.active, palm, palm_up), (self.idle, idle_palm, idle_palm_up)):
            if target is None: continue
            sh = k[SHOULDER[s]]
            swivel = unit(k[ELBOW[s]] - sh)
            normal = PALM_DOWN; fore_normal = None
            if up > 0:
                medial = -SIDE[s] * (self.R_yaw @ np.array([0., 1., 0.]))
                fore_normal = np.cos(np.pi * up) * PALM_DOWN + np.sin(np.pi * up) * medial
            arm = self._arm(s, sh, np.asarray(target, float), normal, fore_normal, None, swivel)
            if fore_normal is not None and (point_at is None or s != self.active):
                arm = self._level_hand(s, sh, np.asarray(target, float), normal, fore_normal, arm, up, swivel)
            if point_at is not None and s == self.active:
                arm['R_hand'] = self._aim_hand(s, arm, np.asarray(point_at, float), aim_blend)
            k[ELBOW[s]] = arm['elbow']; k[WRIST[s]] = arm['wrist']
            for f in ARM_JOINTS[s][3:]:                                  # fingers follow the hand rotation
                k[f] = arm['wrist'] + arm['R_hand'] @ (self.J0[f] - self.J0[WRIST[s]])
            R[SHOULDER[s]], R[ELBOW[s]], R[WRIST[s]] = arm['R_up'], arm['R_fo'], arm['R_hand']
            arms[s] = arm
        a = arms.get(self.active)
        reached = a['palm'] if a is not None else k[WRIST[self.active]]
        return dict(k=k, R=R, palm=reached, reach_error=float(np.linalg.norm(reached - palm)), lean=theta,
                    idle_palm=arms[self.idle]['palm'] if self.idle in arms else self.palm_of(k, self.idle),
                    arms=arms)

    def palm_of(self, k, s):
        return k[[WRIST[s], *FINGER_BASES[s][:4]]].mean(axis=0)

    def apply(self, data, palm, point_at=None, hand_shape='relaxed', palm_up=0., smooth=False, aim_blend=1.,
              shape_from=None, shape_progress=None, idle_palm=None, idle_shape=None, idle_palm_up=0., base=None):
        pose = self.pose(palm, point_at, palm_up, aim_blend, idle_palm, idle_palm_up, base)
        self._lean_prev = pose['lean']
        k, R = pose['k'], pose['R']
        quat = np.zeros(4)
        for i, mid in self.seg_mocap.items():
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(R[i]).ravel())
            data.mocap_pos[mid] = k[i]; data.mocap_quat[mid] = quat
        # hand shapes: one visible variant per side
        for s, shape, frm, prog in ((self.active, hand_shape, shape_from, shape_progress), (self.idle, idle_shape or 'relaxed', None, None)):
            v = self._variant(shape, frm, prog)
            for name, g in self.hand_geoms[s].items():
                self.model.geom_rgba[g, 3] = float(name == v)
            if s == self.active: self.hand_geom = self.hand_geoms[s][v]
        # grasp anchor at the (reached) palm; legacy proxies follow the arm
        data.mocap_pos[self.anchor_mocap] = pose['palm']
        if self.held_object_frame == 'hand':
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(R[WRIST[self.active]]).ravel()); data.mocap_quat[self.anchor_mocap] = quat
        s = self.active
        for name, (p, q) in {'hand': (pose['palm'], None), 'forearm': (k[ELBOW[s]], k[WRIST[s]]), 'upperarm': (k[SHOULDER[s]], k[ELBOW[s]]),
                             'torso': (k[3], k[12]), 'head': (k[15] + [0, 0, .06], None)}.items():
            mid = self.proxy_mocap.get(name)
            if mid is None: continue
            if q is None: data.mocap_pos[mid] = p
            else: data.mocap_pos[mid] = (p + q) / 2; data.mocap_quat[mid] = quat_z_to(q - p)
        pose['segments'] = {}
        return pose

    def keypoints(self, pose, names):
        k = pose['k']; out = []
        for name in names:
            if name == 'head': out.append(k[15]); continue
            s, part = name.split('_', 1)
            if part == 'wrist': out.append(k[WRIST[s]])
            elif part == 'elbow': out.append(k[ELBOW[s]])
            elif part == 'shoulder': out.append(k[SHOULDER[s]])
            elif part in ('index_base', 'index_tip'):
                q = self.index_ray[s][0 if part == 'index_base' else 1] - self.J0[WRIST[s]]
                out.append(k[WRIST[s]] + pose['R'][WRIST[s]] @ q)
            else: raise KeyError(name)
        return np.asarray(out, float)

    def hanging_palm(self, side):
        return self.palm_of(self.base, side)

    def clamp(self, palm):
        return np.asarray(palm, float)
