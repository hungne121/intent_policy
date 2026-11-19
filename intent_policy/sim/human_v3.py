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

Collision masks enable contact with dynamic objects and the robot. Mocap bodies are kinematic: contact does not
stop their commanded motion, and static-world/self intersections need explicit geometry checks. The robot has
to keep clear of the person; the person never yields. Task objects are not blanket-excluded from hand contact.
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
# 61/QĐ-DL: elbow swivel that keeps the arm clear of the trunk (the palm target is unchanged)
SWIVEL_STEPS_DEG = (0., 10., -10., 20., -20., 30., -30., 45., -45., 60., -60., 75., -75., 90., -90.)
SWIVEL_RATE_DEG = 8.          # per control tick (0.05 s): 160 deg/s
SWIVEL_CLEAR_M = 0.003        # solver target clearance arm-trunk (the gate allows 0.5 mm penetration)
WORLD_CLEAR_M = 0.0           # hands may touch rigid objects, never pass into them
PUSH_DECAY = 0.85             # 66/QĐ-DL: per 20 Hz tick, ~0.3 s time constant
TRUNK = ('Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head', 'L_Collar', 'R_Collar')


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
        self.rest_lean_free = bool(cfg.get('rest_lean_free', False))   # 29/QĐ-HL: no lean for the resting arm
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
        # 61/QĐ-DL: collision geoms of each forearm / hand against the trunk (as V3SelfCheck, shoulder excluded)
        coll = lambda bodies: [g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name in bodies
                               and (model.geom_contype[g] or model.geom_conaffinity[g])]
        trunk = coll({f'human_seg_{j}' for j in TRUNK})
        self.self_pairs = {s: [(a, b) for a in coll({f'human_seg_{S}_Elbow', f'human_seg_{S}_Wrist'}) for b in trunk]
                           for s, S in (('l', 'L'), ('r', 'R'))}
        self.swivel = {'l': 0., 'r': 0.}
        self.world_geoms = []          # set by the episode: rigid non-human, non-robot collision geoms
        self.skip_bodies = lambda: set()   # body ids the hands may touch now (objects held by the person)
        self.held_geoms = lambda s: []     # collision geoms of the objects arm s carries: they move with the hand
        self.hand_world_geoms = {s: coll({f'human_seg_{S}_Elbow', f'human_seg_{S}_Wrist'}) for s, S in (('l', 'L'), ('r', 'R'))}
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
        # elbow direction: the base pose's swivel blended with the outward / back / down hint of v2, so an arm that
        # starts from the hanging rest does not fold its forearm through the chest when reaching forward
        outward = unit(self.R_yaw @ [0.35, 0.6 * side, -1.0])
        hint = unit(swivel + outward) if swivel is not None else outward
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

    def pose(self, palm, point_at=None, palm_up=0., aim_blend=1., idle_palm=None, idle_palm_up=0., base=None, swivel=None):
        """World keypoints (52, 3) and body rotations (52) for the task targets on top of a base body pose."""
        k = np.array(self.base if base is None else base, float)
        palm = np.asarray(palm, float)
        swivel_offsets = swivel
        # waist lean fallback (rate limited): rotate the upper body about Spine1 when the palm is out of reach
        theta, axis = self.lean_needed(k, palm, self.active)
        if self.rest_lean_free:        # 29/QĐ-HL: only the lean beyond what the hanging arm itself needs
            theta = max(0., theta - self.lean_needed(k, self.palm_of(self.base, self.active), self.active)[0])
        if idle_palm is not None:      # 56/QĐ-DL: lean for the farther hand (as human_body_v2.pose); C3 brushing hand
            th2, ax2 = self.lean_needed(k, np.asarray(idle_palm, float), self.idle)
            if self.rest_lean_free:
                th2 = max(0., th2 - self.lean_needed(k, self.palm_of(self.base, self.idle), self.idle)[0])
            if th2 > theta: theta, axis = th2, ax2
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
            phi = 0. if swivel_offsets is None else swivel_offsets.get(s, 0.)
            if phi:                                    # 61/QĐ-DL: rotate the preferred elbow about shoulder->target
                swivel = rotation(unit(np.asarray(target, float) - sh), np.radians(phi)) @ swivel
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

    def _write(self, data, pose):
        quat = np.zeros(4)
        for i, mid in self.seg_mocap.items():
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(pose['R'][i]).ravel())
            data.mocap_pos[mid] = pose['k'][i]; data.mocap_quat[mid] = quat

    def arm_clearance(self, data, s, away=False):
        """Min signed distance (m) of arm s (forearm, hand) to the trunk at the current mocap pose; with `away`
        also the unit vector from the closest trunk point to the closest arm point."""
        best, vec, ft = np.inf, None, np.zeros(6)
        for a, b in self.self_pairs[s]:
            dist = mujoco.mj_geomDistance(self.model, data, a, b, .05, ft)
            if dist < best:
                # fromto runs trunk -> arm surface when apart; when they overlap it points INTO the trunk: flip it
                best = dist; v = (ft[:3] - ft[3:]) * (1. if dist >= 0 else -1.); n = np.linalg.norm(v)
                vec = v / n if n > 1e-9 else None
        return (best, vec) if away else best

    def world_clearance(self, data, s):
        """(min signed distance, unit push direction) of arm s (forearm, hand) to rigid world geoms not held now."""
        skip = self.skip_bodies(); best, vec, ft = np.inf, None, np.zeros(6)
        held = list(self.held_geoms(s))
        # carried objects follow the hand only at the next physics step: shift them by the candidate hand motion
        shift = np.zeros(3)
        if held and getattr(self, '_palm_prev', {}).get(s) is not None and getattr(self, '_palm_now', {}).get(s) is not None:
            shift = self._palm_now[s] - self._palm_prev[s]
        saved = data.geom_xpos[held].copy() if held else None
        if held:
            data.geom_xpos[held] += shift
        try:
            W = np.asarray(self.world_geoms, int)
            wpos, wr = data.geom_xpos[W], self.model.geom_rbound[W]
            for a in list(self.hand_world_geoms[s]) + held:
                # broad phase (speed only): bounding spheres farther apart than the 3 cm query cannot be closer
                near = W[np.linalg.norm(wpos - data.geom_xpos[a], axis=1) - wr - self.model.geom_rbound[a] < .03]
                for b in near:
                    if self.model.geom_bodyid[b] in skip:
                        continue
                    dist = mujoco.mj_geomDistance(self.model, data, a, b, .03, ft)
                    if dist < best:
                        best = dist; v = (ft[:3] - ft[3:]) * (1. if dist >= 0 else -1.); n = np.linalg.norm(v)
                        vec = v / n if n > 1e-9 else None
        finally:
            if held:
                data.geom_xpos[held] = saved
        return best, vec

    def _best_swivel(self, data, s, pose, *args):
        """Smallest swivel offset (deg) keeping arm s >= SWIVEL_CLEAR_M from the trunk, preferring the current one.
        Leaves the mocap state of the CURRENT pose in `data` on return."""
        cur = self.swivel[s]
        if self.arm_clearance(data, s) >= SWIVEL_CLEAR_M and abs(cur) < 1e-9:
            return 0.
        palm, point_at, palm_up, aim_blend, idle_palm, idle_palm_up, base, lean_prev = args
        results = []
        for phi in SWIVEL_STEPS_DEG:
            offs = dict(self.swivel); offs[s] = phi
            self._lean_prev = lean_prev
            cand = self.pose(palm, point_at, palm_up, aim_blend, idle_palm, idle_palm_up, base, swivel=offs)
            self._write(data, cand); mujoco.mj_kinematics(self.model, data)
            c = self.arm_clearance(data, s)
            results.append((phi, c))
            if c >= SWIVEL_CLEAR_M and abs(phi) <= abs(cur) + 1e-9:
                break                                   # the smallest |phi| that is clear (back towards 0)
        self._lean_prev = lean_prev
        self._write(data, pose); mujoco.mj_kinematics(self.model, data)
        clear = [r for r in results if r[1] >= SWIVEL_CLEAR_M]
        if clear:
            return min(clear, key=lambda r: (abs(r[0]), abs(r[0] - cur)))[0]
        return max(results, key=lambda r: r[1])[0]       # none clear: the least penetrating

    def palm_of(self, k, s):
        return k[[WRIST[s], *FINGER_BASES[s][:4]]].mean(axis=0)

    def apply(self, data, palm, point_at=None, hand_shape='relaxed', palm_up=0., smooth=False, aim_blend=1.,
              shape_from=None, shape_progress=None, idle_palm=None, idle_shape=None, idle_palm_up=0., base=None):
        lean_prev = self._lean_prev
        # 66/QĐ-DL: push memory. The correction found last tick is kept and decays (time constant ~0.3 s) instead of
        # vanishing: without it a 4 cm push out of the table one tick, a sideways push off a cup the next and nothing
        # the tick after made the C1 hand oscillate into the cup (5-15 mm, up to 936 N).
        mem = getattr(self, '_push_mem', {})
        cmd_palm = np.asarray(palm, float); cmd_idle = None if idle_palm is None else np.asarray(idle_palm, float)
        if mem.get(self.active) is not None:
            palm = cmd_palm + mem[self.active] * PUSH_DECAY
        if cmd_idle is not None and mem.get(self.idle) is not None:
            idle_palm = cmd_idle + mem[self.idle] * PUSH_DECAY
        pose = self.pose(palm, point_at, palm_up, aim_blend, idle_palm, idle_palm_up, base, swivel=dict(self.swivel))
        sides = [x for x in (self.active, self.idle) if x in pose['arms']]
        if getattr(self, 'self_pairs', None) and sides:
            self._write(data, pose); mujoco.mj_kinematics(self.model, data)
            changed = False
            for s in sides:
                want = self._best_swivel(data, s, pose, palm, point_at, palm_up, aim_blend, idle_palm, idle_palm_up, base, lean_prev)
                step = float(np.clip(want - self.swivel[s], -SWIVEL_RATE_DEG, SWIVEL_RATE_DEG))
                if abs(step) > 1e-9:
                    self.swivel[s] += step; changed = True
            if changed:
                self._lean_prev = lean_prev
                pose = self.pose(palm, point_at, palm_up, aim_blend, idle_palm, idle_palm_up, base, swivel=dict(self.swivel))
            # 61/QĐ-DL step 2: swivel alone cannot clear a hand target that lies against the body -> move that palm
            # target away from the trunk by the remaining overlap (a few mm to ~1.5 cm), up to 4 iterations
            self._write(data, pose); mujoco.mj_kinematics(self.model, data)
            palm, idle_palm = np.asarray(palm, float), None if idle_palm is None else np.asarray(idle_palm, float)
            self.self_push_m = {}
            self._palm_now = {self.active: pose['palm'], self.idle: pose['idle_palm']}
            for _ in range(4):
                moved = False
                for s in sides:
                    # 61/QĐ-DL: out of the trunk AND out of rigid world geoms (a kinematic hand, with what it carries,
                    # must not pass into objects / the table: touch = 0); both corrections in the same iteration
                    d = np.zeros(3)
                    c, v = self.arm_clearance(data, s, away=True)
                    if c < SWIVEL_CLEAR_M and v is not None:
                        d += (SWIVEL_CLEAR_M - c) * v
                    c, v = self.world_clearance(data, s) if self.world_geoms else (np.inf, None)
                    if c < WORLD_CLEAR_M and v is not None:
                        d += (WORLD_CLEAR_M - c) * v
                    if not d.any():
                        continue
                    if s == self.active: palm = palm + d
                    else: idle_palm = idle_palm + d
                    self.self_push_m[s] = self.self_push_m.get(s, 0.) + float(np.linalg.norm(d)); moved = True
                if not moved:
                    break
                self._lean_prev = lean_prev
                pose = self.pose(palm, point_at, palm_up, aim_blend, idle_palm, idle_palm_up, base, swivel=dict(self.swivel))
                self._write(data, pose); mujoco.mj_kinematics(self.model, data)
                self._palm_now = {self.active: pose['palm'], self.idle: pose['idle_palm']}
        self._lean_prev = pose['lean']
        self._palm_prev = {self.active: np.asarray(pose['palm']).copy(), self.idle: np.asarray(pose['idle_palm']).copy()}
        self._push_mem = {self.active: np.asarray(palm, float) - cmd_palm,
                          self.idle: None if cmd_idle is None else np.asarray(idle_palm, float) - cmd_idle}
        k, R = pose['k'], pose['R']
        quat = np.zeros(4)
        self._write(data, pose)
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
