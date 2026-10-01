"""Kinematic human mannequin posed from the scripted palm position.

The body is the HY-Motion 1.0 wooden model split into rigid segments (`assets/human`, built by
`scripts/build_human_asset.py`). The scripted behaviours only move the palm; this module turns a palm
position into a whole-body pose:
  - the pelvis and legs stay at the standing spot (`stand_xy` on the floor at `floor_z`), the body facing
    `facing_deg` (world yaw of the facing direction; the asset's T-pose faces -X = 180 deg),
  - the torso leans at the waist towards the palm target just as far as the arm needs to reach it
    (continuous in the target, capped at `max_lean_deg`),
  - the active arm follows by analytic two-link IK (elbow down/out/back, palm facing down; `palm_up` turns the
    forearm and hand palm up through the thumb-up position and bends the wrist so the upturned hand lies level,
    for receiving an object into the palm),
  - a pointing hand aims its index finger ray (knuckle -> fingertip of the HY-Motion hand) at the target,
  - hand shapes change gradually through the blended meshes of the asset (`smooth=True`),
  - the idle arm hangs relaxed beside the body, the head tilts down towards the table.
Segments are rendered as visual-only mocap bodies; capsule/sphere proxies of the arms, torso and head
are used for the human-robot distance and contact checks.
"""
from __future__ import annotations
import json
import mujoco
import numpy as np
from intent_policy.sim.base_env import ROOT

PROXIES = ('hand', 'forearm', 'upperarm', 'torso', 'head')   # invisible distance proxies (mocap bodies human_<name>)

SIDE = {'r': 1.0, 'l': -1.0}          # the human faces -X, so +Y is the human's right
PALM_DOWN = np.array([0.0, 0.0, -1.0])  # palm normal in the T-pose
MAX_WRIST_DEG = 70.0                    # wrist bend limit when aiming the pointing hand


def rotation(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3)
    x, y, z = axis / n
    c, s, C = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def frame(direction, normal) -> np.ndarray:
    """Orthonormal frame [d, n, d x n] with n the part of `normal` orthogonal to d (fallbacks if parallel)."""
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    for cand in (normal, (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)):
        n = np.asarray(cand, float) - np.dot(cand, d) * d
        if np.linalg.norm(n) > 1e-6:
            n /= np.linalg.norm(n)
            return np.column_stack([d, n, np.cross(d, n)])
    raise ValueError('degenerate frame')


def quat_z_to(direction) -> np.ndarray:
    """WXYZ quaternion rotating +Z onto `direction`."""
    d = np.asarray(direction, float) / (np.linalg.norm(direction) + 1e-12)
    axis = np.cross([0.0, 0.0, 1.0], d)
    s, c = np.linalg.norm(axis), float(d[2])
    if s < 1e-9:
        return np.array([1.0, 0, 0, 0]) if c > 0 else np.array([0.0, 1, 0, 0])
    angle = np.arctan2(s, c)
    return np.r_[np.cos(angle / 2), axis / s * np.sin(angle / 2)]


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


class HumanBody:
    def __init__(self, cfg: dict):
        skel = json.loads((ROOT / cfg.get('model', 'assets/human/skeleton.json')).read_text())
        self.skeleton = skel
        offset = np.array([cfg['stand_xy'][0], cfg['stand_xy'][1], cfg['floor_z']], float)
        self.R_yaw = rotation([0.0, 0.0, 1.0], np.radians(float(cfg.get('facing_deg', 180.0)) - 180.0))
        place = lambda p: offset + self.R_yaw @ np.asarray(p, float)
        self.J = {k: place(v) for k, v in skel['joints'].items()}
        self.palm_T = {s: place(p) for s, p in skel['palm'].items()}
        self.forward = self.R_yaw @ np.array([-1.0, 0.0, 0.0])    # facing direction (world)
        self.segments = list(skel['segments'])
        self.active = cfg.get('active_arm', 'r')
        self.idle = 'l' if self.active == 'r' else 'r'
        self.max_lean = np.radians(float(cfg.get('max_lean_deg', 60.0)))
        self.head_pitch = np.radians(float(cfg.get('head_pitch_deg', 15.0)))
        self.reach_margin = float(cfg.get('reach_margin', 0.97))
        self.idle_palm_offset = np.asarray(cfg.get('idle_palm_offset', [0.10, 0.05, 0.50]), float)  # forward, outward, down
        J = self.J
        self.L1 = {s: float(np.linalg.norm(J[f'{S}_Elbow'] - J[f'{S}_Shoulder'])) for s, S in (('l', 'L'), ('r', 'R'))}
        self.L2 = {s: float(np.linalg.norm(J[f'{S}_Wrist'] - J[f'{S}_Elbow'])) for s, S in (('l', 'L'), ('r', 'R'))}
        self.Lp = {s: float(np.linalg.norm(self.palm_T[s] - J[f'{S}_Wrist'])) for s, S in (('l', 'L'), ('r', 'R'))}
        self.torso_length = float(np.linalg.norm(J['Neck'] - J['Spine1']))
        self.index_ray = {s: [place(q) for q in skel['index_ray'][s]] for s in ('l', 'r')} if 'index_ray' in skel else None
        self.shape_order = list(skel['segments'][f'{self.active}_hand'].get('variants', {}))
        self.shape_state = None                  # (from shape, to shape, percent) of the smooth hand-shape change

    # ------------------------------------------------------------------ MuJoCo binding
    def bind(self, model) -> None:
        """Look up the mocap bodies / hand meshes built by `scene_builder` for this body."""
        mocap = lambda name: int(model.body(name).mocapid[0])
        self.model = model
        self.seg_mocap = {seg: mocap(f'human_seg_{seg}') for seg in self.segments}
        self.proxy_mocap = {name: mocap(f'human_{name}') for name in PROXIES}
        self.anchor_mocap = mocap('human_grasp_anchor')
        self.proxy_geoms = [model.geom('human_palm').id] + [model.geom(f'human_{n}_geom').id for n in PROXIES[1:]]
        hand = f'{self.active}_hand'
        self.hand_geom = model.geom(f'human_seg_{hand}_mesh').id
        self.hand_meshes = {v: model.mesh(f'human_{hand}_{v}').id for v in self.skeleton['segments'][hand]['variants']}

    def reset_shape(self) -> None:
        self.shape_state = None

    def _shape_variant(self, target: str, smooth: bool) -> str:
        """Hand mesh variant: `target` at once, or (smooth) one 25 % blend step per call towards it."""
        if not smooth or self.shape_state is None or target not in self.shape_order:
            self.shape_state = (target, target, 100)
            return target
        a, b, pct = self.shape_state
        if target == b:
            pct = min(pct + 25, 100)
        elif target == a:
            pct = max(pct - 25, 0)
        else:                                   # a new target: continue from the nearer end
            a, b, pct = (a if pct <= 50 else b), target, 25
        if pct in (0, 100):
            a = b = a if pct == 0 else b
            pct = 100
        self.shape_state = (a, b, pct)
        if a == b:
            return a
        x, y, q = (a, b, pct) if self.shape_order.index(a) < self.shape_order.index(b) else (b, a, 100 - pct)
        return f'{x}-{y}-{q}'

    def apply(self, data, palm, point_at=None, hand_shape: str = 'relaxed', palm_up: float = 0.0,
              smooth: bool = False) -> dict:
        """Pose the bound model for a palm position: segments, distance proxies, grasp anchor, hand mesh."""
        pose = self.pose(palm, point_at, palm_up)
        self.model.geom_dataid[self.hand_geom] = self.hand_meshes[self._shape_variant(hand_shape, smooth)]
        quat = np.zeros(4)
        for seg, (pos, rot) in pose['segments'].items():
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rot).ravel())
            data.mocap_pos[self.seg_mocap[seg]] = pos
            data.mocap_quat[self.seg_mocap[seg]] = quat
        for name, (a, b) in pose['capsules'].items():
            data.mocap_pos[self.proxy_mocap[name]] = (a + b) / 2
            data.mocap_quat[self.proxy_mocap[name]] = quat_z_to(b - a)
        data.mocap_pos[self.proxy_mocap['head']] = pose['head']
        data.mocap_pos[self.proxy_mocap['hand']] = palm
        data.mocap_pos[self.anchor_mocap] = palm
        return pose

    # ------------------------------------------------------------------ pose
    def _lean_axis(self, palm) -> np.ndarray:
        dh = np.asarray(palm, float)[:2] - self.J['Spine1'][:2]
        dh = unit(dh) if np.linalg.norm(dh) > 1e-6 else self.forward[:2]
        return np.cross([0.0, 0.0, 1.0], [dh[0], dh[1], 0.0])

    def lean_angle(self, palm) -> float:
        """Smallest waist lean that brings the palm target within reach of the active shoulder."""
        palm = np.asarray(palm, float)
        spine, axis, S = self.J['Spine1'], self._lean_axis(palm), self.active.upper()
        reach = (self.L1[self.active] + self.L2[self.active] + self.Lp[self.active]) * self.reach_margin
        gap = lambda th: np.linalg.norm(palm - (spine + rotation(axis, th) @ (self.J[f'{S}_Shoulder'] - spine))) - reach
        if gap(0.0) <= 0:
            return 0.0
        if gap(self.max_lean) > 0:
            return self.max_lean
        lo, hi = 0.0, self.max_lean
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if gap(mid) > 0 else (lo, mid)
        return hi

    def pointing_palm(self, target, length: float = 0.35) -> np.ndarray:
        """Palm position for pointing at a (table) location: `length` out from the active shoulder along the ray."""
        shoulder = self.J[f'{self.active.upper()}_Shoulder']
        return shoulder + length * unit(np.asarray(target, float) - shoulder)

    def clamp(self, palm) -> np.ndarray:
        """A reachable palm position on the way to `palm` (itself when reachable at <= max lean). Iterated because
        the lean direction follows the target."""
        p = np.asarray(palm, float)
        for _ in range(20):
            pose = self.pose(p)
            if pose['reach_error'] < 1e-7:
                break
            p = pose['palm']
        return p

    def _arm(self, s: str, shoulder, target, palm_normal, fore_normal=None, hand=None):
        """Two-link IK: elbow, wrist, reached palm and segment rotations of arm `s` towards `target` (`fore_normal`:
        palm normal of the forearm and hand, default `palm_normal`; `hand`: direction of the hand, default the
        forearm's, the wrist is then placed so that the palm centre still reaches `target`)."""
        S, side = s.upper(), SIDE[s]
        L1, L2 = self.L1[s], self.L2[s] + (self.Lp[s] if hand is None else 0.0)
        end = np.asarray(target, float) - (0.0 if hand is None else self.Lp[s] * np.asarray(hand, float))
        v = end - shoulder
        D = float(np.clip(np.linalg.norm(v), abs(L1 - L2) + 1e-3, L1 + L2 - 1e-4))
        u = unit(v)
        a = np.arccos(np.clip((L1 ** 2 + D ** 2 - L2 ** 2) / (2 * L1 * D), -1.0, 1.0))
        hint = unit(self.R_yaw @ [0.35, 0.6 * side, -1.0])   # elbow backwards, outwards and down
        w = hint - np.dot(hint, u) * u
        w = unit(w) if np.linalg.norm(w) > 1e-6 else np.array([0.0, 0.0, -1.0])
        elbow = shoulder + L1 * (np.cos(a) * u + np.sin(a) * w)
        fore = unit(shoulder + D * u - elbow)
        wrist = elbow + self.L2[s] * fore
        hand = fore if hand is None else np.asarray(hand, float)
        J = self.J
        n = palm_normal if fore_normal is None else fore_normal
        rest = frame(J[f'{S}_Wrist'] - J[f'{S}_Elbow'], PALM_DOWN).T @ self.R_yaw
        R_up = frame(elbow - shoulder, palm_normal) @ frame(J[f'{S}_Elbow'] - J[f'{S}_Shoulder'], PALM_DOWN).T @ self.R_yaw
        return dict(elbow=elbow, wrist=wrist, palm=wrist + self.Lp[s] * hand, R_up=R_up, R_fo=frame(fore, n) @ rest,
                    R_hand=frame(hand, n) @ rest)

    def _level_hand(self, s: str, shoulder, target, palm_normal, fore_normal, arm: dict, palm_up: float) -> dict:
        """Receiving: the wrist bends (at most MAX_WRIST_DEG) so that the upturned hand lies level, in proportion to
        `palm_up`, like a hand held out flat for an object; the palm centre stays on `target` (the wrist moves)."""
        for _ in range(3):                      # the level heading follows the re-solved forearm
            fore = unit(arm['wrist'] - arm['elbow'])
            flat = np.r_[fore[:2], 0.0]
            if np.linalg.norm(flat) < 1e-6:
                return arm
            flat = unit(flat)
            pitch = min(np.arccos(np.clip(np.dot(fore, flat), -1.0, 1.0)), np.radians(MAX_WRIST_DEG)) * palm_up
            hand = rotation(np.cross(fore, flat), pitch) @ fore
            arm = self._arm(s, shoulder, target, palm_normal, fore_normal, hand)
        return arm

    def _aim_hand(self, arm: dict, target) -> np.ndarray:
        """Hand rotation that puts the index finger ray (knuckle -> tip) through `target`; the wrist bends at most
        MAX_WRIST_DEG away from the forearm."""
        S = self.active.upper()
        J, wrist, fore = self.J, arm['wrist'], unit(arm['wrist'] - arm['elbow'])
        rest = frame(J[f'{S}_Wrist'] - J[f'{S}_Elbow'], PALM_DOWN).T
        rot = lambda aim: frame(aim, PALM_DOWN) @ rest              # T-pose (placed) -> posed world
        aim = unit(target - wrist)
        if self.index_ray is not None:                               # correct for the finger's offset from the axis
            k, tip = (q - J[f'{S}_Wrist'] for q in self.index_ray[self.active])
            for _ in range(6):
                R = rot(aim)
                knuckle, finger = wrist + R @ k, unit(R @ (tip - k))
                want = unit(target - knuckle)
                axis, ang = np.cross(finger, want), np.arccos(np.clip(np.dot(finger, want), -1.0, 1.0))
                if ang < 1e-4:
                    break
                aim = unit(rotation(axis, ang) @ aim)
        angle = np.arccos(np.clip(np.dot(fore, aim), -1.0, 1.0))
        if angle > np.radians(MAX_WRIST_DEG):
            aim = rotation(np.cross(fore, aim), np.radians(MAX_WRIST_DEG)) @ fore
        return rot(aim) @ self.R_yaw

    def pose(self, palm, point_at=None, palm_up: float = 0.0) -> dict:
        """Segment transforms {segment: (pivot position, rotation)} and proxy geometry for a palm target.

        With `point_at`, the active hand bends at the wrist so that its index finger ray points at that point (at
        most MAX_WRIST_DEG away from the forearm); otherwise it continues the forearm. `palm_up` in [0, 1] turns
        the active forearm and hand from palm down (0) through thumb up to palm up (1), the hand levelling out as it
        turns (the palm centre stays on the target).
        """
        palm = np.asarray(palm, float)
        J, spine = self.J, self.J['Spine1']
        theta = self.lean_angle(palm)
        Rt = rotation(self._lean_axis(palm), theta)
        on_torso = lambda p: spine + Rt @ (np.asarray(p, float) - spine)
        Ry = self.R_yaw               # mesh vertices are in T-pose axes: rest orientation = facing yaw
        seg = {'pelvis': (J['Pelvis'].copy(), Ry), 'torso': (spine.copy(), Rt @ Ry),
               'head': (on_torso(J['Neck']), Rt @ Ry @ rotation([0.0, -1.0, 0.0], self.head_pitch))}
        arms = {}
        for s in (self.active, self.idle):
            S = s.upper()
            shoulder = on_torso(J[f'{S}_Shoulder'])
            fore_normal = None
            if s == self.active:
                target, normal = palm, PALM_DOWN
                if palm_up > 0:                 # supination: the palm normal turns down -> medial -> up
                    medial = -SIDE[s] * (self.R_yaw @ np.array([0.0, 1.0, 0.0]))
                    fore_normal = np.cos(np.pi * palm_up) * PALM_DOWN + np.sin(np.pi * palm_up) * medial
            else:
                f, o, dn = self.idle_palm_offset
                target = on_torso(J[f'{S}_Shoulder'] + self.R_yaw @ [-f, SIDE[s] * o, -dn])
                normal = self.R_yaw @ np.array([0.0, -SIDE[s], 0.0])
            arm = self._arm(s, shoulder, target, normal, fore_normal)
            if fore_normal is not None and point_at is None:
                arm = self._level_hand(s, shoulder, target, normal, fore_normal, arm, palm_up)
            arm['shoulder'] = shoulder
            arms[s] = arm
            seg[f'{s}_upperarm'] = (shoulder, arm['R_up'])
            seg[f'{s}_forearm'] = (arm['elbow'], arm['R_fo'])
            seg[f'{s}_hand'] = (arm['wrist'], arm['R_hand'])
        a = arms[self.active]
        if point_at is not None:
            seg[f'{self.active}_hand'] = (a['wrist'], self._aim_hand(a, np.asarray(point_at, float)))
        return dict(segments=seg, lean=theta, palm=a['palm'], reach_error=float(np.linalg.norm(a['palm'] - palm)),
                    chest=on_torso(J['Spine3']),
                    capsules={'upperarm': (a['shoulder'], a['elbow']), 'forearm': (a['elbow'], a['wrist']),
                              'torso': (on_torso(J['Spine1']), on_torso(J['Neck']))},
                    head=on_torso(J['Head'] + [0.0, 0.0, 0.06]),
                    face=on_torso(J['Head'] + [0.0, 0.0, 0.05]) + Rt @ self.forward * 0.09)
