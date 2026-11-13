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
SELF_ITER = 10                          # self-clearance re-poses per tick (cfg self_clearance_m)
SELF_PAD_M = 0.05                       # self_depth: arm points farther than this outside a body mesh's box are skipped
                                        #   (their depth is below -SELF_PAD_M; clearances are reported down to -SELF_PAD_M)


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
        self.default_active = self.active               # restored at every scenario reset (I2 switches arms)
        self.idle = 'l' if self.active == 'r' else 'r'
        self.max_lean = np.radians(float(cfg.get('max_lean_deg', 60.0)))
        self.head_pitch = np.radians(float(cfg.get('head_pitch_deg', 15.0)))
        self.reach_margin = float(cfg.get('reach_margin', 0.97))
        # `max_extension` (13/11, user: the wrist whipped as the arm straightened: arccos in the two-link IK has an
        # infinite slope at full extension): the elbow never straightens beyond this fraction of L1 + L2, the
        # waist lean (reach_margin below it) does the rest. Absent: 1.0 (as before).
        self.max_extension = float(cfg.get('max_extension', 1.0))
        self.idle_palm_offset = np.asarray(cfg.get('idle_palm_offset', [0.10, 0.05, 0.50]), float)  # forward, outward, down
        # `self_clearance_m` (decision 46/QĐ-DL, docs/reports/do_tay_xuyen_than_2026-11-06.md): a pose whose forearm or
        # hand sinks into the torso or the hips / legs gets that palm pushed out along the outward normal of the nearest
        # slab-hull edge by the depth plus this clearance, then re-posed (SELF_ITER times). Absent: no correction (the
        # v9 / v10 / v10b poses).
        self.self_clearance = float(cfg['self_clearance_m']) if cfg.get('self_clearance_m') is not None else None
        # `hang_blend_m` (decision 47/QĐ-DL step 1): within this distance of its own hanging rest target the active palm
        # turns from palm down to the hanging orientation (palm to the thigh), as the idle arm hangs. I2 / I3 / C4 rest
        # the active hand at that hanging spot: palm down there, the hand's edge sank 88 mm into the thigh, and no
        # palm-down position within 12 cm cleared it (outputs/diag/self_pen_2026-11-06/rest_search.py). Absent: palm down.
        self.hang_blend = float(cfg['hang_blend_m']) if cfg.get('hang_blend_m') is not None else None
        # `hang_rest_upright` (47/QĐ-DL step 1): the hanging rest of each side (I2 / C4 rest_by_side) is taken with
        # both arms hanging and the torso upright (`hanging_palm`). The old way posed the other arm reaching to
        # hand_rest, which leaned the torso 20.6 deg and left the "hanging" right palm 18 cm low and forward, inside the hip.
        self.hang_rest_upright = bool(cfg.get('hang_rest_upright', False))
        # `hang_blend_palm_up` (49/QĐ-DL, I2 v11): the hang blend also while the palm turns up (palm_up > 0). Without it
        # the first sample of a palm-up motion snapped the hand from the hanging orientation (palm to the thigh) to palm
        # down beside the thigh, its edge 15 mm into the hip (I2 request_out / request_back, i2_path_profile.py 06/11).
        # The supination normal is blended towards the hanging one by the same weight. Absent: as before.
        self.hang_blend_palm_up = bool(cfg.get('hang_blend_palm_up', False))
        self.idle_hang_blend = float(cfg.get('idle_hang_blend_m', 0.0))
        # `lean_rate_deg_s` (13/11, user: the torso snapped upright when the withdrawing hand came back within
        # reach, lean_angle has a kink at the reach boundary; measured 0.56 m/s -> 0 in one tick, I1 preview):
        # the waist lean may change at most this fast between applied poses. Absent: instantaneous (as before).
        self.lean_rate = float(cfg['lean_rate_deg_s']) if cfg.get('lean_rate_deg_s') is not None else None
        self.lean_dt = float(cfg.get('lean_dt_s', 0.05))
        self._lean_prev = None
        self._idle_shift_prev = None    # rate-limited with the lean (13/11): the hanging arm twitched as the torso leaned
        # `idle_clearance_m` (47/QĐ-DL step 1, the idle arm's rest): the hanging idle arm keeps this clearance from the
        # torso and the hips / legs. Its rest target turns with the leaning torso, so a lean towards the working hand put
        # the idle hand 48-71 mm into the hip (I1 lower_coaster, I2 / C1 / C3 pilot 06/11); the rest target is moved out
        # sideways (the body's lateral axis, outward) just far enough. Continuous: the shift grows from 0 as the arm comes
        # within the clearance, so the hand does not jump. Absent: no correction.
        self.idle_clearance = float(cfg['idle_clearance_m']) if cfg.get('idle_clearance_m') is not None else None
        J = self.J
        self.L1 = {s: float(np.linalg.norm(J[f'{S}_Elbow'] - J[f'{S}_Shoulder'])) for s, S in (('l', 'L'), ('r', 'R'))}
        self.L2 = {s: float(np.linalg.norm(J[f'{S}_Wrist'] - J[f'{S}_Elbow'])) for s, S in (('l', 'L'), ('r', 'R'))}
        self.Lp = {s: float(np.linalg.norm(self.palm_T[s] - J[f'{S}_Wrist'])) for s, S in (('l', 'L'), ('r', 'R'))}
        self.torso_length = float(np.linalg.norm(J['Neck'] - J['Spine1']))
        self.index_ray = {s: [place(q) for q in skel['index_ray'][s]] for s in ('l', 'r')} if 'index_ray' in skel else None
        self.shape_order = list(skel['segments'][f'{self.active}_hand'].get('variants', {}))
        self.shape_state = None                  # (from shape, to shape, percent) of the smooth hand-shape change
        # `hand`: the grasp anchor carries the active hand segment's orientation too, so a held object keeps its pose
        # in the hand (a rigid grasp). Default `world` keeps the anchor axis-aligned (legacy scenes, reproducibility).
        self.held_object_frame = cfg.get('held_object_frame', 'world')
        if self.held_object_frame not in ('world', 'hand'):
            raise ValueError('held_object_frame must be world | hand')

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
        idle = f'{self.idle}_hand'
        self.idle_hand_geom = model.geom(f'human_seg_{idle}_mesh').id
        self.idle_hand_meshes = {v: model.mesh(f'human_{idle}_{v}').id
                                 for v in self.skeleton['segments'][idle].get('variants', {})}
        self.compiled_hand_geoms = {}
        for side in (self.active, self.idle):
            segment = f'{side}_hand'
            default = next(v for v, file in self.skeleton['segments'][segment]['variants'].items()
                           if file == self.skeleton['segments'][segment]['mesh'])
            variants = {}
            for variant in self.skeleton['segments'][segment]['variants']:
                name = f'human_seg_{segment}_mesh' if variant == default else f'human_seg_{segment}_variant_{variant}'
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid >= 0:
                    variants[variant] = gid
            if len(variants) > 1:
                self.compiled_hand_geoms[side] = variants
        self._hand_mesh_ids = {self.active: dict(self.hand_meshes), self.idle: dict(self.idle_hand_meshes)}
        self._body_slabs, self._arm_points = None, {}
        if self.self_clearance is not None:
            from intent_policy.sim.human_self_check import _mesh_vertices
            gid = lambda n: model.geom(n).id
            self._self_body = {b: v for b, v in self.body_slabs().items() if b in ('torso', 'pelvis')}
            self._self_arm = {s: {p: (*self._geom_offset(gid(f'human_seg_{s}_{p}_mesh')), _mesh_vertices(model, gid(f'human_seg_{s}_{p}_mesh'))[::3])
                                  for p in ('forearm', 'hand')} for s in ('l', 'r')}

    def _geom_offset(self, g):
        """Frame of geom g in its (mocap) segment body frame: (position, rotation)."""
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.model.geom_quat[g])
        return np.asarray(self.model.geom_pos[g], float), R.reshape(3, 3)

    def body_slabs(self) -> dict:
        """{segment: (geom position, geom rotation, BodySlabs)} of the torso, hips / legs and head meshes, as the
        collection gate's self check (sim/human_self_check.py) cuts them."""
        if self._body_slabs is None:
            from intent_policy.sim.human_self_check import BODY, BodySlabs, _mesh_vertices
            m = self.model
            vert = {b: _mesh_vertices(m, m.geom(f'human_seg_{b}_mesh').id) for b in BODY}
            self._body_slabs = {b: (*self._geom_offset(m.geom(f'human_seg_{b}_mesh').id), BodySlabs(vert[b])) for b in BODY}
            self._slab_box = {b: (v.min(axis=0) - SELF_PAD_M, v.max(axis=0) + SELF_PAD_M) for b, v in vert.items()}
        return self._body_slabs

    def _segment_points(self, side: str, part: str, variant: str | None = None):
        """Every ARM_STRIDE-th mesh vertex of arm segment `side`_`part` in its segment frame, as the gate samples them;
        a hand by shape variant (its compiled geom, or the base geom showing that mesh), None: the base mesh; None for
        a variant the scene does not have."""
        key = (side, part, variant)
        if key not in self._arm_points:
            from intent_policy.sim.human_self_check import ARM_STRIDE
            m = self.model
            g = m.geom(f'human_seg_{side}_{part}_mesh').id
            mid = int(m.geom_dataid[g])
            if part == 'hand' and variant is not None:
                compiled = self.compiled_hand_geoms.get(side, {})
                if variant in compiled:
                    g = compiled[variant]
                    mid = int(m.geom_dataid[g])
                else:
                    mid = self._hand_mesh_ids[side].get(variant)
            if mid is None:
                self._arm_points[key] = None
            else:
                gp, gR = self._geom_offset(g)
                a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                self._arm_points[key] = gp + np.asarray(m.mesh_vert[a:a + n], float)[::ARM_STRIDE] @ gR.T
        return self._arm_points[key]

    def shape_at(self, shape_from: str | None, target: str, progress: float) -> str:
        """The active hand's mesh variant at `progress` (0..1) of a motion forming shape_from -> target, as
        _shape_variant shows it (25 % blend steps); the target when no blend applies."""
        if (shape_from is None or shape_from == target or shape_from not in self.shape_order
                or target not in self.shape_order):
            return target
        pct = int(round(float(np.clip(progress, 0.0, 1.0)) * 4)) * 25
        if pct in (0, 100):
            return shape_from if pct == 0 else target
        x, y, q = (shape_from, target, pct) if self.shape_order.index(shape_from) < self.shape_order.index(target) \
            else (target, shape_from, 100 - pct)
        name = f'{x}-{y}-{q}'
        return name if name in self.hand_meshes else target

    def visible_shape(self, side: str) -> str | None:
        """The hand variant shown now on `side` (compiled variants: the one with alpha > 0); None if not known."""
        for v, g in self.compiled_hand_geoms.get(side, {}).items():
            if self.model.geom_rgba[g, 3] > 0.0:
                return v
        return None

    def idle_clear_pose(self, pose, palm, point_at=None, palm_up=0.0, aim_blend=1.0, idle_palm_up=0.0) -> dict:
        """`pose` with the hanging idle arm's rest target moved outward (lateral axis) until its forearm and hand keep
        idle_clearance_m from the torso and the hips / legs (cfg idle_clearance_m). The shift is the distance, along
        that direction, for the arm points moved idle_clearance_m inwards to leave the slab hulls (exit_along), so it
        is 0 while the arm is farther than the clearance and grows continuously; re-posed SELF_ITER times at most."""
        s, c = self.idle, self.idle_clearance
        out = self.R_yaw @ np.array([0.0, SIDE[s], 0.0])          # the idle side, outward, horizontal
        shift = 0.0
        shapes = (self.visible_shape(s),)
        for _ in range(SELF_ITER):
            seg = pose['segments']
            pts = [seg[f'{s}_forearm'][0] + self._segment_points(s, 'forearm') @ np.asarray(seg[f'{s}_forearm'][1]).T]
            q = self._segment_points(s, 'hand', shapes[0])
            if q is not None:
                pts.append(seg[f'{s}_hand'][0] + q @ np.asarray(seg[f'{s}_hand'][1]).T)
            p = np.concatenate(pts) - c * out
            need = 0.0
            for b in ('torso', 'pelvis'):
                gp, gR, slabs = self.body_slabs()[b]
                pos, R = seg[b]
                Rg = np.asarray(R) @ gR
                t = slabs.exit_along((p - (pos + np.asarray(R) @ gp)) @ Rg, Rg.T @ out)
                need = max(need, float(np.max(t, initial=0.0)))
            if need < 1e-4:
                break
            shift += need
            pose = self.pose(palm, point_at, palm_up, aim_blend, None, idle_palm_up, {s: shift * out})
        if self.lean_rate is not None and self._idle_shift_prev is not None:
            step = .004                                     # m per applied pose (0.08 m/s at 20 Hz)
            limited = float(np.clip(shift, self._idle_shift_prev - step, self._idle_shift_prev + step))
            if abs(limited - shift) > 1e-6:
                shift = limited
                pose = self.pose(palm, point_at, palm_up, aim_blend, None, idle_palm_up, {s: shift * out})
        pose['idle_shift_m'] = shift
        return pose

    def self_depth(self, pose, shapes=None, idle_shapes=None, sides=None) -> float:
        """Deepest penetration (m; < 0: the smallest clearance, down to -SELF_PAD_M) of either forearm or hand into the
        torso, the hips / legs or the head of a pose (`pose()`, not yet applied): the gate's self check
        (sim/human_self_check.py) for a planned pose (human_behavior.body_clearance, decision 47/QĐ-DL step 3).
        `shapes` / `idle_shapes`: hand shape variants of the active / idle hand to test (None: the base hand mesh).
        `sides`: the arms measured (default both)."""
        seg, pts = pose['segments'], []
        for s, variants in ((self.active, shapes), (self.idle, idle_shapes)):
            if sides is not None and s not in sides:
                continue
            pos, R = seg[f'{s}_forearm']
            pts.append(pos + self._segment_points(s, 'forearm') @ np.asarray(R).T)
            pos, R = seg[f'{s}_hand']
            for v in (variants or (None,)):
                q = self._segment_points(s, 'hand', v)
                if q is not None:
                    pts.append(pos + q @ np.asarray(R).T)
        p = np.concatenate(pts)
        depth = -SELF_PAD_M
        for b, (gp, gR, slabs) in self.body_slabs().items():
            pos, R = seg[b]
            local = (p - (pos + np.asarray(R) @ gp)) @ (np.asarray(R) @ gR)
            lo, hi = self._slab_box[b]
            near = local[np.all((local >= lo) & (local <= hi), axis=1)]     # only points near the segment's mesh
            if len(near):
                depth = max(depth, float(slabs.depth(near).max()))
        return depth

    def set_active(self, side: str) -> None:
        """v7 (I2): the other arm becomes the moving one (its palm the scripted position); call only with both hands at
        their relaxed pose (the arm that becomes idle hangs relaxed, the new active one starts where it hangs)."""
        if side not in ('l', 'r'):
            raise ValueError('side must be l or r')
        if side == self.active:
            return
        self.active, self.idle = side, self.active
        self.hand_geom, self.idle_hand_geom = self.idle_hand_geom, self.hand_geom
        self.hand_meshes, self.idle_hand_meshes = self.idle_hand_meshes, self.hand_meshes
        self.shape_state = None

    def reset_shape(self) -> None:
        self.shape_state = None

    def _shape_variant(self, target: str, smooth: bool, shape_from: str | None = None,
                       progress: float | None = None) -> str:
        """Hand mesh variant: `target` at once, or (smooth) one 25 % blend step per call towards it; with `progress`
        (0..1, the gesture's own motion progress) the blend shape_from -> target at that fraction (25 % steps), so the
        hand forms with the arm motion instead of at its first frame."""
        if progress is not None and shape_from is not None and shape_from != target \
                and shape_from in self.shape_order and target in self.shape_order:
            pct = int(round(float(np.clip(progress, 0.0, 1.0)) * 4)) * 25
            self.shape_state = (target, target, 100)
            if pct in (0, 100):
                return shape_from if pct == 0 else target
            x, y, q = (shape_from, target, pct) if self.shape_order.index(shape_from) < self.shape_order.index(target) \
                else (target, shape_from, 100 - pct)
            name = f'{x}-{y}-{q}'
            if name in self.hand_meshes:
                return name
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
              smooth: bool = False, aim_blend: float = 1.0, shape_from: str | None = None,
              shape_progress: float | None = None, idle_palm=None, idle_shape: str | None = None,
              idle_palm_up: float = 0.0) -> dict:
        """Pose the bound model for a palm position: segments, distance proxies, grasp anchor, hand mesh.
        `aim_blend` / `shape_progress`: how far a pointing hand has formed (the arm motion's progress, 2026-10-12).
        `idle_palm` / `idle_shape` / `idle_palm_up`: the other hand's target and shape (two-handed actions)."""
        pose = self.pose(palm, point_at, palm_up, aim_blend, idle_palm, idle_palm_up)
        if self.idle_clearance is not None and idle_palm is None and self.self_clearance is None:
            pose = self.idle_clear_pose(pose, palm, point_at, palm_up, aim_blend, idle_palm_up)
        if self.self_clearance is not None:
            palm = np.asarray(palm, float)
            idle_t = None if idle_palm is None else np.asarray(idle_palm, float)
            rest_shift = {}
            for _ in range(SELF_ITER):
                push = self._self_push(pose)
                if not push:
                    break
                if self.active in push:
                    palm = palm + push[self.active]
                if self.idle in push:
                    if idle_t is None:          # a hanging hand keeps its rest orientation: shift the rest target
                        rest_shift[self.idle] = rest_shift.get(self.idle, 0.0) + push[self.idle]
                    else:
                        idle_t = idle_t + push[self.idle]
                pose = self.pose(palm, point_at, palm_up, aim_blend, idle_t, idle_palm_up, rest_shift)
            pose['self_push'] = push
        variant = self._shape_variant(hand_shape, smooth, shape_from, shape_progress)
        if self.active in self.compiled_hand_geoms:
            self.hand_geom = self._select_compiled_hand(self.active, variant)
        else:
            self.model.geom_dataid[self.hand_geom] = self.hand_meshes[variant]
        if idle_shape in self.idle_hand_meshes:
            if self.idle in self.compiled_hand_geoms:
                self.idle_hand_geom = self._select_compiled_hand(self.idle, idle_shape)
            else:
                self.model.geom_dataid[self.idle_hand_geom] = self.idle_hand_meshes[idle_shape]
        self._lean_prev = float(pose['lean'])
        self._idle_shift_prev = float(pose.get('idle_shift_m', 0.0))
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
        if self.held_object_frame == 'hand':
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(pose['segments'][f'{self.active}_hand'][1]).ravel())
            data.mocap_quat[self.anchor_mocap] = quat
        return pose

    def _select_compiled_hand(self, side, variant):
        geoms = self.compiled_hand_geoms[side]
        for name, geom in geoms.items():
            self.model.geom_rgba[geom, 3] = float(name == variant)
        return geoms[variant]

    def keypoints(self, pose: dict, names) -> np.ndarray:
        """World positions (len(names), 3) of named keypoints of a pose: `<s>_wrist` (hand segment pivot),
        `<s>_index_base` / `<s>_index_tip` (knuckle / end of the HY-Motion index finger ray), `<s>_shoulder` / `<s>_elbow`
        (upper arm / forearm segment pivots), `head`; s = r | l."""
        out = []
        for name in names:
            if name == 'head':                                         # head joint (rendered pose, gaze proxy)
                out.append(np.asarray(pose['head'], float))
                continue
            s, part = name.split('_', 1)
            if part in ('shoulder', 'elbow'):                          # pivots of the upper arm / forearm segments
                out.append(np.asarray(pose['segments'][f"{s}_{'upperarm' if part == 'shoulder' else 'forearm'}"][0], float))
                continue
            wrist, R = pose['segments'][f'{s}_hand']
            if part == 'wrist':
                out.append(wrist)
            elif part in ('index_base', 'index_tip'):
                if self.index_ray is None:
                    raise ValueError('the skeleton has no index finger ray')
                q = self.index_ray[s][0 if part == 'index_base' else 1] - self.J[f'{s.upper()}_Wrist']   # placed T-pose offset
                out.append(wrist + R @ self.R_yaw.T @ q)                      # segment rotations act on T-pose axes
            else:
                raise KeyError(f'unknown keypoint {name!r}')
        return np.asarray(out, float)

    # ------------------------------------------------------------------ pose
    def _lean_axis(self, palm) -> np.ndarray:
        dh = np.asarray(palm, float)[:2] - self.J['Spine1'][:2]
        dh = unit(dh) if np.linalg.norm(dh) > 1e-6 else self.forward[:2]
        return np.cross([0.0, 0.0, 1.0], [dh[0], dh[1], 0.0])

    def lean_angle(self, palm, side: str | None = None) -> float:
        """Smallest waist lean that brings the palm target within reach of the shoulder of `side` (default active)."""
        palm, s = np.asarray(palm, float), side or self.active
        spine, axis, S = self.J['Spine1'], self._lean_axis(palm), s.upper()
        reach = (self.L1[s] + self.L2[s] + self.Lp[s]) * self.reach_margin
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
        D = float(np.clip(np.linalg.norm(v), abs(L1 - L2) + 1e-3, min(L1 + L2 - 1e-4, self.max_extension * (L1 + L2))))
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

    def _aim_hand(self, arm: dict, target, blend: float = 1.0) -> np.ndarray:
        """Hand rotation that puts the index finger ray (knuckle -> tip) through `target`; the wrist bends at most
        MAX_WRIST_DEG away from the forearm. `blend` < 1: only that fraction of the bend from the straight wrist (the
        finger continuing the forearm) towards that aim - a pointing hand forms with the arm motion (2026-10-12: aiming
        at once exposed the target in the gesture's first frame, before the arm moved)."""
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
            angle = np.radians(MAX_WRIST_DEG)
        blend = float(np.clip(blend, 0.0, 1.0))
        if blend < 1.0 and angle > 1e-6:
            aim = rotation(np.cross(fore, aim), blend * angle) @ fore
        return rot(aim) @ self.R_yaw

    def hanging_palm(self, side: str) -> np.ndarray:
        """Palm of `side` hanging relaxed (idle_palm_offset) while the other arm hangs too: no reach, no lean."""
        keep = self.active
        other = 'l' if side == 'r' else 'r'
        self.set_active(other)
        O = other.upper()
        f, o, dn = self.idle_palm_offset
        palm = np.asarray(self.pose(self.J[f'{O}_Shoulder'] + self.R_yaw @ [-f, SIDE[other] * o, -dn])['idle_palm'], float)
        self.set_active(keep)
        return palm

    def _self_push(self, pose) -> dict:
        """{side: palm displacement} for each arm whose forearm or hand sinks into the torso or hips; {} when both arms
        are clear. Torso: forwards (the torso's facing direction) by the distance to leave it + self_clearance, so the
        hand passes in front of the chest and its sideways position, which carries the gesture's meaning (I3 reads the
        side from it), is kept: a push along the nearest edge moved a hand raised to the chest sideways and the early
        I3 expert took the clothes-adjusting distractor for a sweep (PIF §8.13 06/11). Hips / legs: out along the
        nearest slab-hull edge by depth + self_clearance (a hanging hand moves to the side of the thigh)."""
        seg, push = pose['segments'], {}
        forward = seg['torso'][1] @ (self.R_yaw.T @ self.forward)        # the leaning torso's facing direction
        for s in (self.active, self.idle):
            pts = []
            for part in ('forearm', 'hand'):
                pos, R = seg[f'{s}_{part}']
                gp, gR, v = self._self_arm[s][part]
                pts.append(pos + R @ gp + v @ (R @ gR).T)
            p = np.concatenate(pts)
            best = None
            for b, (gp, gR, slabs) in self._self_body.items():
                pos, R = seg[b]
                Rg = R @ gR
                local = (p - (pos + R @ gp)) @ Rg
                if b == 'torso':
                    t = slabs.exit_along(local, Rg.T @ forward)
                    i = int(np.argmax(t))
                    if np.isfinite(t[i]) and t[i] > 0.0 and (best is None or t[i] > best[0]):
                        best = (float(t[i]), forward / np.linalg.norm(forward))
                    continue
                depth, normal = slabs.depth_normal(local)
                i = int(np.argmax(depth))
                if depth[i] > 0.0 and (best is None or depth[i] > best[0]):
                    best = (float(depth[i]), Rg @ normal[i])
            if best is not None:
                push[s] = best[1] * (best[0] + self.self_clearance)
        return push

    def pose(self, palm, point_at=None, palm_up: float = 0.0, aim_blend: float = 1.0, idle_palm=None,
             idle_palm_up: float = 0.0, rest_shift: dict | None = None) -> dict:
        """Segment transforms {segment: (pivot position, rotation)} and proxy geometry for a palm target.

        With `point_at`, the active hand bends at the wrist so that its index finger ray points at that point (at
        most MAX_WRIST_DEG away from the forearm); otherwise it continues the forearm. `palm_up` in [0, 1] turns
        the active forearm and hand from palm down (0) through thumb up to palm up (1), the hand levelling out as it
        turns (the palm centre stays on the target). `idle_palm` (two-handed actions, v7 2026-10-27): a palm target
        for the other arm too, with its own `idle_palm_up`; None keeps that arm hanging relaxed. The waist lean
        follows the palm that needs the larger lean.
        """
        palm = np.asarray(palm, float)
        J, spine = self.J, self.J['Spine1']
        theta, lean_to = self.lean_angle(palm), palm
        if idle_palm is not None and self.lean_angle(idle_palm, self.idle) > theta:   # lean for the farther hand
            theta, lean_to = self.lean_angle(idle_palm, self.idle), np.asarray(idle_palm, float)
        if self.lean_rate is not None and self._lean_prev is not None:
            step = np.radians(self.lean_rate) * self.lean_dt
            theta = float(np.clip(theta, self._lean_prev - step, self._lean_prev + step))
        Rt = rotation(self._lean_axis(lean_to), theta)
        on_torso = lambda p: spine + Rt @ (np.asarray(p, float) - spine)
        Ry = self.R_yaw               # mesh vertices are in T-pose axes: rest orientation = facing yaw
        seg = {'pelvis': (J['Pelvis'].copy(), Ry), 'torso': (spine.copy(), Rt @ Ry),
               'head': (on_torso(J['Neck']), Rt @ Ry @ rotation([0.0, -1.0, 0.0], self.head_pitch))}
        arms = {}
        for s in (self.active, self.idle):
            S = s.upper()
            shoulder = on_torso(J[f'{S}_Shoulder'])
            fore_normal = None
            up = palm_up if s == self.active else idle_palm_up
            if s == self.active or idle_palm is not None:
                target, normal = (palm if s == self.active else np.asarray(idle_palm, float)), PALM_DOWN
                if up > 0:                      # supination: the palm normal turns down -> medial -> up
                    medial = -SIDE[s] * (self.R_yaw @ np.array([0.0, 1.0, 0.0]))
                    fore_normal = np.cos(np.pi * up) * PALM_DOWN + np.sin(np.pi * up) * medial
                    if s == self.active and self.hang_blend and self.hang_blend_palm_up and point_at is None:
                        f, o, dn = self.idle_palm_offset
                        hang = on_torso(J[f'{S}_Shoulder'] + self.R_yaw @ [-f, SIDE[s] * o, -dn])
                        w = float(np.clip(1.0 - np.linalg.norm(target - hang) / self.hang_blend, 0.0, 1.0))
                        if w > 0.0:             # near its hanging rest: towards the hanging orientation, as below
                            hang_n = self.R_yaw @ np.array([0.0, -SIDE[s], 0.0])
                            fore_normal = unit((1.0 - w) * fore_normal + w * hang_n)
                            normal = unit((1.0 - w) * PALM_DOWN + w * hang_n)
                elif (point_at is None and ((s == self.active and self.hang_blend) or
                                                (s == self.idle and idle_palm is not None and self.idle_hang_blend))):
                    blend = self.hang_blend if s == self.active else self.idle_hang_blend
                    f, o, dn = self.idle_palm_offset
                    hang = on_torso(J[f'{S}_Shoulder'] + self.R_yaw @ [-f, SIDE[s] * o, -dn])
                    w = float(np.clip(1.0 - np.linalg.norm(target - hang) / blend, 0.0, 1.0))
                    if w > 0.0:                 # near its hanging rest: turn towards the hanging palm orientation
                        n = (1.0 - w) * PALM_DOWN + w * (self.R_yaw @ np.array([0.0, -SIDE[s], 0.0]))
                        normal = n / np.linalg.norm(n)
            else:
                f, o, dn = self.idle_palm_offset
                target = on_torso(J[f'{S}_Shoulder'] + self.R_yaw @ [-f, SIDE[s] * o, -dn])
                if rest_shift and s in rest_shift:
                    target = target + rest_shift[s]
                normal = self.R_yaw @ np.array([0.0, -SIDE[s], 0.0])
            arm = self._arm(s, shoulder, target, normal, fore_normal)
            if fore_normal is not None and (point_at is None or s != self.active):
                arm = self._level_hand(s, shoulder, target, normal, fore_normal, arm, up)
            arm['shoulder'] = shoulder
            arms[s] = arm
            seg[f'{s}_upperarm'] = (shoulder, arm['R_up'])
            seg[f'{s}_forearm'] = (arm['elbow'], arm['R_fo'])
            seg[f'{s}_hand'] = (arm['wrist'], arm['R_hand'])
        a = arms[self.active]
        if point_at is not None:
            seg[f'{self.active}_hand'] = (a['wrist'], self._aim_hand(a, np.asarray(point_at, float), aim_blend))
        return dict(segments=seg, lean=theta, palm=a['palm'], reach_error=float(np.linalg.norm(a['palm'] - palm)),
                    idle_palm=arms[self.idle]['palm'],
                    idle_reach_error=0.0 if idle_palm is None else float(np.linalg.norm(arms[self.idle]['palm'] - idle_palm)),
                    chest=on_torso(J['Spine3']),
                    capsules={'upperarm': (a['shoulder'], a['elbow']), 'forearm': (a['elbow'], a['wrist']),
                              'torso': (on_torso(J['Spine1']), on_torso(J['Neck']))},
                    head=on_torso(J['Head'] + [0.0, 0.0, 0.06]),
                    face=on_torso(J['Head'] + [0.0, 0.0, 0.05]) + Rt @ self.forward * 0.09)
