"""Explicit geometry-pair audit for collecting demos, independent of the policy.

Caller supplies ALL relevant pairs, including held objects, and phase-specific
permitted contact. This is not an automatic scene certification or a controller.
Bind observe to env.physics_observers; also observe after resets and human pose
updates. A human teleported between control ticks still needs a swept-path check.

Concave human hands: mj_geomDistance measures mesh geoms by their convex hull. A grasping hand is strongly concave
(the hull spans thumb and fingers around a held object), so a hull overlap is not a penetration of the hand. When
the hull flags a pair of a human hand mesh and a primitive (sphere, capsule, cylinder, box), the pair is re-measured
exactly on points sampled over every triangle of the hand mesh (spacing <= SAMPLE_SPACING_M, i.e. features of the
primitive thinner than that could pass between samples) with the primitive's signed distance function.
A flagged pair of a human hand mesh and another (convex-hull) mesh, e.g. a gripper finger: the hand samples in the
signed distance of that mesh's convex hull (its face planes; the robot meshes are measured by their hulls in the
physics too). The open, upturned hand's hull spans the space above the palm where the robot hands the can over:
hull vs hull gave -10.5 mm there (2026-10-30).
Primitive pairs: mj_geomDistance returned -51 mm for a cup wall box over a coaster cylinder with no physics contact
between them (2026-10-30), so a flagged pair of two primitives is re-measured the same way, surface samples of each
in the other's signed distance function.

Physically colliding pairs (collision masks match, no weld / parent / explicit exclusion): mj_geomDistance returned
-11 to -21 mm for a gripper finger mesh against a cup wall box that the physics collision detection found no contact
with (fingers pinching the rim, nothing visibly inside; 2026-10-30). A flagged physical pair is re-measured from the
contacts the solver itself generated for that geom pair at this step (data.contact; none = not overlapping).

`allowed_contact` may return a number: the permitted penetration bound of that contact (else the pair's bound).
"""
from dataclasses import dataclass
import numpy as np
import mujoco


SAMPLE_SPACING_M = 0.0025      # 0.4 px of the high, 0.6-0.8 px of the top policy camera (6.4-6.8 / 3.2-4.5 mm/px, measured)
_SAMPLE_CACHE = {}               # hand mesh surface samples, per mesh (identical across episodes of a scene)
PRIMITIVES = {mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER,
              mujoco.mjtGeom.mjGEOM_BOX}


def surface_samples(vert, face, spacing):
    """Points on every triangle (barycentric grid, edge subdivided so neighbours are <= spacing apart) + vertices."""
    out = [vert]
    A, B, C = vert[face[:, 0]], vert[face[:, 1]], vert[face[:, 2]]
    longest = np.max([np.linalg.norm(B - A, axis=1), np.linalg.norm(C - B, axis=1), np.linalg.norm(A - C, axis=1)], 0)
    n = np.maximum(1, np.ceil(longest / spacing).astype(int))
    for k in np.unique(n):
        if k < 2:
            continue
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing='ij')
        keep = (i + j <= k)
        u, v = i[keep] / k, j[keep] / k
        sel = n == k
        a, b, c = A[sel], B[sel], C[sel]
        out.append((a[:, None] * (1 - u - v)[None, :, None] + b[:, None] * u[None, :, None] +
                    c[:, None] * v[None, :, None]).reshape(-1, 3))
    return np.unique(np.concatenate(out).round(6), axis=0)


def primitive_sdf(kind, size, p):
    """Signed distance of local points p [N, 3] to a MuJoCo primitive (negative inside)."""
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.linalg.norm(p, axis=1) - size[0]
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        q = p.copy(); q[:, 2] -= np.clip(p[:, 2], -size[1], size[1])
        return np.linalg.norm(q, axis=1) - size[0]
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        d = np.stack([np.linalg.norm(p[:, :2], axis=1) - size[0], np.abs(p[:, 2]) - size[1]], 1)
    else:   # box
        d = np.abs(p) - size[:3]
    return np.minimum(d.max(1), 0) + np.linalg.norm(np.maximum(d, 0), axis=1)


def primitive_samples(kind, size, spacing):
    """Surface points of a MuJoCo primitive in its frame, neighbours <= spacing apart."""
    n = lambda L: max(2, int(np.ceil(L / spacing)) + 1)
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        pts = []
        for ax in range(3):
            o = [i for i in range(3) if i != ax]
            u, v = np.meshgrid(np.linspace(-size[o[0]], size[o[0]], n(2 * size[o[0]])),
                               np.linspace(-size[o[1]], size[o[1]], n(2 * size[o[1]])))
            for sgn in (-1, 1):
                q = np.zeros((u.size, 3)); q[:, ax] = sgn * size[ax]; q[:, o[0]] = u.ravel(); q[:, o[1]] = v.ravel()
                pts.append(q)
        return np.concatenate(pts)
    if kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        r, h = size[0], size[1]
        a = np.linspace(0, 2 * np.pi, n(2 * np.pi * r), endpoint=False)
        z = np.linspace(-h, h, n(2 * h))
        A, Z = np.meshgrid(a, z)
        side = np.c_[r * np.cos(A.ravel()), r * np.sin(A.ravel()), Z.ravel()]
        caps = []
        for rr in np.linspace(0, r, n(r)):
            aa = np.linspace(0, 2 * np.pi, max(1, n(2 * np.pi * rr)), endpoint=False)
            for zz in (-h, h):
                if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    caps.append(np.c_[rr * np.cos(aa), rr * np.sin(aa), np.full(len(aa), zz)])
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            sph = primitive_samples(mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), spacing)
            caps = [sph[sph[:, 2] >= 0] + [0, 0, h], sph[sph[:, 2] < 0] - [0, 0, h]]
        return np.concatenate([side] + caps)
    r = size[0]                                   # sphere: Fibonacci lattice
    k = max(8, int(np.ceil(4 * np.pi * r * r / spacing ** 2)))
    i = np.arange(k) + 0.5
    phi, th = np.arccos(1 - 2 * i / k), np.pi * (1 + 5 ** 0.5) * i
    return r * np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


@dataclass(frozen=True)
class PairRule:
    geom_a: str | int
    geom_b: str | int
    clearance_m: float
    max_contact_penetration_m: float
    require_physical_collision: bool = False


class DemoContactAudit:
    def __init__(self, model, rules, allowed_contact=None):
        if not rules:
            raise ValueError('Explicit nonempty geometry coverage is required')
        self.rules = list(rules)
        self.allowed_contact = allowed_contact or (lambda a, b, t: False)
        self.pairs = []
        for r in self.rules:
            if not np.isfinite([r.clearance_m, r.max_contact_penetration_m]).all() or min(
                    r.clearance_m, r.max_contact_penetration_m) < 0:
                raise ValueError('Invalid distance tolerances')
            a, b = model.geom(r.geom_a).id, model.geom(r.geom_b).id
            if a == b:
                raise ValueError('Cannot audit a geom against itself')
            if r.require_physical_collision:
                mask = ((model.geom_contype[a] & model.geom_conaffinity[b]) |
                        (model.geom_contype[b] & model.geom_conaffinity[a]))
                if not mask:
                    raise ValueError(f'Collision masks disabled for {r.geom_a}/{r.geom_b}')
                # Masks alone do not override body excludes or welded-body filtering.
                ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
                wa, wb = int(model.body_weldid[ba]), int(model.body_weldid[bb])
                parent_filtered = wa != 0 and wb != 0 and (
                    model.body_parentid[wa] == wb or model.body_parentid[wb] == wa)
                if wa == wb or parent_filtered:
                    raise ValueError(f'Weld/parent collision filtering: {r.geom_a}/{r.geom_b}')
                signature = (min(ba, bb) << 16) + max(ba, bb)
                if signature in model.exclude_signature:
                    raise ValueError(f'Explicit collision exclusion: {r.geom_a}/{r.geom_b}')
            self.pairs.append((a, b))
        self._physical = np.asarray([self._collides(model, a, b) for a, b in self.pairs], bool)
        # Broad phase (exact, not a tolerance): MuJoCo's bounding sphere geom_rbound encloses each geom about its
        # frame origin, so centre distance - rbound_a - rbound_b > limit proves the pair farther than `limit`, the
        # mj_geomDistance cap. rbound 0 (planes) is unbounded and always measured. Hidden compiled hand variants are
        # resolved per sample from geom_rgba, as before.
        self._a = np.asarray([a for a, _ in self.pairs], int)
        self._b = np.asarray([b for _, b in self.pairs], int)
        self._limit = np.asarray([max(r.clearance_m, r.max_contact_penetration_m, 0.01) + 0.01 for r in self.rules])
        rb = np.asarray(model.geom_rbound, float)
        self._ra, self._rb = rb[self._a], rb[self._b]
        self._unbounded = (self._ra <= 0) | (self._rb <= 0)
        variant = np.asarray([model.geom(g).name.startswith('human_seg_') and (
            '_hand_variant_' in model.geom(g).name or model.geom(g).name.endswith('_hand_mesh'))
            for g in range(model.ngeom)], bool)
        self._variant_a, self._variant_b = variant[self._a], variant[self._b]
        hand = [model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and model.geom(g).name.startswith('human_seg_')
                and '_hand' in model.geom(g).name for g in range(model.ngeom)]
        self._concave = {}
        for a, b in self.pairs:
            for m_, p_ in ((a, b), (b, a)):
                if hand[m_] and model.geom_type[p_] in PRIMITIVES and m_ not in self._concave:
                    mid = model.geom_dataid[m_]
                    va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                    fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
                    key = (model.mesh(mid).name, int(vn), int(fn), float(model.mesh_vert[va].sum()))
                    if key not in _SAMPLE_CACHE:
                        _SAMPLE_CACHE[key] = surface_samples(model.mesh_vert[va:va + vn].astype(float),
                                                             model.mesh_face[fa:fa + fn], SAMPLE_SPACING_M)
                    self._concave[m_] = _SAMPLE_CACHE[key]
        self.refined = 0
        self._prim_samples = {}
        self._planes = {}
        self._contacts_at = None
        self.reset()

    def _refine(self, model, data, a, b, hull):
        """Exact re-measure of a hull-flagged hand-mesh x primitive pair, or of two primitives (module doc); else the
        hull distance."""
        if model.geom_type[a] in PRIMITIVES and model.geom_type[b] in PRIMITIVES:
            best = np.inf
            for g, h in ((a, b), (b, a)):
                key = (int(g), 'prim')
                if key not in self._prim_samples:
                    self._prim_samples[key] = primitive_samples(model.geom_type[g], model.geom_size[g], SAMPLE_SPACING_M)
                world = data.geom_xpos[g] + self._prim_samples[key] @ data.geom_xmat[g].reshape(3, 3).T
                local = (world - data.geom_xpos[h]) @ data.geom_xmat[h].reshape(3, 3)
                best = min(best, float(primitive_sdf(model.geom_type[h], model.geom_size[h], local).min()))
            self.refined += 1
            return best
        for m_, p_ in ((a, b), (b, a)):
            pts = self._concave.get(m_)
            if pts is None:
                continue
            if model.geom_type[p_] == mujoco.mjtGeom.mjGEOM_MESH and p_ not in self._concave:
                eq = self._hull_planes(model, p_)
                world = data.geom_xpos[m_] + pts @ data.geom_xmat[m_].reshape(3, 3).T
                local = (world - data.geom_xpos[p_]) @ data.geom_xmat[p_].reshape(3, 3)
                near = np.linalg.norm(local, axis=1) <= model.geom_rbound[p_] + 0.05
                self.refined += 1
                if not near.any():
                    return 0.05
                sd = (local[near] @ eq[:, :3].T + eq[:, 3]).max(axis=1)   # > 0 outside (a lower bound of the distance)
                return float(sd.min())
            if model.geom_type[p_] not in PRIMITIVES:
                continue
            world = data.geom_xpos[m_] + pts @ data.geom_xmat[m_].reshape(3, 3).T
            local = (world - data.geom_xpos[p_]) @ data.geom_xmat[p_].reshape(3, 3)
            near = np.linalg.norm(local, axis=1) <= model.geom_rbound[p_] + 0.05
            self.refined += 1
            if not near.any():
                return 0.05
            return float(primitive_sdf(model.geom_type[p_], model.geom_size[p_], local[near]).min())
        return hull

    def _hull_planes(self, model, g):
        """Face planes [F, 4] (unit normal, offset; n.x + d <= 0 inside) of the convex hull of a mesh geom's vertices."""
        if g not in self._planes:
            from scipy.spatial import ConvexHull
            mid = model.geom_dataid[g]
            v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]].astype(float)
            self._planes[g] = ConvexHull(v).equations
        return self._planes[g]

    @staticmethod
    def _collides(model, a, b) -> bool:
        if not ((model.geom_contype[a] & model.geom_conaffinity[b]) | (model.geom_contype[b] & model.geom_conaffinity[a])):
            return False
        ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
        wa, wb = int(model.body_weldid[ba]), int(model.body_weldid[bb])
        if wa == wb or (wa != 0 and wb != 0 and (model.body_parentid[wa] == wb or model.body_parentid[wb] == wa)):
            return False
        return ((min(ba, bb) << 16) + max(ba, bb)) not in set(model.exclude_signature)

    def _physics_distance(self, data, a, b) -> float:
        """Deepest solver contact of the geom pair at this step (one vectorised read of data.contact per sample)."""
        if self._contacts_at is None:
            n = data.ncon
            g1, g2 = np.asarray(data.contact.geom1[:n]), np.asarray(data.contact.geom2[:n])
            dist = np.asarray(data.contact.dist[:n])
            key = np.minimum(g1, g2).astype(np.int64) << 20 | np.maximum(g1, g2)
            self._contacts = {}
            for k, x in zip(key.tolist(), dist.tolist()):
                if x < self._contacts.get(k, np.inf):
                    self._contacts[k] = x
            self._contacts_at = True
        return float(self._contacts.get((min(a, b) << 20) | max(a, b), 0.0))

    def reset(self):
        self.samples = 0
        self.first_violation = None
        self.step_violations = []
        self.minimum_distance_m = np.full(len(self.rules), np.inf)
        self.pair_samples = np.zeros(len(self.rules), int)

    def observe(self, model, data):
        self.samples += 1
        self._contacts_at = None                 # solver contacts are read afresh for this sample
        self.step_violations = []                # every violating pair of this sample: (rule index, distance, boundary)
        if not all(np.isfinite(x).all() for x in
                   (data.qpos, data.qvel, data.mocap_pos, data.mocap_quat)):
            if self.first_violation is None:
                self.first_violation = dict(kind='nonfinite_state', t=float(data.time))
            return
        if not np.isfinite(data.geom_xpos).all():
            if self.first_violation is None:
                self.first_violation = dict(kind='nonfinite_geometry', t=float(data.time))
            return
        hidden = np.asarray(model.geom_rgba[:, 3] == 0)
        active = ~((self._variant_a & hidden[self._a]) | (self._variant_b & hidden[self._b]))
        centre = np.linalg.norm(data.geom_xpos[self._a] - data.geom_xpos[self._b], axis=1)
        near = active & (self._unbounded | (centre - self._ra - self._rb <= self._limit))
        self.pair_samples += active
        far = active & ~near
        self.minimum_distance_m[far] = np.minimum(self.minimum_distance_m[far], self._limit[far])
        for i in np.flatnonzero(near):
            r, (a, b) = self.rules[i], self.pairs[i]
            limit = float(self._limit[i])
            distance = float(mujoco.mj_geomDistance(model, data, a, b, limit, None))
            if not np.isfinite(distance):
                if self.first_violation is None:
                    self.first_violation = dict(kind='nonfinite_distance', t=float(data.time),
                                               geom_a=r.geom_a, geom_b=r.geom_b)
                continue
            if distance < 0 and self._physical[i]:
                distance = max(distance, self._physics_distance(data, a, b))
            elif distance < r.clearance_m and (a in self._concave or b in self._concave or (
                    model.geom_type[a] in PRIMITIVES and model.geom_type[b] in PRIMITIVES)):
                distance = max(distance, self._refine(model, data, a, b, distance))
            self.minimum_distance_m[i] = min(self.minimum_distance_m[i], distance)
            if distance >= r.clearance_m:
                continue
            permit = self.allowed_contact(r.geom_a, r.geom_b, float(data.time))
            allowed = bool(permit)
            bound = r.max_contact_penetration_m if permit is True or not allowed else float(permit)
            boundary = -bound if allowed else r.clearance_m
            if distance < boundary:
                self.step_violations.append((i, distance, boundary, allowed))
            if distance < boundary and self.first_violation is None:
                self.first_violation = dict(kind='penetration' if distance < 0 else 'clearance',
                    t=float(data.time), geom_a=r.geom_a, geom_b=r.geom_b,
                    distance_m=distance, required_minimum_m=boundary,
                    contact_permitted=allowed)

    def report(self):
        return dict(scope='configured_pairs_only', samples=self.samples,
                    passed_configured_pairs=bool(self.pair_samples.any()) and self.first_violation is None,
                    first_violation=self.first_violation,
                    pairs=[dict(geom_a=r.geom_a, geom_b=r.geom_b,
                                checked_samples=int(n), minimum_distance_m=float(d) if np.isfinite(d) else None)
                           for r, d, n in zip(self.rules, self.minimum_distance_m, self.pair_samples)])
