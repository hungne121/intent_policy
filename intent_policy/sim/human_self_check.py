"""Penetration of the mannequin's arms into its own body (acceptance record 01/BB-DL, decision 46/QĐ-DL; measurement
docs/reports/do_tay_xuyen_than_2026-11-06.md).

Why not mj_geomDistance: it measures meshes by their convex hull. The `pelvis` segment mesh is the hips and both legs
(295 x 362 x 1031 mm), so its hull fills the space where a relaxed hand hangs beside the thigh: an idle I3 hand read
-114 mm against it. The body segments are instead cut into slabs 1 cm thick along their long axis; each slab's
cross-section of a torso / head / hips is close to convex, so its 2-D hull is tight. A vertex of a forearm or of the
visible hand mesh is inside a body segment when it lies inside the hull of its slab; its depth is the distance to the
nearest hull edge.

The mannequin is kinematic (mocap), so one check per control tick sees every pose.
"""
import numpy as np
from scipy.spatial import ConvexHull

ARM = ('l_forearm', 'l_hand', 'r_forearm', 'r_hand')
BODY = ('torso', 'pelvis', 'head')
SLAB_M = 0.01          # slab thickness along the segment's long axis
WINDOW_M = 0.01        # vertices within this of the slab centre build the slab's hull (meshes are irregular)
ARM_STRIDE = 3         # every third arm vertex (a forearm has 1166, a hand 2299; spacing stays a few mm)


def _mesh_vertices(model, g):
    mid = int(model.geom_dataid[g])
    a, n = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    return np.asarray(model.mesh_vert[a:a + n], float)


class BodySlabs:
    """Slab hulls of one body segment mesh in its geom frame."""

    def __init__(self, vert):
        self.axis = int(np.argmax(np.ptp(vert, axis=0)))
        self.plane = [i for i in range(3) if i != self.axis]
        h = vert[:, self.axis]
        self.h0 = float(h.min())
        self.n = int(np.ceil((h.max() - self.h0) / SLAB_M)) + 1
        self.eq = [None] * self.n            # per slab: hull edge equations [nx, ny, c], inside when n.q + c <= 0
        for k in range(self.n):
            c = self.h0 + (k + 0.5) * SLAB_M
            q = vert[np.abs(h - c) <= WINDOW_M][:, self.plane]
            if len(q) >= 4:
                try:
                    self.eq[k] = ConvexHull(q).equations
                except Exception:                 # degenerate slab (collinear points at a tip)
                    self.eq[k] = None

    def depth(self, p):
        """Depth (m, > 0 inside) of local points p (N, 3); -inf outside the segment's length."""
        return self.depth_normal(p)[0]

    def exit_along(self, p, d):
        """Per local point: distance along local direction d (its in-slab component, normalised) to leave its slab
        hull; -inf for points outside the hull or the segment's length."""
        dq = np.asarray(d, float)[self.plane]
        dq = dq / max(float(np.linalg.norm(dq)), 1e-9)
        k = np.floor((p[:, self.axis] - self.h0) / SLAB_M).astype(int)
        out = np.full(len(p), -np.inf)
        for s in np.unique(k):
            if s < 0 or s >= self.n or self.eq[s] is None:
                continue
            idx = np.flatnonzero(k == s)
            e = self.eq[s]
            vals = p[idx][:, self.plane] @ e[:, :2].T + e[:, 2]          # <= 0 on every edge when inside
            inside = (vals <= 0.0).all(axis=1)
            nd = e[:, :2] @ dq                                             # edges the direction runs into
            t = np.where(nd > 1e-9, -vals / np.where(nd > 1e-9, nd, 1.0), np.inf).min(axis=1)
            out[idx[inside]] = t[inside]
        return out

    def depth_normal(self, p):
        """Depth as `depth`, and per point the outward unit normal (local frame) of the nearest slab-hull edge."""
        k = np.floor((p[:, self.axis] - self.h0) / SLAB_M).astype(int)
        out, normal = np.full(len(p), -np.inf), np.zeros((len(p), 3))
        for s in np.unique(k):
            if s < 0 or s >= self.n or self.eq[s] is None:
                continue
            idx = np.flatnonzero(k == s)
            e = self.eq[s]
            vals = p[idx][:, self.plane] @ e[:, :2].T + e[:, 2]
            j = vals.argmax(axis=1)
            out[idx] = -vals[np.arange(len(idx)), j]
            normal[np.ix_(idx, self.plane)] = e[j, :2]
        return out, normal


class HumanSelfCheck:
    def __init__(self, model):
        self.model = model
        name = lambda g: model.body(model.geom_bodyid[g]).name
        self.body_geom = {}
        for g in range(model.ngeom):
            n = name(g)
            if model.geom_type[g] == 7 and n.startswith('human_seg_') and n[10:] in BODY:
                self.body_geom[n[10:]] = g
        self.slabs = {s: BodySlabs(_mesh_vertices(model, g)) for s, g in self.body_geom.items()}
        self.arm_geoms = {s: [g for g in range(model.ngeom) if model.geom_type[g] == 7 and name(g) == f'human_seg_{s}']
                          for s in ARM}
        self._vert = {}

    def _arm_points(self, data, s):
        """World vertices of arm segment s: the visible mesh geom(s) of that segment (hand variants are hidden by
        alpha 0; the base hand geom may swap meshes, so the mesh id is read at every call)."""
        m = self.model
        pts = []
        for g in self.arm_geoms[s]:
            if m.geom_rgba[g][3] <= 0.0:
                continue
            mid = int(m.geom_dataid[g])
            if mid not in self._vert:
                self._vert[mid] = _mesh_vertices(m, g)[::ARM_STRIDE]
            pts.append(data.geom_xpos[g] + self._vert[mid] @ data.geom_xmat[g].reshape(3, 3).T)
        return np.concatenate(pts) if pts else np.zeros((0, 3))

    def measure(self, data):
        """Deepest penetration (m) of each arm segment into each body segment now: {(arm, body): depth}, only > 0."""
        res = {}
        for s in ARM:
            p = self._arm_points(data, s)
            if not len(p):
                continue
            for b, g in self.body_geom.items():
                local = (p - data.geom_xpos[g]) @ data.geom_xmat[g].reshape(3, 3)
                d = float(self.slabs[b].depth(local).max())
                if d > 0.0:
                    res[(s, b)] = d
        return res
