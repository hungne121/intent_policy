"""Human hand paths from generated motion (HY-Motion 1.0 library, configs/human_motion/hymotion_v6.yaml) for the
scripted human: natural curvature, speed profile, hesitations instead of the straight minimum-jerk line.

A clip is converted to the simulator world exactly as the mannequin was built (scripts/build_human_asset.py:
SMPL_TO_WORLD, facing -X), turned so the person faces as in the scene (`facing_deg`) and placed with its pelvis
on the scene's pelvis. The right palm path is cut into motions (speed above STILL_FRACTION of the clip's peak, the
convention used for the pointing kinematics) and stored. A scripted motion from p0 to p1 then uses a stored
motion of its kind, warped so it starts at p0 and ends at p1 (motion warping: Witkin & Popović, 'Motion Warping',
SIGGRAPH 1995; the offsets are blended in with the motion's own progress) and replayed over the scripted duration
(its own speed profile kept, uniformly time-scaled).
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from intent_policy.sim.base_env import ROOT

SMPL_TO_WORLD = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)   # = scripts/build_human_asset.py
PELVIS, L_HIP, R_HIP, HEAD, R_WRIST = 0, 1, 2, 15, 21
FEET = (7, 8, 10, 11)                   # SMPL-H L/R ankle, L/R foot
R_HAND_BASES = (40, 43, 46, 49)        # SMPL-H right Index1 / Middle1 / Pinky1 / Ring1 (37 + 3k)
STILL_FRACTION = 0.10                   # motion = palm speed above 10 % of the clip's peak
FPS = 30.0                              # HY-Motion output rate (hymotion/pipeline/motion_diffusion.py)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def to_scene(k3d: np.ndarray, pelvis_world: np.ndarray, facing_deg: float, floor_z: float = 0.0,
             height: float | None = None) -> np.ndarray:
    """k3d [L, 52, 3] (HY-Motion axes, Y up) -> world [L, 52, 3]: mannequin axes, scaled to `height` (feet -> head
    joint of the first frame; the generated people are ~1.5 m between those joints), first-frame facing turned to
    `facing_deg` (world yaw of the facing direction), first-frame pelvis on `pelvis_world` (x, y), feet on `floor_z`
    (the clips are root-centred, not grounded, measured 2026-10-24)."""
    w = k3d @ SMPL_TO_WORLD.T
    feet = w[0, list(FEET), 2].min()
    if height is not None:
        w = w * (height / (w[0, HEAD, 2] - feet))
        feet = w[0, list(FEET), 2].min()
    w[..., 2] += floor_z - feet
    hips = w[0, R_HIP] - w[0, L_HIP]                      # the person's right; facing = up x right (mannequin: right
    face = np.cross([0.0, 0.0, 1.0], hips)                #   = +Y, facing -X)
    face[2] = 0.0
    yaw0 = np.arctan2(face[1], face[0])
    w = w @ rot_z(np.radians(facing_deg) - yaw0).T
    w[..., :2] += np.asarray(pelvis_world[:2], float) - w[0, PELVIS, :2]
    return w


def palm_path(w: np.ndarray, hand: str = 'right') -> np.ndarray:
    """Palm centre [L, 3]; left/right come from the same original skeleton."""
    if hand not in ('left', 'right'):
        raise ValueError('hand must be left or right')
    ids = [R_WRIST, *R_HAND_BASES] if hand == 'right' else [20, 25, 28, 31, 34]
    return w[:, ids].mean(axis=1)


def clip_id(k3d: np.ndarray) -> str:
    """Content identity before transforms; duplicates cannot cross dataset splits."""
    x = np.ascontiguousarray(k3d, dtype='<f4')
    return hashlib.sha256(str(x.shape).encode() + x.tobytes()).hexdigest()


@dataclass
class MotionRecord:
    path: np.ndarray
    source: dict
    skeleton: np.ndarray | None = None  # synchronized full skeleton before task-specific retargeting


def segments(p: np.ndarray, min_len_m: float = 0.04) -> list[np.ndarray]:
    """Cut a palm path into motions: runs of frames whose speed is above STILL_FRACTION of the peak, at least
    `min_len_m` long (end to end)."""
    return [p[a:b] for a, b in segment_ranges(p, min_len_m)]


def segment_ranges(p: np.ndarray, min_len_m: float = 0.04) -> list[tuple[int, int]]:
    """Original frame ranges, end exclusive; retain provenance across crops."""
    if len(p) < 2:
        return []
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) * FPS
    moving = v > STILL_FRACTION * v.max()
    out, i = [], 0
    while i < len(moving):
        if moving[i]:
            j = i
            while j < len(moving) and moving[j]:
                j += 1
            seg = p[i:j + 1]
            if np.linalg.norm(seg[-1] - seg[0]) >= min_len_m:
                out.append((i, j + 1))
            i = j
        else:
            i += 1
    return out


class MotionBank:
    """Palm motions per library key (prompt) from the npz files of gen_motions.py, in the scene's frame."""

    def __init__(self, folder: str | Path, pelvis_world, facing_deg: float, floor_z: float, height: float,
                 *, hand: str = 'right', split_manifest: str | Path | None = None, split: str | None = None):
        self.motions: dict[str, list[np.ndarray]] = {}
        self.records: dict[str, list[MotionRecord]] = {}
        if (split_manifest is None) != (split is None):
            raise ValueError('split and split_manifest must be specified together')
        assignments = None
        if split_manifest is not None:
            manifest = json.loads((ROOT / split_manifest).read_text())
            assignments = manifest['clip_splits']
            if split not in set(assignments.values()):
                raise ValueError(f'No clips assigned to split {split}')
        for f in sorted((ROOT / folder).glob('*.npz')):
            with np.load(f, allow_pickle=False) as raw:
                k3d = raw['keypoints3d']                # [B, L, 52, 3]
                if k3d.ndim != 4 or k3d.shape[2:] != (52, 3) or not np.isfinite(k3d).all():
                    raise ValueError(f'Invalid skeleton array: {f}')
                fps = float(raw['fps']) if 'fps' in raw else FPS
                if fps != FPS:
                    raise ValueError(f'{f}: resample explicitly before using {fps} Hz clips')
                seed_values = raw['seeds'].tolist() if 'seeds' in raw else [None] * len(k3d)
            file_hash = hashlib.sha256(f.read_bytes()).hexdigest()
            records = []
            for b in range(len(k3d)):
                cid = clip_id(k3d[b])
                if assignments is not None:
                    if cid not in assignments:
                        raise ValueError(f'Clip absent from split manifest: {f.name} sample {b}')
                    if assignments[cid] != split:
                        continue
                skeleton = to_scene(k3d[b], pelvis_world, facing_deg, floor_z, height)
                palm = palm_path(skeleton, hand)
                for a, z in segment_ranges(palm):
                    records.append(MotionRecord(palm[a:z], dict(clip_id=cid, file=str(f), file_sha256=file_hash,
                        sample_index=b, generation_seed=seed_values[b], start_frame=a, end_frame_exclusive=z,
                        fps=fps, hand=hand, split=split, scene_transform=dict(
                            pelvis_world=np.asarray(pelvis_world).tolist(), facing_deg=float(facing_deg),
                            floor_z=float(floor_z), height=float(height))), skeleton[a:z]))
            self.records[f.stem] = records
            self.motions[f.stem] = [r.path for r in records]

    def keys(self, prefix: str = '') -> list[str]:
        return [k for k, v in self.motions.items() if k.startswith(prefix) and v]

    def pool(self, keys: list[str]) -> list[np.ndarray]:
        """Stored motions of the given library keys (a trailing '*' matches a key prefix)."""
        return [r.path for r in self.pool_records(keys)]

    def pool_records(self, keys: list[str]) -> list[MotionRecord]:
        names = [n for k in keys for n in (self.keys(k[:-1]) if k.endswith('*') else [k]) if n in self.motions]
        return [m for n in dict.fromkeys(names) for m in self.records[n]]


def warp(motion: np.ndarray, p0, p1) -> np.ndarray:
    """Motion warping (Witkin & Popović 1995): the motion displaced so it starts at p0 and ends at p1, the start /
    end offsets blended along the motion's own progress (arc length), so the shape between is kept."""
    m = np.asarray(motion, float)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(m, axis=0), axis=1))]
    s = (arc / arc[-1])[:, None] if arc[-1] > 0 else np.linspace(0, 1, len(m))[:, None]
    d0, d1 = np.asarray(p0, float) - m[0], np.asarray(p1, float) - m[-1]
    return m + (1 - s) * d0 + s * d1


def sample(path: np.ndarray, u: float) -> np.ndarray:
    """Position at progress u in [0, 1] of the path's own time (uniform over its frames, linear between)."""
    x = float(np.clip(u, 0.0, 1.0)) * (len(path) - 1)
    i = min(int(x), len(path) - 2)
    return path[i] + (x - i) * (path[i + 1] - path[i])


def fit(motion: np.ndarray, p0, p1, scale_range=(0.5, 2.0)) -> np.ndarray:
    """A stored motion re-targeted from p0 to p1, its shape kept: turned about the vertical so its horizontal
    direction is p0 -> p1 and its horizontal extent scaled to the distance (within `scale_range`), then the
    remaining start / end offsets warped in (warp)."""
    m = np.asarray(motion, float) - motion[0]
    u, v = m[-1], np.asarray(p1, float) - np.asarray(p0, float)
    nu, nv = np.linalg.norm(u[:2]), np.linalg.norm(v[:2])
    if nu > 1e-3 and nv > 1e-3:
        m = m @ rot_z(np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0])).T
        m[:, :2] *= float(np.clip(nv / nu, *scale_range))
    return warp(np.asarray(p0, float) + m, p0, p1)


def above_table(path: np.ndarray, table_top: float, edge_y: float, margin: float) -> np.ndarray:
    """The generated people do not see the table: palm points over it (y > edge_y) are kept `margin` above its top."""
    p = path.copy()
    over = p[:, 1] > edge_y
    p[over, 2] = np.maximum(p[over, 2], table_top + margin)
    return p
