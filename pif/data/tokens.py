"""Per-frame PIF tokens (handoff §2, updated by HANDOFF_PIF_CORRECTIONS.md #14 / #17 / #18).

Every domain reader (sim: pif/data/episode_reader.py, HABIT: pif/data/habit.py) produces the same *episode* dict at
the PIF rate, positions in the table frame (z = 0 on the table top, metres):

  fps           float
  human_pos     [T, J, 3]  keypoints (config `tokens.keypoints` order)
  human_vis     [T, J]     bool, keypoint seen / valid
  obj_pos       [T, K, 3]  visible things only: objects, containers, place surfaces (no layout zone ids)
  obj_vis       [T, K]     bool
  obj_type      [K]        index into `tokens.object_types`
  obj_has_item  [T, K]     bool, something stands on / in it
  obj_done      [T, K]     bool, the robot has finished with it (delivered / an object was delivered onto it)
  holding       [T]        index of the object the robot holds, -1 = none (never the end-effector pose: copycat, B5)
  meta          dict

`build_tokens` turns it into model inputs:

  human     [T, J, 7]  position (3), velocity (3, backward difference x fps: causal), visible (1)
  objects   [T, K, 6]  position (3), has_item, done, visible
  obj_type  [K]
  holding   [T]

A token at frame t uses frames t and t - 1 only. Hidden keypoints / objects keep their token with zeros and visible = 0
(handoff §2: never removed). Positions and velocities are normalised with train statistics (`norm_stats`); one position
mean / std is shared by human and object tokens so their relative geometry (where a hand points / goes) is kept.
"""
from __future__ import annotations
import numpy as np

HUMAN_DIM, OBJECT_DIM = 7, 6


def backward_velocity(pos: np.ndarray, vis: np.ndarray, fps: float) -> np.ndarray:
    """v_t = (p_t - p_{t-1}) * fps along axis 0; 0 at t = 0 and where p_t or p_{t-1} is not visible."""
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) * fps
    ok = np.zeros(vis.shape, bool)
    ok[1:] = vis[1:] & vis[:-1]
    vel[~ok] = 0.0
    return vel


def norm_stats(episodes: list[dict]) -> dict:
    """Train-set statistics: position mean / std per axis over visible human keypoints and objects, velocity std per
    axis over visible human keypoints (stored with the checkpoint, handoff §2)."""
    pos = np.concatenate([e['human_pos'][e['human_vis']] for e in episodes] + [e['obj_pos'][e['obj_vis']] for e in episodes])
    vel = np.concatenate([backward_velocity(e['human_pos'], e['human_vis'], e['fps'])[e['human_vis']] for e in episodes])
    return dict(pos_mean=pos.mean(0).astype(np.float32), pos_std=pos.std(0).astype(np.float32) + 1e-6,
                vel_std=vel.std(0).astype(np.float32) + 1e-6)


def build_tokens(ep: dict, stats: dict | None = None) -> dict:
    hv, ov = ep['human_vis'], ep['obj_vis']
    vel = backward_velocity(ep['human_pos'], hv, ep['fps'])
    hpos, opos = ep['human_pos'].astype(np.float32), ep['obj_pos'].astype(np.float32)
    if stats is not None:
        hpos = (hpos - stats['pos_mean']) / stats['pos_std']
        opos = (opos - stats['pos_mean']) / stats['pos_std']
        vel = vel / stats['vel_std']
    human = np.concatenate([hpos, vel, hv[..., None]], -1).astype(np.float32)
    human[~hv] = 0.0
    objects = np.concatenate([opos, ep['obj_has_item'][..., None], ep['obj_done'][..., None], ov[..., None]],
                             -1).astype(np.float32)
    objects[~ov, :3] = 0.0
    return dict(human=human, objects=objects, obj_type=np.asarray(ep['obj_type'], np.int64),
                holding=np.asarray(ep['holding'], np.int64))


def permute_objects(ep: dict, perm: np.ndarray) -> dict:
    """Same episode with the object axis reordered by `perm` (new k = old perm[k]); `holding` follows."""
    perm = np.asarray(perm)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    out = dict(ep)
    for key in ('obj_pos', 'obj_vis', 'obj_has_item', 'obj_done'):
        out[key] = ep[key][:, perm]
    out['obj_type'] = np.asarray(ep['obj_type'])[perm]
    if 'obj_names' in ep:
        out['obj_names'] = [ep['obj_names'][i] for i in perm]
    h = np.asarray(ep['holding'])
    out['holding'] = np.where(h >= 0, inv[np.maximum(h, 0)], -1)
    return out


def slice_episode(ep: dict, start: int, stop: int) -> dict:
    """Frames [start, stop) of an episode (per-frame arrays only)."""
    out = dict(ep)
    for key in ('human_pos', 'human_vis', 'obj_pos', 'obj_vis', 'obj_has_item', 'obj_done', 'holding'):
        out[key] = ep[key][start:stop]
    return out
