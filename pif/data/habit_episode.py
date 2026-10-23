"""HABIT perception output (pif/perception/habit_perception.py) -> PIF episode dict (pif/data/tokens.py), 10 Hz,
estimated 3D in the table frame.

Objects: the detection tracks of the episode, without the robot gripper and without tracks seen in fewer than
`min_track_frac` of the frames (spurious detections). A hidden object keeps its token with visible = 0 and its last
seen position (causal hold; frames before its first detection are zero).
holding: when a gripper of the robot is closed (configs/pif/habit_labels.yaml thresholds), the robot holds the
`graspable` object track nearest to the gripper's image position (S8789 / S8791: the bread, S6760: the board);
-1 otherwise. has_item / done are not built for HABIT yet (zeros).

Eval labels (S8789): the target container = the container track nearest (table frame) to the bread's position
after the release, cross-checked with the release-position rule (pif/data/habit_label_check.py).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import yaml

from intent_policy.utils import resolve
from pif.data.habit import gripper_events, read_habit_episode, debounce

GRASPABLE = {'S8789': 'bread roll', 'S8791': 'bread roll', 'S6760': 'circuit board'}


def read_habit_tokens(npz: str | Path, cfg: dict, labels_cfg: dict, min_track_frac: float = 0.1) -> dict:
    r = dict(np.load(npz, allow_pickle=True))
    sid, T = str(r['sid']), int(r['T'])
    names = [str(n) for n in r['keypoints']]
    order = [names.index(k) for k in cfg['tokens']['keypoints']]
    keep = np.flatnonzero(r['obj_vis'].mean(0) >= min_track_frac)
    pos, vis = r['obj_pos'][:, keep].astype(np.float64), r['obj_vis'][:, keep]
    held = pos.copy()
    for k in range(len(keep)):                                   # causal hold of the last seen position
        last = None
        for t in range(T):
            if vis[t, k]:
                last = pos[t, k]
            elif last is not None:
                held[t, k] = last
    labels = [str(x) for x in r['obj_label'][keep]]
    ep = read_habit_episode(int(r['index']))
    g = labels_cfg['gripper']
    closed = np.zeros(T, bool)
    for arm in ('robot0', 'robot1'):
        closed |= debounce(ep[f'{arm}_grip'][:T] > g['threshold'][sid], g['min_frames'])
    grasp_k = [k for k, l in enumerate(labels) if l == GRASPABLE[sid]]
    holding = np.full(T, -1)
    if grasp_k:
        # the graspable track nearest to a detected gripper this frame (image distance), else the last one held
        dets = json.loads(str(r['dets_json']))
        last = grasp_k[0]
        for t in np.flatnonzero(closed):
            gu = r['gripper_uv'][t]
            gu = gu[np.isfinite(gu).all(1)]
            if len(gu):
                cand = []
                for k in grasp_k:
                    if vis[t, k]:
                        uv = _track_uv(dets[t], labels[k], held[t, k], r)
                        if uv is not None:
                            cand.append((np.linalg.norm(gu - uv, axis=1).min(), k))
                if cand:
                    last = min(cand)[1]
            holding[t] = last
    hp = r['human_pos'][:, order].astype(np.float64)
    return dict(fps=10.0, t=np.arange(T) / 10.0, human_pos=hp, human_vis=r['human_vis'][:, order],
                obj_pos=held, obj_vis=vis, obj_type=r['obj_type'][keep].astype(int), obj_has_item=np.zeros((T, len(keep)), bool),
                obj_done=np.zeros((T, len(keep)), bool), holding=holding, obj_names=labels,
                meta=dict(domain='habit', sid=sid, task=sid, episode_index=int(r['index']), stride=1))


def _track_uv(dets_t: list, label: str, xyz, r) -> np.ndarray | None:
    """Image centre of the detection of `label` in this frame closest to the track's 3D position (projected)."""
    f, (h, w) = float(r['focal']), r['image_size']
    pc = np.asarray(xyz) @ r['R'] + r['origin']
    if pc[2] <= 0:
        return None
    uv = np.array([f * pc[0] / pc[2] + w / 2, f * pc[1] / pc[2] + h / 2])
    boxes = [np.array([(d['box'][0] + d['box'][2]) / 2, (d['box'][1] + d['box'][3]) / 2]) for d in dets_t if d['label'] == label]
    return min(boxes, key=lambda b: np.linalg.norm(b - uv)) if boxes else None


def s8789_label(ep: dict, release_frame: int | None) -> int | None:
    """Target container token = the container nearest to the bread after the release (table frame, xy)."""
    names = ep['obj_names']
    cont = [k for k, n in enumerate(names) if n == 'airtight container']
    bread = [k for k, n in enumerate(names) if n == 'bread roll']
    if release_frame is None or not cont or not bread:
        return None
    t = min(release_frame + 10, len(ep['holding']) - 1)            # 1 s after the release
    b = ep['obj_pos'][t, bread[0], :2]
    return cont[int(np.argmin([np.linalg.norm(ep['obj_pos'][t, k, :2] - b) for k in cont]))]
