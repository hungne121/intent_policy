"""HABIT episodes (configinc/HABIT, `full` config, 10 Hz): robot state, decoded subtasks, gripper events.

Notes from PIF_P0 / HANDOFF_PIF_LOCAL §4 that this reader follows:
- subtask indices are decoded with the meta of the SAME config (`full/meta`); `sample` numbers them differently;
- robot0 / robot1 cartesian positions are each in their own arm base frame (robot0 = left arm); no transform between
  the arms and no camera intrinsics, so TCP positions are only compared within one arm;
- gripper closed = gripper_position above a per-task threshold (configs/pif/habit_labels.yaml; LOCAL §4.5's 0.5 fails
  for the soft butter roll of S8789 / S8791).
"""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from intent_policy.utils import resolve

FPS = 10.0
GRIP_CLOSED = 0.5
ARMS = ('robot0', 'robot1')


@lru_cache(maxsize=2)
def habit_meta(root: str = 'outputs/habit') -> dict:
    root = resolve(root) / 'full/meta'
    read = lambda f: [json.loads(line) for line in open(root / f)]
    episodes = {int(e['episode_index']): e for e in read('episodes.jsonl')}
    return dict(episodes=episodes, human={d['task_index']: d['task'] for d in read('human_subtasks.jsonl')},
                robot={d['task_index']: d['task'] for d in read('subtasks.jsonl')})


def episode_indices(sid: str, root: str = 'outputs/habit') -> list[int]:
    return sorted(i for i, e in habit_meta(root)['episodes'].items() if e['sid'] == sid)


def paths(index: int, root: str = 'outputs/habit', cam: str = 'front_view') -> tuple[Path, Path]:
    base, c = resolve(root) / 'full', index // 1000
    return (base / f'data/chunk-{c:03d}/episode_{index:06d}.parquet',
            base / f'videos/chunk-{c:03d}/observation.images.{cam}/episode_{index:06d}.mp4')


def runs(x: np.ndarray) -> list[tuple[int, int, object]]:
    """Run-length segments (start, end inclusive, value) of a 1-D array."""
    cut = np.flatnonzero(np.r_[True, x[1:] != x[:-1]])
    return [(int(a), int(b) - 1, x[a]) for a, b in zip(cut, np.r_[cut[1:], len(x)])]


def read_habit_episode(index: int, root: str = 'outputs/habit') -> dict:
    meta = habit_meta(root)
    df = pd.read_parquet(paths(index, root)[0])
    out = dict(index=index, sid=meta['episodes'][index]['sid'], t=df['frame_index'].to_numpy() / FPS,
               human_subtask=df['human_role_subtask_index'].to_numpy(), robot_subtask=df['low_level_task_index'].to_numpy())
    for arm in ARMS:
        out[f'{arm}_tcp'] = np.stack(df[f'{arm}.observation.state.cartesian_position'].to_numpy())[:, :3]
        out[f'{arm}_grip'] = np.hstack(df[f'{arm}.observation.state.gripper_position'].to_numpy())
    out['human_segments'] = [dict(start=a, end=b, text=meta['human'][int(v)]) for a, b, v in runs(out['human_subtask'])]
    out['robot_segments'] = [dict(start=a, end=b, text=meta['robot'][int(v)]) for a, b, v in runs(out['robot_subtask'])]
    return out


def debounce(x: np.ndarray, min_frames: int) -> np.ndarray:
    """Boolean series where runs shorter than `min_frames` take the previous run's value; the first and the last run
    are kept (cut by the episode bounds, not flicker)."""
    x = x.copy()
    for a, b, v in runs(x):
        if 0 < a and b < len(x) - 1 and b - a + 1 < min_frames:
            x[a:b + 1] = x[a - 1]
    return x


def gripper_events(ep: dict, threshold: float = GRIP_CLOSED, min_frames: int = 1) -> list[dict]:
    """Close / open events of each arm: {arm, frame, kind, tcp} at the first frame of the new state."""
    events = []
    for arm in ARMS:
        closed = debounce(ep[f'{arm}_grip'] > threshold, min_frames)
        for f in np.flatnonzero(closed[1:] != closed[:-1]) + 1:
            events.append(dict(arm=arm, frame=int(f), kind='close' if closed[f] else 'open',
                               tcp=ep[f'{arm}_tcp'][f].round(4).tolist()))
    return sorted(events, key=lambda e: e['frame'])


def motion_onset(tcp: np.ndarray, start: int, end: int, speed_mps: float) -> int | None:
    """First frame in [start, end] where the TCP speed (backward difference) exceeds `speed_mps`."""
    v = np.r_[0.0, np.linalg.norm(np.diff(tcp, axis=0), axis=1) * FPS]
    hit = np.flatnonzero(v[start:end + 1] > speed_mps)
    return int(start + hit[0]) if len(hit) else None
