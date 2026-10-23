"""Sim episodes -> PIF episode dicts (pif/data/tokens.py) at the PIF rate, in the table frame.

Inputs: the LeRobot dataset (meta/hri_episodes.jsonl) and its replay (pif/data/sim_replay.py: per-frame object
positions, grasp state, 11 keypoints). Only the replay's fields are used per frame; the dataset gives the episode
meta (task, role, eval labels).

Object tokens (CORRECTIONS #14): the objects on the table (type from their catalog shape: box -> object, cup ->
container) and the visible tape markers of `sim.surfaces` (type surface, static position at the tape centre).
  has_item  surface: a free object's centre lies inside the tape footprint, below `sim.surface_height_m`;
            container: a free object's centre lies inside its radius and above its base (T3 block in the cup)
  done      object: the robot has grasped and released it (delivered); surface: holds an object that is done
  holding   index of the object whose grasp state is set, -1 = none
"""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from intent_policy.utils import resolve

def _types(cfg: dict) -> dict:
    return {name: i for i, name in enumerate(cfg['tokens']['object_types'])}


@lru_cache(maxsize=4)
def _code_configs(code_root: str) -> tuple[dict, dict]:
    root = Path(code_root)
    layout = yaml.safe_load((root / 'configs/layout.yaml').read_text())
    scene = yaml.safe_load((root / 'configs/scene/common.yaml').read_text())
    return layout, scene


def replay_code_root(replay_dir: Path) -> str:
    """Configs (layout, object catalog) of the code the replay ran with; the working tree without --code-commit."""
    summary = json.loads((replay_dir / 'replay_summary.json').read_text())
    commit = summary.get('code_commit')
    return str(resolve(f'outputs/pif/code/{commit}') if commit else resolve('.'))


def load_meta(dataset_root: Path) -> dict[int, dict]:
    return {m['episode_index']: m for m in map(json.loads, open(dataset_root / 'meta/hri_episodes.jsonl'))}


def surface_specs(cfg: dict, layout: dict) -> list[dict]:
    out = []
    for name in cfg['sim']['surfaces']:
        z = layout['zones'][name]
        size = np.broadcast_to(np.asarray(z.get('size', layout['marker_size']), float), 2)
        out.append(dict(name=name, xy=np.asarray(z['xy'], float), half=size / 2))
    return out


def read_sim_episode(meta: dict, replay_file: Path, cfg: dict, code_root: str, source_fps: int = 20) -> dict:
    """One episode at `cfg['fps']`. Frame i of the output = source frame i * stride (stride = source_fps / fps)."""
    stride = int(round(source_fps / cfg['fps']))
    if abs(source_fps / stride - cfg['fps']) > 1e-9:
        raise ValueError(f"fps {cfg['fps']} is not an integer divisor of the source rate {source_fps}")
    layout, scene = _code_configs(code_root)
    types = _types(cfg)
    z0 = float(cfg['sim']['table_z'])
    r = np.load(replay_file)
    if not bool(r['reproduced']):
        check = json.loads((replay_file.parent / 'replay_summary.json').read_text())['episodes'][str(meta['episode_index'])]
        if not (check['frames'] == check['dataset_frames'] and check['holding_mismatch'] == 0 and check['keypoints'] < 1e-5
                and check['state'] <= cfg['sim']['replay_state_tol']):
            raise ValueError(f'{replay_file} did not reproduce the dataset episode: {check}')
    names = [str(n) for n in r['obj_names']]
    kp_names = [str(n) for n in r['keypoint_names']]
    order = [kp_names.index(k) for k in cfg['tokens']['keypoints']]
    sel = slice(0, None, stride)
    human = r['keypoints'][sel][:, order].astype(np.float64)
    human[..., 2] -= z0
    pos = r['obj_pos'][sel].astype(np.float64)
    pos[..., 2] -= z0
    grasped = r['obj_grasped'][sel]
    T, K = grasped.shape
    catalog = scene['objects_catalog']
    otype = [types[cfg['sim']['shape_type'][catalog[n]['shape']]] for n in names]
    # done: released after a grasp (on the source frames, so a short grasp between two kept frames is not missed)
    g_full = r['obj_grasped']
    released = np.zeros_like(g_full)
    released[1:] = g_full[:-1] & ~g_full[1:]
    done = np.maximum.accumulate(released, axis=0)[sel]
    # containers: an object inside the cup (T3)
    has_item = np.zeros((T, K), bool)
    free = ~grasped
    for c in [k for k in range(K) if otype[k] == types['container']]:
        radius, height = catalog[names[c]]['size']
        for k in range(K):
            if k == c:
                continue
            d = np.linalg.norm(pos[:, k, :2] - pos[:, c, :2], axis=-1)
            dz = pos[:, k, 2] - pos[:, c, 2]
            has_item[:, c] |= free[:, k] & (d < radius) & (dz > -height) & (dz < height)
    # surfaces (static tape markers)
    surf = surface_specs(cfg, layout)
    s_pos = np.zeros((T, len(surf), 3))
    s_has = np.zeros((T, len(surf)), bool)
    s_done = np.zeros((T, len(surf)), bool)
    for j, s in enumerate(surf):
        s_pos[:, j, :2] = s['xy']
        inside = (np.abs(pos[..., :2] - s['xy']) <= s['half']).all(-1) & (pos[..., 2] < cfg['sim']['surface_height_m']) & free
        s_has[:, j] = inside.any(1)
        s_done[:, j] = (inside & done).any(1)
    holding = np.where(grasped.any(1), grasped.argmax(1), -1)
    t = np.arange(T) / cfg['fps']
    return dict(fps=float(cfg['fps']), t=t, human_pos=human, human_vis=np.ones(human.shape[:2], bool),
                obj_pos=np.concatenate([pos, s_pos], 1), obj_vis=np.ones((T, K + len(surf)), bool),
                obj_type=np.array(otype + [types['surface']] * len(surf)),
                obj_has_item=np.concatenate([has_item, s_has], 1), obj_done=np.concatenate([done, s_done], 1),
                holding=holding, obj_names=names + [s['name'] for s in surf],
                meta=dict(domain='sim', episode_index=meta['episode_index'], task=meta['list_task'],
                          task_code=meta['task_code'], role=meta['role'], scenario_id=meta['scenario_id'],
                          dart=meta['dart'], stride=stride, source_fps=source_fps))


def read_sim_dataset(dataset_root: str | Path, replay_dir: str | Path, cfg: dict, episodes=None) -> list[dict]:
    dataset_root, replay_dir = resolve(dataset_root), resolve(replay_dir)
    metas = load_meta(dataset_root)
    code_root = replay_code_root(replay_dir)
    idx = sorted(metas) if episodes is None else list(episodes)
    return [read_sim_episode(metas[i], replay_dir / f'episode_{i:06d}.npz', cfg, code_root) for i in idx]
