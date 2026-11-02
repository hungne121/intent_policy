"""Sim (MuJoCo T1-T5) episodes for PIF: replay, PIF episodes, eval labels, statistics. Settings: configs/pif/sim.yaml.

  python -m pif.data.sim replay --code-commit b5d0c8e     per-frame object positions, grasp state, 11 keypoints
  python -m pif.data.sim stats                            numbers and figures of the PIF data (docs/reports/PIF_P1)

Replay: the dataset (scripts/collect_demos.py) has no per-frame object positions, no "which object the robot holds"
and only 6 keypoints. Each episode is re-run headless from what the dataset saved (experiment config, controller,
seed, spec, DART flag) and compared with it (frames, state, keypoints, holding). The sim code changed after the
collection, so `--code-commit` runs the code of that time (extracted with `git archive` to outputs/pif/code/<commit>).
PIF episode: 10 Hz (every `source_fps / fps`-th frame, never interpolated), table frame. Object tokens = the objects on
the table (box -> object, cup -> container) + the visible tapes of `surfaces` (surface). has_item: something on the
tape / in the cup; done: grasped then released by the robot (a surface: holds a done object); holding: grasped object.
Eval labels: target keys mapped to object tokens, or TARGET_HUMAN (H1/H2) / TARGET_REGION (T4 zones) / TARGET_NONE.
"""
from __future__ import annotations
import argparse
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pif.utils import log_run, resolve, run_dir
from pif.config import load_config

DATASET_KEYPOINTS = ['r_wrist', 'r_index_base', 'r_index_tip', 'l_wrist', 'l_index_base', 'l_index_tip']
KEYPOINTS = DATASET_KEYPOINTS + ['r_shoulder', 'r_elbow', 'l_shoulder', 'l_elbow', 'head']     # J = 11
TARGET_NONE, TARGET_HUMAN, TARGET_REGION = -1, -2, -3


# ------------------------------------------------------------------ replay
def pose_keypoints(scenario) -> np.ndarray:
    """[11, 3] world keypoints of the mannequin as posed: shoulder / elbow = upper-arm / forearm segment pivots,
    head = the head proxy centre (HumanBody.pose)."""
    pose = scenario.human_pose
    seg = pose['segments']
    extra = [seg['r_upperarm'][0], seg['r_forearm'][0], seg['l_upperarm'][0], seg['l_forearm'][0], pose['head']]
    return np.vstack([scenario.human_keypoints(DATASET_KEYPOINTS), np.asarray(extra, float)])


def _use_code(code_root):
    if code_root:
        for name in [m for m in sys.modules if m == 'intent_policy' or m.startswith('intent_policy.')]:
            del sys.modules[name]
        sys.path.insert(0, code_root)


def _replay_one(job: dict) -> dict:
    from intent_policy.benchmark.runner import ExpertAgent, NoisyExpertAgent, ObservationConfig, run_episode
    from intent_policy.scenarios.scenario_registry import make_scenario, scenario_overrides
    from intent_policy.sim.restricted_action import RestrictedActionConfig
    meta, exp = job['meta'], job['exp']
    sc = make_scenario(meta['scenario_id'], scenario_overrides(exp, meta['scenario_id']))
    noise = exp['data'].get('noise') or {}
    agent = NoisyExpertAgent(noise['burst_prob'], noise['burst_ticks']) if meta['dart'] else ExpertAgent()
    rows = defaultdict(list)

    def on_frame(obs, decision, target, scenario):
        rows['state'].append(np.asarray(obs['observation.state'], np.float32))
        rows['kp'].append(pose_keypoints(scenario))
        rows['holding'].append(int(any(scenario.grasp_state.values())))
        rows['obj_pos'].append(np.array([scenario.pos(k) for k in scenario.on_table], float))
        rows['obj_grasped'].append(np.array([scenario.grasp_state[k] for k in scenario.on_table], bool))

    rec = run_episode(sc, agent, int(meta['seed']), RestrictedActionConfig(**job['controller']), ObservationConfig(cameras=()),
                      on_frame=on_frame, keep_trace=False, spec=meta['spec'])
    names = list(sc.on_table)
    sc.close()
    ref, kp = job['ref'], np.stack(rows['kp']).astype(np.float32)
    n = min(len(kp), len(ref['state']))
    dev = dict(state=float(np.abs(np.stack(rows['state'])[:n] - ref['state'][:n]).max()),
               keypoints=float(np.abs(kp[:n, :6] - ref['keypoints'][:n]).max()),
               holding_mismatch=int((np.array(rows['holding'])[:n] != ref['holding'][:n]).sum()))
    ok = bool(len(kp) == len(ref['state']) and rec['success'] and dev['state'] <= job['atol'] and dev['keypoints'] <= job['atol']
              and dev['holding_mismatch'] == 0)
    np.savez_compressed(job['out'], obj_names=np.array(names), obj_pos=np.stack(rows['obj_pos']).astype(np.float32),
                        obj_grasped=np.stack(rows['obj_grasped']), keypoints=kp, keypoint_names=np.array(KEYPOINTS), reproduced=ok)
    return dict(episode_index=meta['episode_index'], scenario_id=meta['scenario_id'], frames=len(kp), dataset_frames=len(ref['state']),
                success=bool(rec['success']), reproduced=ok, **dev)


def replay(cfg: dict, code_commit: str | None, episodes, workers: int, atol: float, overwrite: bool) -> None:
    root, out = resolve(cfg['dataset']), resolve(cfg['replay'])
    out.mkdir(parents=True, exist_ok=True)
    metas = [m for m in map(json.loads, open(root / 'meta/hri_episodes.jsonl')) if episodes is None or m['episode_index'] in episodes]
    exp = json.loads((root / 'meta/hri_experiment_config.json').read_text())
    ctrl = json.loads((root / 'meta/hri_restricted_actions.json').read_text())['controller']
    cols = ['episode_index', 'frame_index', 'observation.state', 'human.keypoints', 'hri.holding_gt']
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted((root / 'data').glob('*/*.parquet'))]).sort_values(cols[:2])
    jobs = []
    for m in metas:
        dst = out / f"episode_{m['episode_index']:06d}.npz"
        if dst.exists() and not overwrite:
            continue
        e = df[df.episode_index == m['episode_index']]
        ref = dict(state=np.stack(e['observation.state'].to_numpy()), holding=np.hstack(e['hri.holding_gt'].to_numpy()).astype(int),
                   keypoints=np.stack([np.stack(k) for k in e['human.keypoints'].to_numpy()]))
        jobs.append(dict(meta=m, exp=exp, controller=ctrl, ref=ref, out=str(dst), atol=atol))
    code_root = None
    if code_commit:
        code_root = resolve(f'outputs/pif/code/{code_commit}')
        if not (code_root / 'intent_policy').exists():
            code_root.mkdir(parents=True, exist_ok=True)
            arc = subprocess.run(['git', 'archive', code_commit], cwd=resolve('.'), check=True, capture_output=True).stdout
            subprocess.run(['tar', '-x', '-C', str(code_root)], input=arc, check=True)
        code_root = str(code_root)
    summary_path = out / 'replay_summary.json'
    results = json.loads(summary_path.read_text())['episodes'] if summary_path.exists() else {}
    t0 = time.time()
    with get_context('spawn').Pool(workers, initializer=_use_code, initargs=(code_root,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_replay_one, jobs)):
            results[str(r['episode_index'])] = r
            print(f"[{i + 1}/{len(jobs)}] ep {r['episode_index']} {r['scenario_id']} dev state {r['state']:.2g} kp {r['keypoints']:.2g} "
                  f"holding {r['holding_mismatch']}{'' if r['reproduced'] else '  NOT REPRODUCED'} [{time.time() - t0:.0f}s]", flush=True)
    summary_path.write_text(json.dumps(dict(dataset=str(root), atol=atol, code_commit=code_commit, keypoints=KEYPOINTS,
                                            episodes=results), indent=1))


# ------------------------------------------------------------------ PIF episodes
@lru_cache(maxsize=4)
def _code_configs(code_root: str) -> tuple[dict, dict]:
    root = Path(code_root)
    return (yaml.safe_load((root / 'configs/layout.yaml').read_text()),
            yaml.safe_load((root / 'configs/scene/common.yaml').read_text()))


def load_meta(cfg: dict) -> dict[int, dict]:
    return {m['episode_index']: m for m in map(json.loads, open(resolve(cfg['dataset']) / 'meta/hri_episodes.jsonl'))}


def read_episode(meta: dict, cfg: dict, fps: float, keypoints: list[str], object_types: list[str]) -> dict:
    replay_dir = resolve(cfg['replay'])
    summary = json.loads((replay_dir / 'replay_summary.json').read_text())
    code_root = str(resolve(f"outputs/pif/code/{summary['code_commit']}") if summary.get('code_commit') else resolve('.'))
    r = np.load(replay_dir / f"episode_{meta['episode_index']:06d}.npz")
    if not bool(r['reproduced']):
        c = summary['episodes'][str(meta['episode_index'])]
        if not (c['frames'] == c['dataset_frames'] and c['holding_mismatch'] == 0 and c['keypoints'] < 1e-5
                and c['state'] <= cfg['replay_state_tol']):
            raise ValueError(f"episode {meta['episode_index']} did not reproduce: {c}")
    stride = int(round(cfg['source_fps'] / fps))
    layout, scene = _code_configs(code_root)
    types = {n: i for i, n in enumerate(object_types)}
    names = [str(n) for n in r['obj_names']]
    kp_names = [str(n) for n in r['keypoint_names']]
    sel = slice(0, None, stride)
    human = r['keypoints'][sel][:, [kp_names.index(k) for k in keypoints]].astype(np.float64)
    human[..., 2] -= cfg['table_z']
    pos = r['obj_pos'][sel].astype(np.float64)
    pos[..., 2] -= cfg['table_z']
    grasped = r['obj_grasped'][sel]
    T, K = grasped.shape
    catalog = scene['objects_catalog']
    otype = [types[cfg['shape_type'][catalog[n]['shape']]] for n in names]
    g = r['obj_grasped']
    released = np.zeros_like(g)
    released[1:] = g[:-1] & ~g[1:]
    done = np.maximum.accumulate(released, axis=0)[sel]          # on the source frames: a short grasp is not missed
    free = ~grasped
    has_item = np.zeros((T, K), bool)
    for c in [k for k in range(K) if otype[k] == types['container']]:
        radius, height = catalog[names[c]]['size']
        for k in range(K):
            if k != c:
                dz = pos[:, k, 2] - pos[:, c, 2]
                has_item[:, c] |= free[:, k] & (np.linalg.norm(pos[:, k, :2] - pos[:, c, :2], axis=-1) < radius) & (np.abs(dz) < height)
    S = len(cfg['surfaces'])
    s_pos, s_has, s_done = np.zeros((T, S, 3)), np.zeros((T, S), bool), np.zeros((T, S), bool)
    for j, name in enumerate(cfg['surfaces']):
        z = layout['zones'][name]
        half = np.broadcast_to(np.asarray(z.get('size', layout['marker_size']), float), 2) / 2
        s_pos[:, j, :2] = z['xy']
        inside = (np.abs(pos[..., :2] - z['xy']) <= half).all(-1) & (pos[..., 2] < cfg['surface_height_m']) & free
        s_has[:, j], s_done[:, j] = inside.any(1), (inside & done).any(1)
    return dict(fps=float(fps), t=np.arange(T) / fps, human_pos=human, human_vis=np.ones(human.shape[:2], bool),
                obj_pos=np.concatenate([pos, s_pos], 1), obj_vis=np.ones((T, K + S), bool),
                obj_type=np.array(otype + [types['surface']] * S), obj_has_item=np.concatenate([has_item, s_has], 1),
                obj_done=np.concatenate([done, s_done], 1), holding=np.where(grasped.any(1), grasped.argmax(1), -1),
                obj_names=names + list(cfg['surfaces']),
                meta=dict(domain='sim', episode_index=meta['episode_index'], task=meta['list_task'], task_code=meta['task_code'],
                          role=meta['role'], scenario_id=meta['scenario_id'], dart=meta['dart'], stride=stride))


def key_to_target(key, names: list[str], cfg: dict) -> int:
    if key is None:
        return TARGET_NONE
    key = cfg['surface_alias'].get(key, key)
    if key in names:
        return names.index(key)
    if key in cfg['human_targets']:
        return TARGET_HUMAN
    if key in cfg['region_targets']:
        return TARGET_REGION
    raise KeyError(f'target {key!r} has no token or code')


def labels(meta: dict, ep: dict, cfg: dict) -> dict:
    """gestures [{label, key, target, onset, clear}], intent_object / intent_dest per frame, protocol steps (PIF frames:
    an event at source frame f is at the first PIF frame i with i * stride >= f)."""
    stride, T, names = ep['meta']['stride'], len(ep['holding']), ep['obj_names']
    at = lambda f: None if f is None else min(math.ceil(f / stride), T - 1)
    gestures = [dict(label=m['label'], key=m['key'], target=key_to_target(m['key'], names, cfg), onset=at(m['start']),
                     clear=at(m['end'])) for m in meta['human_motions']]
    obj, dest = np.full(T, TARGET_NONE), np.full(T, TARGET_NONE)
    for s in meta['human_target_stages']:
        o, _, d = s['name'].partition(' -> ')
        a, b = at(s['start']), at(s['end'])
        obj[a:b + 1] = key_to_target(o if o != 'none' else None, names, cfg)
        dest[a:b + 1] = key_to_target(d or None, names, cfg)
    return dict(gestures=gestures, intent_object=obj, intent_dest=dest, task=meta['list_task'], role=meta['role'],
                steps=[dict(step=s['step'], frame=at(s['frame']), by=s['by']) for s in meta['protocol_steps']])


def read_dataset(full: dict, tasks=None) -> tuple[list[dict], list[dict]]:
    cfg, metas = full['sim'], load_meta(full['sim'])
    eps = [read_episode(m, cfg, full['fps'], full['tokens']['keypoints'], full['tokens']['object_types'])
           for i, m in sorted(metas.items()) if tasks is None or m['list_task'] in tasks]
    return eps, [labels(metas[e['meta']['episode_index']], e, cfg) for e in eps]


def stats(full: dict, out: Path) -> dict:
    eps, labs = read_dataset(full)
    rows = defaultdict(lambda: defaultdict(int))
    for e, l in zip(eps, labs):
        r = rows[e['meta']['task']]
        r['episodes'] += 1
        r['frames_10hz'] += len(e['holding'])
        for g in l['gestures']:
            if g['target'] != TARGET_NONE:
                kind = full['tokens']['object_types'][e['obj_type'][g['target']]] if g['target'] >= 0 else \
                    {TARGET_HUMAN: 'human', TARGET_REGION: 'region'}[g['target']]
                r[f'target_{kind}'] += 1
    res = dict(episodes=len(eps), K=[int(min(len(e['obj_type']) for e in eps)), int(max(len(e['obj_type']) for e in eps))],
               per_task={k: dict(v) for k, v in sorted(rows.items())})
    out.mkdir(parents=True, exist_ok=True)
    (out / 'sim_stats.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['replay', 'stats'])
    p.add_argument('--config', default='configs/pif/pif.yaml')
    p.add_argument('--code-commit')
    p.add_argument('--episodes', nargs='*', type=int)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--atol', type=float, default=1e-5)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--out', default='docs/reports/PIF_P1')
    args = p.parse_args()
    full = load_config(args.config)
    if args.command == 'replay':
        log_run(run_dir('pif_replay', 'sim', None), 'pif.data.sim')
        replay(full['sim'], args.code_commit, set(args.episodes) if args.episodes else None, args.workers, args.atol, args.overwrite)
    else:
        stats(full, resolve(args.out))


if __name__ == '__main__':
    main()
