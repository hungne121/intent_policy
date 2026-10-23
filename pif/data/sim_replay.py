"""Replay the episodes of a sim dataset and store the per-frame fields the PIF tokens need (PIF_P0 §4).

The dataset (scripts/collect_demos.py) has no per-frame object positions, no "which object the robot holds" and only
6 human keypoints. Each kept episode is re-run headless (no cameras) from what the dataset saved: the experiment
config (meta/hri_experiment_config.json), the controller (meta/hri_restricted_actions.json), the list entry (seed,
spec, DART flag). On every recorded frame (same callback point as collect_demos) it stores

  obj_pos      [T, K, 3]  world position of every object on the table (scenario.on_table order = obj_names)
  obj_grasped  [T, K]     the simulator's grasp state of each object
  keypoints    [T, 11, 3] world keypoints KEYPOINTS: the 6 of the dataset + shoulders, elbows, head (from the posed
                          mannequin, scenario.human_pose; nothing in the sim code is changed)

and checks the re-run against the dataset: frame count, observation.state, human.keypoints, hri.holding_gt. An
episode whose deviation exceeds `--atol` is reported and flagged (`reproduced = False`); its file is still written.

The sim code has changed since a dataset was collected, so the replay must run the code of that time:
`--code-commit` extracts that commit (`git archive`) to outputs/pif/code/<commit> and the workers import
`intent_policy` from there. intent_act_late_v5 (collected 2026-10-03 16:04) reproduces exactly with b5d0c8e, the
first commit after the collection; with the current code it does not (frame counts and states differ).
Output: <out>/episode_XXXXXX.npz + <out>/replay_summary.json, run log in outputs/pif_replay/<date>/<time>_<name>.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from intent_policy.utils import log_run, resolve, run_dir

DATASET_KEYPOINTS = ['r_wrist', 'r_index_base', 'r_index_tip', 'l_wrist', 'l_index_base', 'l_index_tip']
EXTRA_KEYPOINTS = ['r_shoulder', 'r_elbow', 'l_shoulder', 'l_elbow', 'head']
KEYPOINTS = DATASET_KEYPOINTS + EXTRA_KEYPOINTS           # J = 11 (CORRECTIONS #17)


def pose_keypoints(scenario) -> np.ndarray:
    """[11, 3] world keypoints of the mannequin as posed this frame. Shoulder / elbow = pivots of the upper-arm /
    forearm segments, head = the head proxy centre (6 cm above the Head joint), as HumanBody.pose() returns them."""
    pose = scenario.human_pose
    seg = pose['segments']
    extra = [seg['r_upperarm'][0], seg['r_forearm'][0], seg['l_upperarm'][0], seg['l_forearm'][0], pose['head']]
    return np.vstack([scenario.human_keypoints(DATASET_KEYPOINTS), np.asarray(extra, float)])


def dataset_frames(root: Path) -> pd.DataFrame:
    cols = ['episode_index', 'frame_index', 'observation.state', 'human.keypoints', 'hri.holding_gt']
    return pd.concat([pd.read_parquet(f, columns=cols) for f in sorted((root / 'data').glob('*/*.parquet'))],
                     ignore_index=True).sort_values(['episode_index', 'frame_index'])


def use_code(code_root: str | None) -> None:
    """Pool initializer: import `intent_policy` from the extracted commit instead of the working tree."""
    if code_root:
        for name in [m for m in sys.modules if m == 'intent_policy' or m.startswith('intent_policy.')]:
            del sys.modules[name]
        sys.path.insert(0, code_root)


def extract_commit(commit: str) -> Path:
    dst = resolve(f'outputs/pif/code/{commit}')
    if not (dst / 'intent_policy').exists():
        dst.mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(['git', 'archive', commit], cwd=resolve('.'), check=True, capture_output=True).stdout
        subprocess.run(['tar', '-x', '-C', str(dst)], input=archive, check=True)
    return dst


def replay_one(job: dict) -> dict:
    from intent_policy.benchmark.runner import ExpertAgent, NoisyExpertAgent, ObservationConfig, run_episode
    from intent_policy.scenarios.scenario_registry import make_scenario, scenario_overrides
    from intent_policy.sim.restricted_action import RestrictedActionConfig

    meta, exp, ctrl = job['meta'], job['exp'], job['controller']
    t0 = time.time()
    sc = make_scenario(meta['scenario_id'], scenario_overrides(exp, meta['scenario_id']))
    noise = exp['data'].get('noise') or {}
    agent = NoisyExpertAgent(noise['burst_prob'], noise['burst_ticks']) if meta['dart'] else ExpertAgent()
    rows = dict(state=[], kp=[], holding=[], obj_pos=[], obj_grasped=[])

    def on_frame(obs, decision, target, scenario):
        rows['state'].append(np.asarray(obs['observation.state'], np.float32))
        rows['kp'].append(pose_keypoints(scenario))
        rows['holding'].append(int(any(scenario.grasp_state.values())))
        rows['obj_pos'].append(np.array([scenario.pos(k) for k in scenario.on_table], float))
        rows['obj_grasped'].append(np.array([scenario.grasp_state[k] for k in scenario.on_table], bool))

    rec = run_episode(sc, agent, int(meta['seed']), RestrictedActionConfig(**ctrl), ObservationConfig(cameras=()),
                      on_frame=on_frame, keep_trace=False, spec=meta['spec'])
    names = list(sc.on_table)
    sc.close()
    ref = job['ref']
    kp = np.stack(rows['kp']).astype(np.float32)
    n_ok = len(kp) == len(ref['state'])
    n = min(len(kp), len(ref['state']))
    dev = dict(state=float(np.abs(np.stack(rows['state'])[:n] - ref['state'][:n]).max()),
               keypoints=float(np.abs(kp[:n, :6] - ref['keypoints'][:n]).max()),
               holding_mismatch=int((np.array(rows['holding'])[:n] != ref['holding'][:n]).sum()))
    reproduced = bool(n_ok and rec['success'] and dev['state'] <= job['atol'] and dev['keypoints'] <= job['atol']
                      and dev['holding_mismatch'] == 0)
    np.savez_compressed(job['out'], obj_names=np.array(names), obj_pos=np.stack(rows['obj_pos']).astype(np.float32),
                        obj_grasped=np.stack(rows['obj_grasped']), keypoints=kp, keypoint_names=np.array(KEYPOINTS),
                        reproduced=reproduced)
    return dict(episode_index=meta['episode_index'], scenario_id=meta['scenario_id'], frames=len(kp),
                dataset_frames=len(ref['state']), success=bool(rec['success']), reproduced=reproduced,
                wall_s=round(time.time() - t0, 1), **dev)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='outputs/datasets/intent_act_late_v5')
    p.add_argument('--out', help='default: outputs/pif/sim_replay/<dataset name>')
    p.add_argument('--episodes', nargs='*', type=int, help='subset of episode indices (default: all)')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--atol', type=float, default=1e-5)
    p.add_argument('--overwrite', action='store_true', help='re-run episodes whose file exists')
    p.add_argument('--code-commit', help='replay with the sim code of this commit (default: the working tree)')
    p.add_argument('--log-dir')
    args = p.parse_args()
    root = resolve(args.dataset)
    out = resolve(args.out or f'outputs/pif/sim_replay/{root.name}')
    out.mkdir(parents=True, exist_ok=True)
    log_run(run_dir('pif_replay', root.name, args.log_dir), 'pif.data.sim_replay')
    metas = [json.loads(line) for line in open(root / 'meta/hri_episodes.jsonl')]
    if args.episodes is not None:
        metas = [m for m in metas if m['episode_index'] in set(args.episodes)]
    exp = json.loads((root / 'meta/hri_experiment_config.json').read_text())
    ctrl = json.loads((root / 'meta/hri_restricted_actions.json').read_text())['controller']
    df = dataset_frames(root)
    jobs = []
    for m in metas:
        dst = out / f"episode_{m['episode_index']:06d}.npz"
        if dst.exists() and not args.overwrite:
            continue
        e = df[df.episode_index == m['episode_index']]
        ref = dict(state=np.stack(e['observation.state'].to_numpy()), holding=np.hstack(e['hri.holding_gt'].to_numpy()).astype(int),
                   keypoints=np.stack([np.stack(k) for k in e['human.keypoints'].to_numpy()]))
        jobs.append(dict(meta=m, exp=exp, controller=ctrl, ref=ref, out=str(dst), atol=args.atol))
    code_root = str(extract_commit(args.code_commit)) if args.code_commit else None
    print(f'sim code: {code_root or "working tree"}', flush=True)
    print(f'{len(jobs)} episodes to replay ({len(metas) - len(jobs)} already done) -> {out}', flush=True)
    summary_path = out / 'replay_summary.json'
    results = json.loads(summary_path.read_text())['episodes'] if summary_path.exists() else {}
    t0 = time.time()
    with get_context('spawn').Pool(args.workers, initializer=use_code, initargs=(code_root,)) as pool:
        for i, r in enumerate(pool.imap_unordered(replay_one, jobs)):
            results[str(r['episode_index'])] = r
            flag = '' if r['reproduced'] else '  NOT REPRODUCED'
            print(f"[{i + 1}/{len(jobs)}] ep {r['episode_index']} {r['scenario_id']} frames {r['frames']}/{r['dataset_frames']} "
                  f"dev state {r['state']:.2g} kp {r['keypoints']:.2g} holding {r['holding_mismatch']}{flag} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            if (i + 1) % 20 == 0 or i + 1 == len(jobs):
                summary_path.write_text(json.dumps(dict(dataset=str(root), atol=args.atol, code_commit=args.code_commit, keypoints=KEYPOINTS,
                                                        episodes=results), indent=1))
    bad = [k for k, r in results.items() if not r['reproduced']]
    print(f'replayed {len(results)} episodes, not reproduced: {len(bad)} {bad[:20]}')


if __name__ == '__main__':
    main()
