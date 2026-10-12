"""Re-run recorded dataset episodes with an extra world camera that sees the whole human ("sensor" view), as input for
a real perception model (DeePoint: scripts/deepoint_infer.py, scripts/deepoint_export_dp.py).

The episode is the dataset's own: same scenario, seed, spec and experiment config (meta/hri_experiment_config.json),
scripted expert; the extra camera is only added to the scene (`scene.cameras` override), it changes nothing else
(the replayed keypoints are compared with the dataset's: `max_keypoint_diff_m`, 0 = the same episode).

Per episode and view:
  <tag>_<view>.mp4        every control frame (20 fps)
  <tag>_<view>_15fps.mp4  DeePoint's rate (it was trained at 15 fps): 15 fps frame k = 20 fps frame round(k * 4 / 3)
                          (`idx20_of_15` in the sidecar), so every 15 fps frame has an exact simulator frame
Per episode: <tag>.npz (schema keypoints, COCO-17 3D joints of the mannequin - nose, shoulders, elbows, wrists, the
rest NaN -, camera poses and intrinsics, idx20_of_15) and <tag>.json (dataset meta needed for labels: spec, task, fold,
ground-truth gesture segments of intent/labels.segments_from_sim, target positions of intent/places.py).

Views (look-at, world frame; the human stands at (0.30, -0.48) facing +Y, configs/scene/common.yaml):
  front    across the table from the human, further back than the policy camera `high` (whole upper body, head-on)
  side     from the human's right (+X), the pointing arm towards the table in profile (the whole body is visible)
  oblique  between the two, from the robot's side
`--probe` renders one frame per view (at --probe-time s) to pick a view. Episodes: `--episodes` or `--per-task N`
(N non-DART episodes per list task, taken round-robin over the 5 seed folds of intent/folds.fold_of).

  ./run.sh -m scripts.render_sensor_view --dataset-root outputs/datasets/intent_act_late_v5 --episodes 0 --probe
  ./run.sh -m scripts.render_sensor_view --dataset-root outputs/datasets/intent_act_late_v5 --per-task 17 --views side
"""
import argparse
import json
from collections import defaultdict

import av
import numpy as np
from PIL import Image
from intent_policy.benchmark.runner import ExpertAgent, run_episode
from intent_policy.intent.folds import fold_of
from intent_policy.intent.labels import load_schema, segments_from_sim
from intent_policy.intent.places import target_positions
from intent_policy.scenarios.config import _deep_merge
from intent_policy.scenarios.scenario_registry import make_scenario, scenario_overrides
from intent_policy.utils import controller_config, observation_config, resolve, run_dir

VIEWS = {
    'front': {'pos': [0.30, 1.90, 1.45], 'target': [0.30, -0.48, 0.85], 'fovy': 50},
    'side': {'pos': [2.30, -0.48, 1.25], 'target': [0.30, -0.48, 0.85], 'fovy': 50},
    'oblique': {'pos': [1.80, 1.20, 1.50], 'target': [0.30, -0.48, 0.85], 'fovy': 50},
}
FPS, FPS_DP, FOLDS = 20, 15, 5
COCO = {'nose': 0, 'l_shoulder': 5, 'r_shoulder': 6, 'l_elbow': 7, 'r_elbow': 8, 'l_wrist': 9, 'r_wrist': 10}


def camera_pose(sc, name: str) -> tuple[np.ndarray, np.ndarray]:
    """World position and rotation (MuJoCo camera frame: x right, y up, looking along -z) of camera `name`."""
    cid = sc.env.model.camera(name).id
    return sc.env.data.cam_xpos[cid].copy(), sc.env.data.cam_xmat[cid].reshape(3, 3).copy()


def coco_joints(pose: dict) -> np.ndarray:
    """(17, 3) COCO-ordered 3D joints of the mannequin pose: segment pivots (upper arm = shoulder, forearm = elbow,
    hand = wrist) and the face point as the nose; joints the mannequin has no point for are NaN."""
    out = np.full((17, 3), np.nan)
    seg = pose['segments']
    for s in ('l', 'r'):
        out[COCO[f'{s}_shoulder']] = seg[f'{s}_upperarm'][0]
        out[COCO[f'{s}_elbow']] = seg[f'{s}_forearm'][0]
        out[COCO[f'{s}_wrist']] = seg[f'{s}_hand'][0]
    out[COCO['nose']] = pose['face']
    return out


def select(metas: list[dict], per_task: int) -> list[int]:
    """`per_task` non-DART episodes per list task, round-robin over the seed folds (balanced folds)."""
    by = defaultdict(lambda: defaultdict(list))
    for m in metas:
        if not (m.get('dart') or m.get('noise_frames')):
            by[m.get('list_task')][fold_of(int(m['seed']), FOLDS)].append(m['episode_index'])
    out = []
    for task, folds in sorted(by.items()):
        picked, i = [], 0
        while len(picked) < per_task and any(len(v) > i for v in folds.values()):
            picked += [folds[f][i] for f in sorted(folds) if len(folds[f]) > i][:per_task - len(picked)]
            i += 1
        out += picked
    return sorted(out)


class Writer:
    def __init__(self, path, W, H, fps):
        self.c = av.open(str(path), 'w')
        self.s = self.c.add_stream('libx264', rate=fps)
        self.s.width, self.s.height, self.s.pix_fmt, self.s.options = W, H, 'yuv420p', {'crf': '18', 'preset': 'veryfast'}

    def add(self, img):
        for pkt in self.s.encode(av.VideoFrame.from_ndarray(img, format='rgb24')):
            self.c.mux(pkt)

    def close(self):
        for pkt in self.s.encode():
            self.c.mux(pkt)
        self.c.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--episodes', type=int, nargs='+')
    p.add_argument('--per-task', type=int, help='instead of --episodes: N non-DART episodes per list task')
    p.add_argument('--views', nargs='+', default=list(VIEWS), choices=list(VIEWS))
    p.add_argument('--size', default='1280x720', help='WxH of the sensor camera')
    p.add_argument('--probe', action='store_true', help='one PNG per view at --probe-time instead of videos')
    p.add_argument('--probe-time', type=float, default=2.5)
    p.add_argument('--skip-existing', action='store_true')
    p.add_argument('--out', help='default: outputs/sensor_view/<date>/<time>_<dataset>')
    args = p.parse_args()
    root = resolve(args.dataset_root)
    exp = json.loads((root / 'meta/hri_experiment_config.json').read_text())
    metas = [json.loads(line) for line in (root / 'meta/hri_episodes.jsonl').read_text().splitlines()]
    schema = load_schema(exp['intent']['schema'])
    targets = {k: v.round(4).tolist() for k, v in target_positions(schema).items()}
    names = list(schema['keypoints'])
    W, H = map(int, args.size.split('x'))
    out = run_dir('sensor_view', root.name, args.out)
    out.mkdir(parents=True, exist_ok=True)
    obs_cfg, ctrl_cfg = observation_config(exp), controller_config(exp)
    from scripts.build_intent_labels import load_episodes
    recorded = load_episodes(root)
    cams = {v: VIEWS[v] for v in args.views}
    episodes = select(metas, args.per_task) if args.per_task else args.episodes
    (out / 'episodes.json').write_text(json.dumps(dict(dataset=str(root), episodes=episodes, views=args.views), indent=1))
    for n, ep in enumerate(episodes):
        m = metas[ep]
        if m.get('dart') or m.get('noise_frames'):
            print(f'episode {ep}: DART episode (random perturbations), skipped')
            continue
        sid, seed = m['scenario_id'], int(m['seed'])
        tag = f"episode_{ep:03d}_{m.get('list_task', sid)}_seed{seed}"
        if args.skip_existing and (out / f'{tag}.json').exists():
            continue
        sc = make_scenario(sid, _deep_merge(scenario_overrides(exp, sid), {'scene': {'cameras': cams}}))
        writers, kp, coco, poses, idx15 = {}, [], [], {}, []
        probe_frame = int(round(args.probe_time * FPS))

        def on_frame(obs, decision, target, scenario):
            t = len(kp)
            kp.append(scenario.human_keypoints(names).astype(np.float32))
            coco.append(coco_joints(scenario.human_pose).astype(np.float32))
            take15 = round(len(idx15) * FPS / FPS_DP) == t
            if take15:
                idx15.append(t)
            for v in cams:
                if v not in poses:
                    poses[v] = camera_pose(scenario, v)
                if args.probe:
                    if t == probe_frame:
                        Image.fromarray(scenario.env.render(v, W, H)).save(out / f'{tag}_{v}_probe.png')
                    continue
                if v not in writers:
                    writers[v] = (Writer(out / f'{tag}_{v}.mp4', W, H, FPS), Writer(out / f'{tag}_{v}_15fps.mp4', W, H, FPS_DP))
                img = scenario.env.render(v, W, H)
                writers[v][0].add(img)
                if take15:
                    writers[v][1].add(img)

        rec = run_episode(sc, ExpertAgent(), seed, ctrl_cfg, obs_cfg, on_frame=on_frame, keep_trace=False, spec=m['spec'])
        for w20, w15 in writers.values():
            w20.close()
            w15.close()
        sc.close()
        kp = np.stack(kp)
        ref = recorded[ep]['keypoints']
        k = min(len(kp), len(ref))
        diff = float(np.abs(kp[:k] - ref[:k]).max()) if k else float('nan')
        np.savez_compressed(out / f'{tag}.npz', keypoints=kp, keypoint_names=np.array(names), coco3d=np.stack(coco),
                            idx20_of_15=np.array(idx15), fps=FPS, width=W, height=H,
                            **{f'{v}_pos': poses[v][0] for v in cams}, **{f'{v}_xmat': poses[v][1] for v in cams},
                            **{f'{v}_fy': H / 2 / np.tan(np.radians(cams[v]['fovy']) / 2) for v in cams},
                            episode_index=ep, dataset_frames=len(ref), max_keypoint_diff_m=diff)
        segs = segments_from_sim(m['human_motions'], m['human_stages'], m['spec'], schema, len(kp))
        (out / f'{tag}.json').write_text(json.dumps(dict(
            tag=tag, episode_index=ep, seed=seed, fold=fold_of(seed, FOLDS), list_task=m.get('list_task'),
            task_code=m['task_code'], spec=m['spec'], segments=segs, targets=targets, schema_targets=schema['targets'],
            object_desc=schema['instruction']['object_desc'], views=args.views, n_frames=len(kp), n_frames_15=len(idx15),
            max_keypoint_diff_m=diff, success=rec['success']), indent=1, default=str))
        print(f"[{n + 1}/{len(episodes)}] episode {ep} ({sid}, seed {seed}, fold {fold_of(seed, FOLDS)}): {len(kp)} frames "
              f"(dataset {len(ref)}), keypoint diff {diff:.1e} m", flush=True)


if __name__ == '__main__':
    main()
