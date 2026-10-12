"""Write simulated sensor-view episodes (scripts/render_sensor_view.py) as a DP dataset (DeePoint, ICCV 2023), so the
authors' own `DPDataset`, `PointingModule` and loss train on it unchanged (scripts/deepoint_finetune.py).

Per frame the inputs are made by DeePoint's released inference pipeline, as at test time (scripts/deepoint_infer.py):
`MovieDataset` on the 15 fps video (OpenPifPaf shufflenetv2k16 keypoints / bbox / score) and the frozen ResNet-34
layer3 joint features of `prep/resnet_preprocess.process_single_batch` (stored as the `use_preprocessed` feature
pickles; frames without a confident person are never used by DPDataset and get none).

Pointing labels (`--label-start`, docs/requirements/INTENT_ACT_v3.2.md §4.2c):
  onset  (decision 2026-10-11, default) the whole pointing gesture from the start of the arm motion, [t_onset, t_end]:
         the labelling of online / early action detection (De Geest et al., ECCV 2016; Shou et al., ECCV 2018) over
         Kendon's gesture phases (preparation, stroke, hold), so the detector responds during the preparation;
  clear  the DP convention, measured on the DP labels + 3D keypoints (5981 pointings): a label starts when the pointing
         pose is reached (median 0.07 s after the hand has slowed below 20 % of its approach speed) and ends when the
         retraction starts (retraction half-way accelerated 0.13 s after the end), i.e. [t_clear, t_end].
t_onset / t_clear / t_end are a pointing segment's simulator frames (intent/labels.py); the positive 15 fps frames are
those whose simulator frame is in the interval. Every other frame (rest, hand offer, reach, intrusion, 'wait',
withdrawal, retraction) is a negative, as DP's non-pointing frames. The direction
target is DeePoint's own (`hand2marker_direction`): hand = wrist + 0.2 (wrist - elbow) of the 3D joints -> the target
place (intent/places.py), stored as a marker whose 4 corners sit on the target (so its 'number' point is the target).

Layout (one venue per seed fold, one session per episode, the sensor view is camera 00 of the 15 DP camera slots):
  <out>/frames/sim-f<k>/venue-info/{params.pickle, marker_corners.npz}
  <out>/featvec-result-scratch/sim-f<k>/<tag>/00/<frame+1:010d>.jpg.pickle
  <out>/labels/sim-f<k>/{timeinfo.yaml, <tag>.txt}
  <out>/keypoints/{collected_json.pickle, triangulation.pickle}

  scripts/deepoint.sh scripts/deepoint_export_dp.py --src outputs/sensor_view/2026-10-11/pilot_side --view side \\
      --out outputs/deepoint_dp/pilot_side
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from einops import rearrange

REPO = Path(__file__).resolve().parents[1] / 'third_party' / 'deepoint'
sys.path.insert(0, str(REPO / 'src'))
import dataset as dp_dataset                                   # noqa: E402
from prep.resnet_preprocess import process_single_batch        # noqa: E402

N_CAMERAS = 15            # DPDataset iterates over 15 camera slots per session
# `--classes multi` (decision 2026-10-12, docs/requirements/INTENT_ACT_v3.2.md §4.2e): the gesture classes of the intent
# contract (configs/intent_schema.yaml gestures + withdraw); class c >= 1 is the DP action-id block [100 (c-1),
# 100 (c-1) + 99] (DP: pointing = ids 0-99), the id inside the block the target marker (99: no target -> no direction)
GESTURE_CLASSES = ['none', 'point', 'reach', 'palm_up', 'palm_forward', 'withdraw']
NO_TARGET = 99


def opencv_rotation(xmat) -> np.ndarray:
    """Camera-to-world rotation with columns = image x (right), image y (down), optical axis: DP's params['R']."""
    return np.asarray(xmat, float) @ np.diag([1.0, -1.0, -1.0])


def pointing_intervals(meta: dict, idx20: np.ndarray, start: str = 'onset') -> list[tuple[int, int, int]]:
    """(marker id, start, end) in 15 fps frames, end exclusive: frames whose simulator frame is in [t_<start>, t_end]."""
    out = []
    for s in meta['segments']:
        if s['gesture'] != 'point':
            continue
        frames = np.flatnonzero((idx20 >= s[f't_{start}']) & (idx20 <= s['t_end']))
        if len(frames):
            out.append((meta['schema_targets'].index(s['target']) - 1, int(frames[0]), int(frames[-1]) + 1))
    return out


def gesture_intervals(meta: dict, idx20: np.ndarray, motions: list[dict], start: str = 'onset') -> list[tuple[int, int, int]]:
    """(action id, start, end) of every gesture segment ([t_<start>, t_end]) and every change-of-mind withdrawal
    (motion `withdraw_change`, schema retract label of the belief's `withdraw` class), 15 fps frames, end exclusive."""
    out = []
    spans = [(s['gesture'], s['target'], s[f't_{start}'], s['t_end']) for s in meta['segments']]
    spans += [('withdraw', 'none', m['start'], m['end'] if m['end'] is not None else int(idx20[-1]))
              for m in motions if m['label'] == 'withdraw_change']
    for gesture, target, t0, t1 in spans:
        c = GESTURE_CLASSES.index(gesture)
        m = meta['schema_targets'].index(target) - 1 if target in meta['schema_targets'][1:] else NO_TARGET
        frames = np.flatnonzero((idx20 >= t0) & (idx20 <= t1))
        if len(frames):
            out.append((100 * (c - 1) + m, int(frames[0]), int(frames[-1]) + 1))
    return out


def marker_corners(meta: dict, classes: str) -> dict:
    """DP marker table: every target place as a marker whose 4 corners sit on it (its 'number' point = the place),
    repeated in every gesture block that has a target (multi)."""
    names = meta['schema_targets'][1:]
    pts = np.stack([np.repeat(np.asarray(meta['targets'][k], float)[None], 4, 0) for k in names])
    if classes == 'point':
        return dict(valid_ids=np.arange(len(names)), corners=pts)
    blocks = [GESTURE_CLASSES.index(g) - 1 for g in ('point', 'reach', 'palm_up')]
    corners = np.zeros((100 * (len(GESTURE_CLASSES) - 1), 4, 3))
    ids = []
    for b in blocks:
        corners[100 * b:100 * b + len(names)] = pts
        ids += list(range(100 * b, 100 * b + len(names)))
    return dict(valid_ids=np.array(ids), corners=corners)


def export_episode(meta: dict, side: np.lib.npyio.NpzFile, video: Path, feat_dir: Path, device: str) -> list[list[dict]]:
    """Per 15 fps frame: [OpenPifPaf detection] (or []) as MovieDataset makes it; writes the joint feature pickles."""
    ds = dp_dataset.MovieDataset(str(video), 'r', 1, device)
    dets = []
    feat_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i in range(ds.num_frames):
            ds.prev_frame = i                      # sequential read (no seek)
            it = ds[i]
            det = dict(ds.predicted_joints[i])
            ok = det.get('score', 0) > 0 and any(det['keypoints'][2::3])
            dets.append([det] if ok else [])
            if ok and det['score'] >= dp_dataset.DPDataset.SCORE_THRESH:
                detection = {'keypoints': rearrange(it['keypoints'], 'bs joint xy -> joint xy bs'),
                             'bbox': rearrange(it['bbox'], 'bs xywh -> xywh bs')}
                kp_feat, _, _ = process_single_batch(it['images'], detection)
                with (feat_dir / f'{i + 1:010d}.jpg.pickle').open('wb') as f:
                    pickle.dump((kp_feat[0].numpy().astype(np.float32), None, None), f)
    return dets


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', required=True, help='scripts/render_sensor_view.py output folder')
    p.add_argument('--view', default='side')
    p.add_argument('--out', required=True)
    p.add_argument('--label-start', default='onset', choices=['onset', 'clear'])
    p.add_argument('--features-from', help='an earlier export of the same episodes: link its inputs, write labels only')
    p.add_argument('--classes', default='point', choices=['point', 'multi'],
                   help='point: pointing / not (DeePoint); multi: the gesture classes GESTURE_CLASSES')
    args = p.parse_args()
    src, out = Path(args.src), Path(args.out)
    feat = Path(args.features_from).resolve() if args.features_from else None
    if feat is not None:                  # inputs linked; venue-info (marker table of the label scheme) written here
        out.mkdir(parents=True, exist_ok=True)
        for sub in ('featvec-result-scratch', 'keypoints'):
            if not (out / sub).exists():
                (out / sub).symlink_to(feat / sub)
    dataset = Path(json.loads((src / 'episodes.json').read_text())['dataset'])
    motions = {m['episode_index']: m['human_motions'] for m in map(json.loads, (dataset / 'meta/hri_episodes.jsonl').read_text().splitlines())}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metas = [json.loads(f.read_text()) for f in sorted(src.glob('episode_*.json'))]
    collected, triang = defaultdict(dict), defaultdict(dict)
    sessions = defaultdict(list)
    kp_path = out / 'keypoints'
    if (kp_path / 'collected_json.pickle').exists():         # resume
        collected.update(pickle.load((kp_path / 'collected_json.pickle').open('rb')))
        triang.update(pickle.load((kp_path / 'triangulation.pickle').open('rb')))
    for n, meta in enumerate(metas):
        tag, venue = meta['tag'], f"sim-f{meta['fold']}"
        side = np.load(src / f'{tag}.npz')
        idx20 = side['idx20_of_15']
        vdir = out / 'frames' / venue / 'venue-info'
        if not (vdir / 'marker_corners.npz').exists():
            vdir.mkdir(parents=True, exist_ok=True)
            cam = {'R': opencv_rotation(side[f'{args.view}_xmat']), 't': np.asarray(side[f'{args.view}_pos'], float),
                   'fy': float(side[f'{args.view}_fy']), 'width': int(side['width']), 'height': int(side['height']),
                   'filepath': f'sim:{args.view}'}
            with (vdir / 'params.pickle').open('wb') as f:
                pickle.dump({f'{c:02d}': cam for c in range(N_CAMERAS)}, f)
            np.savez(vdir / 'marker_corners.npz', **marker_corners(meta, args.classes))
        sessions[venue].append(tag)
        lab = out / 'labels' / venue
        lab.mkdir(parents=True, exist_ok=True)
        intervals = (pointing_intervals(meta, idx20, args.label_start) if args.classes == 'point'
                     else gesture_intervals(meta, idx20, motions[meta['episode_index']], args.label_start))
        (lab / f'{tag}.txt').write_text(f'0, {len(idx20)}\n' + ''.join(f'{m}, {s}, {e}, r\n' for m, s, e in intervals))
        if args.features_from or tag in collected.get(venue, {}):
            continue
        dets = export_episode(meta, side, src / f'{tag}_{args.view}_15fps.mp4',
                              out / 'featvec-result-scratch' / venue / tag / '00', device)
        n15 = len(idx20)
        assert len(dets) == n15, f'{tag}: {len(dets)} video frames vs {n15} simulator frames'
        collected[venue][tag] = {c: (dets if c == 0 else [[] for _ in range(n15)]) for c in range(N_CAMERAS)}
        triang[venue][tag] = [side['coco3d'][t].T.astype(np.float64) for t in idx20]   # (3, 17) per frame
        kp_path.mkdir(parents=True, exist_ok=True)
        with (kp_path / 'collected_json.pickle').open('wb') as f:
            pickle.dump(dict(collected), f)
        with (kp_path / 'triangulation.pickle').open('wb') as f:
            pickle.dump(dict(triang), f)
        pos = sum(e - s for _, s, e in intervals)
        seen = sum(bool(d) for d in dets)
        print(f'[{n + 1}/{len(metas)}] {tag} ({venue}): {n15} frames, person {seen}/{n15}, pointing frames {pos}', flush=True)
    for venue, tags in sessions.items():
        (out / 'labels' / venue / 'timeinfo.yaml').write_text(yaml.safe_dump({'skip': 1, 'filt': 1, 'sessions': sorted(tags)}))
    (out / 'export.json').write_text(json.dumps(dict(src=str(src), view=args.view, label_start=args.label_start,
                                                     classes=args.classes,
                                                     class_names=['none', 'point'] if args.classes == 'point' else GESTURE_CLASSES,
                                                     features_from=args.features_from,
                                                     venues={k: sorted(v) for k, v in sessions.items()}), indent=1))


if __name__ == '__main__':
    main()
