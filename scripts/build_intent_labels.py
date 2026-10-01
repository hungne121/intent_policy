"""Add per-timestep intent labels to a recorded LeRobotDataset, in place (docs/requirements/INTENT_ACT_GUIDE.md §1.3-1.4).

New per-frame features (the same episodes then serve both the no-intent and the intent policies):
  intent.obj    [1]        int64    What-target (configs/intent_schema.yaml obj_vocab; 0 = none)
  intent.act    [1]        int64    What-action of the human (act_vocab; 0 = none)
  intent.phase  [1]        float32  When: progress of the current gesture in [0, 1]
  intent.tte    [1]        float32  When: time to its event [s], capped at tte_max_s
  intent.xi     [M, J, D]  float32  How: future keypoint waypoints relative to the current keypoints (unnormalised)
plus meta/intent_segments.json ({episode_index: segments}) and meta/intent_schema.json (the schema used).

Segments: `--segments auto` derives them from simulator ground truth (meta/hri_episodes.jsonl: human motions and
stages, scripts/collect_demos.py); `--segments <file.json>` reads {episode_id: [{t_onset, t_clear, t_event, obj,
act[, t_end]}]} in frame indices, e.g. exported from ELAN / CVAT / Label Studio.
Keypoints: the recorded `human.keypoints` (simulator ground truth). Datasets without them (real cameras) would need
a hand-keypoint detector (MediaPipe Hands, `--kp-camera`) and depth lifting: not implemented, TODO(user).

Example:
  ./run.sh -m scripts.build_intent_labels --dataset-root outputs/datasets/intent_late --segments auto
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import datasets
import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.feature_utils import get_hf_features_from_features

from intent_policy.intent.labels import SCHEMA_PATH, episode_labels, load_schema, segments_from_sim
from intent_policy.utils import resolve

INTENT_FEATURES = ('intent.obj', 'intent.act', 'intent.phase', 'intent.tte', 'intent.xi')


def intent_feature_specs(schema: dict) -> dict:
    M, J, D = schema['n_waypoints'], len(schema['keypoints']), schema['keypoint_dim']
    return {'intent.obj': {'dtype': 'int64', 'shape': (1,), 'names': ['obj']},
            'intent.act': {'dtype': 'int64', 'shape': (1,), 'names': ['act']},
            'intent.phase': {'dtype': 'float32', 'shape': (1,), 'names': ['phase']},
            'intent.tte': {'dtype': 'float32', 'shape': (1,), 'names': ['tte']},
            'intent.xi': {'dtype': 'float32', 'shape': (M, J, D), 'names': ['waypoint', 'keypoint', 'xyz']}}


def mediapipe_keypoints(root: Path, episode: int, camera: str, schema: dict) -> np.ndarray:
    """TODO(user): 2D hand keypoints from camera images (MediaPipe Hands) and, for keypoint_dim 3, lifting with depth
    and camera extrinsics. Not needed for simulator data (human.keypoints is recorded)."""
    raise NotImplementedError('the dataset has no human.keypoints; keypoint detection on images is not implemented')


def data_files(root: Path) -> list[Path]:
    files = sorted((root / 'data').glob('*/*.parquet'))
    if not files:
        raise SystemExit(f'no parquet data files under {root / "data"}')
    return files


def write_parquet(columns: dict, path: Path, features: dict) -> None:
    """Rewrite one data file with the dataset's (LeRobot) feature schema."""
    hf = get_hf_features_from_features(features)
    table = datasets.Dataset.from_dict({k: columns[k] for k in hf}, features=hf, split='train').with_format('arrow')[:]
    pq.write_table(table, path, compression='snappy', use_dictionary=True)


def stats(segments: dict, labels: dict, schema: dict) -> str:
    fps = schema['fps']
    obj = Counter(schema['obj_vocab'][i] for a in labels.values() for i in a['obj'])
    act = Counter(schema['act_vocab'][i] for a in labels.values() for i in a['act'])
    n = sum(obj.values())
    lines = [f'frames {n}, episodes {len(labels)}, segments {sum(len(s) for s in segments.values())}',
             'obj: ' + ', '.join(f'{k} {v / n:.1%}' for k, v in obj.most_common()),
             'act: ' + ', '.join(f'{k} {v / n:.1%}' for k, v in act.most_common()),
             'per act: segments | t_event - t_onset [s] | t_clear - t_onset [s] | label span t_end - t_onset [s] (mean, min-max)']
    by = {}
    for segs in segments.values():
        for s in segs:
            by.setdefault(s['act'], []).append((s['t_event'] - s['t_onset'], s['t_clear'] - s['t_onset'],
                                                s.get('t_end', s['t_event']) - s['t_onset']))
    f = lambda x: f'{np.mean(x) / fps:.2f} ({np.min(x) / fps:.2f}-{np.max(x) / fps:.2f})'
    for a, v in sorted(by.items()):
        v = np.asarray(v, float)
        lines.append(f'  {a:20s} {len(v):4d} | {f(v[:, 0])} | {f(v[:, 1])} | {f(v[:, 2])}')
    empty = [e for e, s in segments.items() if not s]
    if empty:
        lines.append(f'episodes without any segment: {empty}')
    return '\n'.join(lines)


def build_labels(root: Path, schema_path=SCHEMA_PATH, segments_arg: str = 'auto', overwrite: bool = False,
                 kp_camera: str = 'high') -> tuple[dict, dict, dict]:
    """Compute the intent labels of every episode and write them into the dataset at `root`.
    Returns (segments, labels, schema), keyed by episode index."""
    schema = load_schema(schema_path)
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    if info['fps'] != schema['fps']:
        raise SystemExit(f"dataset fps {info['fps']} != schema fps {schema['fps']}")
    present = [k for k in INTENT_FEATURES if k in info['features']]
    if present and not overwrite:
        raise SystemExit(f'{present} already exist; pass --overwrite to replace them')
    files = data_files(root)
    tables = [pq.read_table(f, columns=[c for c in ('episode_index', 'frame_index', 'human.keypoints')
                                        if c in pq.read_schema(f).names]) for f in files]
    ep_col = np.concatenate([t.column('episode_index').to_numpy() for t in tables])
    fr_col = np.concatenate([t.column('frame_index').to_numpy() for t in tables])
    has_kp = all('human.keypoints' in t.column_names for t in tables)
    kp_col = np.concatenate([np.asarray(t.column('human.keypoints').to_pylist(), np.float32) for t in tables]) if has_kp else None
    episodes = sorted(int(e) for e in np.unique(ep_col))

    if segments_arg == 'auto':
        meta = {m['episode_index']: m for m in map(json.loads, (root / 'meta/hri_episodes.jsonl').read_text().splitlines())}
    else:
        given = {int(k): v for k, v in json.loads(resolve(segments_arg).read_text()).items()}

    segments, labels = {}, {}
    for ep in episodes:
        rows = np.flatnonzero(ep_col == ep)
        rows = rows[np.argsort(fr_col[rows])]
        T = len(rows)
        if has_kp:
            kp = kp_col[rows]
        else:
            kp = mediapipe_keypoints(root, ep, kp_camera, schema)
        if segments_arg == 'auto':
            m = meta[ep]
            if 'human_motions' not in m:
                raise SystemExit('meta/hri_episodes.jsonl has no human_motions: re-collect with the current collect_demos.py')
            segs = segments_from_sim(m['human_motions'], m['human_stages'], m['spec'], schema, T)
        else:
            segs = given.get(ep, [])
        segments[ep] = segs
        labels[ep] = episode_labels(T, segs, kp, schema)

    specs = intent_feature_specs(schema)
    features = {k: v for k, v in info['features'].items() if k not in INTENT_FEATURES}
    features.update({k: {**v, 'shape': list(v['shape'])} for k, v in specs.items()})
    for f in files:
        cols = pq.read_table(f).to_pydict()
        for name in INTENT_FEATURES:
            comp = name.removeprefix('intent.')
            vals = [labels[int(e)][comp][int(i)] for e, i in zip(cols['episode_index'], cols['frame_index'])]
            cols[name] = vals if comp == 'xi' else np.asarray(vals)
        write_parquet(cols, f, {k: {**v, 'shape': tuple(v['shape'])} for k, v in features.items()})
    info['features'] = features
    info_path.write_text(json.dumps(info, indent=4))
    (root / 'meta/intent_segments.json').write_text(json.dumps({str(k): v for k, v in segments.items()}, indent=1))
    (root / 'meta/intent_schema.json').write_text(json.dumps(dict(schema, source=str(schema_path), segments=segments_arg),
                                                             indent=2))
    return segments, labels, schema


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--schema', default=SCHEMA_PATH)
    p.add_argument('--segments', default='auto', help="'auto' (simulator ground truth) or a segments JSON file")
    p.add_argument('--kp-camera', default='high', help='camera for keypoint detection when human.keypoints is missing')
    p.add_argument('--overwrite', action='store_true', help='replace existing intent.* features')
    args = p.parse_args()
    root = resolve(args.dataset_root)
    segments, labels, schema = build_labels(root, args.schema, args.segments, args.overwrite, args.kp_camera)
    print(stats(segments, labels, schema))
    print(f'wrote {", ".join(INTENT_FEATURES)} to {root}')


if __name__ == '__main__':
    main()
