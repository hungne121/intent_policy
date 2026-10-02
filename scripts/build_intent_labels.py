"""Add intent fields of the contract (INTENT_ACT_GUIDE_v2.md §2) to a recorded LeRobotDataset, in place.

Per frame and per source (columns `intent_<src>.<field>`, validated by intent/contract.validate_intent):
  hs   HindsightSource           ground truth + true future (ceiling)
  pp   PerfectPerceptionSource   causal, clean keypoints
  pr0 .. pr<R-1>  PredictedSource R noisy replicas (noise_scale drawn per replica; 0 until measured: "chưa hiệu chỉnh")
plus `hri.task_id` (T1-T4), meta/intent_segments.json (ground-truth segments) and meta/intent_schema.json.
The robot state of every source comes from the recorded observation (TCP, measured gripper opening -> holding);
`hri.holding_gt` (simulator grasp state) is only used to report the holding detector's quality.

Segments: `--segments auto` derives them from simulator ground truth (meta/hri_episodes.jsonl: human motions and
stages, scripts/collect_demos.py); `--segments <file.json>` reads {episode_index: [segment, ...]} in frame indices.

Example:
  ./run.sh -m scripts.build_intent_labels --dataset-root outputs/datasets/intent_act_early --sources hindsight perfect predicted
"""
import argparse
import json
from pathlib import Path

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.feature_utils import get_hf_features_from_features

from intent_policy.intent.contract import FIELDS, dataset_features, validate_intent
from intent_policy.intent.labels import SCHEMA_PATH, load_schema, segments_from_sim
from intent_policy.intent.sources import HindsightSource, HumanObs
from intent_policy.intent.tracker import HoldingDetector, RobotState
from intent_policy.utils import resolve

SOURCE_NAMES = ('hindsight', 'perfect', 'predicted')
TASK_ID = 'hri.task_id'


def data_files(root: Path) -> list[Path]:
    files = sorted((root / 'data').glob('*/*.parquet'))
    if not files:
        raise SystemExit(f'no parquet data files under {root / "data"}')
    return files


def column_numpy(table: pa.Table, name: str) -> np.ndarray:
    """A (nested / extension) list column as a dense numpy array (N, *shape) without Python objects."""
    arr = table.column(name).combine_chunks()
    if isinstance(arr, pa.ExtensionArray):
        arr = arr.storage
    shape = []
    while pa.types.is_list(arr.type) or pa.types.is_fixed_size_list(arr.type):
        n = len(arr)
        arr = arr.flatten()
        shape.append(len(arr) // max(n, 1))
    return arr.to_numpy(zero_copy_only=False).reshape((table.num_rows, *shape))


def to_arrow(values: np.ndarray, pa_type: pa.DataType) -> pa.Array:
    """(N, *shape) numpy -> Arrow array of the HF feature type (Value, fixed-size list or ArrayND extension)."""
    if isinstance(pa_type, pa.ExtensionType):
        arr = pa.array(values.reshape(-1))
        for size in reversed(values.shape[1:]):
            arr = pa.ListArray.from_arrays(pa.array(np.arange(0, len(arr) + 1, size, dtype=np.int32)), arr)
        return pa.ExtensionArray.from_storage(pa_type, arr)
    if pa.types.is_fixed_size_list(pa_type):
        return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa_type.value_type), pa_type.list_size)
    return pa.array(values.reshape(-1), type=pa_type)


def write_parquet(table: pa.Table, new: dict[str, np.ndarray], path: Path, features: dict) -> None:
    """Rewrite one data file: the kept columns of `table` plus the `new` numpy columns, with the dataset's (LeRobot /
    HF) feature schema."""
    schema = datasets.Features(get_hf_features_from_features(features)).arrow_schema
    arrays = [to_arrow(new[f.name], f.type) if f.name in new else table.column(f.name) for f in schema]
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), path, compression='snappy', use_dictionary=True)


def load_episodes(root: Path) -> dict[int, dict]:
    """Per episode (frame order): keypoints (T, J, 3), state (T, 10), holding_gt (T,)."""
    files = data_files(root)
    cols = [c for c in ('episode_index', 'frame_index', 'observation.state', 'human.keypoints', 'hri.holding_gt')]
    tables = [pq.read_table(f, columns=[c for c in cols if c in pq.read_schema(f).names]) for f in files]
    if not all('human.keypoints' in t.column_names for t in tables):
        raise SystemExit('the dataset has no human.keypoints (collect with an experiment config that sets intent.schema)')
    cat = lambda c: np.concatenate([column_numpy(t, c) if t.schema.field(c).type.num_fields or isinstance(t.schema.field(c).type, pa.ExtensionType)
                                    else t.column(c).to_numpy() for t in tables])
    ep, fr = cat('episode_index'), cat('frame_index')
    kp, st = cat('human.keypoints').astype(np.float32), cat('observation.state').astype(np.float32)
    gt = cat('hri.holding_gt').reshape(-1) if all('hri.holding_gt' in t.column_names for t in tables) else None
    out = {}
    for e in np.unique(ep):
        rows = np.flatnonzero(ep == e)
        rows = rows[np.argsort(fr[rows])]
        out[int(e)] = dict(keypoints=kp[rows], state=st[rows], holding_gt=None if gt is None else gt[rows].astype(bool))
    return out


def robot_states(state: np.ndarray, schema: dict) -> list[RobotState]:
    """observation.state [10] = 6 joints, gripper opening, TCP xyz -> RobotState with the detected holding flag."""
    det = HoldingDetector(**schema['holding'])
    return [RobotState(tcp=s[7:10].astype(float), gripper=float(s[6]), holding=det.step(float(s[6]))) for s in state]


def run_source(source, meta: dict, ep: dict, robots: list[RobotState]) -> dict[str, np.ndarray]:
    source.reset(meta)
    frames = [source.step(HumanObs(t, ep['keypoints'][t]), robots[t]) for t in range(len(robots))]
    return {k: np.stack([f[k] for f in frames]).astype(np.float32) for k in FIELDS}


def make_sources(names: list[str], schema: dict, episodes: dict, segments: dict, train: list[int]) -> dict:
    """{column prefix: (source factory per episode)}; perfect / predicted are calibrated on the training episodes."""
    out = {}
    for name in names:
        if name == 'hindsight':
            out['intent_hs'] = lambda e: HindsightSource(schema)
        elif name in ('perfect', 'predicted'):
            from intent_policy.intent import perception
            out.update(perception.make_sources(name, schema, episodes, segments, train))
        else:
            raise SystemExit(f'unknown source {name!r}; choose from {SOURCE_NAMES}')
    return out


def build_labels(root: Path, schema_path=SCHEMA_PATH, sources=('hindsight',), segments_arg: str = 'auto',
                 val_every: int = 10) -> dict:
    """Compute and write the intent fields of every episode. Returns {segments, outputs, robots, episodes, schema}."""
    schema = load_schema(schema_path)
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    if info['fps'] != schema['fps']:
        raise SystemExit(f"dataset fps {info['fps']} != schema fps {schema['fps']}")
    episodes = load_episodes(root)
    meta = {m['episode_index']: m for m in map(json.loads, (root / 'meta/hri_episodes.jsonl').read_text().splitlines())}
    if segments_arg == 'auto':
        if any('human_motions' not in m for m in meta.values()):
            raise SystemExit('meta/hri_episodes.jsonl has no human_motions: re-collect with the current collect_demos.py')
        segments = {e: segments_from_sim(meta[e]['human_motions'], meta[e]['human_stages'], meta[e]['spec'], schema,
                                         len(ep['state'])) for e, ep in episodes.items()}
    else:
        segments = {int(k): v for k, v in json.loads(resolve(segments_arg).read_text()).items()}
    train = [e for e in sorted(episodes) if not (val_every and e % val_every == val_every - 1)]   # = train_policy split
    factories = make_sources(list(sources), schema, episodes, segments, train)
    robots = {e: robot_states(ep['state'], schema) for e, ep in episodes.items()}
    outputs = {}
    for e, ep in episodes.items():
        m = dict(segments=segments.get(e, []), keypoints=ep['keypoints'], episode_index=e, spec=meta[e]['spec'])
        for prefix, factory in factories.items():
            out = run_source(factory(e), m, ep, robots[e])
            validate_intent(out, schema, batched=True)
            outputs.setdefault(prefix, {})[e] = out

    task_ids = {e: schema['tasks'].index(schema['task_of_scenario'][meta[e]['scenario_id']]) for e in episodes}
    features = {k: v for k, v in info['features'].items() if not k.startswith('intent_') and k != TASK_ID}
    for prefix in outputs:
        features.update(dataset_features(schema, prefix))
    features[TASK_ID] = {'dtype': 'int64', 'shape': (1,), 'names': ['task_id']}
    tuples = {k: {**v, 'shape': tuple(v['shape'])} for k, v in features.items()}
    for f in data_files(root):
        table = pq.read_table(f)
        ep_col, fr_col = table.column('episode_index').to_numpy(), table.column('frame_index').to_numpy()
        new = {TASK_ID: np.asarray([task_ids[int(e)] for e in ep_col], np.int64)}
        for prefix, per_ep in outputs.items():
            for field in FIELDS:
                some = next(iter(per_ep.values()))[field]
                col = np.zeros((len(ep_col),) + some.shape[1:], np.float32)
                for e in np.unique(ep_col):
                    rows = np.flatnonzero(ep_col == e)
                    col[rows] = per_ep[int(e)][field][fr_col[rows]]
                new[f'{prefix}.{field}'] = col
        write_parquet(table, new, f, tuples)
        del table, new
    info['features'] = {k: {**v, 'shape': list(v['shape'])} for k, v in features.items()}
    info_path.write_text(json.dumps(info, indent=4))
    (root / 'meta/intent_segments.json').write_text(json.dumps({str(k): v for k, v in segments.items()}, indent=1))
    (root / 'meta/intent_schema.json').write_text(json.dumps(dict(schema, source=str(schema_path), segments=segments_arg,
                                                                  sources=list(outputs), train_episodes=train), indent=2))
    return dict(segments=segments, outputs=outputs, robots=robots, episodes=episodes, schema=schema, meta=meta)


def holding_report(res: dict) -> str:
    """Holding detector (gripper opening) vs the simulator grasp state: precision, recall, onset / release latency."""
    tp = fp = fn = 0
    lat_on, lat_off = [], []
    for e, ep in res['episodes'].items():
        gt = ep['holding_gt']
        if gt is None:
            continue
        det = np.array([r.holding for r in res['robots'][e]])
        tp, fp, fn = tp + int((det & gt).sum()), fp + int((det & ~gt).sum()), fn + int((~det & gt).sum())
        for a, b, lat in ((gt, det, lat_on), (~gt, ~det, lat_off)):
            for t in np.flatnonzero(a[1:] & ~a[:-1]) + 1:           # ground-truth rising edges
                hit = np.flatnonzero(b[t:])
                if len(hit):
                    lat.append(int(hit[0]))
    if tp + fp + fn == 0:
        return 'holding: no ground truth in the dataset'
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    f = lambda x: f'{np.mean(x):.2f} (max {np.max(x)})' if x else 'n/a'
    return (f'holding detector vs sim grasp state: precision {p:.3f}, recall {r:.3f}; latency [frames @20 Hz] grasp '
            f'{f(lat_on)}, release {f(lat_off)}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--schema', default=SCHEMA_PATH)
    p.add_argument('--sources', nargs='+', default=['hindsight'], choices=SOURCE_NAMES)
    p.add_argument('--segments', default='auto', help="'auto' (simulator ground truth) or a segments JSON file")
    p.add_argument('--val-every', type=int, default=10, help='same split as scripts/train_policy.py')
    args = p.parse_args()
    root = resolve(args.dataset_root)
    res = build_labels(root, args.schema, args.sources, args.segments, args.val_every)
    print(holding_report(res))
    print(f"wrote {', '.join(res['outputs'])} ({len(res['episodes'])} episodes) and {TASK_ID} to {root}")


if __name__ == '__main__':
    main()
