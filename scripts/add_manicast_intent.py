"""Add D's intent input to a v6 dataset, in place: the ManiCast forecast (docs/PIF.md §2.2).

Columns `intent_pp.<field>` of the v2 contract (intent/contract.py) with configs/intent_schema_v6.yaml:
  built by intent_policy/intent/manicast_intent.py (the same code the closed-loop evaluation uses): xi = the ManiCast
  forecast displacement of the 6 arm joints from the recorded `human.keypoints`, the other fields neutral
plus `hri.task_id` (T1-T4), `hri.instr_*` = 0 and meta/intent_schema.json, as scripts/build_intent_labels.py writes them,
so scripts/train_policy.py reads the dataset unchanged (--intent-source perfect).
  ./run.sh -m scripts.add_manicast_intent --dataset-root outputs/datasets/v6_early
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from scripts.build_intent_labels import TASK_ID, data_files, load_episodes, write_parquet
from intent_policy.intent.contract import FIELDS, dataset_features, validate_intent
from intent_policy.intent.manicast_intent import MANICAST, forecast_xi, horizons, load_forecaster, neutral_contract
from intent_policy.intent.instruction import INSTR_DEST, INSTR_OBJECT
from intent_policy.intent.labels import load_schema
from intent_policy.utils import resolve

PREFIX = 'intent_pp'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset-root', required=True)
    ap.add_argument('--schema', default='configs/intent_schema_v6.yaml')
    ap.add_argument('--val-every', type=int, default=10)
    args = ap.parse_args()
    root, schema = resolve(args.dataset_root), load_schema(args.schema)
    fc = load_forecaster()
    exp = json.loads((root / 'meta/hri_experiment_config.json').read_text())
    recorded = list(exp['intent']['keypoints'])
    sel = [recorded.index(j) for j in MANICAST]
    if list(schema['keypoints']) != MANICAST:
        raise SystemExit(f"schema keypoints {schema['keypoints']} != {MANICAST}")
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    fps = float(info['fps'])
    t_out = horizons(schema)
    episodes = load_episodes(root)
    meta = {m['episode_index']: m for m in map(json.loads, (root / 'meta/hri_episodes.jsonl').read_text().splitlines())}
    out = {}
    for e, ep in episodes.items():
        T = len(ep['state'])
        xi = forecast_xi(ep['keypoints'][:, sel], fc, fps, t_out)
        d = neutral_contract(schema, xi)
        validate_intent(d, schema, batched=True)
        out[e] = d
        print(f'episode {e}: {T} frames, |xi| max {np.abs(xi).max():.3f} m', flush=True)
    task = {e: schema['tasks'].index(schema['task_of_scenario'][meta[e]['scenario_id']]) for e in episodes}
    features = {k: v for k, v in info['features'].items()
                if not k.startswith('intent_') and k not in (TASK_ID, INSTR_OBJECT, INSTR_DEST)}
    features.update(dataset_features(schema, PREFIX))
    for k, name in ((TASK_ID, 'task_id'), (INSTR_OBJECT, 'instr_object'), (INSTR_DEST, 'instr_dest')):
        features[k] = {'dtype': 'int64', 'shape': (1,), 'names': [name]}
    tuples = {k: {**v, 'shape': tuple(v['shape'])} for k, v in features.items()}
    for f in data_files(root):
        table = pq.read_table(f)
        ep_col, fr_col = table.column('episode_index').to_numpy(), table.column('frame_index').to_numpy()
        new = {TASK_ID: np.asarray([task[int(e)] for e in ep_col], np.int64),
               INSTR_OBJECT: np.zeros(len(ep_col), np.int64), INSTR_DEST: np.zeros(len(ep_col), np.int64)}
        for field in FIELDS:
            some = next(iter(out.values()))[field]
            col = np.zeros((len(ep_col),) + some.shape[1:], np.float32)
            for e in np.unique(ep_col):
                rows = np.flatnonzero(ep_col == e)
                col[rows] = out[int(e)][field][fr_col[rows]]
            new[f'{PREFIX}.{field}'] = col
        write_parquet(table, new, f, tuples)
        del table, new
    info['features'] = {k: {**v, 'shape': list(v['shape'])} for k, v in features.items()}
    info_path.write_text(json.dumps(info, indent=4))
    train = [e for e in sorted(episodes) if not (args.val_every and e % args.val_every == args.val_every - 1)]
    (root / 'meta/intent_schema.json').write_text(json.dumps(dict(
        schema, source=str(args.schema), sources=[PREFIX], train_episodes=train,
        intent_pp='ManiCast forecast in xi (scripts/add_manicast_intent.py), other fields neutral'), indent=2))
    print(f'{root}: {PREFIX} written for {len(out)} episodes')


if __name__ == '__main__':
    main()
