"""Add D's intent input to a v6 dataset, in place: the ManiCast forecast (docs/PIF.md §2.2).

Columns `intent_pp.<field>` of the v2 contract (intent/contract.py) with configs/intent_schema_v6.yaml:
  xi (M, J, 3)  the forecast displacement of the 6 arm joints at (m + 1) / M x 1 s, from the frozen ManiCast fine-tuned
                on v6 motion (configs/pif/pif.yaml v6_expert.checkpoint_path), causal: the last 0.4 s of the recorded
                `human.keypoints` (frames before the start repeat the first one)
  the other fields neutral (none / rest / zero occupancy): D is trained with --intent-keep motion
plus `hri.task_id` (T1-T4), `hri.instr_*` = 0 and meta/intent_schema.json, as scripts/build_intent_labels.py writes them,
so scripts/train_policy.py reads the dataset unchanged (--intent-source perfect).
  ./run.sh -m scripts.add_manicast_intent --dataset-root outputs/datasets/v6_early
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pif.config import load_config
from pif.forecast import ManiCastForecaster
from scripts.build_intent_labels import TASK_ID, data_files, load_episodes, write_parquet
from intent_policy.intent.contract import FIELDS, confidence, dataset_features, one_hot, validate_intent
from intent_policy.intent.instruction import INSTR_DEST, INSTR_OBJECT
from intent_policy.intent.labels import load_schema
from intent_policy.utils import resolve

PREFIX = 'intent_pp'
MANICAST = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist']   # pif/forecast.py OUR_JOINTS[1:]


def forecast_xi(kp: np.ndarray, fc: ManiCastForecaster, fps: float, t_out: np.ndarray) -> np.ndarray:
    """kp [T, 6, 3] (MANICAST order, world) -> xi [T, M, 6, 3]: forecast joint position - current position."""
    n = int(round(0.4 * fps)) + 1                                   # 0.4 s of history (ManiCast's input)
    pad = np.concatenate([np.repeat(kp[:1], n - 1, 0), kp])
    mk, waist = fc.to_comad({j: pad[:, i] for i, j in enumerate(MANICAST)})
    H = np.stack([mk[t:t + n] for t in range(len(kp))])
    W = np.stack([waist[t:t + n] for t in range(len(kp))])
    t_hist = np.arange(-(n - 1), 1) / fps
    out = np.concatenate([fc.forecast(H[a:a + 512], W[a:a + 512], t_hist, t_out) for a in range(0, len(H), 512)])
    fut = fc.from_comad(out[:, :, 1:7])                             # markers 1..6 = MANICAST joints, [T, M, 6, 3]
    now = fc.from_comad(H[:, -1, 1:7])                              # the same markers now (offsets cancel)
    return (fut - now[:, None]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset-root', required=True)
    ap.add_argument('--schema', default='configs/intent_schema_v6.yaml')
    ap.add_argument('--val-every', type=int, default=10)
    args = ap.parse_args()
    root, schema = resolve(args.dataset_root), load_schema(args.schema)
    cfg = load_config('configs/pif/pif.yaml', [])
    fc = ManiCastForecaster(dict(cfg, forecast=dict(cfg['forecast'], checkpoint_path=cfg['v6_expert']['checkpoint_path'])))
    exp = json.loads((root / 'meta/hri_experiment_config.json').read_text())
    recorded = list(exp['intent']['keypoints'])
    sel = [recorded.index(j) for j in MANICAST]
    if list(schema['keypoints']) != MANICAST:
        raise SystemExit(f"schema keypoints {schema['keypoints']} != {MANICAST}")
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    fps = float(info['fps'])
    M = int(schema['n_waypoints'])
    t_out = (np.arange(M) + 1) / M * float(schema['horizon_s'])
    K, W, P = len(schema['targets']), len(schema['who']), len(schema['phases'])
    episodes = load_episodes(root)
    meta = {m['episode_index']: m for m in map(json.loads, (root / 'meta/hri_episodes.jsonl').read_text().splitlines())}
    out = {}
    for e, ep in episodes.items():
        T = len(ep['state'])
        xi = forecast_xi(ep['keypoints'][:, sel], fc, fps, t_out)
        p_who, p_target = np.tile(one_hot(0, W), (T, 1)), np.tile(one_hot(0, K), (T, 1))
        d = dict(p_who=p_who, p_target=p_target, c_who=p_who.copy(), c_target=p_target.copy(),
                 occupancy=np.zeros((T, 7), np.float32), tte=np.zeros((T, 1), np.float32),
                 tte_std=np.zeros((T, 1), np.float32), phase=np.tile(one_hot(0, P), (T, 1)), xi=xi,
                 confidence=confidence(p_target, p_who))
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
