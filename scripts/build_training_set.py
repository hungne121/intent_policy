"""Merge the chunks of a paired dataset (scripts/collect_pairs.py: <root>/<task>/<case>/<split>/<mode>/) of one mode into
one LeRobot dataset for scripts/train_policy.py (docs/PIF.md §8.13 G0): LeRobot's aggregate_datasets for the data,
videos and metadata, plus meta/hri_episodes.jsonl concatenated in the same order with the new episode indices (the
episode's `split` field is kept: train_policy's split_source episode_metadata reads it) and meta/hri_sources.json.
Only complete chunks (DONE) are merged; *.incomplete-* folders never.
  ./run.sh scripts/build_training_set.py --mode early [--root outputs/datasets/ten_task_v9] [--tasks I1 I2 ...]
    -> <root>_<mode>/ (repo id local/<root name>_<mode>)
  observed (PIF §8.13 06/11): --mode observed --skip-cases forecast_fault
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def chunks(root: Path, mode: str, tasks: list[str] | None, skip_cases: list[str] | None = None) -> list[Path]:
    out = []
    for d in sorted(root.glob(f'*/*/*/{mode}')):
        task, case, split = d.parts[-4:-1]
        if (tasks and task not in tasks) or '.incomplete' in split or not (d.parent / 'DONE').exists():
            continue
        if skip_cases and case in skip_cases:
            continue
        out.append(d)
    return out


def _flat_copy(src: Path, dst: Path, key: str) -> list[int]:
    """A copy of chunk `src` whose 2-D feature `key` is flattened ([9, 3] -> [27]) in the data parquet and info.json;
    meta/ copied, videos/ and images/ symlinked (read only). Returns the original shape."""
    import shutil
    import pyarrow as pa
    import pyarrow.parquet as pq
    dst.mkdir(parents=True)
    shutil.copytree(src / 'meta', dst / 'meta')
    for d in ('videos', 'images'):
        if (src / d).exists():
            (dst / d).symlink_to((src / d).resolve(), target_is_directory=True)
    info = json.loads((src / 'meta' / 'info.json').read_text())
    shape = list(info['features'][key]['shape'])
    n = int(np.prod(shape))
    for f in sorted((src / 'data').rglob('*.parquet')):
        t = pq.read_table(f)
        col = t.column(key).combine_chunks()
        if isinstance(col.type, pa.ExtensionType):  # HF datasets Array2D (registered once `datasets` is imported)
            col = col.storage
        flat = pa.FixedSizeListArray.from_arrays(col.flatten().flatten(), n)          # [rows] x n, same order
        flat = flat.cast(pa.list_(col.type.value_type.value_type))
        t = t.set_column(t.schema.get_field_index(key), key, flat).replace_schema_metadata(None)
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, out)
    info['features'][key] = {**info['features'][key], 'shape': [n], 'names': [f'{key}_{i}' for i in range(n)]}
    (dst / 'meta' / 'info.json').write_text(json.dumps(info, indent=4))
    return shape


def _unflatten(root: Path, key: str, shape: list[int], feature: dict, hf_meta: bytes) -> None:
    """Back to the original 2-D feature in the merged dataset (data parquet, info.json; the HF schema metadata of a
    source chunk, whose columns are the same)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    for f in sorted((root / 'data').rglob('*.parquet')):
        t = pq.read_table(f)
        col = t.column(key).combine_chunks()
        vt = col.type.value_type
        inner = pa.FixedSizeListArray.from_arrays(col.flatten(), shape[1]).cast(pa.list_(vt))
        nested = pa.FixedSizeListArray.from_arrays(inner, shape[0]).cast(pa.list_(pa.list_(vt)))
        t = t.set_column(t.schema.get_field_index(key), key, nested).replace_schema_metadata({b'huggingface': hf_meta})
        pq.write_table(t, f)
    info = json.loads((root / 'meta' / 'info.json').read_text())
    info['features'][key] = feature
    (root / 'meta' / 'info.json').write_text(json.dumps(info, indent=4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='outputs/datasets/ten_task_v9')
    p.add_argument('--mode', required=True, choices=['early', 'observed'])
    p.add_argument('--tasks', nargs='*')
    p.add_argument('--skip-cases', nargs='*', help='cases left out of this mode (docs/PIF.md §8.13 06/11: the observed '
                   'side of forecast_fault, whose fault only the early rule reads)')
    p.add_argument('--out')
    args = p.parse_args()
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    import pyarrow.parquet as pq
    root = Path(args.root)
    out = Path(args.out or f'{root}_{args.mode}')
    if out.exists():
        raise SystemExit(f'{out} exists: remove it first (the merge never overwrites)')
    src = chunks(root, args.mode, args.tasks, args.skip_cases)
    if not src:
        raise SystemExit('no complete chunks')
    ids = ['local/' + '_'.join(d.parts[-4:]) for d in src]
    # LeRobot's aggregate moves the data through pandas, which breaks on the 2-D feature human.keypoints [9, 3]
    # (3 patch attempts failed, docs/PIF.md §8.13 G0): merge flattened copies, then restore the feature
    key = 'human.keypoints'
    feature = json.loads((src[0] / 'meta' / 'info.json').read_text())['features'][key]
    hf_meta = pq.read_schema(next((src[0] / 'data').rglob('*.parquet'))).metadata[b'huggingface']
    tmp = out.parent / f'_flat_tmp_{out.name}'
    if tmp.exists():
        raise SystemExit(f'{tmp} exists: remove it first')
    flat = []
    for d, i in zip(src, ids):
        shape = _flat_copy(d, tmp / i.split('/')[1], key)
        flat.append(tmp / i.split('/')[1])
    print(f'merging {len(src)} chunks -> {out}', flush=True)
    aggregate_datasets(repo_ids=ids, aggr_repo_id=f'local/{out.name}', roots=flat, aggr_root=out)
    _unflatten(out, key, shape, feature, hf_meta)
    # hri episode metadata in the aggregation order, re-indexed
    rows, sources, offset = [], [], 0
    for d in src:
        n = LeRobotDatasetMetadata('local/x', root=d).total_episodes
        ms = [json.loads(l) for l in (d / 'meta' / 'hri_episodes.jsonl').read_text().splitlines()]
        if len(ms) != n:
            raise SystemExit(f'{d}: {len(ms)} hri episodes for {n} LeRobot episodes')
        for m in sorted(ms, key=lambda m: m['episode_index']):
            m['source_episode_index'], m['source_chunk'] = m['episode_index'], str(d)
            m['episode_index'] = offset + m['source_episode_index']
            rows.append(m)
        sources.append(dict(chunk=str(d), episodes=n, first_episode=offset))
        offset += n
    meta = LeRobotDatasetMetadata(f'local/{out.name}', root=out)
    if meta.total_episodes != len(rows):
        raise SystemExit(f'merged {meta.total_episodes} episodes but {len(rows)} hri rows')
    eps = pd.concat(pd.read_parquet(f) for f in sorted((out / 'meta' / 'episodes').rglob('*.parquet')))
    length = dict(zip(eps['episode_index'].astype(int), eps['length'].astype(int)))
    bad = [m['episode_index'] for m in rows if length[m['episode_index']] != int(m['n_frames'])]
    if bad:
        raise SystemExit(f'frame counts differ from the hri metadata on episodes {bad[:10]}')
    (out / 'meta' / 'hri_episodes.jsonl').write_text(''.join(json.dumps(m) + '\n' for m in rows))
    (out / 'meta' / 'hri_sources.json').write_text(json.dumps(dict(mode=args.mode, root=str(root), chunks=sources), indent=1))
    first = src[0] / 'meta' / 'hri_experiment_config.json'
    if first.exists():
        (out / 'meta' / 'hri_experiment_config.json').write_text(first.read_text())
    import shutil
    shutil.rmtree(tmp)                           # the flattened copies (their videos / images were symlinks)
    print(f'{len(rows)} episodes, {meta.total_frames} frames -> {out}')


if __name__ == '__main__':
    main()
