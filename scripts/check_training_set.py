"""Check a dataset merged by scripts/build_training_set.py against its source chunks (docs/PIF.md §8.13 G0): per
merged episode, the source chunk / episode from meta/hri_episodes.jsonl; equal frame counts; for sampled frames
(first, middle, last of every episode, or every frame with --all) equal numeric features (human.keypoints, state,
action, ...) and camera frames; hri metadata identical apart from the episode index fields; the split field kept.
  ./run.sh scripts/check_training_set.py outputs/datasets/ten_task_v9_early [--all]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root')
    p.add_argument('--all', action='store_true')
    p.add_argument('--image-tol', type=float, default=0.0, help='max |difference| of decoded camera frames')
    args = p.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = Path(args.root)
    merged = LeRobotDataset('local/merged', root=root)
    rows = [json.loads(l) for l in (root / 'meta' / 'hri_episodes.jsonl').read_text().splitlines()]
    assert len(rows) == merged.meta.total_episodes, 'hri rows != episodes'
    keys = [k for k, v in merged.meta.features.items() if v['dtype'] in ('float32', 'float64', 'int64')
            and k not in ('index', 'episode_index', 'task_index')]
    cams = [k for k, v in merged.meta.features.items() if v['dtype'] == 'video']
    srcs, worst, checked = {}, {k: 0.0 for k in keys + cams}, 0
    first = {int(e['episode_index']): int(e['dataset_from_index']) for e in merged.meta.episodes}
    for m in rows:
        e, chunk, se = int(m['episode_index']), m['source_chunk'], int(m['source_episode_index'])
        if chunk not in srcs:
            srcs[chunk] = (LeRobotDataset('local/src', root=chunk),
                           {json.loads(l)['episode_index']: json.loads(l)
                            for l in (Path(chunk) / 'meta' / 'hri_episodes.jsonl').read_text().splitlines()})
        src, src_rows = srcs[chunk]
        s_first = {int(x['episode_index']): int(x['dataset_from_index']) for x in src.meta.episodes}
        n = int(m['n_frames'])
        assert merged.meta.episodes[e]['length'] == src.meta.episodes[se]['length'] == n, f'episode {e}: length'
        a = {k: v for k, v in m.items() if k not in ('episode_index', 'source_episode_index', 'source_chunk')}
        b = {k: v for k, v in src_rows[se].items() if k != 'episode_index'}
        assert a == b, f'episode {e}: hri metadata differs'
        assert m['split'] in ('train', 'val'), f'episode {e}: split {m["split"]}'
        frames = range(n) if args.all else sorted({0, n // 2, n - 1})
        for f in frames:
            x, y = merged[first[e] + f], src[s_first[se] + f]
            assert int(x['frame_index']) == int(y['frame_index']) == f
            for k in keys + cams:
                d = float((x[k].to(torch.float64) - y[k].to(torch.float64)).abs().max())
                worst[k] = max(worst[k], d)
            checked += 1
    print(f'{len(rows)} episodes from {len(srcs)} chunks, {checked} frames compared')
    for k, d in worst.items():
        print(f'  {k}: max |merged - source| = {d:.3g}')
    bad = [k for k in keys if worst[k] > 0] + [k for k in cams if worst[k] > args.image_tol]
    if bad:
        raise SystemExit(f'MISMATCH: {bad}')
    print('OK')


if __name__ == '__main__':
    main()
