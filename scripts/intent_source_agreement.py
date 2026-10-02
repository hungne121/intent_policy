"""Agreement of the causal intent sources with the hindsight timeline (INTENT_ACT_GUIDE_v2.md M2 check).

For p_target, p_who and c_target: the fraction of frames whose argmax equals the hindsight argmax, split by the
frame's place in the ground-truth segments (onset..clear, clear..end, outside every segment, all).

  ./run.sh -m scripts.intent_source_agreement outputs/datasets/intent_act_early [--sources pp pr0]
"""
import argparse
import json

import numpy as np
import pyarrow.parquet as pq

from intent_policy.utils import resolve
from scripts.build_intent_labels import column_numpy, data_files


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dataset_root')
    p.add_argument('--sources', nargs='+', default=['pp', 'pr0'])
    args = p.parse_args()
    root = resolve(args.dataset_root)
    fields = ('p_target', 'p_who', 'c_target')
    segs = {int(k): v for k, v in json.loads((root / 'meta/intent_segments.json').read_text()).items()}
    cols = ['episode_index', 'frame_index'] + [f'intent_{s}.{f}' for s in ['hs'] + args.sources for f in fields]
    tables = [pq.read_table(f, columns=cols) for f in data_files(root)]
    ep = np.concatenate([t.column('episode_index').to_numpy() for t in tables])
    fr = np.concatenate([t.column('frame_index').to_numpy() for t in tables])
    g = {c: np.concatenate([column_numpy(t, c) for t in tables]).argmax(1) for c in cols[2:]}
    rel = np.full(len(ep), np.nan)
    for e, ss in segs.items():
        rows = np.flatnonzero(ep == e)
        for s in ss:
            m = (fr[rows] >= s['t_onset']) & (fr[rows] <= s['t_end'])
            rel[rows[m]] = (fr[rows[m]] - s['t_clear'])
    win = {'onset..clear': rel < 0, 'clear..end': rel >= 0, 'outside': np.isnan(rel), 'all': np.ones(len(ep), bool)}
    for src in args.sources:
        for f in fields:
            a = g[f'intent_{src}.{f}'] == g[f'intent_hs.{f}']
            print(src, f, ' '.join(f'{k} {a[m].mean():.3f}' for k, m in win.items()))


if __name__ == '__main__':
    main()
