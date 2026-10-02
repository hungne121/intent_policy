"""Print the intent contract fields of one episode side by side for every source (INTENT_ACT_GUIDE_v2.md M1/M2
deliverable: "bảng T2"). Rows: the ground-truth milestones of every segment (t_onset, t_evident, t_clear, t_event,
t_end) and a few frames before / between them.

  ./run.sh -m scripts.intent_table --dataset-root outputs/datasets/intent_act_early --task T2
"""
import argparse
import json

import numpy as np
import pyarrow.parquet as pq

from intent_policy.utils import resolve

SHORT = {'hs': 'hindsight', 'pp': 'perfect', 'pr0': 'predicted'}


def episode_rows(root, episode: int, prefixes: list[str]) -> dict:
    cols = ['episode_index', 'frame_index', 'hri.task_id'] + [f'intent_{p}.{f}' for p in prefixes
                                                             for f in ('p_target', 'p_who', 'c_target', 'c_who', 'phase', 'tte', 'tte_std', 'confidence', 'occupancy')]
    out = {}
    for f in sorted((root / 'data').glob('*/*.parquet')):
        t = pq.read_table(f, columns=cols)
        ep = np.asarray(t.column('episode_index'))
        m = ep == episode
        if m.any():
            order = np.argsort(np.asarray(t.column('frame_index'))[m])
            for c in cols:
                v = np.asarray(t.column(c).to_pylist(), dtype=object)[m][order]
                out[c] = np.asarray(list(v)) if c.startswith('intent_') else v
    return out


def fmt(schema, rows, p, t) -> str:
    g = lambda f: rows[f'intent_{p}.{f}'][t]
    pt, pw, ct, cw, ph = g('p_target'), g('p_who'), g('c_target'), g('c_who'), g('phase')
    i, w = int(np.argmax(pt)), int(np.argmax(pw))
    return (f"{schema['targets'][i]}:{pt[i]:.2f} {schema['who'][w]}:{pw[w]:.2f} | c {schema['targets'][int(np.argmax(ct))]}"
            f"/{schema['who'][int(np.argmax(cw))]} | {schema['phases'][int(np.argmax(ph))]} | tte {float(np.ravel(g('tte'))[0]):.2f}"
            f"±{float(np.ravel(g('tte_std'))[0]):.2f} | conf {g('confidence')[0]:.2f}/{g('confidence')[1]:.2f} occ {g('occupancy')[6]:.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--episode', type=int, help='episode index (default: the first of --task)')
    p.add_argument('--task', default='T2', help='list task (T1 T2 T3 T4 T4neg T5) when --episode is not given')
    p.add_argument('--sources', nargs='+', default=['hs', 'pp', 'pr0'])
    args = p.parse_args()
    root = resolve(args.dataset_root)
    schema = json.loads((root / 'meta/intent_schema.json').read_text())
    meta = [json.loads(l) for l in (root / 'meta/hri_episodes.jsonl').read_text().splitlines()]
    ep = args.episode if args.episode is not None else next(m['episode_index'] for m in meta if m['list_task'] == args.task)
    m = next(x for x in meta if x['episode_index'] == ep)
    segs = json.loads((root / 'meta/intent_segments.json').read_text())[str(ep)]
    rows = episode_rows(root, ep, [f'{s}' for s in args.sources])
    T = len(rows['frame_index'])
    marks = {}
    for k, s in enumerate(segs):
        for key in ('t_onset', 't_evident', 't_clear', 't_event', 't_end'):
            marks.setdefault(s[key], []).append(f"{key}#{k}")
        marks.setdefault(max(s['t_onset'] - 5, 0), []).append(f'onset#{k}-5')
        if s.get('t_retract'):
            marks.setdefault(s['t_retract'], []).append(f'retracted#{k}')
    marks.setdefault(T - 1, []).append('last')
    print(f"episode {ep} ({m['list_task']}, {m['scenario_id']}, seed {m['seed']}), task_id {int(rows['hri.task_id'][0])} "
          f"({schema['tasks'][int(rows['hri.task_id'][0])]}), {T} frames @ {schema['fps']} Hz")
    for k, s in enumerate(segs):
        print(f"  segment #{k}: {s['gesture']} -> {s['target']} (who {s['who']}), onset {s['t_onset']} evident {s['t_evident']} "
              f"clear {s['t_clear']} event {s['t_event']} end {s['t_end']} retract {s.get('t_retract')}")
    print(f"columns: p_target:prob p_who:prob | c_target/c_who | phase | tte±std [s] | confidence[0]/[1] occupancy conf")
    for t in sorted(marks):
        print(f"t={t:4d} ({t / schema['fps']:5.2f}s) {', '.join(marks[t])}")
        for s in args.sources:
            print(f"    {SHORT.get(s, s):10s} {fmt(schema, rows, s, t)}")


if __name__ == '__main__':
    main()
