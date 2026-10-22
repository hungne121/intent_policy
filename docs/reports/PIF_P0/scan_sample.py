"""Human/robot subtask runs of the HABIT `sample` config (1 episode per task), decoded with sample/meta (its
subtask ids differ from full/meta). Prints the episodes with a pointing / indicating subtask, or of the sids given.
usage: scan_sample.py habit_root [sid ...]"""
import json, sys, glob, collections
import pandas as pd
R, M = sys.argv[1], 'sample'
hs = {json.loads(l)['task_index']: json.loads(l)['task'] for l in open(f'{R}/{M}/meta/human_subtasks.jsonl')}
ss = {json.loads(l)['task_index']: json.loads(l)['task'] for l in open(f'{R}/{M}/meta/subtasks.jsonl')}
eps = {int(json.loads(l)['episode_index']): json.loads(l) for l in open(f'{R}/sample/meta/episodes.jsonl')}
full = collections.Counter(json.loads(l)['sid'] for l in open(f'{R}/full/meta/episodes.jsonl'))
kw = ('oint', 'indicat', 'held')
for f in sorted(glob.glob(f'{R}/sample/data/*/*.parquet')):
    df = pd.read_parquet(f, columns=['episode_index', 'frame_index', 'human_role_subtask_index', 'low_level_task_index'])
    ei = int(df.episode_index.iloc[0]); e = eps[ei]
    def runs(col, names):
        out, prev = [], None
        for i, v in enumerate(df[col].tolist()):
            if v != prev: out.append([v, i, i]); prev = v
            else: out[-1][2] = i
        return [(names.get(v, v), round(a / 10, 1), round(b / 10 + .1, 1)) for v, a, b in out]
    h = runs('human_role_subtask_index', hs); r = runs('low_level_task_index', ss)
    if any(any(k in str(x[0]) for k in kw) for x in h + r) or e['sid'] in sys.argv[2:]:
        print(f"\n## ep{ei} {e['sid']} full={full[e['sid']]} len={len(df)/10:.1f}s :: {e['high_level_instruction']}")
        print('  H:', ' | '.join(f'{n} [{a}-{b}]' for n, a, b in h))
        print('  R:', ' | '.join(f'{n} [{a}-{b}]' for n, a, b in r))
