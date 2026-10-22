"""Human/robot subtask runs of full-config episodes, decoded with full/meta. usage: scan_full.py habit_root parquet..."""
import json, sys
import pandas as pd
R = sys.argv[1]
hs = {json.loads(l)['task_index']: json.loads(l)['task'] for l in open(f'{R}/full/meta/human_subtasks.jsonl')}
ss = {json.loads(l)['task_index']: json.loads(l)['task'] for l in open(f'{R}/full/meta/subtasks.jsonl')}
for f in sys.argv[2:]:
    df = pd.read_parquet(f, columns=['episode_index', 'human_role_subtask_index', 'low_level_task_index'])
    def runs(col, names):
        out, prev = [], None
        for i, v in enumerate(df[col].tolist()):
            if v != prev: out.append([v, i, i]); prev = v
            else: out[-1][2] = i
        return [(names.get(v, v), round(a / 10, 1), round(b / 10 + .1, 1)) for v, a, b in out]
    print(f"\n## ep{int(df.episode_index.iloc[0])} len={len(df)/10:.1f}s")
    print('  H:', ' | '.join(f'{n} [{a}-{b}]' for n, a, b in runs('human_role_subtask_index', hs)))
    print('  R:', ' | '.join(f'{n} [{a}-{b}]' for n, a, b in runs('low_level_task_index', ss)))
