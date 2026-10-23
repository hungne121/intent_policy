"""Per task: how much each robot arm is used. Files named <sid>_episode_*.parquet.
closed = gripper > 0.5; moving = TCP speed > 0.02 m/s (10 Hz backward difference).
usage: arm_usage.py parquet..."""
import sys, os, collections
import numpy as np, pandas as pd
rows = collections.defaultdict(list)
for f in sys.argv[1:]:
    sid = os.path.basename(f).split('_')[0]
    df = pd.read_parquet(f)
    g = [np.stack(df[f'robot{a}.observation.state.gripper_position'].values)[:, 0] > 0.5 for a in (0, 1)]
    p = [np.stack(df[f'robot{a}.observation.state.cartesian_position'].values)[:, :3] for a in (0, 1)]
    v = [np.r_[0, np.linalg.norm(np.diff(q, axis=0), axis=1) * 10] for q in p]
    m = [x > 0.02 for x in v]
    path = [np.linalg.norm(np.diff(q, axis=0), axis=1).sum() for q in p]
    rows[sid].append(dict(n=len(df), path0=path[0], path1=path[1], c0=g[0].mean(), c1=g[1].mean(),
                          both_closed=(g[0] & g[1]).mean(), both_moving=(m[0] & m[1]).mean(),
                          any_moving=(m[0] | m[1]).mean()))
print(f"{'sid':6} {'ep':>2} {'path L/R m':>11} {'closed L/R':>10} {'both closed':>11} {'both move / any move':>21}")
for sid, r in rows.items():
    a = {k: np.mean([x[k] for x in r]) for k in r[0]}
    print(f"{sid:6} {len(r):2d} {a['path0']:5.1f}/{a['path1']:<5.1f} {a['c0']:4.2f}/{a['c1']:<5.2f} {a['both_closed']:11.2f} {a['both_moving'] / max(a['any_moving'], 1e-9):21.2f}")
