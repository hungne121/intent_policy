"""Gripper-close events (arm, time, TCP xyz in that arm's frame) of HABIT episodes. usage: grasp_xy.py parquet..."""
import sys, os
import numpy as np, pandas as pd
for f in sys.argv[1:]:
    df = pd.read_parquet(f)
    ev = []
    for a in (0, 1):
        g = np.stack(df[f'robot{a}.observation.state.gripper_position'].values)[:, 0]
        p = np.stack(df[f'robot{a}.observation.state.cartesian_position'].values)[:, :3]
        closed = g > 0.5
        for i in np.flatnonzero(closed[1:] & ~closed[:-1]) + 1:
            ev.append((i / 10, a, p[i].round(3).tolist()))
    ev.sort()
    print(os.path.basename(f), ' '.join(f't={t:.1f} arm{a} {xyz}' for t, a, xyz in ev))
