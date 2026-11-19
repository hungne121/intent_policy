"""Event-based TSync of 18/QĐ-KT (contract locked 19/11/2026 21:13; unchanged logic, moved out of tools/hribench_table.py so it can be unit-tested)."""
import math
import numpy as np


def tsync_event(rec, u, tau):
    m = rec.get('metrics') or {}
    req = m.get('steps_required') or []
    done = {s: t for s, t in (m.get('steps_done') or [])}
    need = [s for s in req if not (s in done and done[s] <= 0.1)]
    if not need:
        return None, 0
    by = {q.get('step'): q for q in (m.get('rounds') or [])}
    sc, measurable = [], 0
    for s in need:
        q = by.get(s)
        tr, th = (q or {}).get('t_R_ready'), (q or {}).get('t_H_ready')
        if tr is None or th is None:
            sc.append(0.0)
        else:
            measurable += 1
            sc.append(math.exp(-max(tr - th - u, 0.0) / tau))
    return float(np.mean(sc)), measurable
