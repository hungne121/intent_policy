"""Phase 1.2-1.3 (26/QĐ-HL): rule-based failure taxonomy from the stored rollouts (trajectory npz + episodes.jsonl).
Every rule uses only logged signals (gripper command, TCP, env 'robot_grasp' events, grasp_drops, idle time); no new runs.
Gripper 'close' = commanded opening falls under 0.04 after being open; 'attach' = env robot_grasp event (both fingers touched, 61/QĐ-DL).
Output: outputs/diag/phase1_1120/taxonomy.jsonl and a printed table."""
import glob, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRS = ['outputs/eval_full_pod/eval_full_2026-11-19', 'outputs/eval_full_pod2/eval_full_b']
ATTACH_WAIT = 1.6      # s: guarded close takes 1.05 s in the expert (dwell) -> attach must appear within this after a close
CAKES = {'Lx': (.11, -.10), 'Ly': (.21, -.10), 'Rx': (.35, -.10), 'Ry': (.45, -.10)}


def i3_info(seed, variant):
    from intent_policy.scenarios import i3
    return i3.build(seed, variant)[3]


def classify(r, z):
    t, a, tcp = z['t'], z['action_exec'], z['tcp']
    g = a[:, 6]
    closed = g < 0.04
    closes = [i for i in range(1, len(g)) if closed[i] and not closed[i - 1]]
    if closed[0]:
        closes = [0] + closes
    att = sorted(e['t'] for e in r['events'] if e.get('event') == 'robot_grasp')
    sp = np.linalg.norm(np.diff(tcp, axis=0), axis=1) / np.maximum(np.diff(t), 1e-6)
    late = t[1:] > t[-1] - 10
    idle_late = float((sp[late] < 0.005).mean()) if late.any() else 0.
    ok_closes, bad = 0, []
    for i in closes:
        k = i + int(np.argmax(np.linalg.norm(tcp[i:] - tcp[i], axis=1) > 0.006)) if np.linalg.norm(tcp[i:] - tcp[i], axis=1).max() > 0.006 else len(t) - 1
        hold = float(t[k] - t[i])
        has = any(t[i] <= x <= t[i] + ATTACH_WAIT + 0.3 for x in att) or any(t[i] - 0.05 <= x for x in att if x <= t[i] + 3.0)
        if has: ok_closes += 1
        else: bad.append(dict(t=float(t[i]), tcp=tcp[i].round(3).tolist(), hold=round(hold, 2)))
    d = dict(n_close=len(closes), n_attach=len(att), n_bad_close=len(bad), bad=bad[:4], idle_late=round(idle_late, 2),
             drops=r['grasp_drops'])
    if r['success']:
        d['cls'] = 'success'; return d
    task = r['task']
    if task == 'C2':
        d['cls'] = 'C2 wiping/handover (xem C2_AUDIT)'; return d
    if not closes:
        d['cls'] = 'không đóng gripper bao giờ (treo/chậm)' if idle_late > .5 else 'không đóng gripper (di chuyển liên tục, không bắt)'
    elif not att:
        early = np.mean([b['hold'] < 1.0 for b in bad]) > .5
        if len(bad) >= 3: d['cls'] = 'thiếu recovery: lặp đóng/mở không attach' + (' (nhấc sớm)' if early else '')
        else: d['cls'] = 'đóng không attach (' + ('nhấc sớm <1,0 s' if early else 'hụt vị trí') + ')'
    else:
        if r['grasp_drops']: d['cls'] = 'rơi vật sau nắm'
        elif idle_late > .6: d['cls'] = 'đã nắm, treo không hoàn tất'
        else: d['cls'] = 'đã nắm, không hoàn tất đặt/giao'
    if task == 'I3' and closes:
        info = i3_info(r['seed'], r['variant'])
        near = lambda i: min(CAKES, key=lambda c: np.hypot(tcp[i][0] - CAKES[c][0], tcp[i][1] - CAKES[c][1]))
        first = [i for i in closes if tcp[i][1] < -0.05]
        if first:
            c0 = near(first[0]); d['i3_A'] = info['A']; d['i3_B'] = info['B']; d['i3_first_close_cake'] = c0
            tgt = info['A']
            d['i3_first_target_correct'] = (c0 == tgt)
    return d


def main():
    out = open(ROOT / 'outputs/diag/phase1_1120/taxonomy.jsonl', 'w')
    tab = defaultdict(Counter)
    for dd in DIRS:
        for f in sorted(glob.glob(str(ROOT / dd / '**/episodes.jsonl'), recursive=True)):
            base = Path(f).parent
            for l in open(f):
                r = json.loads(l)
                z = np.load(base / 'trajectories' / f"{r['task']}_{r['variant']}_{r['seed']}_true.npz", allow_pickle=True)
                d = classify(r, z)
                d.update(model=r['model'], task=r['task'], variant=r['variant'], seed=r['seed'], success=r['success'])
                out.write(json.dumps(d) + '\n')
                tab[(r['task'], r['variant'], r['model'])][d['cls']] += 1
    for k in sorted(tab):
        print(k, dict(tab[k]))


main()
