"""Classify the real `check` reasons captured by tools.audit_policy_reason (30/QĐ-HL items 4-5).
    ./run.sh -m tools.analyze_audit_reasons outputs/diag/phase3_1120/audit_policy/Mstar_after ...
"""
import collections, json, sys
from pathlib import Path
import numpy as np

STAY_MM = 10.0     # scenarios/i3.py STAY_TOL = 0.01 m (evaluator tolerance, not changed)


def classify(r):
    A, B = r['info'].get('A'), r['info'].get('B'); tgt = B or A
    fin, ini = r['final'], r['initial']
    d = lambda k: 1000 * float(np.linalg.norm(np.array(fin[k][:2]) - np.array(ini[k][:2])))
    cakes = [k for k in ('Lx', 'Ly', 'Rx', 'Ry') if k in fin]
    gr = [g[0] for g in r['grasps']]
    if r['check_ok']:
        return 'OK'
    dropped = [k for k in cakes if fin[k][2] < 0.5]
    if dropped:
        return f'cake rơi khỏi bàn ({",".join(dropped)})'
    tgt_in_tray = abs(fin[tgt][0] - 0.28) < 0.045 and abs(fin[tgt][1] - 0.08) < 0.035 and 0.63 < fin[tgt][2] < 0.66
    moved = [k for k in cakes if k != tgt and d(k) > STAY_MM]
    if not gr:
        return 'không gắp'
    if tgt_in_tray and moved:
        mm = max(d(k) for k in moved)
        return f'đích vào khay, cake khác lệch {"/".join(moved)} ({mm:.0f} mm; ngưỡng {STAY_MM:.0f})'
    if tgt_in_tray:
        return 'đích vào khay, sai trình tự gắp ' + str(gr)
    if A and A != tgt and gr and gr[-1] == A:
        return 'bỏ qua đổi ý: gắp/đặt A'
    if gr[-1] != tgt:
        return f'gắp cake sai ({gr[-1]}), đích {tgt}'
    return 'đã gắp đích nhưng không vào khay'


def main():
    for d in sys.argv[1:]:
        R = [json.loads(l) for l in open(Path(d) / 'audit_reasons.jsonl') if l.strip()]
        c = collections.Counter()
        for r in R:
            k = classify(r)
            c[k.split(' (')[0] if k.startswith('đích vào khay, cake khác') else k] += 1
        print(Path(d).name, len(R), 'OK', c.get('OK', 0))
        for k, v in c.most_common():
            if k != 'OK': print('   ', v, k)


if __name__ == '__main__':
    main()
