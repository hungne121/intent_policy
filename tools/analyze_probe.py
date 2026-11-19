"""27/QĐ-HL probe analysis: (A) M0 checkpoint sweep 20k/40k/50k vs 60k, (B) n_action_steps 1 vs 10 for M0 / M1-noH / M*, paired on seeds 300820-300839."""
import glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1] / 'outputs'
rows = []
for f in glob.glob(str(R / 'eval_probe_pod/**/episodes.jsonl'), recursive=True):
    for l in open(f):
        r = json.loads(l); r['_dir'] = str(Path(f).parent); rows.append(r)
base = []
for d in ('eval_full_pod/eval_full_2026-11-19', 'eval_full_pod2/eval_full_b'):
    for f in glob.glob(str(R / d / '**/episodes.jsonl'), recursive=True):
        for l in open(f):
            r = json.loads(l)
            if 300820 <= r['seed'] <= 300839: r['_dir'] = str(Path(f).parent); base.append(r)
cond = lambda r: f"{r['task']}_{r['variant']}"
K = lambda r: (r['model'], r['step'], r['n_action_steps'])
S = defaultdict(dict)
for r in rows + base: S[(K(r), cond(r))][r['seed']] = r
cfgs = sorted({k for k, _ in S}); conds = sorted({c for _, c in S})
print('CSR (n seeds) per config x condition, seeds 300820-300839')
print('config'.ljust(28) + ''.join(c[:14].ljust(16) for c in conds))
for k in cfgs:
    line = str(k).ljust(28)
    for c in conds:
        d = S.get((k, c), {})
        line += (f"{sum(v['success'] for v in d.values())}/{len(d)}".ljust(16) if d else '-'.ljust(16))
    print(line)
def pair(k1, k2, c):
    a, b = S.get((k1, c), {}), S.get((k2, c), {})
    s = sorted(set(a) & set(b))
    o1 = sum(1 for x in s if a[x]['success'] and not b[x]['success']); o2 = sum(1 for x in s if b[x]['success'] and not a[x]['success'])
    from math import comb
    n = o1 + o2; p = 1. if n == 0 else min(1., 2 * sum(comb(n, i) for i in range(min(o1, o2) + 1)) / 2 ** n)
    return len(s), o1, o2, round(p, 3)
print('\nPaired n_action_steps 1 (second) vs 10 (first): n, only10, only1, p')
for m, st in (('m0', 60000), ('m1_noh', 60000), ('m_star', 60000)):
    for c in conds:
        if (( m, st, 1), c) in S: print(m, c.ljust(26), pair((m, st, 10), (m, st, 1), c))

# ---- (A) sweep: paired later checkpoint vs 60k, pooled; (B) n=1 failure reasons
C7 = ['C2_nominal', 'I2_nominal', 'I3_nominal', 'I3_switch_after_commit', 'I3_switch_before_commit', 'W1_nominal', 'W2_nominal']
print('\nM0 sweep, pooled over 7 conditions x 20 seeds:')
for st in (20000, 40000, 50000, 60000):
    k = ('m0', st, 10); print(' ', st, sum(v['success'] for c in C7 for v in S[(k, c)].values()), '/', sum(len(S[(k, c)]) for c in C7))
print('M0 paired vs 60k (only earlier, only 60k, p):')
for st in (40000, 50000):
    for c in C7:
        print(' ', st, c.ljust(26), pair(('m0', st, 10), ('m0', 60000, 10), c)[1:])
import collections
for m in ('m0', 'm1_noh', 'm_star'):
    x = [r for (k, c), d in S.items() if k == (m, 60000, 1) for r in d.values()]
    print(m, 'n=1: fail reasons', dict(collections.Counter(r['failure_reason'] for r in x)), 'policy_ms/tick', round(float(np.mean([r['policy_ms_per_tick'] for r in x])), 1), 'n', len(x))
