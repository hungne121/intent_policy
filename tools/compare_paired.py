"""21/QĐ-HL: compare two tools.eval_policy paired runs (true vs masked) of two checkpoints, same seeds.
    ./run.sh -m tools.compare_paired <dirA> <labelA> <dirB> <labelB> [--md out.md]"""
import argparse, glob, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('a', type=Path); ap.add_argument('la'); ap.add_argument('b', type=Path); ap.add_argument('lb'); ap.add_argument('--md', type=Path); x = ap.parse_args()


def load(d):
    rows = [json.loads(l) for f in sorted(glob.glob(str(d / '*/*/episodes.jsonl'))) for l in open(f)]
    by = defaultdict(dict)
    for r in rows: by[(r['task'], r['seed'])][r['forecast_condition']] = r
    return by


def tcp_div(d, by, k, c):
    r = by[k][c]
    z = np.load(next(iter(d.glob(f'*/forecast_{c}/trajectories/{r["task"]}_{r["variant"]}_{r["seed"]}_{c}.npz'))), allow_pickle=True)
    return z


def first_div(d, by, k):
    a, b = tcp_div(d, by, k, 'true'), tcp_div(d, by, k, 'masked')
    n = min(len(a['tcp']), len(b['tcp']))
    dd = np.linalg.norm(a['tcp'][:n] - b['tcp'][:n], axis=1) * 1000
    i = np.flatnonzero(dd > 20)
    return float(a['t'][i[0]]) if len(i) else None


def met(r):
    m = r['metrics']
    an = [q for q in m.get('anticipation') or [] if q.get('t_useful_onset') is not None]
    rs = [q['t_response'] - q['t_trigger'] for q in m.get('rounds') or [] if q.get('t_response') is not None and q.get('t_trigger') is not None]
    return dict(useful=an[0]['t_useful_onset'] if an else None, am=an[0].get('AM') if an else None, resp=rs[0] if rs else None,
                steps=len(m.get('steps_done') or []), viol=bool(m.get('safety_distance_violation')), contact=bool(m.get('human_robot_contact')))


A, B = load(x.a), load(x.b)
keys = sorted(set(A) & set(B))
f = lambda v, d=1: '–' if v is None else f'{v:.{d}f}'
med = lambda v: None if not [q for q in v if q is not None] else float(np.median([q for q in v if q is not None]))
tasks = sorted({k[0] for k in keys})
L = [f'| task | n | {x.la} true | {x.la} masked | {x.lb} true | {x.lb} masked | {x.la} true − masked | {x.lb} true − masked | {x.lb} − {x.la} (true) |', '|---|---|---|---|---|---|---|---|---|']
tot = [0, 0, 0, 0]
for t in tasks + ['all']:
    ks = [k for k in keys if t == 'all' or k[0] == t]
    s = lambda D, c: sum(D[k][c]['success'] for k in ks)
    v = [s(A, 'true'), s(A, 'masked'), s(B, 'true'), s(B, 'masked')]
    L.append(f'| {t} | {len(ks)} | {v[0]} | {v[1]} | {v[2]} | {v[3]} | {v[0] - v[1]:+d} | {v[2] - v[3]:+d} | {v[2] - v[0]:+d} |')
P = ['| task | seed | true ' + x.la + ' | true ' + x.lb + ' | CT s ' + x.la + ' → ' + x.lb + ' | useful onset s ' + x.la + ' → ' + x.lb + ' | response s ' + x.la + ' → ' + x.lb + ' | steps done ' + x.la + ' → ' + x.lb + ' | first executed TCP divergence true/masked s ' + x.la + ' → ' + x.lb + ' |', '|---|---|---|---|---|---|---|---|---|']
better = worse = 0
for k in keys:
    a, b = A[k]['true'], B[k]['true']
    better += (b['success'] and not a['success']); worse += (a['success'] and not b['success'])
    ma, mb = met(a), met(b)
    P.append(f"| {k[0]} | {k[1]} | {'OK' if a['success'] else 'FAIL'} | {'OK' if b['success'] else 'FAIL'} | {f(a['ct_s'])} → {f(b['ct_s'])} | {f(ma['useful'])} → {f(mb['useful'])} | {f(ma['resp'], 2)} → {f(mb['resp'], 2)} | {ma['steps']} → {mb['steps']} | {f(first_div(x.a, A, k), 2)} → {f(first_div(x.b, B, k), 2)} |")
p = 2 * sum(math.comb(better + worse, i) for i in range(0, min(better, worse) + 1)) * 0.5 ** (better + worse) if better + worse else 1.0
S = [f'Seeds where {x.lb} true succeeds and {x.la} true fails: {better}; the reverse: {worse}; exact sign test two-sided p = {p:.3f} (20 pairs).']
for lab, D in ((x.la, A), (x.lb, B)):
    tr = [met(D[k]['true']) for k in keys]
    S.append(f"{lab} true: min robot-person {min(D[k]['true']['min_robot_person_m'] or 9 for k in keys):.3f} m; safety_distance_violation in {sum(m['viol'] for m in tr)}/20; human_robot_contact in {sum(m['contact'] for m in tr)}/20; physics_valid {sum(bool(D[k]['true']['physics_valid']) for k in keys)}/20; "
             f"median CT of successes {f(med([D[k]['true']['ct_s'] for k in keys if D[k]['true']['success']]))} s; median first executed TCP divergence true vs masked {f(med([first_div(x.a if D is A else x.b, D, k) for k in keys]), 2)} s")
txt = '## Success (n = 5 seeds per task, timeouts counted as failures)\n\n' + '\n'.join(L) + '\n\n' + '\n\n'.join(S) + '\n\n## Per seed (true condition)\n\n' + '\n'.join(P) + '\n'
print(txt)
if x.md: x.md.write_text(txt)
