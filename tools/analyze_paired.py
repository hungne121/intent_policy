"""20/QĐ-HL: paired true-vs-masked forecast closed-loop report from tools.eval_policy --forecast-condition true masked --trace.
    ./run.sh -m tools.analyze_paired <eval out dir> [--md report.md]"""
import argparse, glob, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('out', type=Path); ap.add_argument('--md', type=Path); a = ap.parse_args()
rows = [json.loads(l) for f in sorted(glob.glob(str(a.out / '*/*/episodes.jsonl'))) for l in open(f)]
by = defaultdict(dict)
for r in rows:
    by[(r['task'], r['seed'])][r['forecast_condition']] = r
tasks = sorted({k[0] for k in by})
traj = lambda r, c: np.load(next(iter(a.out.glob(f'*/forecast_{c}/trajectories/{r["task"]}_{r["variant"]}_{r["seed"]}_{c}.npz'))), allow_pickle=True)


def metrics(r):
    m = r['metrics']
    useful = [x['t_useful_onset'] for x in m.get('anticipation') or [] if x.get('t_useful_onset') is not None]
    am = [x['AM'] for x in m.get('anticipation') or [] if x.get('AM') is not None]
    resp = [x['t_response'] - x['t_trigger'] for x in m.get('rounds') or [] if x.get('t_response') is not None and x.get('t_trigger') is not None]
    return dict(useful=useful[0] if useful else None, am=am[0] if am else None, resp=resp[0] if resp else None,
                steps=len(m.get('steps_done') or []), need=len(m.get('steps_required') or []), ordered=m.get('steps_ordered'),
                viol=bool(m.get('safety_distance_violation')), contact=bool(m.get('human_robot_contact')))


def fmt(x, d=1): return '–' if x is None else f'{x:.{d}f}'
L = []
L.append('| task | seed | true: result / CT s | masked: result | steps done true / masked | useful onset s (true) | response s (true) | AM s (true) | dmin m true / masked | physics_valid true / masked | chunk diff TCP mm median (true run) |')
L.append('|---|---|---|---|---|---|---|---|---|---|---|')
n_t = n_m = 0; disc = [0, 0]; per = {t: [0, 0, 0] for t in tasks}
for (t, s), d in sorted(by.items()):
    T, M = d['true'], d['masked']
    mt, mm = metrics(T), metrics(M)
    n_t += T['success']; n_m += M['success']
    if T['success'] and not M['success']: disc[0] += 1
    if M['success'] and not T['success']: disc[1] += 1
    per[t][0] += 1; per[t][1] += T['success']; per[t][2] += M['success']
    L.append(f"| {t} | {s} | {'OK ' + fmt(T['ct_s']) if T['success'] else 'FAIL ' + str(T['failure_reason'])} | {'OK ' + fmt(M['ct_s']) if M['success'] else 'FAIL ' + str(M['failure_reason'])} | "
             f"{mt['steps']}/{mt['need']} / {mm['steps']}/{mm['need']} | {fmt(mt['useful'])} | {fmt(mt['resp'], 2)} | {fmt(mt['am'])} | {fmt(T['min_robot_person_m'], 3)} / {fmt(M['min_robot_person_m'], 3)} | "
             f"{T['physics_valid']} / {M['physics_valid']} | {fmt((T.get('intervention') or {}).get('chunk_tcp_mm_median'), 0)} |")
n = len(by)
p = 2 * sum(math.comb(sum(disc), k) for k in range(0, min(disc) + 1)) * 0.5 ** sum(disc) if sum(disc) else 1.0
S = ['| task | n | success true | success masked |', '|---|---|---|---|'] + [f'| {t} | {v[0]} | {v[1]} | {v[2]} |' for t, v in per.items()] + [f'| all | {n} | {n_t} | {n_m} |']
S.append(f'\nPaired discordant seeds: true-only success {disc[0]}, masked-only success {disc[1]}; exact sign test two-sided p = {p:.4f} (n = {n} pairs, 4 tasks pooled; timeouts counted as failures).')
# execution check
chk = []
for (t, s), d in sorted(by.items()):
    for c in ('true', 'masked'):
        z = traj(d[c], c)
        ex, ct, cm, n_act = z['chunk_executed'], z['chunk_true'], z['chunk_masked'], int(z['n_action_steps'])
        ref = ct if c == 'true' else cm
        act, ft = z['action_exec'], z['fire_tick']
        # the executed action of the n ticks after each chunk = first n steps of that chunk
        errs = []
        for k, tick in enumerate(ft):
            seg = act[tick:tick + n_act]
            errs.append(np.abs(seg - ex[k][:len(seg)]).max() if len(seg) else 0.0)
        chk.append((t, s, c, float(np.abs(ex - ref).max()), float(max(errs)) if errs else None))
E1 = max(x[3] for x in chk); E2 = max(x[4] for x in chk if x[4] is not None)
# divergence of the executed TCP between conditions (same seed)
div = []
for (t, s), d in sorted(by.items()):
    a1, a2 = traj(d['true'], 'true'), traj(d['masked'], 'masked')
    k = min(len(a1['tcp']), len(a2['tcp']))
    dd = np.linalg.norm(a1['tcp'][:k] - a2['tcp'][:k], axis=1) * 1000
    idx = np.flatnonzero(dd > 20)
    ft1, ft2 = a1['fire_tick'], a2['fire_tick']
    first_chunk_diff = float(np.linalg.norm((a1['chunk_true'][0] - a1['chunk_masked'][0])[:, :6], axis=1).max())
    div.append((t, s, None if not len(idx) else round(float(a1['t'][idx[0]]), 2), float(dd[:min(k, 100)].max()),
                float(np.median(np.abs(a1['chunk_true'] - a1['chunk_masked'])[..., :6]))))
X = ['| task | seed | first time executed TCP differs > 20 mm (s) | max TCP diff first 5 s (mm) | median joint diff of the two chunks (rad) |', '|---|---|---|---|---|']
X += [f'| {t} | {s} | {fmt(f, 2)} | {m:.0f} | {j:.3f} |' for t, s, f, m, j in div]
txt = '## Paired results\n\n' + '\n'.join(S) + '\n\n## Per seed\n\n' + '\n'.join(L) + \
      f'\n\n## Execution check\n\nmax |executed chunk − the chunk of its own condition| = {E1:.2e} (shadow forwards reproduce the executed chunk); max |executed action − first n_action_steps of that chunk| = {E2:.2e}.\n\n' + '\n'.join(X) + '\n'
print(txt)
if a.md: a.md.write_text(txt)
