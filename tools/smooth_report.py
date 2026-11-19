"""23/QĐ-HL: smoothness / early-completion / change-of-mind report from tools.eval_policy runs (episodes.jsonl + trajectories).
    ./run.sh -m tools.smooth_report --run M-star=<dir> --run M1-noH=<dir> ... [--cond true] [--md out.md]
Per (model, task, variant): success, CT, useful onset, response time (t_response - t_trigger), AM, robot idle time, TCP path, RMS jerk,
min robot-person distance, distance violations, contacts; change-of-mind / fault: disrupted, t_disruption, t_recovery, recovered."""
import argparse, glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('--run', action='append', required=True, help='label=dir'); ap.add_argument('--cond', default='true'); ap.add_argument('--md', type=Path)
ap.add_argument('--ref', default=None, help='label of the reference for paired success / CT differences'); a = ap.parse_args()
DT = 0.05
f = lambda v, d=1: '–' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.{d}f}'
med = lambda v: None if not [x for x in v if x is not None] else float(np.median([x for x in v if x is not None]))
mean = lambda v: None if not [x for x in v if x is not None] else float(np.mean([x for x in v if x is not None]))


def load(d):
    out = {}
    for fp in glob.glob(str(Path(d) / '**/episodes.jsonl'), recursive=True):
        if '/forecast_' in fp and f'/forecast_{a.cond}/' not in fp:
            continue
        for l in open(fp):
            r = json.loads(l)
            if r.get('forecast_condition', a.cond) != a.cond:
                continue
            out[(r['task'], r['variant'], r['seed'], r.get('fault', {}).get('fault_type') if r.get('fault') else r.get('fault_type'))] = r
    return out


def traj(d, r):
    for pat in (f"{r['task']}_{r['variant']}_{r['seed']}_{a.cond}.npz", f"{r['task']}_{r['variant']}_{r['seed']}.npz"):
        g = glob.glob(str(Path(d) / f'**/trajectories/{pat}'), recursive=True)
        if g:
            return np.load(g[0], allow_pickle=True)
    return None


def kin(z, r):
    if z is None:
        return dict(idle=None, path=None, jerk=None)
    t, tcp = z['t'], np.asarray(z['tcp'], float)
    end = r['ct_s'] if r['success'] and r.get('ct_s') else float(t[-1])
    k = t <= end
    p = tcp[k]
    v = np.diff(p, axis=0) / DT
    sp = np.linalg.norm(v, axis=1)
    jerk = np.diff(p, n=3, axis=0) / DT ** 3
    return dict(idle=float((sp < 0.02).sum() * DT), path=float(sp.sum() * DT), jerk=float(np.sqrt((np.linalg.norm(jerk, axis=1) ** 2).mean())) if len(jerk) else None)


def mt(r):
    m = r['metrics']
    an = [q for q in m.get('anticipation') or [] if q.get('t_useful_onset') is not None]
    rs = [q['t_response'] - q['t_trigger'] for q in m.get('rounds') or [] if q.get('t_response') is not None and q.get('t_trigger') is not None]
    return dict(useful=an[0]['t_useful_onset'] if an else None, am=mean([q.get('AM') for q in an]), resp=med(rs),
                viol=bool(m.get('safety_distance_violation')), contact=bool(m.get('human_robot_contact')),
                disrupted=m.get('disrupted'), t_dis=m.get('t_disruption'), t_rec=m.get('t_recovery'), recovered=m.get('recovered'))


runs = {}
for x in a.run:
    lab, d = x.split('=', 1)
    runs[lab] = (d, load(d))
keys = sorted({k[:2] + (k[3],) for _, (d, R) in runs.items() for k in R}, key=str)
L = ['| task / variant / fault | model | n | success | CT s (mean of successes) | useful onset s | response s | AM s | idle s | path m | RMS jerk m/s³ | min dist m | dist. violations | contacts | disrupted | recovered | t_recovery − t_disruption s |', '|' + '---|' * 17]
for key in keys:
    for lab, (d, R) in runs.items():
        eps = [r for k, r in R.items() if (k[0], k[1], k[3]) == key]
        if not eps:
            continue
        ks = [(r, kin(traj(d, r), r), mt(r)) for r in eps]
        succ = [r for r, _, _ in ks if r['success']]
        rec = [(m['t_rec'] - m['t_dis']) for _, _, m in ks if m['t_rec'] is not None and m['t_dis'] is not None]
        L.append(f"| {key[0]} {key[1]} {key[2] or ''} | {lab} | {len(eps)} | {len(succ)}/{len(eps)} | {f(mean([r['ct_s'] for r in succ]))} | {f(med([m['useful'] for _, _, m in ks]))} | {f(med([m['resp'] for _, _, m in ks]), 2)} | {f(med([m['am'] for _, _, m in ks]))} | "
                 f"{f(med([k['idle'] for _, k, _ in ks]))} | {f(med([k['path'] for _, k, _ in ks]), 2)} | {f(med([k['jerk'] for _, k, _ in ks]), 1)} | {f(min([r['min_robot_person_m'] for r in eps if r['min_robot_person_m'] is not None], default=None), 3)} | "
                 f"{sum(m['viol'] for _, _, m in ks)}/{len(eps)} | {sum(m['contact'] for _, _, m in ks)}/{len(eps)} | {sum(bool(m['disrupted']) for _, _, m in ks)}/{len(eps)} | {sum(bool(m['recovered']) for _, _, m in ks)}/{len(eps)} | {f(med(rec), 2)} |")
txt = '\n'.join(L) + '\n'
if a.ref:
    txt += f'\n## Paired against {a.ref} (same task / variant / seed / fault)\n\n| task / variant | model | pairs | success {a.ref} | success model | CT difference (both succeed) s | seeds model-only success | seeds ref-only success |\n|---|---|---|---|---|---|---|---|\n'
    ref = runs[a.ref][1]
    for key in keys:
        for lab, (d, R) in runs.items():
            if lab == a.ref:
                continue
            ps = [(ref[k], R[k]) for k in R if k in ref and (k[0], k[1], k[3]) == key]
            if not ps:
                continue
            both = [(m['ct_s'] - r['ct_s']) for r, m in ps if r['success'] and m['success']]
            txt += f"| {key[0]} {key[1]} {key[2] or ''} | {lab} | {len(ps)} | {sum(r['success'] for r, _ in ps)} | {sum(m['success'] for _, m in ps)} | {f(med(both), 2)} (n={len(both)}) | {sum(m['success'] and not r['success'] for r, m in ps)} | {sum(r['success'] and not m['success'] for r, m in ps)} |\n"
print(txt)
if a.md:
    a.md.write_text(txt)
