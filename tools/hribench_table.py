"""HRIBench-style comparison table (17/QĐ-KT, 18/QĐ-KT). EVALUATOR hribench_table 1.0 (19/11/2026).

    ./run.sh -m tools.hribench_table <eval dir> [<eval dir> ...] [--md out.md] [--json out.json]

Reads every episodes.jsonl below the directories (tools.eval_policy output), groups by model / task / variant.
Verification status (17/QĐ-KT): the formulas of HRIBench (arXiv:2607.13056, v1) are NOT published (Appendix A is missing from the arXiv PDF and
HTML, no official evaluator found). Therefore:
  * CSR, CT, Rsp, OC, CFR, DSR: names match the paper, formulas are the project's (intent_policy/benchmark/metrics.py episode_metrics); labelled UNVERIFIED.
  * TSync: "HRIBench-inspired, event-based implementation (MAD-TN grounded)", contract of 18/QĐ-KT (locked 19/11/2026 21:13 before any TSync value of
    M0 / M1-noH / M* was seen): per required step s (steps_required minus steps satisfied at t <= 0.1 s) the round of that step gives
    Delta = t_R_ready - t_H_ready, e = max(Delta - u, 0) (l = -inf), score = exp(-e / tau); a step without a round or without both times scores 0;
    episode TSync = mean over the steps; a task with no measurable round in any episode is N/A. Main u = 0.5 s, tau = 1.0 s (agent-set numbers, no
    literature value); sensitivity u {0.25, 0.5, 1.0} x tau {0.5, 1.0, 2.0} always reported.
  * HCS (the project's N_safe / N differs from the paper's Human Contact Safety), IR (not implemented), CIR (no contradictory instructions in these
    tasks): N/A, never 0.
Timeouts and failures stay in CSR / CFR; metrics not applicable are n/a (17/QĐ-KT 6)."""
import argparse, glob, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

from intent_policy.benchmark import metrics as M

VERSION = 'hribench_table 1.0 (2026-11-19)'
U_MAIN, TAU_MAIN = 0.5, 1.0
GRID = [(u, t) for u in (0.25, 0.5, 1.0) for t in (0.5, 1.0, 2.0)]
rng = np.random.default_rng(0)


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


def wilson(k, n):
    if n == 0:
        return None
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def boot(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return None
    b = [rng.choice(v, len(v)).mean() for _ in range(1000)]
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def fmt(x, d=2):
    return 'n/a' if x is None else f'{x:.{d}f}'


def cell(v, ci, d=2):
    return 'n/a' if v is None else (f'{v:.{d}f} [{ci[0]:.{d}f}, {ci[1]:.{d}f}]' if ci else f'{v:.{d}f}')


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('dirs', nargs='+', type=Path); ap.add_argument('--md', type=Path); ap.add_argument('--json', type=Path)
a = ap.parse_args()
recs, seen = [], set()
for d in a.dirs:
    for f in sorted(glob.glob(str(d / '**/episodes.jsonl'), recursive=True)):
        for l in open(f):
            r = json.loads(l)
            key = (r.get('model'), r.get('task'), r.get('variant'), r.get('seed'), r.get('forecast_condition', 'true'))
            if key in seen or r.get('forecast_condition', 'true') != 'true':
                continue
            seen.add(key); recs.append(r)
bad = [r for r in recs if not 300800 <= int(r['seed']) <= 300899]
if bad:
    raise SystemExit(f'{len(bad)} episodes outside the validation seeds 300800-300899 (test / train seeds refused)')
models = sorted({r['model'] for r in recs})
dets = sorted({(r.get('metrics') or {}).get('detector_version') for r in recs} - {None})
rows = defaultdict(list)
for r in recs:
    rows[(r['model'], r['task'], r['variant'])].append(r)
conds = sorted({(k[1], k[2]) for k in rows})
tsync_ok = {c: any(tsync_event(r, U_MAIN, TAU_MAIN)[1] > 0 for k, rs in rows.items() if (k[1], k[2]) == c for r in rs) for c in conds}
out = {}
L = [f'# HRIBench-style comparison (evaluator {VERSION})', '',
     f'Models: {", ".join(models)}. Episodes: {len(recs)}. Seeds (validation): {min(int(r["seed"]) for r in recs)}-{max(int(r["seed"]) for r in recs)}. Detector: {", ".join(dets)}.', '',
     '**Status:** HRIBench (arXiv:2607.13056 v1) publishes no formulas (Appendix A absent, no evaluator). CSR, CT, Rsp, OC, CFR, DSR = project definitions under the paper\'s names, UNVERIFIED. '
     'TSync = HRIBench-inspired, event-based implementation (MAD-TN grounded), contract 18/QĐ-KT (u = 0.5 s, tau = 1.0 s agent-set). HCS, IR, CIR = N/A. Not an official HRIBench result; validation seeds only.', '']
hdr = '| condition | model | n | CSR | CT s | Rsp s | OC | CFR | DSR | TSync-inspired | HCS | IR | CIR |'
L += ['## Per condition (95 % CI: Wilson for rates, bootstrap over episodes for means)', '', hdr, '|' + '---|' * 13]
macro = defaultdict(lambda: defaultdict(list))
for c in conds:
    for m in models:
        rs = rows.get((m, c[0], c[1]))
        if not rs:
            continue
        em = [M.episode_metrics(r)['metrics'] for r in rs]
        n = len(rs)
        k = sum(bool(r['success']) for r in rs)
        csr = k / n
        cts = [e['CT']['value'] for e in em if e['CT']['den']]
        rsp = [e['Rsp']['value'] for e in em if e['Rsp']['den']]
        ocn, ocd = sum(e['OC']['num'] for e in em), sum(e['OC']['den'] for e in em)
        cf = [e['CFR']['value'] for e in em if e['CFR']['den']]
        dn, dd = sum(e['DSR']['num'] for e in em), sum(e['DSR']['den'] for e in em)
        ts = [tsync_event(r, U_MAIN, TAU_MAIN)[0] for r in rs]; ts = [x for x in ts if x is not None]
        tsv = float(np.mean(ts)) if (ts and tsync_ok[c]) else None
        vals = dict(CSR=csr, CT=float(np.mean(cts)) if cts else None, Rsp=float(np.mean(rsp)) if rsp else None, OC=ocn / ocd if ocd else None,
                    CFR=float(np.mean(cf)) if cf else None, DSR=dn / dd if dd else None, TSync=tsv)
        out[f'{m}|{c[0]}|{c[1]}'] = dict(n=n, **vals, n_success=k, n_rsp=len(rsp), n_dsr=dd, n_tsync=len(ts))
        for key, v in vals.items():
            if v is not None:
                macro[m][key].append(v)
        L.append(f"| {c[0]} {c[1]} | {m} | {n} | {cell(csr, wilson(k, n))} | {cell(vals['CT'], boot(cts), 1)} (n={len(cts)}) | {cell(vals['Rsp'], boot(rsp), 2)} (n={len(rsp)}) | "
                 f"{cell(vals['OC'], None)} | {cell(vals['CFR'], wilson(sum(cf), len(cf)) if cf else None)} | {cell(vals['DSR'], wilson(dn, dd) if dd else None)} (n={dd}) | "
                 f"{cell(tsv, boot(ts) if tsync_ok[c] else None)} | n/a | n/a | n/a |")
L += ['', '## Macro mean over conditions (each condition one vote; n/a conditions skipped)', '', '| model | conditions | CSR | CT s | Rsp s | OC | CFR | DSR | TSync-inspired |', '|' + '---|' * 9]
for m in models:
    g = macro[m]
    L.append(f"| {m} | {len(g['CSR'])} | " + ' | '.join(f"{fmt(float(np.mean(g[x])), 2) if g[x] else 'n/a'} (n={len(g[x])})" for x in ('CSR', 'CT', 'Rsp', 'OC', 'CFR', 'DSR', 'TSync')) + ' |')
    out[f'{m}|macro'] = {x: (float(np.mean(g[x])) if g[x] else None) for x in ('CSR', 'CT', 'Rsp', 'OC', 'CFR', 'DSR', 'TSync')}
L += ['', '## TSync-inspired sensitivity (mean over measurable conditions, u x tau)', '', '| model | ' + ' | '.join(f'u={u}, tau={t}' for u, t in GRID) + ' |', '|' + '---|' * (len(GRID) + 1)]
for m in models:
    cells = []
    for u, t in GRID:
        per = []
        for c in conds:
            if not tsync_ok[c]:
                continue
            v = [tsync_event(r, u, t)[0] for r in rows.get((m, c[0], c[1]), [])]; v = [x for x in v if x is not None]
            if v:
                per.append(float(np.mean(v)))
        cells.append(fmt(float(np.mean(per)), 3) if per else 'n/a')
    L.append(f'| {m} | ' + ' | '.join(cells) + ' |')
L += ['', f'Conditions where TSync is N/A (no measurable round in any episode): {", ".join(f"{c[0]} {c[1]}" for c in conds if not tsync_ok[c]) or "none"}.']
txt = '\n'.join(L) + '\n'
print(txt)
if a.md: a.md.write_text(txt)
if a.json: a.json.write_text(json.dumps(dict(evaluator=VERSION, results=out), indent=1))
