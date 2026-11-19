"""Phase 1.1 (26/QĐ-HL): manifest audit + paired success analysis over all validation rollouts (no new runs).
    ./run.sh -m tools.audit_phase1 --out outputs/diag/phase1_1120"""
import argparse, glob, json, itertools
from collections import defaultdict, Counter
from pathlib import Path

DIRS = ['outputs/eval_full_pod/eval_full_2026-11-19', 'outputs/eval_full_pod2/eval_full_b']
ROOT = Path(__file__).resolve().parents[1]


def load():
    rows = []
    for d in DIRS:
        for f in sorted(glob.glob(str(ROOT / d / '**/episodes.jsonl'), recursive=True)):
            for l in open(f):
                r = json.loads(l)
                r['_file'] = str(Path(f).relative_to(ROOT))
                r['_traj'] = None
                rows.append(r)
    return rows


def main():
    a = argparse.ArgumentParser(); a.add_argument('--out', type=Path, required=True); a = a.parse_args()
    rows = load()
    for r in rows:
        r['cond'] = f"{r['task']}_{r['variant']}"
        r['mdl'] = r['model']
    out = dict(n_rows=len(rows))
    out['models'] = Counter(r['mdl'] for r in rows)
    out['forecast_condition'] = Counter(r.get('forecast_condition') for r in rows)
    out['n_action_steps'] = Counter(r['n_action_steps'] for r in rows)
    out['steps'] = Counter((r['mdl'], r['step']) for r in rows).__len__()
    key = defaultdict(list)
    for r in rows:
        key[(r['mdl'], r['cond'], r['seed'])].append(r)
    dup = {k: len(v) for k, v in key.items() if len(v) > 1}
    out['duplicate_keys'] = {str(k): v for k, v in dup.items()}
    conds = sorted({r['cond'] for r in rows})
    models = sorted({r['mdl'] for r in rows})
    tab = {}
    for c in conds:
        tab[c] = {}
        for m in models:
            ss = sorted({r['seed'] for r in rows if r['cond'] == c and r['mdl'] == m})
            tab[c][m] = dict(n=len(ss), seed_min=ss[0] if ss else None, seed_max=ss[-1] if ss else None,
                             n_succ=sum(1 for r in rows if r['cond'] == c and r['mdl'] == m and r['success']))
        seedsets = {m: {r['seed'] for r in rows if r['cond'] == c and r['mdl'] == m} for m in models}
        tab[c]['same_seeds_across_models'] = len({frozenset(v) for v in seedsets.values()}) == 1
    out['manifest'] = tab
    out['seed_range_ok'] = all(300800 <= r['seed'] <= 300899 for r in rows)
    # paired: per cond per model-pair
    pairs = {}
    for c in conds:
        pairs[c] = {}
        for m1, m2 in itertools.combinations(models, 2):
            s = sorted({r['seed'] for r in rows if r['cond'] == c and r['mdl'] == m1} & {r['seed'] for r in rows if r['cond'] == c and r['mdl'] == m2})
            g = lambda m, sd: next(r['success'] for r in rows if r['cond'] == c and r['mdl'] == m and r['seed'] == sd)
            both = sum(1 for sd in s if g(m1, sd) and g(m2, sd)); neither = sum(1 for sd in s if not g(m1, sd) and not g(m2, sd))
            o1 = sum(1 for sd in s if g(m1, sd) and not g(m2, sd)); o2 = sum(1 for sd in s if g(m2, sd) and not g(m1, sd))
            # exact McNemar (two-sided binomial on discordant)
            from math import comb
            n = o1 + o2
            p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(0, min(o1, o2) + 1)) / 2 ** n)
            pairs[c][f'{m1}|{m2}'] = dict(n=len(s), both=both, neither=neither, only_first=o1, only_second=o2, mcnemar_p=round(p, 4))
    out['paired'] = pairs
    # failure reasons per model/cond
    fr = defaultdict(Counter)
    for r in rows:
        if not r['success']:
            fr[f"{r['mdl']}|{r['cond']}"][r.get('failure_reason')] += 1
    out['failure_reasons'] = {k: dict(v) for k, v in fr.items()}
    # trajectories present?
    miss = 0
    for r in rows:
        d = ROOT / Path(r['_file']).parent / 'trajectories' / f"{r['task']}_{r['variant']}_{r['seed']}_true.npz"
        r['has_traj'] = d.exists()
        miss += (not d.exists())
    out['rows_without_trajectory_npz'] = miss
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'manifest_paired.json').write_text(json.dumps(out, indent=1, default=str))
    print('rows', len(rows), dict(out['models']), 'fc', dict(out['forecast_condition']), 'nas', dict(out['n_action_steps']))
    print('dups', len(dup), 'seed in val range', out['seed_range_ok'], 'rows w/o traj npz', miss)
    for c in conds:
        t = tab[c]
        print(c.ljust(26), ' '.join(f"{m}:{t[m]['n']}[{t[m]['seed_min']}-{t[m]['seed_max']}] {t[m]['n_succ']}" for m in models), 'same' if t['same_seeds_across_models'] else 'DIFF')


main()
