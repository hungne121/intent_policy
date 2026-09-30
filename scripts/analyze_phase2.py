"""Phase-2 analysis: No Information (masked) vs Oracle, corruptions and timing dose-response.

Reads outputs/eval/phase2/<noinfo|oracle>_s<seed>/<condition>/results.json (per-episode metrics) and
reports, per split / scenario / role / overall and per metric:
  * mean with a 95% cluster-bootstrap CI (training seeds, then episodes);
  * the paired Oracle - NoInfo difference (same training-seed index and evaluation seed) with a
    bootstrap CI and a paired test (McNemar exact for binary metrics, Wilcoxon signed-rank otherwise),
    Holm-corrected over the primary metrics x scenarios family;
  * every oracle corruption against the correct oracle (paired on the same checkpoints and seeds);
  * lead / delay dose-response (Spearman trend) for the timing metrics.
The pre-registered pass criteria of the protocol decide PASS / FAIL. IR alone never counts as gain.

  ./run.sh -m scripts.analyze_phase2 --output docs/reports/phase2_results
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
from intent_policy.benchmark.metrics import ALL_METRICS, METRIC_CLASSES
from intent_policy.utils import load_yaml, resolve

BINARY = {'CSR', 'CFR', 'HCS', 'WCR', 'DSR'}
DOSE = [('lead_200ms', -0.2), ('lead_100ms', -0.1), ('correct', 0.0), ('delay_100ms', 0.1), ('delay_200ms', 0.2),
        ('delay_500ms', 0.5)]
B = 2000


def load_runs(eval_root: Path) -> dict:
    """{(policy, condition): {seed: {(split, scenario): [episode metric dicts]}}}"""
    runs = defaultdict(dict)
    for res in eval_root.glob('*_s*/*/results.json'):
        policy, seed = res.parent.parent.name.rsplit('_s', 1)
        data = json.loads(res.read_text())
        runs[(policy, res.parent.name)][int(seed)] = {
            (split, sid): block['episode_metrics'] for split, s in data['splits'].items() for sid, block in s['by_scenario'].items()}
    return runs


def roles(eval_root: Path) -> dict:
    for res in eval_root.glob('*_s*/*/results.json'):
        data = json.loads(res.read_text())
        return {sid: block['role'] for s in data['splits'].values() for sid, block in s['by_scenario'].items()}
    return {}


def values(run: dict, metric: str, cells) -> dict:
    """seed -> list of (episode position key, value) with None/n-a removed."""
    out = {}
    for seed, blocks in run.items():
        vals = []
        for cell in cells:
            for i, m in enumerate(blocks.get(cell, [])):
                v = m.get(metric)
                if v not in (None, 'n/a'):
                    vals.append(((cell, i), float(v)))
        out[seed] = vals
    return out


def cluster_boot(per_seed: dict, rng, fn=np.mean):
    seeds = [s for s, v in per_seed.items() if v]
    if not seeds:
        return None, (None, None)
    point = fn(np.concatenate([np.array([x for _, x in per_seed[s]]) for s in seeds]))
    boots = []
    for _ in range(B):
        pick = rng.choice(seeds, len(seeds))
        sample = np.concatenate([rng.choice(np.array([x for _, x in per_seed[s]]), len(per_seed[s])) for s in pick])
        boots.append(fn(sample))
    return float(point), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def paired(a: dict, b: dict) -> dict:
    """Per seed: list of (a_value, b_value) for episodes present in both (same eval seed)."""
    out = {}
    for seed in set(a) & set(b):
        bm = dict(b[seed])
        out[seed] = [(x, bm[k]) for k, x in a[seed] if k in bm]
    return out


def compare(a: dict, b: dict, metric: str, rng) -> dict | None:
    """b - a (e.g. oracle - noinfo) with cluster-bootstrap CI and a paired test."""
    pr = paired(a, b)
    pairs = [p for v in pr.values() for p in v]
    if len(pairs) < 3:
        return None
    diffs = {s: [(i, y - x) for i, (x, y) in enumerate(v)] for s, v in pr.items()}
    d, ci = cluster_boot(diffs, rng)
    x, y = np.array(pairs).T
    if metric in BINARY:
        n01, n10 = int(((x == 0) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
        p = float(stats.binomtest(n01, n01 + n10).pvalue) if n01 + n10 else 1.0
        test = f'McNemar exact ({n01} vs {n10} discordant)'
    else:
        nz = y - x
        p = float(stats.wilcoxon(nz).pvalue) if np.any(nz != 0) else 1.0
        test = 'Wilcoxon signed-rank'
    return dict(diff=d, ci=ci, p=p, n=len(pairs), test=test)


def better(metric: str, diff: float) -> bool:
    return (diff > 0) == METRIC_CLASSES[metric].higher_is_better


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run_max = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run_max = max(run_max, min(1.0, (m - i) * p))
        out[k] = run_max
    return out


def fmt_ci(v, ci, digits=3):
    if v is None:
        return '—'
    return f'{v:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--eval-root', default='outputs/eval/phase2')
    p.add_argument('--protocol', default='configs/benchmark/phase2_protocol_v1.yaml')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    rng = np.random.default_rng(0)
    eval_root = resolve(args.eval_root)
    protocol = load_yaml(args.protocol)
    crit = protocol['pass_criteria']
    runs = load_runs(eval_root)
    role_of = roles(eval_root)
    noinfo, oracle = runs.get(('noinfo', 'none'), {}), runs.get(('oracle', 'correct'), {})
    cells_all = sorted({c for r in runs.values() for blocks in r.values() for c in blocks})
    groups = {f'{split}/{sid}': [(split, sid)] for split, sid in cells_all}
    for split in dict.fromkeys(s for s, _ in cells_all):
        for role in dict.fromkeys(role_of.values()):
            cells = [c for c in cells_all if c[0] == split and role_of.get(c[1]) == role]
            if cells:
                groups[f'{split}/role:{role}'] = cells
        groups[f'{split}/overall'] = [c for c in cells_all if c[0] == split]

    main_rows, pvals = [], {}
    for g, cells in groups.items():
        for metric in ALL_METRICS:
            a, b = values(noinfo, metric, cells), values(oracle, metric, cells)
            if not any(a.values()) or not any(b.values()):
                continue
            ma, ca = cluster_boot(a, rng)
            mb, cb = cluster_boot(b, rng)
            cmp = compare(a, b, metric, rng)
            row = dict(group=g, metric=metric, noinfo=ma, noinfo_ci=ca, oracle=mb, oracle_ci=cb, cmp=cmp)
            main_rows.append(row)
            if cmp and metric in crit['primary_metrics'] and '/overall' not in g and 'role:' not in g:
                pvals[(g, metric)] = cmp['p']
    adj = holm(pvals)
    for row in main_rows:
        row['p_holm'] = adj.get((row['group'], row['metric']))

    gains = [r for r in main_rows if r['p_holm'] is not None and r['p_holm'] < crit['alpha'] and r['cmp']
             and better(r['metric'], r['cmp']['diff']) and not (crit.get('ir_alone_is_not_gain') and r['metric'] == 'IR')]
    safety_worse = [r for r in main_rows if r['metric'] in crit['safety_metrics'] and r['cmp'] and r['cmp']['p'] < crit['alpha']
                    and not better(r['metric'], r['cmp']['diff'])]

    # corruption: every oracle condition vs correct oracle, on the gain cells (or all primary cells)
    corr_rows = []
    targets = gains or [r for r in main_rows if r['metric'] in crit['primary_metrics'] and '/overall' in r['group']]
    for r in targets:
        cells = groups[r['group']]
        base = values(oracle, r['metric'], cells)
        for (pol, cond), run in sorted(runs.items()):
            if pol != 'oracle' or cond == 'correct':
                continue
            v = values(run, r['metric'], cells)
            mv, cv = cluster_boot(v, rng)
            cmp = compare(base, v, r['metric'], rng)
            reduces = bool(cmp and cmp['p'] < crit['alpha'] and not better(r['metric'], cmp['diff']))
            corr_rows.append(dict(group=r['group'], metric=r['metric'], condition=cond, value=mv, ci=cv, vs_correct=cmp,
                                  reduces_benefit=reduces))

    dose_rows = []
    for g in [k for k in groups if '/overall' in k or k.endswith('/T2') or k.endswith('/T3') or k.endswith('/T5')]:
        for metric in ('AM', 'CT', 'CSR', 'Rsp', 'CRsp', 'WCR'):
            pts = []
            for cond, x in DOSE:
                run = runs.get(('oracle', cond))
                if not run:
                    continue
                m, ci = cluster_boot(values(run, metric, groups[g]), rng)
                if m is not None:
                    pts.append((x, m, ci, cond))
            if len(pts) >= 3:
                rho, pv = stats.spearmanr([q[0] for q in pts], [q[1] for q in pts])
                dose_rows.append(dict(group=g, metric=metric, points=pts, spearman=float(rho), p=float(pv)))

    gain_keys = {(r['group'], r['metric']) for r in gains}
    wrong_shuffle_reduce = {(c['group'], c['metric']) for c in corr_rows if c['condition'] in ('wrong', 'shuffled') and c['reduces_benefit']}
    corrupted_reduces = bool(gain_keys) and all(k in wrong_shuffle_reduce for k in gain_keys)
    passed = bool(gains) and not safety_worse and corrupted_reduces
    summary = dict(protocol=protocol['protocol'], seeds=dict(noinfo=sorted(noinfo), oracle=sorted(oracle)),
                   conditions=sorted(c for p_, c in runs if p_ == 'oracle'), passed=passed,
                   gains=[dict(group=r['group'], metric=r['metric'], diff=r['cmp']['diff'], ci=r['cmp']['ci'], p_holm=r['p_holm']) for r in gains],
                   safety_worse=[dict(group=r['group'], metric=r['metric'], diff=r['cmp']['diff']) for r in safety_worse],
                   corrupted_reduces_benefit=corrupted_reduces)
    out = resolve(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'analysis.json').write_text(json.dumps(dict(summary=summary, main=main_rows, corruption=corr_rows, dose=dose_rows),
                                                  indent=2, default=str))

    lines = [f"# Phase 2 analysis ({protocol['protocol']})", '',
             f"No-Information seeds {summary['seeds']['noinfo']}, Oracle seeds {summary['seeds']['oracle']}; oracle conditions: "
             f"{', '.join(summary['conditions'])}", '',
             f"**Pre-registered criteria: {'PASS' if passed else 'FAIL'}** — significant downstream gains (Holm, alpha "
             f"{crit['alpha']}): {len(gains)}; safety worse: {len(safety_worse)}; wrong/shuffled reduce every gain: {corrupted_reduces}", '',
             '## Oracle vs No Information', '',
             '| group | metric | No-Info mean [95% CI] | Oracle mean [95% CI] | Oracle − No-Info [95% CI] | n pairs | p | p (Holm) |',
             '|---|---|---|---|---|---|---|---|']
    for r in main_rows:
        c = r['cmp']
        lines.append(f"| {r['group']} | {r['metric']} | {fmt_ci(r['noinfo'], r['noinfo_ci'])} | {fmt_ci(r['oracle'], r['oracle_ci'])} | "
                     + (f"{fmt_ci(c['diff'], c['ci'])} | {c['n']} | {c['p']:.3g} | " if c else '— | — | — | ')
                     + (f"{r['p_holm']:.3g} |" if r['p_holm'] is not None else '— |'))
    lines += ['', '## Corrupted oracle vs correct oracle', '',
              '| group | metric | condition | mean [95% CI] | − correct [95% CI] | p | reduces benefit |', '|---|---|---|---|---|---|---|']
    for c in corr_rows:
        v = c['vs_correct']
        lines.append(f"| {c['group']} | {c['metric']} | {c['condition']} | {fmt_ci(c['value'], c['ci'])} | "
                     + (f"{fmt_ci(v['diff'], v['ci'])} | {v['p']:.3g} | " if v else '— | — | ') + f"{c['reduces_benefit']} |")
    lines += ['', '## Timing dose-response (oracle information lead < 0 < delay)', '',
              '| group | metric | ' + ' | '.join(f'{x:+.1f} s' for _, x in DOSE) + ' | Spearman ρ (p) |', '|---|---|' + '---|' * (len(DOSE) + 1)]
    for d in dose_rows:
        by_x = {q[0]: q[1] for q in d['points']}
        lines.append(f"| {d['group']} | {d['metric']} | " + ' | '.join(f'{by_x[x]:.3f}' if x in by_x else '—' for _, x in DOSE)
                     + f" | {d['spearman']:.2f} ({d['p']:.2g}) |")
    (out / 'analysis.md').write_text('\n'.join(lines) + '\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for d in [x for x in dose_rows if x['metric'] in ('AM', 'CT', 'CSR') and '/overall' in x['group']]:
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            xs = [q[0] for q in d['points']]
            ax.errorbar(xs, [q[1] for q in d['points']], yerr=[[q[1] - q[2][0] for q in d['points']], [q[2][1] - q[1] for q in d['points']]],
                        marker='o', capsize=3)
            base = next((r for r in main_rows if r['group'] == d['group'] and r['metric'] == d['metric']), None)
            if base and base['noinfo'] is not None:
                ax.axhline(base['noinfo'], color='gray', ls='--', label='No Information')
                ax.legend(frameon=False)
            ax.set_xlabel('oracle timing shift [s] (lead < 0 < delay)')
            ax.set_ylabel(d['metric'])
            ax.set_title(d['group'], fontsize=9)
            fig.tight_layout()
            fig.savefig(out / f"dose_{d['group'].replace('/', '_')}_{d['metric']}.png", dpi=130)
            plt.close(fig)
    except Exception as e:           # plots are optional
        print('plotting skipped:', e)
    print('\n'.join(lines[:8]))


if __name__ == '__main__':
    main()
