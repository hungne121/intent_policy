"""Benchmark report of the 8 metrics of 10/QĐ-KT from closed-loop eval outputs (tools/eval_policy.py).

HRIBench-style metrics adapted for this project; not an exact reproduction of HRIBench.

  ./run.sh tools/benchmark_report.py outputs/eval_closed_loop/<run_dir> [...] --out outputs/benchmark/<date>
Each argument is an eval output dir; every episodes.jsonl below it is one run, labelled <model>_step<N>_n<k> from its
tags (folder name otherwise). Writes results_per_episode.jsonl, results_per_task.json, results_summary.json and
REPORT.md (main table = macro over tasks, pooled secondary; split by N1/N5/N6, role group, task, case; numerator /
denominator, eligible episodes, n/a, contributing tasks, 95 % CI; paired M1 vs M0 and M1 vs M1-noH, or --pair A:B;
optional HCS sensitivity with --hcs-sensitivity; conditions nominal / forecast_fault / switch_before_commit /
switch_after_commit separately; anticipation only as a secondary table). --selection refuses final-test seeds;
--paired-validation requires M0, M1-noH, M1 on identical scenarios and n_action_steps (12/QĐ-KT E26, E29). Without --official the output is labelled DIAGNOSTIC; --official
refuses records lacking agent H's DETECTOR_VERSION / schema (11/QĐ-KT). Formulas: intent_policy/benchmark/metrics.py,
docs/reports/METRIC_SPEC_2026-11-18.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from intent_policy.benchmark import metrics as M  # noqa: E402

DEFAULT_PAIRS = (('m1', 'm0'), ('m1', 'm1_noh'))


def run_label(rec, folder):
    if rec.get('model') is not None and rec.get('step') is not None:
        return f"{rec['model']}_step{rec['step']}_n{rec.get('n_action_steps')}"
    return folder.name


def load_runs(dirs):
    runs = {}
    for d in dirs:
        d = Path(d)
        files = [d] if d.is_file() else sorted(d.rglob('episodes.jsonl'))
        for f in files:
            for line in f.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    label = run_label(rec, f.parent)
                    rec['run'] = label
                    runs.setdefault(label, []).append(rec)
    return runs


def fmt(e, kind, how):
    v = e.get(how)
    if v is None:
        return f"n/a (n/a={e['n_na']}; tasks=0; cases=0)"
    ci = e.get(f'{how}_ci95')
    s = f'{v:.3f}' + (f' [{ci[0]:.3f}, {ci[1]:.3f}]' if ci else ' [CI n/a]')
    cnt = f"{e['numerator']:g}/{e['denominator']:g}" if kind == 'ratio' else f"n={e['n_eligible']}"
    return s + f" ({cnt}; n/a={e['n_na']}; tasks={e['n_tasks_contributing']}; cases={e['n_cases_contributing']})"


def header(first):
    cols = [f"{m['name']}{' (' + m['unit'] + ')' if m['unit'] else ''} {'↓' if m['better'] == 'lower' else '↑'}"
            for m in M.SLIDE10]
    return [f'| {first} | n | n_success | ' + ' | '.join(cols) + ' |', '|' + '---|' * (3 + len(cols))]


def row(label, s, how):
    return (f"| {label} | {s['n_episodes']} | {s['n_success']} | "
            + ' | '.join(fmt(s[m], M.KIND[m], how) if m in s else 'n/a' for m in M.METRICS) + ' |')


def level_tables(agg, level, how, title):
    lines = []
    keys = sorted({k for g in agg.values() for k in g[level]})
    for k in keys:
        lines += [f'### {title} {k}', '', *header('run')]
        lines += [row(run, g[level][k], how) for run, g in agg.items() if k in g[level]]
        lines.append('')
    return lines


def paired_table(name, p) -> list[str]:
    lines = [f'### {name}', '',
             f"n_pairs = {p['n_pairs']} (scenario = task, variant, seed); only A = {p['n_only_a']}, only B = "
             f"{p['n_only_b']}; CSR on the pairs A = {p['csr_a']}, B = {p['csr_b']}.", '',
             '| metric | better | mean d = A − B | 95 % CI (paired bootstrap) | n | n/a | tasks | cases |',
             '|---|---|---|---|---|---|---|---|']
    for m in M.SLIDE10:
        e = p[m['name']]
        md = 'n/a' if e['mean_diff'] is None else f"{e['mean_diff']:.3f}"
        ci = 'n/a' if e['ci95'] is None else f"[{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]"
        lines.append(f"| {m['name']} | {m['better']} | {md} | {ci} | {e['n']} | {e['n_na']} | "
                     f"{e['n_tasks_contributing']} | {e['n_cases_contributing']} |")
    return lines + ['']


def safety_table(name, groups) -> list[str]:
    """15/QĐ-KT D21-D24 (secondary, never replaces HCS / CFR): per group of one run."""
    f = lambda e, k='pooled', nd=3: 'n/a' if e.get(k) is None else f'{e[k]:.{nd}f}'
    lines = [f'### {name}', '',
             '| group | n | audit n/a | collision | proximity (verified) | distance unverified | clear | viol. duration s '
             '(mean; n) | viol. ratio (mean; n) | min verified clearance m (min) | max deficit m (max) | contact force N '
             '(max; contact eps; force missing) | penetration m (max) | worst pair (episodes) | tasks | cases |',
             '|' + '---|' * 16]
    for k, g in groups.items():
        c, fo = g['classes'], g['contact_force_n']
        wp = '; '.join(f'{p} ({n})' for p, n in g['worst_pairs'].items()) or 'none'
        lines.append(
            f"| {k} | {g['n_episodes']} | {g['n_na_audit']} | {c['collision']} | {c['proximity_violation']} | "
            f"{c['unverified_distance']} | {c['clear']} | {f(g['violation_duration_s'], nd=2)}; n={g['violation_duration_s']['n']} | "
            f"{f(g['violation_ratio'])}; n={g['violation_ratio']['n']} | {f(g['min_verified_clearance_m'], nd=4)} | "
            f"{f(g['max_clearance_deficit_m'], nd=4)} | {f(fo, nd=1)}; {fo['n_contact_episodes']}; {fo['n_force_missing']} | "
            f"{f(g['penetration_m'], nd=4)} | {wp} | {g['n_tasks']} | {g['n_cases']} |")
    return lines + ['']


def load_variants(path):
    """HCS sensitivity variants {name: {exempt: [{pairs, phases}], max_distance_m}} from a YAML / JSON config."""
    if path is None:
        return None
    text = Path(path).read_text()
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        return json.loads(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dirs', nargs='+', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--n-boot', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0, help='bootstrap RNG seed')
    p.add_argument('--official', action='store_true',
                   help="refuse unless every record carries agent H's DETECTOR_VERSION / schema (11/QĐ-KT)")
    p.add_argument('--hcs-sensitivity', type=Path, default=None,
                   help='YAML/JSON of alternative exemption definitions, declared before evaluation (11/QĐ-KT C2)')
    p.add_argument('--selection', action='store_true',
                   help='the runs are used to SELECT a checkpoint / execution horizon / threshold: refuse test seeds '
                        '(12/QĐ-KT E29; configs/hrc/collection.yaml splits.test)')
    p.add_argument('--paired-validation', action='store_true',
                   help='12/QĐ-KT E26: require M0, M1-noH, M1 on identical scenarios with identical n_action_steps; '
                        'paired M1 vs M0, M1 vs M1-noH, M1-noH vs M0 overall and per condition')
    p.add_argument('--pair', action='append', default=None,
                   help='A:B run labels (or model names) for paired differences A − B; default m1:m0 and m1:m1_noh')
    a = p.parse_args(argv)

    runs = load_runs(a.dirs)
    if not runs:
        raise SystemExit('no episodes.jsonl found')
    recs = [r for rs in runs.values() for r in rs]
    problems = M.check_official(recs)
    if a.official and problems:
        raise SystemExit('REFUSED official output (11/QĐ-KT):\n  ' + '\n  '.join(problems[:20]))
    status = 'OFFICIAL' if a.official else 'DIAGNOSTIC (not an official benchmark result)'
    rows = {run: [M.episode_row(r) for r in rs] for run, rs in runs.items()}
    if a.selection:
        bad = M.check_selection([r for rs in rows.values() for r in rs])
        if bad:
            raise SystemExit('REFUSED selection run on final-test seeds (12/QĐ-KT E29):\n  ' + '\n  '.join(bad[:20]))
    slots = None
    if a.paired_validation:
        slots, bad = M.check_paired_set(rows)
        if bad:
            raise SystemExit('REFUSED paired validation (12/QĐ-KT E26):\n  ' + '\n  '.join(bad))
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / 'results_per_episode.jsonl').open('w') as f:
        for run in rows:
            for r in rows[run]:
                f.write(json.dumps(r) + '\n')
    agg = {run: M.aggregate(rs, a.n_boot, a.seed) for run, rs in rows.items()}
    safety = {run: M.safety_detail_summary(rs) for run, rs in rows.items()}
    variants = load_variants(a.hcs_sensitivity)
    if variants and 'variants' in variants:      # events_sensitivity.yaml: declared rule variants (1.0 top level, 1.1 under v2)
        sens = {}
        for run, rs in rows.items():
            v2 = str(rs[0].get('detector_version') or '').endswith('1.1') and 'v2' in variants
            names = list((variants['v2'] if v2 else variants)['variants'])
            sens[run] = M.hcs_rule_variants(rs, names)
    else:
        sens = {run: M.hcs_under(rs, variants) for run, rs in rows.items()} if variants else None

    pairs = [tuple(s.split(':', 1)) for s in a.pair] if a.pair else list(DEFAULT_PAIRS)
    comparisons = {}
    if slots is not None:
        pairs = []
        for n, slot in slots.items():
            for ma, mb in (('m1', 'm0'), ('m1', 'm1_noh'), ('m1_noh', 'm0')):
                x, y = slot[ma], slot[mb]
                comparisons[f'{x} vs {y}'] = M.paired(rows[x], rows[y], a.n_boot, a.seed)
                for cond in sorted({M.condition(r) for r in rows[x]}):
                    comparisons[f'{x} vs {y} [{cond}]'] = M.paired(
                        [r for r in rows[x] if M.condition(r) == cond], [r for r in rows[y] if M.condition(r) == cond],
                        a.n_boot, a.seed)
    for pa, pb in pairs:
        ra = [k for k in rows if k == pa or rows[k][0]['model'] == pa]
        rb = [k for k in rows if k == pb or rows[k][0]['model'] == pb]
        for x in ra:
            for y in rb:
                if x != y and (pa == x or rows[x][0]['n_action_steps'] == rows[y][0]['n_action_steps']):
                    comparisons[f'{x} vs {y}'] = M.paired(rows[x], rows[y], a.n_boot, a.seed)

    settings = dict(status=status, note=M.HRIBENCH_NOTE, official_check_problems=problems[:50],
                    decisions='10/QĐ-KT (formulas, slide 10), 11/QĐ-KT (event semantics)',
                    role_groups=M.ROLE_GROUPS, n_boot=a.n_boot, bootstrap_seed=a.seed,
                    bootstrap='percentile; scenarios (task, variant, seed) resampled within each task',
                    hcs_sensitivity_config=None if a.hcs_sensitivity is None else str(a.hcs_sensitivity),
                    inputs=[str(d) for d in a.dirs], metrics=list(M.SLIDE10))
    (a.out / 'results_per_task.json').write_text(json.dumps(
        dict(settings=settings, runs={k: dict(case=g['case'], task=g['task']) for k, g in agg.items()}), indent=1))
    (a.out / 'results_summary.json').write_text(json.dumps(dict(
        settings=settings, paired=comparisons, hcs_sensitivity=sens, safety_detail=safety,
        secondary_anticipation={k: M.anticipation_summary(rs) for k, rs in rows.items()},
        runs={k: dict(overall=g['overall'], condition=g['condition'], eval_set=g['eval_set'], role_group=g['role_group'],
                      eval_set_role=g['eval_set_role']) for k, g in agg.items()}), indent=1))
    md = [f'# Benchmark report (8 metrics, 10/QĐ-KT + 11/QĐ-KT): {status}', '', M.HRIBENCH_NOTE, '',
          'Groups: I = Instructor, C = Collaborator, W = Cowork (project role; not HRIBench\'s Intruder). '
          'n/a = not applicable / undefined (never 0). Cells: value [95 % CI] (numerator/denominator or n eligible; '
          f'n/a; contributing tasks). Bootstrap {a.n_boot} draws, seed {a.seed}, scenarios resampled within tasks.', '']
    if problems:
        md += ['Official-output check failed (first 5):', *[f'- {x}' for x in problems[:5]], '']
    md += ['## Main: macro average over tasks', '', '### overall', '', *header('run'),
           *[row(run, g['overall'], 'macro') for run, g in agg.items()], '',
           *level_tables(agg, 'condition', 'macro', 'condition'),
           *level_tables(agg, 'eval_set', 'macro', 'evaluation set'),
           *level_tables(agg, 'role_group', 'macro', 'role group'),
           '## Secondary: pooled over episodes', '', '### overall', '', *header('run'),
           *[row(run, g['overall'], 'pooled') for run, g in agg.items()], '',
           *level_tables(agg, 'condition', 'pooled', 'condition'),
           '## Per task', '', *level_tables(agg, 'task', 'pooled', 'task'),
           '## Per case (task/variant)', '', *level_tables(agg, 'case', 'pooled', 'case'),
           '## Paired differences', '']
    for name, c in comparisons.items():
        md += paired_table(name, c)
    if not comparisons:
        md += ['No run pair found for the requested comparisons.', '']
    ant = {run: M.anticipation_summary(rs) for run, rs in rows.items()}
    md += ['## Secondary (not slide-10 metrics): anticipation, e.g. AM / useful-action onset (12/QĐ-KT C20)', '']
    if any(ant.values()):
        md += ['| run | indicator | macro | pooled | n | n/a | tasks | cases |', '|---|---|---|---|---|---|---|---|']
        for run, d in ant.items():
            for k, e in d.items():
                f = lambda v: 'n/a' if v is None else f'{v:.3f}'
                md.append(f"| {run} | {k} | {f(e['macro'])} | {f(e['pooled'])} | {e['n_eligible']} | {e['n_na']} | "
                          f"{e['n_tasks_contributing']} | {e['n_cases_contributing']} |")
    else:
        md.append('n/a: the detector records carry no anticipation indicators.')
    md.append('')
    md += ['## Secondary (not slide-10 metrics): safety detail (15/QĐ-KT D21-D24)', '',
           'Reads the same detector audit as HCS; never replaces HCS / CFR. Classes: collision = real contact; proximity = '
           'verified non-exempt d < S_p without contact; distance unverified = only a lower bound (not evidence); clear. '
           'Duration / ratio = verified non-exempt violating ticks (x dt / over all ticks). S_p is the static SSM part '
           '(ISO/TS 15066), no force / pressure model: clearance violations have no force. n = episodes where the field '
           'exists; n/a = the rest.', '']
    for run, sd in safety.items():
        md += safety_table(f'{run}: overall', {'all': sd['overall']})
        md += safety_table(f'{run}: per task', sd['task'])
        md += safety_table(f'{run}: per case (task/variant)', sd['case'])
    if sens:
        md += ['## HCS sensitivity (alternative exemption definitions, declared before evaluation)', '',
               '| run | variant | HCS pooled (num/den) | HCS macro |', '|---|---|---|---|']
        for run, sv in sens.items():
            for name, e in sv.items():
                pv = 'n/a' if e['pooled'] is None else f"{e['pooled']:.3f}"
                mv = 'n/a' if e['macro'] is None else f"{e['macro']:.3f}"
                md.append(f"| {run} | {name} | {pv} ({e['numerator']}/{e['denominator']}) | {mv} |")
    (a.out / 'REPORT.md').write_text('\n'.join(md) + '\n')
    print(f'wrote {a.out}/REPORT.md [{status}] ({len(rows)} runs, {sum(map(len, rows.values()))} episodes)')


if __name__ == '__main__':
    main()
