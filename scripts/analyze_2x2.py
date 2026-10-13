"""2x2 analysis of the INTENT_ACT models (docs/requirements/INTENT_ACT_v3.2.md §7): late / early x without / with intent.

Reads the evaluation runs of A (late), B (early), C (late + intent), D (early + intent) on the same episode list
(per-episode metrics of episodes/<split>/<task>/*.json.gz) and reports per task and overall:
  - each model's mean of CSR, CT, Rsp, WCR, WCD, AM, CRsp, DSR (benchmark/metrics.py; the HRIBench-style set only,
    no metric of our own: the user, 2026-10-12);
  - the intent effect late (C - A), early (D - B) and the interaction (D - B) - (C - A), with 95 % paired bootstrap
    intervals over the episodes all four models share (H3 of the design).

  ./run.sh -m scripts.analyze_2x2 --runs A=outputs/eval/<A run> B=... C=... D=... --output docs/reports/intent_2x2.md
"""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from intent_policy.utils import resolve

METRICS = ('CSR', 'CT', 'Rsp', 'WCR', 'WCD', 'AM', 'CRsp', 'DSR')          # HRIBench-style set
LOWER_IS_BETTER = {'CT', 'Rsp', 'WCR', 'WCD', 'CRsp'}


def load_run(run: Path) -> dict[str, dict]:
    """episode id -> {task, metric: value} (numeric values only)."""
    out = {}
    for f in sorted((run / 'episodes').glob('*/*/*.json.gz')):
        with gzip.open(f, 'rt') as fh:
            rec = json.load(fh)
        task = rec.get('list_task') or f.parent.name
        vals = {k: float(v) for k, v in (rec.get('metrics') or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        vals.update({k: float(v) for k, v in (rec.get('metrics') or {}).items() if isinstance(v, bool)})
        out[rec['episode_id']] = dict(task=task, **vals)
    return out


def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def table(runs: dict[str, dict], tasks: list[str]) -> str:
    lines = []
    for scope in tasks + ['all']:
        lines += [f'### {scope}', '', '| metric | A late | B early | C late+intent | D early+intent | C − A | D − B | (D − B) − (C − A) [95% CI] | n paired |',
                  '|---|---|---|---|---|---|---|---|---|']
        for m in METRICS:
            mean, common = {}, None
            for k, eps in runs.items():
                vals = {e: r[m] for e, r in eps.items() if m in r and (scope == 'all' or r['task'] == scope)}
                mean[k] = np.mean(list(vals.values())) if vals else None
                common = set(vals) if common is None else common & set(vals)
            if all(v is None for v in mean.values()):
                continue
            fmt = lambda v: '—' if v is None else f'{v:.3f}'
            if common and all(k in runs for k in 'ABCD'):
                ids = sorted(common)
                x = {k: np.array([runs[k][e][m] for e in ids]) for k in 'ABCD'}
                inter = (x['D'] - x['B']) - (x['C'] - x['A'])
                lo, hi = boot_ci(inter)
                cells = [f"{np.mean(x['C'] - x['A']):+.3f}", f"{np.mean(x['D'] - x['B']):+.3f}",
                         f'{inter.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]', str(len(ids))]
            else:
                cells = ['—', '—', '—', '0']
            lines.append(f"| {m}{' ↓' if m in LOWER_IS_BETTER else ''} | " + ' | '.join(fmt(mean.get(k)) for k in 'ABCD')
                         + ' | ' + ' | '.join(cells) + ' |')
        lines.append('')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', nargs='+', required=True, help='MODEL=eval run folder, MODEL in A B C D')
    p.add_argument('--output', help='write the tables to this markdown file')
    args = p.parse_args()
    runs = {}
    for a in args.runs:
        k, path = a.split('=', 1)
        runs[k] = load_run(resolve(path))
    tasks = sorted({r['task'] for eps in runs.values() for r in eps.values()})
    md = '# INTENT_ACT 2×2 (late / early × without / with intent)\n\n' + '\n'.join(f'- {k}: `{a.split("=", 1)[1]}`' for k, a in zip(runs, args.runs)) \
        + '\n\n' + table(runs, tasks)
    print(md)
    if args.output:
        out = resolve(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)


if __name__ == '__main__':
    main()
