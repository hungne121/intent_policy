"""D conditioning study: does each D variant use its intent, and does it help? (INTENT_ACT_v3.2.md §5.3)

For every variant, two closed-loop evaluations on the same episode list: with the intent (`with`) and with every
intent token masked (`masked`, scripts/evaluate.py --intent-keep none). Reported on the intent tasks (T1, T2, T3, T5),
from the per-episode HRIBench-style metrics (benchmark/metrics.py):
  use       the share of episodes whose outcome (success / failure reason) changes when the intent is masked, and the
            paired benefit with vs masked (> 0 = the intent helps; lower-is-better metrics flipped) with its 95 % paired
            bootstrap interval (scripts/analyze_2x2_detail.py)
  vs B      the paired benefit of the variant (with its intent) over B, the early model without intent: Q3 of the thesis
  offline   scripts/intent_reliance.py ratios on the validation frames, when given
The variant uses its intent when masking changes its outcomes; the architecture is useful when the use is a benefit.

  ./run.sh -m scripts.compare_conditioning --variant D=<eval>,<eval masked>[,<reliance dir>] ... --base B=<eval> \
      --output <report.md>
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_2x2_detail import INTENT_TASKS, METRICS, benefit, bootstrap, fmt, load_run

SHOWN = ('CSR', 'CT', 'WCR', 'WCD', 'AM', 'CRsp')


def paired(a: dict, b: dict, metric: str, ids) -> np.ndarray:
    keep = [e for e in ids if metric in a[e]['metrics'] and metric in b[e]['metrics']]
    return benefit(metric, np.array([a[e]['metrics'][metric] - b[e]['metrics'][metric] for e in keep]))


def cell(x: np.ndarray, unit: str = '') -> str:
    if len(x) < 2:
        return f'n {len(x)}'
    lo, hi, p = bootstrap(x)
    return f'{x.mean():+.3f}{unit} [{lo:+.3f}, {hi:+.3f}] (n {len(x)})'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--variant', action='append', required=True, metavar='NAME=EVAL,MASKED[,RELIANCE]')
    p.add_argument('--base', metavar='NAME=EVAL', help='reference run without intent (B)')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    base_name, base = None, None
    if args.base:
        base_name, path = args.base.split('=', 1)
        base = load_run(path)
    lines = ['# D conditioning study', '',
             f'Intent tasks {", ".join(INTENT_TASKS)}. Benefit > 0 = better (lower-is-better metrics flipped); '
             '95 % paired bootstrap intervals over episodes.', '']
    summary = {}
    for spec in args.variant:
        name, paths = spec.split('=', 1)
        parts = paths.split(',')
        run, masked = load_run(parts[0]), load_run(parts[1])
        ids = sorted(e for e in set(run) & set(masked) if run[e]['task'] in INTENT_TASKS)
        changed = [e for e in ids if (run[e]['success'], run[e]['failure']) != (masked[e]['success'], masked[e]['failure'])]
        s = dict(n=len(ids), outcome_changed=len(changed),
                 csr_with=float(np.mean([run[e]['success'] for e in ids])),
                 csr_masked=float(np.mean([masked[e]['success'] for e in ids])))
        lines += [f'## {name}', '',
                  f'- episodes {len(ids)}; outcome changed by masking the intent: **{len(changed)} / {len(ids)}**',
                  f"- CSR with intent {s['csr_with']:.3f}, masked {s['csr_masked']:.3f}"]
        if len(parts) > 2 and (Path(parts[2]) / 'reliance.json').exists():
            r = json.loads((Path(parts[2]) / 'reliance.json').read_text())['results']
            rel = {t: r[t] for t in ('T1', 'T2', 'T3', 'T4', 'all') if t in r}
            s['reliance'] = rel
            lines.append('- offline reliance L1(masked)/L1(full), L1(permuted)/L1(full): ' + '; '.join(
                f"{t} {v['reliance_masked']:.3f} / {v['reliance_permuted']:.3f}" for t, v in rel.items()))
        head = '| metric | with | masked | with vs masked |' + (f' with vs {base_name} |' if base else '')
        lines += ['', head, '|---' * (head.count('|') - 1) + '|']
        for m in SHOWN:
            w = [run[e]['metrics'][m] for e in ids if m in run[e]['metrics']]
            k = [masked[e]['metrics'][m] for e in ids if m in masked[e]['metrics']]
            unit = 's' if m in ('CT', 'WCD', 'AM', 'CRsp') else ''
            row = f'| {m} | {fmt(float(np.mean(w)) if w else None)} | {fmt(float(np.mean(k)) if k else None)} | ' \
                  f'{cell(paired(run, masked, m, ids), unit)} |'
            s[f'{m}_with_vs_masked'] = paired(run, masked, m, ids).tolist()
            if base:
                bid = [e for e in ids if e in base]
                row += f' {cell(paired(run, base, m, bid), unit)} |'
            lines.append(row)
        lines.append('')
        summary[name] = s
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text('\n'.join(lines) + '\n')
    Path(args.output).with_suffix('.json').write_text(json.dumps(summary, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
