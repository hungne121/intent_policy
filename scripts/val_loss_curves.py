"""Validation-loss curves of training runs (scripts/train_policy.py train_log.jsonl: val_loss every 10 x log_every
steps), for the plateau check of the ACT tuning tips ("train ... 3-4 times the length after the loss has plateaued",
github.com/tonyzhaozh/act). Per run: the curve, the step of its minimum, and the last relative improvement.

  ./run.sh -m scripts.val_loss_curves --runs A=<run> B=<run> C=<run> D=<run> --output <out.md>
"""
import argparse
import json
from pathlib import Path

from intent_policy.utils import resolve


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True, help='NAME=run folder')
    ap.add_argument('--output')
    args = ap.parse_args()
    md = ['| run | val_loss by step | minimum at | last change |', '|---|---|---|---|']
    for spec in args.runs:
        name, run = spec.split('=', 1)
        f = resolve(run) / 'train_log.jsonl'
        rows = [r for r in map(json.loads, f.read_text().splitlines()) if 'val_loss' in r] if f.exists() else []
        if not rows:
            md.append(f'| {name} | — | — | — |')
            continue
        best = min(rows, key=lambda r: r['val_loss'])
        last = f"{(rows[-1]['val_loss'] - rows[-2]['val_loss']) / rows[-2]['val_loss']:+.1%}" if len(rows) > 1 else '—'
        md.append(f"| {name} | " + ', '.join(f"{r['step']}: {r['val_loss']:.4f}" for r in rows)
                  + f" | {best['step']} | {last} |")
    print('\n'.join(md))
    if args.output:
        Path(args.output).write_text('# Validation loss\n\n' + '\n'.join(md) + '\n')


if __name__ == '__main__':
    main()
