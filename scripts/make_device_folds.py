"""Assemble device_folds.json (configs/intent_schema.yaml `device.folds_file`, intent/device.py) from the five fold
models of workstation/job1_device.sh: per fold k its best checkpoint (scripts/deepoint_finetune.py metrics.json,
model k: test fold k, validation fold k + 1) and its calibration on fold k + 1 (scripts/calibrate_device_rules.py
--json). Paths are stored relative to the project root, so the file works on every machine with the same outputs/.
`--single K` (the laptop's preliminary run, docs/requirements/INTENT_ACT_v3.2.md §4.2i): only model K, used for every
episode; the seeds it was trained or validated on (rendered episodes outside its test fold K) are listed as
`seen_seeds`, and intent/device.py refuses to measure them.

  ./run.sh -m scripts.make_device_folds --run outputs/device/<date>
  ./run.sh -m scripts.make_device_folds --run outputs/prelim/<date>/device --single 0
"""
import argparse
import json
from pathlib import Path

from intent_policy.intent.folds import FOLDS
from intent_policy.intent.labels import load_schema
from intent_policy.utils import resolve


def rel(p) -> str:
    """Project-relative path (also for a path written on another machine: from its outputs/ on)."""
    s = str(p)
    if '/outputs/' in s:
        return 'outputs/' + s.split('/outputs/', 1)[1]
    p, root = Path(p).resolve(), resolve('.').resolve()
    return str(p.relative_to(root)) if p.is_relative_to(root) else str(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='outputs/device/<date> with fold<k>/train/metrics.json, fold<k>/calibration.json')
    ap.add_argument('--out', help='default <run>/device_folds.json')
    ap.add_argument('--single', type=int, help='one model (this test fold) for every episode, its seen seeds refused')
    args = ap.parse_args()
    run = resolve(args.run)
    folds = {}
    for k in ([args.single] if args.single is not None else range(FOLDS)):
        m = json.loads((run / f'fold{k}' / 'train' / 'metrics.json').read_text())
        cal = json.loads((run / f'fold{k}' / 'calibration.json').read_text())
        if cal['test_fold'] != k or cal['val_fold'] != (k + 1) % FOLDS:
            raise SystemExit(f'fold {k}: calibration of test fold {cal["test_fold"]} / val fold {cal["val_fold"]}')
        ck = rel(m['best_checkpoint'])                # Lightning's best_model_path (absolute or from the root)
        if not resolve(ck).exists():
            raise SystemExit(f'fold {k}: checkpoint {ck} not found')
        folds[str(k)] = dict(checkpoint=rel(ck), val_fold=cal['val_fold'], sigma_deg=cal['sigma_deg'],
                             sigma_deg_by_frames=cal['sigma_deg_by_frames'], hold_frames=cal['hold_frames'],
                             hold_not_reached=cal['hold_not_reached'], pause_timeout_s=cal['pause_timeout_s'],
                             pause_intervals=cal['pause_intervals'])
    out = Path(args.out) if args.out else run / 'device_folds.json'
    doc = dict(classes=load_schema()['device']['classes'], run=rel(run), folds=folds)
    if args.single is not None:
        seen = sorted(int(m['seed']) for m in map(lambda j: json.loads(j.read_text()), (run / 'render').glob('episode_*.json'))
                      if m['fold'] != args.single)
        doc.update(assign='single', model=args.single, seen_seeds=seen)
        print(f'single model {args.single}: {len(seen)} seen seeds (trained / validated on) are refused')
    out.write_text(json.dumps(doc, indent=1))
    print(f'{out}: {len(folds)} folds')
    for k, f in folds.items():
        print(f"  fold {k}: {f['checkpoint']}  hold {f['hold_frames']}  pause {f['pause_timeout_s']} s"
              + (f"  (hold not reached: {f['hold_not_reached']})" if f['hold_not_reached'] else ''))


if __name__ == '__main__':
    main()
