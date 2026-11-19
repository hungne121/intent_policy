"""Technical smoke gate of M* (19/QĐ-HL item 5): after a 200-500 step run, check tensors (finite losses), gradients
(forecast-encoder gradient > 0 in every log row), NaN, encoder collapse (z_H variance > 0, dead-unit ratio of the
forecast encoder == 0) and that the run finished. Not a research round: it only tells whether the full run may start.
    ./run.sh -m tools.check_smoke --run <train output dir> [--out report.json]      exit 1 = failed
"""
import argparse
import json
import math
from pathlib import Path


def rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def check(run: Path) -> dict:
    tr, dg, va = rows(run / 'train_log.jsonl'), rows(run / 'diag_log.jsonl'), rows(run / 'val_log.jsonl')
    log = (run.parent.parent / f'smoke_{run.name}.log')
    done = log.exists() and 'TRAIN_FINISHED' in log.read_text()
    meta = json.loads((run / 'run_meta.json').read_text()) if (run / 'run_meta.json').exists() else {}
    fin = lambda v: v is not None and isinstance(v, (int, float)) and math.isfinite(v)
    has_fc = 'forecast_grad_norm' in (tr[0] if tr else {})
    c = {
        'finished (TRAIN_FINISHED, no COLLAPSE.json)': done and not (run / 'COLLAPSE.json').exists(),
        'train log rows': len(tr) > 0,
        'all losses finite': all(fin(r.get('loss')) and fin(r.get('l1_loss')) for r in tr),
        'val L1 finite': bool(va) and all(fin(r.get('val_l1')) for r in va),
        'forecast gradient > 0 in every log row': has_fc and all(r['forecast_grad_norm'] > 0 for r in tr),
        'diagnostics logged': len(dg) > 0 and all(all(fin(v) for v in r.values()) for r in dg),
        'z_H variance > 0 in every diagnostic': bool(dg) and all(r['zh_var'] > 0 for r in dg),
        'dead-unit ratio == 0 in every diagnostic': bool(dg) and all(r['dead_unit_ratio'] == 0 for r in dg),
        'decision mask active (share > 0)': (meta.get('decision_window_share') or {}).get('train', {}).get('all', 0) > 0,
    }
    return dict(checks=c, ok=all(c.values()), last_diag=dg[-1] if dg else None, last_train=tr[-1] if tr else None)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    r = check(a.run)
    for k, v in r['checks'].items():
        print(('PASS ' if v else 'FAIL ') + k)
    if a.out:
        a.out.write_text(json.dumps(r, indent=1))
    raise SystemExit(0 if r['ok'] else 1)
