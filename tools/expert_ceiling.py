"""Expert-only success on validation seeds (diagnosis / ceiling, not a policy claim) — 28/QĐ-HL Phase 3 task grouping.
    ./run.sh -m tools.expert_ceiling --task C2 --variant nominal --seeds 300820 ... --out outputs/diag/phase3_1120/ceiling
"""
import argparse, json
from pathlib import Path
from intent_policy.shared.manicast_adapter import load_predictor
from tools.preview import run_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True); p.add_argument('--variant', default='nominal')
    p.add_argument('--seeds', type=int, nargs='+', required=True); p.add_argument('--mode', default='forecast')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True); pred = load_predictor()
    for seed in a.seeds:
        s = run_episode(a.task, a.mode, seed, pred, review=False, variant=a.variant)
        s.pop('_arrays', None)
        row = dict(task=a.task, variant=a.variant, seed=seed, success=s['success'], reason=s['failure_reason'], physics_valid=s.get('physics_valid'), dur_s=s['duration_s'])
        with open(a.out / f'ceiling_{a.task}_{a.variant}.jsonl', 'a') as f: f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
