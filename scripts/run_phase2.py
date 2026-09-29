"""Phase-2 orchestration: train No-Information (masked) and Oracle policies for every training seed,
then evaluate every checkpoint on the versioned protocol. Finished work is skipped, partial training
is resumed, so the script can be re-run at any time.

  ./run.sh -m scripts.run_phase2 --stage train                   # sequential GPU training jobs
  ./run.sh -m scripts.run_phase2 --stage eval --wait --workers 4 # CPU eval of checkpoints as they appear

Layout: outputs/train/phase2/<condition>_s<seed>/ and outputs/eval/phase2/<condition>_s<seed>/<oracle condition>/
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from scripts.common import load_yaml, resolve

CONDITIONS = {'noinfo': ['intent_mask=true'], 'oracle': ['intent_mask=false']}
TRACE_CONDITIONS = {'none', 'correct'}          # keep per-step traces only where the analyses need them


def ckpt_dir(root: Path, cond: str, seed: int) -> Path:
    return root / f'{cond}_s{seed}' / 'checkpoints' / 'last' / 'pretrained_model'


def run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(' '.join(cmd), '>', log, flush=True)
    with open(log, 'a') as f:
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)


def train(args, exp, train_root):
    for seed in args.seeds:
        for cond in args.conditions:
            out = train_root / f'{cond}_s{seed}'
            if (ckpt_dir(train_root, cond, seed) / 'model.safetensors').exists():
                print(f'skip trained {out}', flush=True)
                continue
            cmd = [sys.executable, '-m', 'scripts.train_policy', '--config', args.config, '--output-dir', str(out),
                   '--seed', str(seed), '--keep-last', '1', '--device', 'cuda', '--policy-set', *CONDITIONS[cond]]
            if (out / 'checkpoints').exists():
                cmd.append('--resume')
            if run(cmd, resolve('outputs/logs') / f'train_phase2_{cond}_s{seed}.log') != 0:
                raise SystemExit(f'training failed: {out}')
            if args.prune:                   # disk is tight: keep only the final model (checkpoints/last)
                for d in (out / 'checkpoints').glob('[0-9]*'):
                    shutil.rmtree(d)


def evaluate(args, exp, train_root, eval_root):
    """Evaluate in priority order (main comparison first, then the corruption and timing conditions), each
    job as soon as its checkpoint exists."""
    protocol = load_yaml(args.protocol)
    names = [c for c in args.eval_conditions if c in protocol['conditions']]
    jobs = [(0, seed, 'noinfo', 'none') for seed in args.seeds if 'noinfo' in args.conditions]
    if 'oracle' in args.conditions:
        jobs += [(1 + (n != 'correct') + (n not in ('correct', 'wrong', 'shuffled')), seed, 'oracle', n)
                 for n in names for seed in args.seeds]
    pending = [j[1:] for j in sorted(jobs, key=lambda j: j[0])]
    while pending:
        ready = [j for j in pending if (ckpt_dir(train_root, j[1], j[0]) / 'model.safetensors').exists()]
        if not ready:
            if not args.wait:
                print(f'not trained yet: {pending}', flush=True)
                return
            time.sleep(120)
            continue
        seed, cond, name = ready[0]
        pending.remove(ready[0])
        out = eval_root / f'{cond}_s{seed}' / name
        if (out / 'results.json').exists():
            continue
        cmd = [sys.executable, '-m', 'scripts.evaluate', '--config', args.config, '--protocol', args.protocol,
               '--checkpoint', str(ckpt_dir(train_root, cond, seed)), '--condition', 'correct' if name == 'none' else name,
               '--output-dir', str(out), '--workers', str(args.workers), '--device', args.device]
        if name not in TRACE_CONDITIONS:
            cmd.append('--no-traces')
        if run(cmd, resolve('outputs/logs') / f'eval_phase2_{cond}_s{seed}_{name}.log') != 0:
            raise SystemExit(f'evaluation failed: {out}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/experiments/phase2_oracle.yaml')
    p.add_argument('--protocol', default='configs/benchmark/phase2_protocol_v1.yaml')
    p.add_argument('--stage', choices=['train', 'eval'], required=True)
    p.add_argument('--seeds', type=int, nargs='*')
    p.add_argument('--conditions', nargs='*', default=list(CONDITIONS))
    p.add_argument('--eval-conditions', nargs='*', default=['correct', 'wrong', 'shuffled', 'noisy', 'delay_100ms',
                                                            'delay_200ms', 'delay_500ms', 'lead_100ms', 'lead_200ms'])
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--device', default='cpu', help='eval inference device; never share the GPU with a training job')
    p.add_argument('--wait', action='store_true', help='eval: keep polling for checkpoints that are still training')
    p.add_argument('--no-prune', dest='prune', action='store_false', help='keep the numbered checkpoints of finished runs')
    args = p.parse_args()
    exp = load_yaml(args.config)
    args.seeds = args.seeds if args.seeds is not None else exp['training']['seeds']
    train_root = resolve(exp['training']['output_root'])
    eval_root = resolve('outputs/eval/phase2')
    (train if args.stage == 'train' else lambda a, e, t: evaluate(a, e, t, eval_root))(args, exp, train_root)


if __name__ == '__main__':
    main()
