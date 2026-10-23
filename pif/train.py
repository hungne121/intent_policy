"""Train PIF on the sim episodes (handoff §6). `python -m pif.train --config configs/pif/base.yaml [--set key=value]`.

Split: per task, every `split.test_every`-th episode (sorted by index) is held out (test), never trained on.
Run folder outputs/pif_train/<date>/<time>_<name>: config.yaml (resolved), meta.json (seed, git commit, split,
normalisation stats), log.txt, metrics.jsonl, checkpoints (model + stats). Intent labels are not loaded for training
unless `supervised_warmstart` (handoff §0.4; asserted).
"""
from __future__ import annotations
import argparse
import json
import subprocess
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from intent_policy.utils import log_run, resolve, run_dir
from pif.data.dataset import PIFWindowDataset, collate
from pif.data.episode_reader import read_sim_dataset
from pif.data.tokens import norm_stats
from pif.losses import SIGReg, pif_losses
from pif.model.pif import PIF


def set_key(cfg: dict, assignment: str) -> None:
    key, value = assignment.split('=', 1)
    *path, last = key.split('.')
    d = cfg
    for p in path:
        d = d.setdefault(p, {})
    d[last] = yaml.safe_load(value)


def split_episodes(eps: list[dict], every: int) -> tuple[list[int], list[int]]:
    by = {}
    for i, e in enumerate(eps):
        by.setdefault(e['meta']['task'], []).append((e['meta']['episode_index'], i))
    train, test = [], []
    for task in sorted(by):
        for n, (_, i) in enumerate(sorted(by[task])):
            (test if n % every == every - 1 else train).append(i)
    return train, test


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def load_data(cfg: dict):
    if cfg.get('data', {}).get('domain', 'sim') == 'habit':
        from pif.data.habit_episode import read_habit_tokens
        lab = yaml.safe_load(resolve(cfg['data']['habit_labels']).read_text())
        files = sorted(f for d in cfg['data']['habit_dirs'] for s in cfg['data']['sids'] for f in (resolve(d) / s).glob('episode_*.npz'))
        eps = [read_habit_tokens(f, cfg, lab) for f in files]
    else:
        eps = read_sim_dataset(cfg['sim']['dataset'], cfg['sim']['replay'], cfg)
    train_idx, test_idx = split_episodes(eps, cfg['split']['test_every'])
    stats = norm_stats([eps[i] for i in train_idx])
    return eps, train_idx, test_idx, stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/pif/base.yaml')
    p.add_argument('--set', nargs='*', default=[], help='overrides, e.g. train.steps=2000 ablation.no_filter=true')
    p.add_argument('--name', default='pif')
    p.add_argument('--log-dir')
    args = p.parse_args()
    cfg = yaml.safe_load(resolve(args.config).read_text())
    for a in args.set:
        set_key(cfg, a)
    assert not cfg.get('supervised_warmstart'), 'supervised warm start needs the user permission (handoff §6)'
    out = run_dir('pif_train', args.name, args.log_dir)
    log_run(out, 'pif.train')
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eps, train_idx, test_idx, stats = load_data(cfg)
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=resolve('.'), capture_output=True, text=True).stdout.strip()
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / 'meta.json').write_text(json.dumps(dict(seed=cfg['seed'], commit=commit, train=[eps[i]['meta']['episode_index'] for i in train_idx],
                                                   test=[eps[i]['meta']['episode_index'] for i in test_idx],
                                                   stats={k: v.tolist() for k, v in stats.items()}), indent=1))
    ds = PIFWindowDataset([eps[i] for i in train_idx], cfg, stats, train=True, seed=cfg['seed'])
    tc = cfg['train']
    dl = DataLoader(ds, batch_size=tc['batch_size'], shuffle=True, collate_fn=collate, num_workers=tc.get('num_workers', 2),
                    drop_last=True, persistent_workers=tc.get('num_workers', 2) > 0)
    model = PIF(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'PIF {n_params / 1e6:.2f} M parameters, {len(train_idx)} train / {len(test_idx)} test episodes, '
          f'{len(ds)} windows, device {device}, git {commit}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['optim']['lr'], weight_decay=cfg['optim']['weight_decay'])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / tc.get('warmup', 500)))
    sigreg = SIGReg(tc.get('sigreg_slices', 1024))
    rng = torch.Generator(device=device).manual_seed(cfg['seed'])
    step, t0, acc = 0, time.time(), {}
    mfile = open(out / 'metrics.jsonl', 'a')
    while step < tc['steps']:
        for batch in dl:
            batch = to_device(batch, device)
            losses = pif_losses(model, batch, cfg, step, sigreg, tc['n_sample'], rng)
            opt.zero_grad(set_to_none=True)
            losses['total'].backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get('grad_clip', 1.0))
            opt.step()
            sched.step()
            for k, v in losses.items():
                acc[k] = acc.get(k, 0.0) + float(v)
            acc['grad_norm'] = acc.get('grad_norm', 0.0) + float(gn)
            step += 1
            if step % tc['log_every'] == 0:
                row = {k: round(v / tc['log_every'], 5) for k, v in acc.items()}
                row.update(step=step, s_per_step=round((time.time() - t0) / step, 3))
                mfile.write(json.dumps(row) + '\n')
                mfile.flush()
                print(' '.join(f'{k}={v}' for k, v in row.items()), flush=True)
                acc = {}
            if step % tc['save_every'] == 0 or step == tc['steps']:
                ck = out / 'checkpoints' / f'{step:06d}.pt'
                ck.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(model=model.state_dict(), cfg=cfg, stats=stats, step=step, commit=commit), ck)
            if step >= tc['steps']:
                break
    print(f'done: {out}', flush=True)


if __name__ == '__main__':
    main()
