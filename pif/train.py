"""Train PIF (docs/PIF.md): python -m pif.train --config configs/pif/pif.yaml [--set train.steps=3000 data.tasks=[S8789]]

Data, model, loss and schedule all come from the config. Split: per task every `data.test_every`-th episode is held
out. Run folder outputs/pif_train/<date>/<time>_<name>: config.yaml (resolved), meta.json (seed, git commit, split,
normalisation stats), log.txt, metrics.jsonl, checkpoints/ (model + config + stats). Eval labels never reach training
unless `train.supervised_warmstart` (docs/PIF.md; asserted).
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

from pif.utils import log_run, resolve, run_dir
from pif.config import load_config
from pif.data import load_episodes, split
from pif.data.dataset import PIFWindowDataset, collate
from pif.data.tokens import norm_stats
from pif.losses import SIGReg, pif_losses
from pif.model import PIF


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/pif/pif.yaml')
    p.add_argument('--set', nargs='*', default=[])
    p.add_argument('--name', default='pif')
    p.add_argument('--log-dir')
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    tc = cfg['train']
    assert not tc['supervised_warmstart'], 'supervised warm start needs the user permission (docs/PIF.md)'
    out = run_dir('pif_train', args.name, args.log_dir)
    log_run(out, 'pif.train')
    torch.manual_seed(tc['seed'])
    np.random.seed(tc['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eps, _ = load_episodes(cfg)
    train_idx, test_idx = split(eps, cfg['data']['test_every'])
    stats = norm_stats([eps[i] for i in train_idx])
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=resolve('.'), capture_output=True, text=True).stdout.strip()
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / 'meta.json').write_text(json.dumps(dict(seed=tc['seed'], commit=commit,
                                                   train=[eps[i]['meta']['episode_index'] for i in train_idx],
                                                   test=[eps[i]['meta']['episode_index'] for i in test_idx],
                                                   stats={k: v.tolist() for k, v in stats.items()}), indent=1))
    ds = PIFWindowDataset([eps[i] for i in train_idx], cfg, stats, train=True, seed=tc['seed'])
    dl = DataLoader(ds, batch_size=tc['batch_size'], shuffle=True, collate_fn=collate, num_workers=tc['num_workers'],
                    drop_last=True, persistent_workers=tc['num_workers'] > 0)
    model = PIF(cfg).to(device)
    print(f"PIF {sum(q.numel() for q in model.parameters()) / 1e6:.2f} M parameters; {cfg['data']['domain']} {cfg['data']['tasks']}: "
          f"{len(train_idx)} train / {len(test_idx)} test episodes, {len(ds)} windows; device {device}; git {commit}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=tc['lr'], weight_decay=tc['weight_decay'])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / tc['warmup']))
    sigreg = SIGReg(cfg['loss']['sigreg_slices'])
    rng = torch.Generator(device=device).manual_seed(tc['seed'])
    step, t0, acc = 0, time.time(), {}
    metrics = open(out / 'metrics.jsonl', 'a')
    while step < tc['steps']:
        for batch in dl:
            losses = pif_losses(model, to_device(batch, device), cfg, step, sigreg, cfg['loss']['n_sample'], rng)
            opt.zero_grad(set_to_none=True)
            losses['total'].backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc['grad_clip'])
            opt.step()
            sched.step()
            for k, v in dict(losses, grad_norm=gn).items():
                acc[k] = acc.get(k, 0.0) + float(v.detach() if torch.is_tensor(v) else v)
            step += 1
            if step % tc['log_every'] == 0:
                row = dict({k: round(v / tc['log_every'], 5) for k, v in acc.items()}, step=step,
                           s_per_step=round((time.time() - t0) / step, 3))
                metrics.write(json.dumps(row) + '\n')
                metrics.flush()
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
