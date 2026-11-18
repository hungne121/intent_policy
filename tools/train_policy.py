"""Step-2 training entry for M0 / M1-noH / M1 (07/QĐ-HL C, D, F, G, H; 03/TTr-HL).

Why not `lerobot-train`: its data loader always uses LeRobot's EpisodeAwareSampler (lerobot/scripts/lerobot_train.py,
no sampler hook), and 03/TTr-HL #3 needs task-balanced sampling; 07/QĐ-HL H also needs the loss per task / variant.
Everything else is LeRobot's: LeRobotDataset (+ delta_timestamps -> action chunk and action_is_pad: chunks never
cross an episode end), dataset_to_policy_features, the ACT pre/post-processors (normalisation from the train split's
stats.json, recomputed by tools/build_train_set.py), ACTPolicy.forward (masked L1 + kl_weight x KL, 07/QĐ-HL F),
the ACT optimizer preset (AdamW, backbone lr group, no scheduler) and LeRobot's grad-clip default.

Sampling (03/TTr-HL #3): frame weights 1 / (frames of the task) x optional group weight, drawn with replacement, so
every task carries the same total weight; N1 / N5 / N6 keep their natural ratio unless `group_weights` is set.
Validation: L1 with z = 0 (ACT inference) over the val split every `val_every` steps; the lowest val L1 is
checkpoints/best (07/QĐ-HL H: chosen on val, never on test). Saved per run: run_meta.json (model config, training
config, action definition, normalisation + forecast statistics, seeds, parameter counts, git commit), manifest.jsonl
(the train + val episodes), train_log.jsonl / val_log.jsonl, curves.png.

    ./run.sh -m tools.train_policy --policy-config configs/policy/m1.yaml --train-set <build_train_set out> \
        --output-dir outputs/train_step2/<run>/M1 [--steps N] [--batch-size B] [--num-workers W] [--device cuda]
"""
import argparse
import json
import platform
import random
import subprocess
import shutil
import signal
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from tools.train_checkpoint_io import publish_checkpoint, verify_checkpoint

ROOT = Path(__file__).resolve().parents[1]
ACTION_DEFINITION = ('action (7): joint position targets of the 6 UR3e joints [rad] (shoulder_pan, shoulder_lift, elbow, '
                     'wrist_1, wrist_2, wrist_3) + gripper command [m], exactly as sent to the servo at that 20 Hz tick '
                     '(intent_policy/data/dataset_schema.py); chunk = the next chunk_size ticks, padded + masked past '
                     'the episode end')


def git_commit() -> str:
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return sha + ('-dirty' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def build_config(policy_yaml: dict, meta, device: str):
    from lerobot.configs.types import FeatureType
    from lerobot.utils.feature_utils import dataset_to_policy_features
    from intent_policy.policies.forecast_token import ur3e_tcp_chain
    from intent_policy.policies.hri_act import HRIACTConfig
    feats = dataset_to_policy_features(meta.features)
    cams = [f'observation.images.{c}' for c in policy_yaml['train_data']['cameras']]
    missing = [k for k in cams + ['observation.state', 'action'] if k not in feats]
    if missing:
        raise KeyError(f'dataset lacks {missing}')
    inputs = {k: feats[k] for k in cams}                          # camera order = config order
    inputs['observation.state'] = feats['observation.state']
    assert all(feats[k].type is FeatureType.VISUAL for k in cams)
    kwargs = {k: v for k, v in policy_yaml.items() if k not in ('type', 'train_data')}
    if kwargs.get('use_forecast'):
        for k in (kwargs['forecast_key'], kwargs['forecast_valid_key']):
            if k not in meta.features:
                raise KeyError(f'dataset lacks {k}')
        kwargs['forecast_fk_chain'] = ur3e_tcp_chain()
    return HRIACTConfig(input_features=inputs, output_features={'action': feats['action']}, device=device,
                        push_to_hub=False, **kwargs)


def frame_weights(ep_index: np.ndarray, ep_meta: dict, mode: str, group_weights: dict | None) -> np.ndarray:
    """Per-frame sampling weights; task_balanced: each task's frames sum to the same total (03/TTr-HL #3)."""
    task = np.array([ep_meta[int(e)]['task_name'] for e in ep_index])
    w = np.ones(len(ep_index))
    if mode == 'task_balanced':
        for t in np.unique(task):
            w[task == t] = 1.0 / (task == t).sum()
    elif mode != 'uniform':
        raise ValueError(f'sampling {mode!r}')
    if group_weights:
        g = np.array([ep_meta[int(e)]['group'] for e in ep_index])
        w *= np.array([float(group_weights.get(x, 1.0)) for x in g])
    return w / w.sum()


class Groups:
    """Running per-task / per-variant sums of the per-sample L1."""

    def __init__(self, ep_meta):
        self.ep_meta, self.s, self.n = ep_meta, defaultdict(float), defaultdict(int)

    def add(self, episodes, per):
        for e, v in zip(episodes.tolist(), per.tolist()):
            m = self.ep_meta[int(e)]
            for key in (f'task/{m["task_name"]}', f'variant/{m["task_name"]}/{m["task_variant"]}', f'group/{m["group"]}'):
                self.s[key] += v
                self.n[key] += 1

    def result(self, prefix):
        out = {f'{prefix}l1/{k}': self.s[k] / self.n[k] for k in sorted(self.s)}
        self.s, self.n = defaultdict(float), defaultdict(int)
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--policy-config', required=True)
    p.add_argument('--train-config', default='configs/policy/train_m0m1.yaml')
    p.add_argument('--train-set', type=Path, required=True, help='tools/build_train_set.py output')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--steps', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--num-workers', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--val-every', type=int)
    p.add_argument('--save-every', type=int)
    p.add_argument('--log-every', type=int)
    p.add_argument('--val-max-batches', type=int)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--video-backend', default=None, help='torchcodec | pyav (LeRobot default if unset)')
    p.add_argument('--resume', action='store_true', help='continue from the newest numbered checkpoint in --output-dir')
    a = p.parse_args()

    import lerobot
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from intent_policy.policies.forecast_token import forecast_position_stats
    from intent_policy.policies.hri_act import HRIACTPolicy, make_hri_act_pre_post_processors

    if a.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but unavailable; refusing to fall back to the CPU')
    pol_yaml = yaml.safe_load((ROOT / a.policy_config).read_text())
    tr = yaml.safe_load((ROOT / a.train_config).read_text())
    for k in ('steps', 'batch_size', 'num_workers', 'seed', 'val_every', 'save_every', 'log_every', 'val_max_batches', 'amp'):
        if getattr(a, k) is not None:
            tr[k] = getattr(a, k)
    tr['amp'] = bool(tr['amp']) and a.device == 'cuda'
    seed = int(tr['seed'])
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    dname = pol_yaml['train_data']['dataset']
    train_root, val_root = a.train_set / dname / 'train', a.train_set / dname / 'val'
    if not (train_root / 'meta/stats_source.json').exists():
        raise SystemExit(f'{train_root}: no meta/stats_source.json (stats not recomputed from train: run tools/build_train_set.py)')
    meta = LeRobotDatasetMetadata(f'local/{dname}_train', root=train_root)
    cfg = build_config(pol_yaml, meta, a.device)
    delta = {'action': [i / meta.fps for i in range(cfg.chunk_size)]}
    kw = dict(delta_timestamps=delta, **({'video_backend': a.video_backend} if a.video_backend else {}))
    train_ds = LeRobotDataset(f'local/{dname}_train', root=train_root, **kw)
    val_ds = LeRobotDataset(f'local/{dname}_val', root=val_root, **kw) if (val_root / 'meta/info.json').exists() else None
    tr_meta = {m['episode_index']: m for m in read_jsonl(train_root / 'meta/hrc_episodes.jsonl')}
    va_meta = {m['episode_index']: m for m in read_jsonl(val_root / 'meta/hrc_episodes.jsonl')} if val_ds else {}
    missing_val = {m['task_name'] for m in tr_meta.values()} - {m['task_name'] for m in va_meta.values()}
    if missing_val:
        raise SystemExit(f'validation lacks training tasks: {sorted(missing_val)}')

    ep_col = np.asarray(train_ds.hf_dataset.with_format('numpy', columns=['episode_index'])['episode_index'])
    weights = frame_weights(ep_col, tr_meta, tr['sampling'], tr.get('group_weights'))
    gen = torch.Generator().manual_seed(seed)
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(train_ds),
                                                     replacement=True, generator=gen)
    lk = dict(batch_size=int(tr['batch_size']), num_workers=int(tr['num_workers']), pin_memory=a.device == 'cuda',
              persistent_workers=int(tr['num_workers']) > 0)
    train_loader = torch.utils.data.DataLoader(train_ds, sampler=sampler, drop_last=True, **lk)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, drop_last=False, **lk) if val_ds else None

    policy = HRIACTPolicy(cfg).to(a.device)
    stats = meta.stats
    fstats = None
    if cfg.use_forecast:
        cols = train_ds.hf_dataset.with_format('numpy', columns=[cfg.forecast_key, cfg.forecast_valid_key])
        mu, sd = forecast_position_stats(np.asarray(cols[cfg.forecast_key]), np.asarray(cols[cfg.forecast_valid_key]))
        qm = np.asarray(stats['observation.state']['mean'], np.float64)[:6]
        qs = np.asarray(stats['observation.state']['std'], np.float64)[:6]
        policy.set_forecast_stats(mu, sd, qm, qs)
        fstats = dict(pos_mean=mu.tolist(), pos_std=sd.tolist(), q_mean=qm.tolist(), q_std=qs.tolist(),
                      source='valid forecasts of the train split (all horizons, joints); qpos = train stats.json')
    preprocessor, postprocessor = make_hri_act_pre_post_processors(cfg, dataset_stats=stats)
    opt_cfg = cfg.get_optimizer_preset()
    optimizer = opt_cfg.build(policy.get_optim_params())
    clip = float(tr.get('grad_clip_norm', opt_cfg.grad_clip_norm))
    scaler = torch.amp.GradScaler('cuda', enabled=tr['amp'])
    n_params = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_fc = sum(p.numel() for p in policy.forecast_encoder.parameters()) if policy.forecast_encoder is not None else 0
    n_used = n_params - sum(p.numel() for p in policy.restricted_head.parameters())

    out = a.output_dir
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    step, best = 0, float('inf')
    if a.resume:
        ck = [p for p in (out / 'checkpoints').iterdir()
              if (p.name.isdigit() or p.name == 'resume') and (p / 'training_state.pt').exists()
              and (p / 'COMPLETE.json').exists()]
        if not ck:
            raise SystemExit('--resume requested but no complete training state exists')
        ck = max(ck, key=lambda p: json.loads((p / 'pretrained_model/run_meta.json').read_text())['step'])
        verify_checkpoint(ck)
        st = torch.load(ck / 'training_state.pt', weights_only=False)
        policy.load_state_dict(st['model']), optimizer.load_state_dict(st['optimizer']), scaler.load_state_dict(st['scaler'])
        step, best = st['step'], st['best']
        random.setstate(st['rng']['python'])
        np.random.set_state(st['rng']['numpy'])
        torch.set_rng_state(st['rng']['torch'])
        if a.device == 'cuda':
            torch.cuda.set_rng_state_all(st['rng']['cuda'])
        gen.set_state(st['rng']['sampler'])
        print(f'resumed from {ck} at step {step}; sampler starts a new draw (not bitwise replay)', flush=True)
    else:
        for f in ('train_log.jsonl', 'val_log.jsonl'):
            (out / f).write_text('')
    manifest = read_jsonl(a.train_set / 'manifest.jsonl')
    with (out / 'manifest.jsonl').open('w') as f:
        for r in manifest:
            if r['dataset'] == dname:
                f.write(json.dumps(r) + '\n')
    run_meta = dict(
        model=Path(a.policy_config).stem, policy_config_path=a.policy_config, train_config_path=a.train_config,
        policy_config=json.loads(json.dumps(asdict(cfg), default=str)), training=tr, seed=seed, dataset=dname,
        train_set=str(a.train_set), train_root=str(train_root), val_root=str(val_root) if val_ds else None,
        train_episodes=train_ds.meta.total_episodes, train_frames=train_ds.meta.total_frames,
        val_episodes=val_ds.meta.total_episodes if val_ds else 0, val_frames=val_ds.meta.total_frames if val_ds else 0,
        action_definition=ACTION_DEFINITION, normalization=dict(mapping={k: str(v) for k, v in cfg.normalization_mapping.items()},
                                                               stats_file='stats.json (train split only)'),
        forecast_stats=fstats, policy_inputs=list(cfg.input_features) + ([cfg.forecast_key, cfg.forecast_valid_key] if cfg.use_forecast else []),
        params=dict(total=n_params, trainable=n_train, used_by_continuous_act=n_used, forecast_encoder=n_fc),
        optimizer=dict(type='AdamW (LeRobot ACT preset)', lr=cfg.optimizer_lr, lr_backbone=cfg.optimizer_lr_backbone,
                       weight_decay=cfg.optimizer_weight_decay, grad_clip_norm=clip, scheduler=None),
        git_commit=git_commit(), created=datetime.now(timezone.utc).isoformat(),
        versions=dict(python=platform.python_version(), torch=torch.__version__, lerobot=getattr(lerobot, '__version__', '?')),
        device=torch.cuda.get_device_name(0) if a.device == 'cuda' else 'cpu')
    (out / 'run_meta.json').write_text(json.dumps(run_meta, indent=1))
    (out / 'stats.json').write_text((train_root / 'meta/stats.json').read_text())
    print(f'{run_meta["model"]}: {dname} train {run_meta["train_episodes"]} ep / {run_meta["train_frames"]} fr, val '
          f'{run_meta["val_episodes"]} ep; params {n_params / 1e6:.3f}M (continuous ACT {n_used / 1e6:.3f}M, forecast '
          f'{n_fc}); steps {tr["steps"]} batch {tr["batch_size"]} sampling {tr["sampling"]}; device {run_meta["device"]}', flush=True)

    def save(d: Path, state: bool):
        def write(temp):
            md = temp / 'pretrained_model'
            policy.save_pretrained(md)
            preprocessor.save_pretrained(md)
            postprocessor.save_pretrained(md)
            (md / 'run_meta.json').write_text(json.dumps(dict(run_meta, step=step, best_val_l1=best), indent=1))
            if state:
                rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                           cuda=torch.cuda.get_rng_state_all() if a.device == 'cuda' else [], sampler=gen.get_state())
                torch.save(dict(model=policy.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                                step=step, best=best, rng=rng), temp / 'training_state.pt')
        publish_checkpoint(d, write)
        if d.name.isdigit():
            old = sorted(p for p in d.parent.iterdir() if p.name.isdigit() and (p / 'COMPLETE.json').exists())
            for previous in old[:-2]:
                shutil.rmtree(previous)

    stopping = False
    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    @torch.no_grad()
    def validate() -> dict:
        policy.eval()
        g = Groups(va_meta)
        tot, n = 0.0, 0
        for i, b in enumerate(val_loader):
            if tr.get('val_max_batches') and i >= int(tr['val_max_batches']):
                break
            eps = b['episode_index']
            b = preprocessor(b)
            with torch.autocast('cuda', dtype=torch.float16, enabled=tr['amp']):
                policy.forward(b)
            per = policy.per_sample_l1(b).float().cpu()
            g.add(eps, per)
            tot, n = tot + float(per.sum()), n + len(per)
        policy.train()
        return dict(step=step, val_l1=tot / max(n, 1), val_samples=n, **g.result('val_'))

    policy.train()
    if not a.resume:
        save(out / 'checkpoints' / '000000', state=True)
    groups = Groups(tr_meta)
    acc = defaultdict(list)
    t0 = time.time()
    s0 = step
    while step < int(tr['steps']):
        for b in train_loader:
            eps = b['episode_index']
            b = preprocessor(b)
            with torch.autocast('cuda', dtype=torch.float16, enabled=tr['amp']):
                loss, info = policy.forward(b)
            groups.add(eps, policy.per_sample_l1(b).detach().float().cpu())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if stopping:
                save(out / 'checkpoints' / f'{step:06d}', state=True)
                print(f'TRAIN_PAUSED step {step}', flush=True)
                return
            for k, v in dict(loss=loss.item(), grad_norm=float(gn), **info).items():
                acc[k].append(v)
            last = step >= int(tr['steps'])
            if step % int(tr['log_every']) == 0 or last:
                row = dict(step=step, **{k: float(np.mean(v)) for k, v in acc.items()}, **groups.result('train_'),
                           elapsed_s=round(time.time() - t0, 1), steps_per_s=round((step - s0) / max(time.time() - t0, 1e-6), 3))
                acc.clear()
                with (out / 'train_log.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                print(f'step={step} loss={row["loss"]:.4f} l1={row.get("l1_loss", float("nan")):.4f} '
                      f'kld={row.get("kld_loss", float("nan")):.4f} steps/s={row["steps_per_s"]}', flush=True)
            if val_loader is not None and (step % int(tr['val_every']) == 0 or last):
                v = validate()
                with (out / 'val_log.jsonl').open('a') as f:
                    f.write(json.dumps(v) + '\n')
                print(f'step={step} val_l1={v["val_l1"]:.4f} (best {min(best, v["val_l1"]):.4f})', flush=True)
                if v['val_l1'] < best:
                    best = v['val_l1']
                    save(out / 'checkpoints' / 'best', state=False)
                    (out / 'checkpoints' / 'best' / 'best.json').write_text(json.dumps(dict(step=step, val_l1=best)))
            if tr.get('save_every') and step % int(tr['save_every']) == 0 and not last:
                save(out / 'checkpoints' / f'{step:06d}', state=True)
            if last:
                break
    save(out / 'checkpoints' / f'{step:06d}', state=True)
    save(out / 'checkpoints' / 'last', state=True)
    try:
        plot_curves(out)
    except Exception as e:                       # plotting never fails a finished run
        print('curves not plotted:', e, flush=True)
    print(f'TRAIN_FINISHED {run_meta["model"]} step {step} best_val_l1 {best:.5f}', flush=True)


def plot_curves(out: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tr, va = read_jsonl(out / 'train_log.jsonl'), read_jsonl(out / 'val_log.jsonl')
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot([r['step'] for r in tr], [r.get('l1_loss', r['loss']) for r in tr], label='train L1')
    if va:
        ax[0].plot([r['step'] for r in va], [r['val_l1'] for r in va], 'o-', label='val L1 (z = 0)')
    ax[0].set_xlabel('step'), ax[0].set_ylabel('L1 (normalised action)'), ax[0].legend(), ax[0].set_yscale('log')
    keys = sorted({k for r in va for k in r if k.startswith('val_l1/task/')})
    for k in keys:
        ax[1].plot([r['step'] for r in va], [r.get(k, np.nan) for r in va], 'o-', label=k.split('/')[-1])
    ax[1].set_xlabel('step'), ax[1].set_ylabel('val L1 per task'), ax[1].legend(fontsize=7)
    fig.suptitle(out.name)
    fig.tight_layout()
    fig.savefig(out / 'curves.png', dpi=110)


if __name__ == '__main__':
    main()
