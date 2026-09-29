"""Train the shared LeRobot-ACT policy (restricted or continuous head) on recorded demos.

Uses LeRobot components end to end: LeRobotDataset (+ delta_timestamps for action chunks),
the ACT pre/post-processors (normalisation from dataset stats), ACTPolicy.get_optim_params
and save_pretrained. The only project-specific pieces are carrying the restricted label
(`restricted_action`, not an ACT input/output feature) around the preprocessor and, when the
policy config sets `use_intent`, declaring the oracle intention features as extra inputs.

Checkpoints: <output>/checkpoints/<step>/{pretrained_model, training_state.pt}; the newest
`--keep-last` are kept and the final model is also written to checkpoints/last. `--resume`
continues exactly (optimizer, AMP scaler, RNG states and the position inside the shuffled epoch),
also to extend a finished run with a larger `--steps`.
"""
import argparse
import json
import platform
import random
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
import lerobot
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.feature_utils import dataset_to_policy_features

from intent_policy.policies.hri_act import HRIACTConfig, HRIACTPolicy, make_hri_act_pre_post_processors
from intent_policy.utils import load_yaml, resolve


def intent_keys(policy_yaml: dict) -> list[str]:
    """Oracle features the policy reads (none without intent or for the masked No-Information control)."""
    if not policy_yaml.get('use_intent') or policy_yaml.get('intent_mask'):
        return []
    return [k for keys in (policy_yaml.get('intent_branches') or {}).values() for k in keys]


def build_config(policy_yaml: dict, meta: LeRobotDatasetMetadata, device: str, cameras: list[str]) -> HRIACTConfig:
    feats = dataset_to_policy_features(meta.features)
    image_keys = [f'observation.images.{c}' for c in cameras]
    kwargs = {k: v for k, v in policy_yaml.items() if k != 'type'}
    if policy_yaml.get('use_intent'):
        branches = policy_yaml.get('intent_branches') or {}
        missing = {k for keys in branches.values() for k in keys} - set(meta.features)
        if missing:
            raise KeyError(f'intent features missing from the dataset: {sorted(missing)}')
        kwargs['intent_branch_dims'] = {b: int(sum(meta.features[k]['shape'][0] for k in keys)) for b, keys in branches.items()}
    wanted = set(intent_keys(policy_yaml))
    inputs = {k: f for k, f in feats.items()
              if (f.type is FeatureType.VISUAL and k in image_keys) or k == 'observation.state' or k in wanted}
    outputs = {'action': feats['action']}
    return HRIACTConfig(input_features=inputs, output_features=outputs, device=device, push_to_hub=False, **kwargs)


def split_episodes(n: int, val_every: int) -> tuple[list[int], list[int]]:
    val = [i for i in range(n) if val_every and i % val_every == val_every - 1]
    return [i for i in range(n) if i not in val], val


class EpochSampler(torch.utils.data.Sampler):
    """Shuffled order fixed by (seed, epoch); `start` skips already consumed samples (exact resume)."""

    def __init__(self, n: int, seed: int):
        self.n, self.seed, self.epoch, self.start = n, seed, 0, 0

    def set_position(self, epoch: int, start: int) -> None:
        self.epoch, self.start = epoch, start

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 100003 + self.epoch)
        return iter(torch.randperm(self.n, generator=g)[self.start:].tolist())

    def __len__(self):
        return self.n - self.start


def run_batch(policy, preprocessor, batch, device, label_key, amp: bool = False):
    label = batch.pop(label_key)                       # not an ACT feature: bypass the processor
    batch = preprocessor(batch)
    batch[label_key] = label.to(device)
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
        return policy.forward(batch)


@torch.no_grad()
def evaluate(policy, preprocessor, loader, device, label_key, amp, max_batches=20) -> dict:
    policy.eval()
    acc = {}
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        loss, info = run_batch(policy, preprocessor, batch, device, label_key, amp)
        for k, v in {'loss': loss.item(), **info}.items():
            acc.setdefault(k, []).append(v)
    policy.train()
    return {f'val_{k}': float(np.mean(v)) for k, v in acc.items()}


def rng_state() -> dict:
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def set_rng_state(s: dict) -> None:
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'])
    if s.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s['cuda'])


def latest_checkpoint(out: Path) -> Path | None:
    steps = sorted(p for p in (out / 'checkpoints').glob('[0-9]*') if (p / 'training_state.pt').exists())
    return steps[-1] if steps else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/experiments/foundation_baseline.yaml')
    p.add_argument('--policy-config', help='override experiment policy.config')
    p.add_argument('--policy-set', nargs='*', default=[], metavar='KEY=VALUE',
                   help='override policy config fields (YAML values), e.g. intent_mask=true')
    p.add_argument('--dataset-root')
    p.add_argument('--repo-id')
    p.add_argument('--output-dir')
    p.add_argument('--steps', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--num-workers', type=int)
    p.add_argument('--seed', type=int, help='override experiment seed (training seed)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--val-every', type=int, default=10, help='every k-th episode is held out for validation')
    p.add_argument('--save-every', type=int, help='override training.save_every')
    p.add_argument('--keep-last', type=int, default=2, help='numbered checkpoints to keep')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=None, help='fp16 autocast (default: training.amp)')
    p.add_argument('--video-backend', help='LeRobot video backend (torchcodec | pyav)')
    p.add_argument('--resume', action='store_true', help='continue from the newest checkpoint in --output-dir')
    args = p.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but unavailable (driver error?); refusing to fall back to a CPU run')
    exp = load_yaml(args.config)
    tr = exp['training']
    policy_cfg_path = args.policy_config or exp['policy']['config']
    policy_yaml = load_yaml(policy_cfg_path)
    for item in args.policy_set:
        key, value = item.split('=', 1)
        policy_yaml[key] = yaml.safe_load(value)
    root = resolve(args.dataset_root or exp['data']['root'])
    repo_id = args.repo_id or exp['data']['repo_id']
    out = resolve(args.output_dir or tr['output_dir'])
    steps = args.steps or tr['steps']
    batch_size = args.batch_size or tr['batch_size']
    num_workers = tr['num_workers'] if args.num_workers is None else args.num_workers
    save_every = args.save_every or tr['save_every']
    amp = (tr.get('amp', False) if args.amp is None else args.amp) and args.device == 'cuda'
    seed = int(exp['seed'] if args.seed is None else args.seed)
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    cfg = build_config(policy_yaml, meta, args.device, exp['observation']['cameras'])
    label_key = cfg.restricted_label_key
    delta = {k: [i / meta.fps for i in range(cfg.chunk_size)] for k in ('action', label_key)}
    train_eps, val_eps = split_episodes(meta.total_episodes, args.val_every)
    ds_kw = dict(root=root, delta_timestamps=delta, **({'video_backend': args.video_backend} if args.video_backend else {}))
    train_ds = LeRobotDataset(repo_id, episodes=train_eps, **ds_kw)
    val_ds = LeRobotDataset(repo_id, episodes=val_eps, **ds_kw) if val_eps else None
    loader_kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=args.device == 'cuda', drop_last=True,
                     persistent_workers=False)
    sampler = EpochSampler(len(train_ds), seed)
    # own generator: creating a loader iterator must not consume the global torch RNG (dropout), or resume drifts
    train_loader = torch.utils.data.DataLoader(train_ds, sampler=sampler, generator=torch.Generator().manual_seed(seed),
                                               **loader_kw)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, generator=torch.Generator().manual_seed(seed),
                                             **loader_kw) if val_ds else None

    policy = HRIACTPolicy(cfg).to(args.device)
    preprocessor, postprocessor = make_hri_act_pre_post_processors(cfg, dataset_stats=meta.stats)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=cfg.optimizer_lr, weight_decay=cfg.optimizer_weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / 'train_log.jsonl'
    step, epoch, consumed, elapsed0 = 0, 0, 0, 0.0
    if args.resume and (ckpt_dir := latest_checkpoint(out)) is not None:
        state = torch.load(ckpt_dir / 'training_state.pt', weights_only=False)
        policy.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        set_rng_state(state['rng'])
        step, epoch, consumed, elapsed0 = state['step'], state['epoch'], state['consumed'], state.get('elapsed_s', 0.0)
        print(f'resumed from {ckpt_dir} (step {step}, epoch {epoch}, {consumed} samples into the epoch)', flush=True)
    else:
        log_path.write_text('')
    gpu = torch.cuda.get_device_name(0) if args.device == 'cuda' else 'cpu'
    print(f'train episodes {len(train_eps)} ({len(train_ds)} frames), val episodes {len(val_eps)}; mode={cfg.action_mode}; '
          f'intent={cfg.use_intent}; params={sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M; steps={steps}; '
          f'batch={batch_size}; amp={amp}; device={gpu}', flush=True)

    def metadata(tag_step: int, last_info: dict) -> dict:
        return dict(experiment=exp, experiment_config_path=str(args.config), policy_config_path=str(policy_cfg_path),
                    policy_config=json.loads(json.dumps(asdict(cfg), default=str)), dataset_root=str(root),
                    dataset_repo_id=repo_id, dataset_episodes=meta.total_episodes, dataset_frames=meta.total_frames,
                    train_episodes=train_eps, val_episodes=val_eps, seed=seed, step=tag_step, steps_total=steps,
                    batch_size=batch_size, amp=amp, num_workers=num_workers, last_log=last_info,
                    created=datetime.now(timezone.utc).isoformat(),
                    versions=dict(python=platform.python_version(), torch=torch.__version__,
                                  lerobot=getattr(lerobot, '__version__', 'unknown')),
                    hardware=dict(device=gpu, host=platform.node()),
                    intent_provider=exp['intent']['provider'], intent_features=intent_keys(policy_yaml),
                    policy_overrides=list(args.policy_set))

    def save(ckpt_dir: Path, last_info: dict, with_state: bool) -> Path:
        model_dir = ckpt_dir / 'pretrained_model'
        policy.save_pretrained(model_dir)
        preprocessor.save_pretrained(model_dir)
        postprocessor.save_pretrained(model_dir)
        (model_dir / 'training_metadata.json').write_text(json.dumps(metadata(step, last_info), indent=2))
        if with_state:
            torch.save(dict(model=policy.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                            rng=rng_state(), step=step, epoch=epoch, consumed=consumed,
                            elapsed_s=elapsed0 + time.time() - t0), ckpt_dir / 'training_state.pt')
            numbered = sorted(p for p in (out / 'checkpoints').glob('[0-9]*'))
            for old in numbered[:-args.keep_last]:
                shutil.rmtree(old)
        return model_dir

    policy.train()
    t0, info_acc, last = time.time(), {}, {}
    while step < steps:
        sampler.set_position(epoch, consumed)
        for batch in train_loader:
            loss, info = run_batch(policy, preprocessor, batch, args.device, label_key, amp)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            consumed += batch_size
            for k, v in {'loss': loss.item(), 'grad_norm': grad.item(), **info}.items():
                info_acc.setdefault(k, []).append(v)
            if step % tr['log_every'] == 0 or step == steps:
                last = {k: float(np.mean(v)) for k, v in info_acc.items()}
                info_acc = {}
                if val_loader is not None and (step % (tr['log_every'] * 10) == 0 or step == steps):
                    last.update(evaluate(policy, preprocessor, val_loader, args.device, label_key, amp))
                elapsed = elapsed0 + time.time() - t0
                last.update(step=step, epoch=epoch, elapsed_s=round(elapsed, 1), steps_per_s=round(step / max(elapsed, 1e-6), 2))
                with open(log_path, 'a') as f:
                    f.write(json.dumps(last) + '\n')
                print(' '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in last.items()), flush=True)
            if step % save_every == 0 and step != steps:
                save(out / 'checkpoints' / f'{step:06d}', last, with_state=True)
            if step >= steps:
                break
        else:
            epoch, consumed = epoch + 1, 0
    save(out / 'checkpoints' / f'{step:06d}', last, with_state=True)     # lets --resume extend a finished run
    ckpt = save(out / 'checkpoints' / 'last', last, with_state=False)
    print(f'saved {ckpt}')


if __name__ == '__main__':
    main()
