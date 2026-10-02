"""Train the shared LeRobot-ACT policy (restricted or continuous head) on recorded demos.

Uses LeRobot components end to end: LeRobotDataset (+ delta_timestamps for action chunks),
the ACT pre/post-processors (normalisation from dataset stats), ACTPolicy.get_optim_params
and save_pretrained. The only project-specific pieces are carrying the restricted label
(`restricted_action`, not an ACT input/output feature) around the preprocessor and, when the
policy config sets `use_intent`, declaring the oracle intention features as extra inputs (fusion) or carrying the
intent contract around the preprocessor (intent tokens, docs/requirements/INTENT_ACT_GUIDE_v2.md §5.2):
  --[no-]use-task-token    task token T1-T4 (default: experiment `policy.use_task_token`, else off)
  --use-intent             intent tokens (default off: the dataset's intent_* columns are ignored)
  --intent-source          hindsight | perfect | predicted (columns intent_hs / intent_pp / intent_pr<r>, read as intent.*)
  --intent-source-mix      e.g. predicted:0.7,perfect:0.3 (source drawn per sample; predicted: replica drawn uniformly)
  --noise-scale-range      must match the range the predicted replicas were built with (schema predicted.*)
  --p-drop-group 0.15  --p-drop-all 0.1  --intent-keep semantic,spatial,memory,time,motion  --[no-]intent-in-cvae
  --action-mode continuous | discrete (= restricted 9-action head)   --intent-film   --n-action-steps   --ckpt-rule last
xi is normalised with mean / std over the training episodes (std >= 1e-2) of the (first) source, stored in the model.
Metadata records the full config, seed, git commit and the checkpoint rule.

Checkpoints: <output>/checkpoints/<step>/{pretrained_model, training_state.pt}; the newest
`--keep-last` are kept and the final model is also written to checkpoints/last. `--resume`
continues exactly (optimizer, AMP scaler, RNG states and the position inside the shuffled epoch),
also to extend a finished run with a larger `--steps`.
"""
import argparse
import json
import subprocess
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

from intent_policy.intent.labels import SCHEMA_PATH, load_schema
from intent_policy.policies.hri_act import HRIACTConfig, HRIACTPolicy, make_hri_act_pre_post_processors
from intent_policy.utils import load_yaml, resolve

INTENT_COLUMN_PREFIX = 'intent_'
TASK_ID = 'hri.task_id'
SOURCE_PREFIX = {'hindsight': ['intent_hs'], 'perfect': ['intent_pp'], 'predicted': None}    # predicted: intent_pr<r>
CONTRACT_KEYS = ('targets', 'who', 'phases', 'tasks', 'n_waypoints', 'keypoints', 'keypoint_dim', 'fps', 'horizon_s', 'tte_max_s')


def intent_keys(policy_yaml: dict) -> list[str]:
    """Oracle features the fusion policy reads (none without intent, for the masked No-Information control or for
    intent tokens)."""
    if not policy_yaml.get('use_intent') or policy_yaml.get('intent_mask') or policy_yaml.get('intent_arch') == 'tokens':
        return []
    return [k for keys in (policy_yaml.get('intent_branches') or {}).values() for k in keys]


def uses_intent_tokens(policy_yaml: dict) -> bool:
    return bool(policy_yaml.get('use_intent')) and policy_yaml.get('intent_arch') == 'tokens'


def source_prefixes(source: str, meta: LeRobotDatasetMetadata) -> list[str]:
    """Dataset column prefixes of an intent source (predicted: every replica present)."""
    if source == 'predicted':
        out = sorted({k.split('.')[0] for k in meta.features if k.startswith('intent_pr')})
    else:
        out = [p for p in SOURCE_PREFIX[source] if f'{p}.p_target' in meta.features]
    if not out:
        raise KeyError(f'intent source {source!r} missing from the dataset (run scripts/build_intent_labels.py --sources {source})')
    return out


def parse_mix(source: str, mix: str | None) -> dict[str, float]:
    if not mix:
        return {source: 1.0}
    out = {}
    for item in mix.split(','):
        name, w = item.split(':')
        if name.strip() not in SOURCE_PREFIX:
            raise ValueError(f'unknown intent source {name!r} in --intent-source-mix')
        out[name.strip()] = float(w)
    return out


def check_intent_dataset(schema: dict, meta: LeRobotDatasetMetadata, sources: list[str] = ('hindsight',)) -> None:
    """The dataset must carry the contract of every requested source, built with the same schema sizes."""
    for src in sources:
        source_prefixes(src, meta)
    if TASK_ID not in meta.features:
        raise KeyError(f'{TASK_ID} missing from the dataset (run scripts/build_intent_labels.py)')
    built = json.loads((meta.root / 'meta/intent_schema.json').read_text())
    for k in CONTRACT_KEYS:
        if built.get(k) != schema.get(k):
            raise ValueError(f'intent schema {k!r} differs from the one the dataset was built with: {schema.get(k)} != {built.get(k)}')


class IntentSelector:
    """Picks, per sample, the intent source (weights of --intent-source-mix; predicted: a replica uniformly) and
    exposes its columns as intent.<field>. Evaluation uses the first source (first replica) only."""

    def __init__(self, mix: dict[str, list[str]], weights: dict[str, float], seed: int):
        self.names = list(mix)
        self.prefixes = mix
        w = np.array([weights[n] for n in self.names], float)
        self.p = torch.as_tensor(w / w.sum())
        self.g = torch.Generator().manual_seed(seed + 17)

    @property
    def primary(self) -> str:
        return self.prefixes[self.names[0]][0]

    def select(self, cols: dict, train: bool) -> dict:
        fields = sorted({k.split('.', 1)[1] for k in cols if k.startswith(self.primary + '.')})
        b = cols[f'{self.primary}.p_target'].shape[0]
        if not train or (len(self.names) == 1 and len(self.prefixes[self.names[0]]) == 1):
            return {f'intent.{f}': cols[f'{self.primary}.{f}'] for f in fields}
        choice = []
        for src in torch.multinomial(self.p, b, replacement=True, generator=self.g).tolist():
            reps = self.prefixes[self.names[src]]
            choice.append(reps[int(torch.randint(len(reps), (1,), generator=self.g))])
        out = {}
        for f in fields:
            stack = {p: cols[f'{p}.{f}'] for p in set(choice)}
            out[f'intent.{f}'] = torch.stack([stack[p][i] for i, p in enumerate(choice)])
        return out


def git_commit() -> str:
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=resolve('.'), capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=resolve('.'), capture_output=True, text=True).stdout.strip()
        return sha + ('-dirty' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def build_config(policy_yaml: dict, meta: LeRobotDatasetMetadata, device: str, cameras: list[str]) -> HRIACTConfig:
    feats = dataset_to_policy_features(meta.features)
    image_keys = [f'observation.images.{c}' for c in cameras]
    kwargs = {k: v for k, v in policy_yaml.items() if k != 'type'}
    if uses_intent_tokens(policy_yaml):
        check_intent_dataset(policy_yaml['intent_schema'], meta, list(parse_mix(policy_yaml.get('intent_source', 'hindsight'),
                                                                                policy_yaml.pop('intent_source_mix', None))))
        kwargs.pop('intent_source_mix', None)
    elif policy_yaml.get('use_task_token') and TASK_ID not in meta.features:
        raise KeyError(f'{TASK_ID} missing from the dataset (run scripts/build_intent_labels.py)')
    if policy_yaml.get('use_intent') and policy_yaml.get('intent_arch', 'fusion') == 'fusion':
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


def run_batch(policy, preprocessor, batch, device, label_key, amp: bool = False, selector: IntentSelector | None = None):
    # the restricted label, the task id and the intent columns are not ACT features: bypass the processor
    extra = {k: batch.pop(k) for k in list(batch) if k in (label_key, TASK_ID) or k.startswith(INTENT_COLUMN_PREFIX)}
    batch = preprocessor(batch)
    if selector is not None:
        extra.update(selector.select(extra, policy.training))
    batch.update({k: v.to(device) for k, v in extra.items() if not k.startswith(INTENT_COLUMN_PREFIX)})
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
        return policy.forward(batch)


@torch.no_grad()
def evaluate(policy, preprocessor, loader, device, label_key, amp, max_batches=20, selector=None) -> dict:
    policy.eval()
    acc = {}
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        loss, info = run_batch(policy, preprocessor, batch, device, label_key, amp, selector)
        for k, v in {'loss': loss.item(), **info}.items():
            acc.setdefault(k, []).append(v)
    policy.train()
    return {f'val_{k}': float(np.mean(v)) for k, v in acc.items()}


def xi_stats(ds: LeRobotDataset, prefix: str = 'intent_hs') -> tuple[np.ndarray, np.ndarray]:
    """Mean / std of raw xi of one source over every frame of `ds` (the training episodes)."""
    col = f'{prefix}.xi'
    xi = np.asarray(ds.hf_dataset.with_format('numpy', columns=[col])[col], np.float64)
    return xi.mean(0), xi.std(0)


def guide_overrides(args) -> dict:
    """INTENT_ACT_GUIDE_v2.md §5.2 flags -> policy config fields (only the flags that were given)."""
    out = {}
    if args.use_intent:
        out.update(use_intent=True, intent_arch='tokens', intent_schema=load_schema(args.intent_schema))
    if args.use_task_token is not None:
        out['use_task_token'] = args.use_task_token
        if args.use_task_token and 'intent_schema' not in out:
            out['intent_schema'] = load_schema(args.intent_schema)
    if args.intent_keep is not None:
        out['intent_keep'] = [g.strip() for g in args.intent_keep.split(',') if g.strip()]
    for flag, key in (('intent_source', 'intent_source'), ('intent_source_mix', 'intent_source_mix'),
                      ('intent_in_cvae', 'intent_in_cvae'), ('p_drop_group', 'intent_p_drop_group'),
                      ('p_drop_all', 'intent_p_drop_all'), ('intent_film', 'intent_film'), ('n_action_steps', 'n_action_steps')):
        if getattr(args, flag) is not None:
            out[key] = getattr(args, flag)
    if args.action_mode is not None:
        out['action_mode'] = 'restricted' if args.action_mode == 'discrete' else args.action_mode
    return out


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
    p.add_argument('--keep-last', type=int, help='numbered checkpoints to keep (0: all; default training.keep_last or 2)')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=None, help='fp16 autocast (default: training.amp)')
    p.add_argument('--video-backend', help='LeRobot video backend (torchcodec | pyav)')
    p.add_argument('--resume', action='store_true', help='continue from the newest checkpoint in --output-dir')
    g = p.add_argument_group('task / intent tokens (docs/requirements/INTENT_ACT_GUIDE_v2.md §5.2)')
    g.add_argument('--use-task-token', '--use_task_token', action=argparse.BooleanOptionalAction, default=None,
                   help='task token T1-T4 (default: experiment policy.use_task_token, else off)')
    g.add_argument('--use-intent', '--use_intent', action='store_true', help='intent tokens (default: off)')
    g.add_argument('--intent-schema', default=SCHEMA_PATH)
    g.add_argument('--intent-source', '--intent_source', choices=['hindsight', 'perfect', 'predicted'],
                   help='intent source (default: hindsight)')
    g.add_argument('--intent-source-mix', '--intent_source_mix', help='e.g. predicted:0.7,perfect:0.3')
    g.add_argument('--noise-scale-range', '--noise_scale_range',
                   help='lo,hi: must equal the predicted replicas\' build range (meta/intent_schema.json)')
    g.add_argument('--intent-in-cvae', '--intent_in_cvae', action=argparse.BooleanOptionalAction, default=None,
                   help='extra tokens also in the CVAE encoder (default: on; unused by the restricted head)')
    g.add_argument('--p-drop-group', '--p_drop_group', type=float, help='per-group dropout probability (default 0.15)')
    g.add_argument('--p-drop-all', '--p_drop_all', type=float, help='drop-all-intent probability (default 0.1)')
    g.add_argument('--intent-keep', '--intent_keep', help='groups kept at evaluation (default: all)')
    g.add_argument('--action-mode', '--action_mode', choices=['continuous', 'discrete', 'restricted'],
                   help='override the policy action mode (discrete = restricted 9-action head)')
    g.add_argument('--intent-film', '--intent_film', action=argparse.BooleanOptionalAction, default=None,
                   help='FiLM fallback on the backbone features (default: off)')
    g.add_argument('--n-action-steps', '--n_action_steps', type=int, help='override the policy n_action_steps')
    g.add_argument('--ckpt-rule', '--ckpt_rule', default='last', choices=['last'],
                   help='checkpoint selection rule recorded in the metadata (never chosen on test results)')
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
    if args.use_task_token is None and exp['policy'].get('use_task_token') is not None:
        args.use_task_token = bool(exp['policy']['use_task_token'])
    policy_yaml.update(guide_overrides(args))
    if isinstance(policy_yaml.get('intent_schema'), str):
        policy_yaml['intent_schema'] = load_schema(policy_yaml['intent_schema'])
    mix_spec = policy_yaml.get('intent_source_mix')
    root = resolve(args.dataset_root or exp['data']['root'])
    repo_id = args.repo_id or exp['data']['repo_id']
    out = resolve(args.output_dir or tr['output_dir'])
    steps = args.steps or tr['steps']
    batch_size = args.batch_size or tr['batch_size']
    num_workers = tr['num_workers'] if args.num_workers is None else args.num_workers
    save_every = args.save_every or tr['save_every']
    keep_last = args.keep_last if args.keep_last is not None else int(tr.get('keep_last', 2))
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
    selector = None
    if policy.intent_encoder is not None:
        weights = parse_mix(cfg.intent_source, mix_spec)
        selector = IntentSelector({s: source_prefixes(s, meta) for s in weights}, weights, seed)
        if args.noise_scale_range and 'predicted' in weights:
            built = json.loads((root / 'meta/intent_schema.json').read_text())['predicted']['noise_scale_range']
            if [float(x) for x in args.noise_scale_range.split(',')] != [float(x) for x in built]:
                raise SystemExit(f'--noise-scale-range {args.noise_scale_range} != the dataset build range {built}: rebuild '
                                 'the predicted replicas (configs/intent_schema.yaml predicted.noise_scale_range)')
        mean, std = xi_stats(train_ds, selector.primary)
        policy.intent_encoder.set_xi_stats(mean, std)
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
          f'intent={cfg.use_intent and cfg.intent_arch}{"/" + cfg.intent_source if selector else ""}; task_token={cfg.use_task_token}; '
          f'cvae={"used" if cfg.use_vae else "unused (restricted head)"}; params={sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M; steps={steps}; '
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
                    intent_tokens=dict(source=cfg.intent_source, source_mix=mix_spec, columns=selector.prefixes if selector else None,
                                       in_cvae=cfg.intent_in_cvae, cvae_used=bool(cfg.use_vae), p_drop_group=cfg.intent_p_drop_group,
                                       p_drop_all=cfg.intent_p_drop_all, keep=list(cfg.intent_keep), film=cfg.intent_film,
                                       calibrated=not (cfg.intent_source == 'predicted' or 'predicted' in str(mix_spec)) or
                                       bool(cfg.intent_schema.get('predicted', {}).get('noise')))
                    if uses_intent_tokens(policy_yaml) else None,
                    use_task_token=cfg.use_task_token, git_commit=git_commit(), ckpt_rule=args.ckpt_rule, keep_last=keep_last,
                    policy_overrides=list(args.policy_set) + [f'{k}={v}' for k, v in guide_overrides(args).items()
                                                              if k != 'intent_schema'])

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
            for old in (numbered[:-keep_last] if keep_last else []):
                shutil.rmtree(old)
        return model_dir

    policy.train()
    t0, info_acc, last = time.time(), {}, {}
    while step < steps:
        sampler.set_position(epoch, consumed)
        for batch in train_loader:
            loss, info = run_batch(policy, preprocessor, batch, args.device, label_key, amp, selector)
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
                    last.update(evaluate(policy, preprocessor, val_loader, args.device, label_key, amp, selector=selector))
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
