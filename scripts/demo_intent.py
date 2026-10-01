"""Does the intent reach the policy? Offline counterfactual demo (docs/requirements/INTENT_ACT_GUIDE.md §3.2).

For recorded episodes of a labelled dataset, run an intent-token checkpoint (z = 0, eval mode) at every timestep in
four modes:
  original   the episode's own intent labels
  swap       one component replaced: obj / act by `--to <vocab name>`; tau / xi from `--from-episode` (same frame
             index, clipped to its length)
  drop       that component replaced by its learned UNKNOWN / null token (keep[c] = False)
  drop_all   every component UNKNOWN (should behave like the no-intent model B, `--baseline-ckpt`)
and compare the first action of each predicted chunk (restricted / discrete head: argmax differs; continuous head:
L1 of the unnormalised joint/gripper target > `--eps`).

Output per episode in <out>/ep<k>/: actions.png (first action over time, 4 lines; dotted / dashed / solid vertical
lines at t_onset / t_clear / t_event of every intent segment) and summary.txt (fraction of timesteps whose action
changes w.r.t. the original, overall, before / after the first segment's t_clear and per segment).

Example:
  ./run.sh -m scripts.demo_intent --ckpt outputs/train/intent_act/D/checkpoints/last/pretrained_model \\
      --dataset-root outputs/datasets/intent_act_early --episodes 3 --swap act --to give_to_robot \\
      --out outputs/demo_intent/act --baseline-ckpt outputs/train/intent_act/B/checkpoints/last/pretrained_model
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from intent_policy.policies.policy_agent import load_policy
from intent_policy.sim.restricted_action import RestrictedAction
from intent_policy.utils import resolve

MODES = ('original', 'swap', 'drop', 'drop_all')
COMPONENTS = ('obj', 'act', 'tau', 'xi')


def episode_frames(root: Path, episode: int, video_backend: str | None = None) -> tuple[LeRobotDataset, list[int]]:
    ds = LeRobotDataset(root.name, root=root, episodes=[episode], **({'video_backend': video_backend} if video_backend else {}))
    return ds, list(range(len(ds)))


def intent_arrays(ds: LeRobotDataset) -> dict[str, np.ndarray]:
    cols = ['intent.obj', 'intent.act', 'intent.phase', 'intent.tte', 'intent.xi']
    d = ds.hf_dataset.with_format('numpy', columns=cols)[:]
    return {k.removeprefix('intent.'): np.asarray(d[k]) for k in cols}


def mode_intent(base: dict, mode: str, swap: str, swap_value, idx: np.ndarray, components) -> tuple[dict, dict | None]:
    """Raw intent arrays of the frames `idx` for one mode, and its keep mask (None: keep everything)."""
    it = {k: v[idx] for k, v in base.items()}
    keep = None
    if mode == 'swap':
        if swap in ('obj', 'act'):
            it[swap] = np.full_like(it[swap], swap_value)
        else:
            j = np.minimum(idx, len(swap_value['phase']) - 1)
            if swap == 'tau':
                it['phase'], it['tte'] = swap_value['phase'][j], swap_value['tte'][j]
            else:
                it['xi'] = swap_value['xi'][j]
    elif mode == 'drop':
        keep = {c: np.full(len(idx), c != swap) for c in components}
    elif mode == 'drop_all':
        keep = {c: np.zeros(len(idx), bool) for c in components}
    return it, keep


@torch.no_grad()
def first_actions(policy, pre, post, ds, frames, intent=None, keep_fn=None, batch_size: int = 32) -> np.ndarray:
    """First action of the predicted chunk at every frame: restricted ids (T,) or continuous targets (T, 7).
    intent(idx) -> raw intent arrays, keep_fn(idx) -> keep mask; both None for a no-intent policy."""
    policy.eval()
    dev = policy.config.device
    keys = [k for k in policy.config.input_features]
    out = []
    for s in range(0, len(frames), batch_size):
        idx = np.asarray(frames[s:s + batch_size])
        items = [ds[int(i)] for i in idx]
        batch = pre({k: torch.stack([it[k] for it in items]) for k in keys})
        if intent is not None:
            it = intent(idx)
            enc = policy.intent_encoder
            batch['intent'] = dict(obj=torch.as_tensor(it['obj']).reshape(-1).long().to(dev),
                                   act=torch.as_tensor(it['act']).reshape(-1).long().to(dev),
                                   tau=torch.as_tensor(np.stack([it['phase'].reshape(-1), it['tte'].reshape(-1)], -1)).float().to(dev),
                                   xi=enc.normalize_xi(torch.as_tensor(it['xi']).float().to(dev)))
            keep = keep_fn(idx) if keep_fn is not None else None
            batch['intent_keep'] = None if keep is None else {c: torch.as_tensor(v).bool().to(dev) for c, v in keep.items()}
        if policy.config.action_mode == 'restricted':
            out.append(policy.predict_restricted_chunk(batch)[:, 0].argmax(-1).cpu().numpy())
        else:
            out.append(post(policy.predict_action_chunk(batch)[:, 0]).cpu().numpy())
    return np.concatenate(out)


def changed(a: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    return a != b if a.ndim == 1 else np.abs(a - b).sum(-1) > eps


def frac(mask: np.ndarray, where: np.ndarray) -> str:
    n = int(where.sum())
    return f'{mask[where].mean():6.1%} ({int(mask[where].sum())}/{n})' if n else '   n/a (0/0)'


def summary_text(actions: dict, segments: list[dict], eps: float, header: str, baseline=None) -> str:
    T = len(actions['original'])
    t = np.arange(T)
    lines = [header, f'frames {T}; segments ' + '; '.join(
        f"{s['act']}@{s['obj']} onset {s['t_onset']} clear {s['t_clear']} event {s['t_event']} end {s.get('t_end', s['t_event'])}"
        for s in segments), '']
    first = segments[0]['t_clear'] if segments else T
    lines.append(f"{'action changed vs original':32s} {'all frames':18s}   {'t < t_clear(1st seg)':18s}   t >= t_clear(1st seg)")
    for m in MODES[1:]:
        c = changed(actions[m], actions['original'], eps)
        lines.append(f'  {m:30s} {frac(c, t >= 0):18s}   {frac(c, t < first):18s}   {frac(c, t >= first)}')
    if baseline is not None:
        c = changed(actions['drop_all'], baseline, eps)
        lines.append(f"  {'drop_all vs baseline B':30s} {frac(c, t >= 0):18s}   {frac(c, t < first):18s}   {frac(c, t >= first)}")
        c = changed(actions['original'], baseline, eps)
        lines.append(f"  {'original vs baseline B':30s} {frac(c, t >= 0):18s}   {frac(c, t < first):18s}   {frac(c, t >= first)}")
    lines += ['', f"{'per segment (swap vs original)':32s} {'[t_onset, t_clear)':18s}   [t_clear, t_end]"]
    c = changed(actions['swap'], actions['original'], eps)
    for s in segments:
        pre = (t >= s['t_onset']) & (t < s['t_clear'])
        post = (t >= s['t_clear']) & (t <= s.get('t_end', s['t_event']))
        lines.append(f"  {s['act'] + '@' + s['obj']:30s} {frac(c, pre):18s}   {frac(c, post)}")
    lines.append(f"  {'outside segments':30s} {frac(c, ~np.any([(t >= s['t_onset']) & (t <= s.get('t_end', s['t_event'])) for s in segments] or [np.zeros(T, bool)], 0))}")
    return '\n'.join(lines) + '\n'


def plot(actions: dict, segments: list[dict], path: Path, title: str, baseline=None, fps: float = 20.0) -> None:
    disc = actions['original'].ndim == 1
    dims = 1 if disc else actions['original'].shape[1]
    fig, axes = plt.subplots(dims, 1, figsize=(12, 3.2 if disc else 1.6 * dims), sharex=True, squeeze=False)
    t = np.arange(len(actions['original'])) / fps
    series = dict(actions, **({'baseline B': baseline} if baseline is not None else {}))
    for d, ax in enumerate(axes[:, 0]):
        for k, (name, a) in enumerate(series.items()):
            y = a if disc else a[:, d]
            ax.step(t, y + (0.06 * (k - 1.5) if disc else 0.0), where='post', lw=1.4 if name == 'original' else 1.0,
                    label=name, alpha=0.9)
        for s in segments:
            for key, ls in (('t_onset', ':'), ('t_clear', '--'), ('t_event', '-')):
                ax.axvline(s[key] / fps, color='k', ls=ls, lw=0.8, alpha=0.6)
        if disc:
            ax.set_yticks(range(len(RestrictedAction)), [a.name for a in RestrictedAction], fontsize=7)
        else:
            ax.set_ylabel(f'a[{d}]', fontsize=8)
    axes[0, 0].set_title(title + '   (vertical lines: dotted t_onset, dashed t_clear, solid t_event)', fontsize=9)
    axes[0, 0].legend(fontsize=7, ncol=5, loc='upper right')
    axes[-1, 0].set_xlabel('t [s]')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True, help='intent-token checkpoint (pretrained_model directory)')
    p.add_argument('--dataset-root', required=True, help='dataset with intent.* labels')
    p.add_argument('--episodes', type=int, nargs='+', required=True)
    p.add_argument('--swap', choices=COMPONENTS, required=True)
    p.add_argument('--to', help='obj / act: the vocabulary name to swap in')
    p.add_argument('--from-episode', type=int, help='tau / xi: take the component from this episode')
    p.add_argument('--baseline-ckpt', help='no-intent model (B) for comparison with drop_all')
    p.add_argument('--eps', type=float, default=0.05, help='continuous head: L1 change threshold [rad, m]')
    p.add_argument('--out', default='outputs/demo_intent')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--video-backend')
    args = p.parse_args()

    policy, pre, post = load_policy(args.ckpt, args.device)
    cfg = policy.config
    if policy.intent_encoder is None:
        raise SystemExit(f'{args.ckpt} is not an intent-token policy')
    components = policy.intent_encoder.components
    if args.swap not in components:
        raise SystemExit(f'the model has no {args.swap!r} component ({components})')
    root = resolve(args.dataset_root)
    schema = cfg.intent_schema
    if args.swap in ('obj', 'act'):
        vocab = schema[f'{args.swap}_vocab']
        if args.to not in vocab:
            raise SystemExit(f'--to must be one of {vocab}')
        swap_value, swap_desc = vocab.index(args.to), f'{args.swap} -> {args.to}'
    else:
        if args.from_episode is None:
            raise SystemExit('--from-episode is required to swap tau / xi')
        donor, _ = episode_frames(root, args.from_episode, args.video_backend)
        swap_value, swap_desc = intent_arrays(donor), f'{args.swap} from episode {args.from_episode}'
    base_model = load_policy(args.baseline_ckpt, args.device) if args.baseline_ckpt else None
    segs_all = json.loads((root / 'meta/intent_segments.json').read_text())
    meta = {m['episode_index']: m for m in map(json.loads, (root / 'meta/hri_episodes.jsonl').read_text().splitlines())} \
        if (root / 'meta/hri_episodes.jsonl').exists() else {}

    for ep in args.episodes:
        ds, frames = episode_frames(root, ep, args.video_backend)
        base = intent_arrays(ds)
        segments = segs_all.get(str(ep), [])
        actions = {}
        for mode in MODES:
            fn = lambda idx, mode=mode: mode_intent(base, mode, args.swap, swap_value, idx, components)
            actions[mode] = first_actions(policy, pre, post, ds, frames, intent=lambda idx, fn=fn: fn(idx)[0],
                                          keep_fn=lambda idx, fn=fn: fn(idx)[1], batch_size=args.batch_size)
        baseline = first_actions(*base_model, ds, frames, batch_size=args.batch_size) if base_model else None
        task = meta.get(ep, {}).get('list_task') or meta.get(ep, {}).get('scenario_id', '')
        header = (f'episode {ep} ({task}, scenario {meta.get(ep, {}).get("scenario_id")}, seed {meta.get(ep, {}).get("seed")}); '
                  f'model {args.ckpt} ({cfg.action_mode}, components {",".join(components)}); swap {swap_desc}; '
                  f'{"argmax differs" if cfg.action_mode == "restricted" else f"L1 > {args.eps}"}')
        out = resolve(args.out) / f'ep{ep}'
        out.mkdir(parents=True, exist_ok=True)
        text = summary_text(actions, segments, args.eps, header, baseline)
        (out / 'summary.txt').write_text(text)
        plot(actions, segments, out / 'actions.png', f'ep {ep} {task}: swap {swap_desc}', baseline, ds.meta.fps)
        np.savez_compressed(out / 'actions.npz', **actions, **({'baseline': baseline} if baseline is not None else {}))
        print(text)
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
