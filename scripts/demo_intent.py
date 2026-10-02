"""Does the policy use the intent? Offline swap / drop demo (INTENT_ACT_GUIDE_v2.md §6, M5).

On held-out episodes of a labelled dataset (default: the validation episodes of scripts/train_policy.py), every frame,
z = 0, the intent-token checkpoint runs in these modes:
  original        the episode's intent (source of the checkpoint unless --intent-source)
  swap_spatial    p_target and c_target moved to a valid place of the same kind (benchmark/counterfactual.py)
  swap_semantic   p_who and c_who: robot <-> human
  swap_time       tte shifted by +1 s (-1 s past the event, clipped), phase taken from the frame half an episode away
  drop_<group>    one information group replaced by its null tokens (semantic, spatial, memory, time, motion)
  drop_all        every intent group dropped
and the no-intent model B (--baseline-ckpt) runs on the same frames.
Metrics per bin of t - t_clear (0.2 s bins over [-1.5, 1.1) s, the nearest segment's t_clear; `pre` = t < t_clear):
  ICR   fraction of the swapped frames whose action (argmax of the first chunk step) changes
  CGR   among the changed frames, fraction whose new action is correct for the new intent (rule (a); spatial and
        semantic swaps; `coverage` = fraction of changed frames where a correct set exists)
  accuracy vs the expert label, false starts (non-HOLD actions before the first t_onset; T1-T3 episodes)
  agreement of drop_all with B
`--verify-cgr N` checks rule (a) against the re-simulated expert (b) on N sampled frames.
Output: <out>/metrics.json, <out>/summary.txt, <out>/ep<k>/{actions.png, summary.txt, actions.npz}.

  ./run.sh -m scripts.demo_intent --ckpt outputs/train/intent_act/D/checkpoints/last/pretrained_model \\
      --baseline-ckpt outputs/train/intent_act/B/checkpoints/last/pretrained_model \\
      --dataset-root outputs/datasets/intent_act_early --config configs/experiments/intent_act_early.yaml --verify-cgr 25
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from intent_policy.benchmark.counterfactual import (correct_actions_semantic, correct_actions_spatial, expert_counterfactual,
                                                    permute_targets, swap_partners, swap_who)
from intent_policy.intent.contract import FIELDS
from intent_policy.intent.places import target_positions
from intent_policy.policies.hri_act.configuration_hri_act import INTENT_GROUPS
from intent_policy.policies.policy_agent import load_policy
from intent_policy.sim.restricted_action import RestrictedAction
from intent_policy.utils import load_yaml, resolve

SWAPS = ('swap_spatial', 'swap_semantic', 'swap_time')
MODES = ('original',) + SWAPS + tuple(f'drop_{g}' for g in INTENT_GROUPS) + ('drop_all',)
BIN_EDGES = np.round(np.arange(-1.5, 1.1 + 1e-9, 0.2), 2)
SRC_PREFIX = {'hindsight': 'intent_hs', 'perfect': 'intent_pp', 'predicted': 'intent_pr0'}
HOLD = int(RestrictedAction.HOLD)


def episode_arrays(ds: LeRobotDataset, prefix: str) -> dict:
    cols = [f'{prefix}.{f}' for f in FIELDS] + ['hri.task_id', 'restricted_action', 'observation.state']
    d = ds.hf_dataset.with_format('numpy', columns=cols)[:]
    out = {f: np.asarray(d[f'{prefix}.{f}'], np.float32) for f in FIELDS}
    for f in ('tte', 'tte_std'):
        out[f] = out[f].reshape(-1, 1)
    out['task_id'] = np.asarray(d['hri.task_id']).reshape(-1)
    out['label'] = np.asarray(d['restricted_action']).reshape(-1)
    out['tcp'] = np.asarray(d['observation.state'])[:, 7:10]
    return out


def mode_inputs(base: dict, mode: str, partners: dict, schema: dict) -> tuple[dict, dict | None, np.ndarray]:
    """Intent arrays of a mode, its keep mask (None: all kept) and the per-frame 'input changed' flag."""
    it = {k: base[k].copy() for k in FIELDS}
    T, keep = len(it['tte']), None
    if mode == 'swap_spatial':
        it['p_target'] = permute_targets(it['p_target'], partners, schema['targets'])
        it['c_target'] = permute_targets(it['c_target'], partners, schema['targets'])
    elif mode == 'swap_semantic':
        it['p_who'], it['c_who'] = swap_who(it['p_who'], schema['who']), swap_who(it['c_who'], schema['who'])
    elif mode == 'swap_time':
        tte = it['tte']
        it['tte'] = np.where(tte + 1.0 <= schema['tte_max_s'], tte + 1.0, np.maximum(tte - 1.0, 0.0)).astype(np.float32)
        it['phase'] = it['phase'][(np.arange(T) + T // 2) % T]
    elif mode.startswith('drop_'):
        gone = INTENT_GROUPS if mode == 'drop_all' else (mode.removeprefix('drop_'),)
        keep = {g: np.full(T, g not in gone) for g in INTENT_GROUPS}
    changed = np.zeros(T, bool)
    for k in FIELDS:
        changed |= np.abs(it[k] - base[k]).reshape(T, -1).max(1) > 1e-6
    if keep is not None:
        changed[:] = True
    return it, keep, changed


@torch.no_grad()
def run_modes(policy, pre, ds, arr: dict, inputs: dict, baseline=None, batch_size: int = 32) -> dict:
    """First restricted action of the chunk at every frame for every mode (and B)."""
    policy.eval()
    dev = policy.config.device
    keys = list(policy.config.input_features)
    T = len(arr['label'])
    out = defaultdict(list)
    for s in range(0, T, batch_size):
        idx = np.arange(s, min(s + batch_size, T))
        items = [ds[int(i)] for i in idx]
        obs = {k: torch.stack([it[k] for it in items]) for k in keys}
        task = torch.as_tensor(arr['task_id'][idx]).long()
        for mode, (it, keep, _) in inputs.items():
            b = pre(dict(obs))
            b['hri.task_id'] = task.to(dev)
            b.update({f'intent.{f}': torch.as_tensor(it[f][idx]).to(dev) for f in FIELDS})
            b['intent_keep'] = None if keep is None else {g: torch.as_tensor(v[idx]).to(dev) for g, v in keep.items()}
            out[mode].append(policy.predict_restricted_chunk(b)[:, 0].argmax(-1).cpu().numpy())
        if baseline is not None:
            bp, bpre = baseline
            b = bpre(dict(obs))
            b['hri.task_id'] = task.to(bp.config.device)
            out['B'].append(bp.predict_restricted_chunk(b)[:, 0].argmax(-1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def frame_bins(T: int, segments: list[dict], fps: float) -> np.ndarray:
    """Bin index of t - t_clear (nearest segment) per frame; -1 outside every window."""
    out = np.full(T, -1)
    if not segments:
        return out
    clears = np.array([s['t_clear'] for s in segments])
    rel = (np.arange(T)[:, None] - clears[None]) / fps
    r = rel[np.arange(T), np.abs(rel).argmin(1)]
    inside = (r >= BIN_EDGES[0]) & (r < BIN_EDGES[-1])
    out[inside] = np.clip(np.searchsorted(BIN_EDGES, r[inside], side='right') - 1, 0, len(BIN_EDGES) - 2)
    return out


def correct_sets(mode: str, it: dict, arr: dict, schema: dict, places: dict) -> list:
    sets = []
    for t in range(len(arr['label'])):
        tgt_c, tgt_p = int(it['c_target'][t].argmax()), int(it['p_target'][t].argmax())
        target = schema['targets'][tgt_c if tgt_c else tgt_p] if (tgt_c or tgt_p) else None
        if mode == 'swap_spatial':
            sets.append(None if target is None else correct_actions_spatial(arr['tcp'][t], target, places))
        else:
            who = schema['who'][int(it['c_who'][t].argmax()) or int(it['p_who'][t].argmax())]
            sets.append(correct_actions_semantic(arr['tcp'][t], who, target, places))
    return sets


def plot(actions: dict, segments: list, path: Path, title: str, fps: float) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.4))
    t = np.arange(len(actions['original'])) / fps
    for k, name in enumerate([m for m in ('original', 'swap_spatial', 'swap_semantic', 'drop_all', 'B') if m in actions]):
        ax.step(t, actions[name] + 0.07 * (k - 2), where='post', lw=1.5 if name == 'original' else 1.0, label=name, alpha=0.9)
    for s in segments:
        for key, ls in (('t_onset', ':'), ('t_clear', '--'), ('t_event', '-')):
            ax.axvline(s[key] / fps, color='k', ls=ls, lw=0.8, alpha=0.6)
    ax.set_yticks(range(len(RestrictedAction)), [a.name for a in RestrictedAction], fontsize=7)
    ax.set_title(title + '   (dotted t_onset, dashed t_clear, solid t_event)', fontsize=9)
    ax.legend(fontsize=7, ncol=5, loc='upper right')
    ax.set_xlabel('t [s]')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def rate(num, den) -> float | None:
    return None if den == 0 else round(float(num) / float(den), 4)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True, help='intent-token checkpoint (pretrained_model directory)')
    p.add_argument('--baseline-ckpt', help='no-intent model B')
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--config', help='experiment config (needed for --verify-cgr: scenario overrides, controller)')
    p.add_argument('--episodes', type=int, nargs='*', help='default: the validation episodes (every --val-every-th)')
    p.add_argument('--val-every', type=int, default=10)
    p.add_argument('--intent-source', choices=list(SRC_PREFIX), help="default: the checkpoint's source")
    p.add_argument('--verify-cgr', type=int, default=0, help='frames checked against the re-simulated expert')
    p.add_argument('--out', default='outputs/demo_intent')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()

    policy, pre, _ = load_policy(args.ckpt, args.device)
    cfg = policy.config
    if policy.intent_encoder is None or cfg.action_mode != 'restricted':
        raise SystemExit('demo_intent needs a restricted-head intent-token checkpoint')
    baseline = load_policy(args.baseline_ckpt, args.device)[:2] if args.baseline_ckpt else None
    root, out = resolve(args.dataset_root), resolve(args.out)
    schema = cfg.intent_schema
    fps = float(schema['fps'])
    source = args.intent_source or cfg.intent_source
    meta = {m['episode_index']: m for m in map(json.loads, (root / 'meta/hri_episodes.jsonl').read_text().splitlines())}
    segs_all = {int(k): v for k, v in json.loads((root / 'meta/intent_segments.json').read_text()).items()}
    episodes = args.episodes or [e for e in sorted(meta) if e % args.val_every == args.val_every - 1]
    places = target_positions(schema)
    n_bins = len(BIN_EDGES) - 1
    acc = {m: dict(changed=np.zeros(n_bins + 2), swapped=np.zeros(n_bins + 2), covered=np.zeros(n_bins + 2),
                   correct=np.zeros(n_bins + 2)) for m in MODES[1:]}            # bins + pre + post
    totals = defaultdict(lambda: np.zeros(2))
    by_task = defaultdict(lambda: defaultdict(lambda: np.zeros(2)))
    samples, per_episode = [], []
    out.mkdir(parents=True, exist_ok=True)
    for e in episodes:
        ds = LeRobotDataset(root.name, root=root, episodes=[e])
        arr = episode_arrays(ds, SRC_PREFIX[source])
        segs = segs_all.get(e, [])
        m = meta[e]
        partners = swap_partners(m['spec'], schema)
        inputs = {mode: mode_inputs(arr, mode, partners, schema) for mode in MODES}
        actions = run_modes(policy, pre, ds, arr, inputs, baseline, args.batch_size)
        T = len(arr['label'])
        bins = frame_bins(T, segs, fps)
        rel_pre = np.zeros(T, bool)
        for t in range(T):
            if bins[t] >= 0:
                rel_pre[t] = BIN_EDGES[bins[t]] < 0
        sets = {mode: correct_sets(mode, inputs[mode][0], arr, schema, places) for mode in ('swap_spatial', 'swap_semantic')}
        ep_rows = {}
        for mode in MODES[1:]:
            swapped = inputs[mode][2]
            ch = (actions[mode] != actions['original']) & swapped
            cov = np.array([ch[t] and mode in sets and sets[mode][t] is not None for t in range(T)])
            cor = np.array([cov[t] and int(actions[mode][t]) in sets[mode][t] for t in range(T)])
            for slot, mask in [(b, bins == b) for b in range(n_bins)] + [(n_bins, (bins >= 0) & rel_pre), (n_bins + 1, (bins >= 0) & ~rel_pre)]:
                a = acc[mode]
                a['swapped'][slot] += (swapped & mask).sum()
                a['changed'][slot] += (ch & mask).sum()
                a['covered'][slot] += (cov & mask).sum()
                a['correct'][slot] += (cor & mask).sum()
            ep_rows[mode] = dict(icr=rate(ch.sum(), swapped.sum()), icr_pre=rate((ch & rel_pre).sum(), (swapped & rel_pre).sum()),
                                 cgr=rate(cor.sum(), cov.sum()) if mode in sets else None)
            if mode == 'swap_spatial':
                for t in np.flatnonzero(swapped & ~np.array([arr['label'][t] in (7, 8) for t in range(T)])):
                    if sets[mode][t] is not None:
                        samples.append((e, int(t), schema['targets'][int(inputs[mode][0]['c_target'][t].argmax()) or
                                                                    int(inputs[mode][0]['p_target'][t].argmax())], sets[mode][t]))
        task = m.get('list_task') or m['scenario_id']
        first_onset = min([s['t_onset'] for s in segs], default=T)
        for name in [k for k in ('original', 'drop_all', 'B') if k in actions]:
            totals[f'accuracy_{name}'] += [(actions[name] == arr['label']).sum(), T]
            by_task[task][f'accuracy_{name}'] += [(actions[name] == arr['label']).sum(), T]
            if task in ('T1', 'T2', 'T3', 'T5'):
                totals[f'false_start_{name}'] += [(actions[name][:first_onset] != HOLD).sum(), first_onset]
        if task in ('T1', 'T2', 'T3', 'T5'):
            totals['false_start_expert'] += [(arr['label'][:first_onset] != HOLD).sum(), first_onset]
        if 'B' in actions:
            totals['drop_all_agrees_with_B'] += [(actions['drop_all'] == actions['B']).sum(), T]
            totals['original_agrees_with_B'] += [(actions['original'] == actions['B']).sum(), T]
        d = out / f'ep{e}'
        d.mkdir(exist_ok=True)
        plot(actions, segs, d / 'actions.png', f'ep {e} {task} ({source})', fps)
        np.savez_compressed(d / 'actions.npz', **actions, label=arr['label'])
        lines = [f'episode {e} {task} seed {m["seed"]}; source {source}; partners {partners}'] + \
                [f"  {s['gesture']}->{s['target']} ({s['who']}) onset {s['t_onset']} clear {s['t_clear']} event {s['t_event']} end {s['t_end']}" for s in segs] + \
                [f'  {k:15s} {v}' for k, v in ep_rows.items()]
        (d / 'summary.txt').write_text('\n'.join(lines) + '\n')
        per_episode.append(dict(episode=e, task=task, **{k: v for k, v in ep_rows.items()}))
        print(lines[0], '| swap_spatial', ep_rows['swap_spatial'], flush=True)

    labels = [f'[{BIN_EDGES[i]:+.1f},{BIN_EDGES[i + 1]:+.1f})' for i in range(n_bins)] + ['pre', 'post']
    modes_out = {}
    for mode, a in acc.items():
        modes_out[mode] = {lab: dict(swapped=int(a['swapped'][i]), ICR=rate(a['changed'][i], a['swapped'][i]),
                                     CGR=rate(a['correct'][i], a['covered'][i]) if mode in ('swap_spatial', 'swap_semantic') else None,
                                     CGR_coverage=rate(a['covered'][i], a['changed'][i]) if mode in ('swap_spatial', 'swap_semantic') else None)
                           for i, lab in enumerate(labels)}
    summary = {k: rate(*v) for k, v in totals.items()}
    verify = None
    if args.verify_cgr and samples:
        if not args.config:
            raise SystemExit('--verify-cgr needs --config (experiment config of the dataset)')
        exp = load_yaml(args.config)
        rng = np.random.default_rng(0)
        pick = [samples[i] for i in rng.choice(len(samples), min(args.verify_cgr, len(samples)), replace=False)]
        rows = []
        for e, t, tgt, ok in pick:
            ds = LeRobotDataset(root.name, root=root, episodes=[e])
            labels_e = np.asarray(ds.hf_dataset.with_format('numpy', columns=['restricted_action'])[:]['restricted_action']).reshape(-1)
            r = expert_counterfactual(exp, meta[e], labels_e, t, tgt)
            rows.append(dict(episode=e, frame=t, new_target=tgt, rule_a=sorted(RestrictedAction(a).name for a in ok),
                             expert_b=RestrictedAction(r['action']).name, stage=r['stage'], match=r['action'] in ok))
        verify = dict(n=len(rows), match_rate=rate(sum(r['match'] for r in rows), len(rows)), frames=rows)
    pre_sp = modes_out['swap_spatial']['pre']
    criteria = dict(spatial_ICR_pre_ge_0_3=bool(pre_sp['ICR'] is not None and pre_sp['ICR'] >= 0.3),
                    spatial_CGR_pre_ge_0_6=bool(pre_sp['CGR'] is not None and pre_sp['CGR'] >= 0.6),
                    drop_all_agrees_with_B=summary.get('drop_all_agrees_with_B'))
    result = dict(checkpoint=str(args.ckpt), baseline=args.baseline_ckpt, dataset=str(root), intent_source=source,
                  calibrated=source != 'predicted' or bool(schema.get('predicted', {}).get('noise')),
                  episodes=episodes, bins_s=BIN_EDGES.tolist(), modes=modes_out, summary=summary,
                  by_task={t: {k: rate(*v) for k, v in d.items()} for t, d in by_task.items()},
                  criteria=criteria, cgr_verification=verify, per_episode=per_episode)
    (out / 'metrics.json').write_text(json.dumps(result, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    txt = [f'model {args.ckpt} | source {source} | episodes {len(episodes)}', f'criteria {criteria}', f'summary {summary}']
    for mode in MODES[1:]:
        r = modes_out[mode]
        txt.append(f"{mode:15s} pre: ICR {r['pre']['ICR']} CGR {r['pre']['CGR']} (cov {r['pre']['CGR_coverage']}) | "
                   f"post: ICR {r['post']['ICR']} CGR {r['post']['CGR']}")
    txt.append('swap_spatial by bin: ' + ', '.join(f"{lab} {v['ICR']}/{v['CGR']}" for lab, v in modes_out['swap_spatial'].items()))
    if verify:
        txt.append(f"CGR rule (a) vs re-simulated expert (b): {verify['match_rate']} over {verify['n']} frames")
    (out / 'summary.txt').write_text('\n'.join(txt) + '\n')
    print('\n'.join(txt))


if __name__ == '__main__':
    main()
