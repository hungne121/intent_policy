"""Does a v3.2 intent policy use its intent? Offline check on held-out frames (INTENT_ACT_v3.2.md §5.3).

On the checkpoint's validation episodes (never trained on), the action chunk is predicted three times per frame:
  full      the intent as recorded,
  masked    every intent token masked (the policy's own no-intent mode, trained with p_drop_all),
  permuted  the intent of another frame of the batch (batches drawn across episodes): model reliance, the loss
            with the feature permuted over the loss with it intact (Fisher, Rudin & Dominici, JMLR 2019).
Reported per task and overall: the action L1 (normalised units, padding excluded, as the training loss), the
reliance ratios L1(masked) / L1(full) and L1(permuted) / L1(full), and the mean |a_full - a_masked|. A policy that
ignores its intent has both ratios ≈ 1 and a near-zero action change. Images, state and task are the same in all three.

`--window decision` keeps only the frames where the intent is ahead of the scene: from the human's cue onset to the
moment the object cue becomes evident (meta/hri_episodes.jsonl: human_cue_onset_t, intention_evident), T1-T3; the
expert's early decision falls inside (median 0.35 s after the onset, 0.35 s before the evidence). It adds `swapped`:
p_target / c_target of the alternative the human did not choose (benchmark/counterfactual.slot_partners), and splits
the frames by whether the expert's first decision was the human's object (`agree`) or the alternative (`twin_first`:
there the intent and the scene disagree in the demonstration itself). `--split train` measures the training episodes
(did the network learn to use the intent where it matters), `val` the held-out ones.

  ./run.sh -m scripts.intent_reliance --checkpoint <run>/train/D/checkpoints/last/pretrained_model --out <run>/reliance/D
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from intent_policy.benchmark.counterfactual import SWAP_TASKS, permute_targets, slot_partners
from intent_policy.policies.intent_encoder_v32 import TOKENS_V32
from intent_policy.policies.policy_agent import load_policy
from scripts.train_policy import (INSTR_DEST, INSTR_OBJECT, INTENT_COLUMN_PREFIX, TASK_ID, IntentSelector,
                                  source_prefixes)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--split', choices=['val', 'train'], default='val')
    p.add_argument('--window', choices=['all', 'decision'], default='all')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    meta_run = json.loads((ckpt / 'training_metadata.json').read_text())
    policy, pre, _ = load_policy(ckpt, args.device)
    policy.eval()
    cfg = policy.config
    if not (cfg.use_intent and cfg.intent_version == 'v32'):
        raise SystemExit('intent_reliance needs an intent v3.2 checkpoint')
    tasks = cfg.intent_schema['tasks']
    meta = LeRobotDatasetMetadata(meta_run['dataset_repo_id'], root=meta_run['dataset_root'])
    label_key = cfg.restricted_label_key
    delta = {k: [i / meta.fps for i in range(cfg.chunk_size)] for k in ('action', label_key)}
    episodes = meta_run['val_episodes'] if args.split == 'val' else meta_run['train_episodes']
    ds = LeRobotDataset(meta_run['dataset_repo_id'], root=meta_run['dataset_root'], episodes=episodes,
                        delta_timestamps=delta)
    hri = {m['episode_index']: m for m in map(json.loads, (Path(meta_run['dataset_root']) / 'meta/hri_episodes.jsonl')
                                              .read_text().splitlines())}
    data = ds
    if args.window == 'decision':
        win = {}
        for e in episodes:
            m = hri[e]
            ev = next((x['frame'] for x in m.get('intention_evident', []) if x['kind'] == 'target_object'), None)
            if m['list_task'] in SWAP_TASKS and ev is not None and m.get('human_cue_onset_t') is not None \
                    and m.get('expert_commits'):
                win[e] = (int(round(m['human_cue_onset_t'] * meta.fps)), ev)
        cols = ds.hf_dataset.with_format('numpy', columns=['episode_index', 'frame_index'])
        ep, fr = np.asarray(cols['episode_index']).ravel(), np.asarray(cols['frame_index']).ravel()
        keep = [i for i in range(len(ep)) if ep[i] in win and win[ep[i]][0] <= fr[i] <= win[ep[i]][1]]
        data = torch.utils.data.Subset(ds, keep)
        print(f'decision windows: {len(win)} episodes, {len(keep)} frames')
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                         generator=torch.Generator().manual_seed(args.seed))
    selector = IntentSelector({cfg.intent_source: source_prefixes(cfg.intent_source, meta)}, {cfg.intent_source: 1.0},
                              args.seed)
    g = torch.Generator().manual_seed(args.seed + 1)
    acc = defaultdict(lambda: defaultdict(list))               # task -> quantity -> per-sample values
    with torch.no_grad():
        for batch in loader:
            eps_b = batch['episode_index'].reshape(-1).tolist()
            extra = {k: batch.pop(k) for k in list(batch)
                     if k in (label_key, TASK_ID, INSTR_OBJECT, INSTR_DEST) or k.startswith(INTENT_COLUMN_PREFIX)}
            b = pre(batch)
            extra.update(selector.select(extra, False))
            b.update({k: v.to(args.device) for k, v in extra.items() if not k.startswith(INTENT_COLUMN_PREFIX)})
            n = b['observation.state'].shape[0]
            perm = torch.randperm(n, generator=g)
            if (perm == torch.arange(n)).any():               # every frame gets another frame's intent
                perm = torch.roll(torch.arange(n), 1)
            none = {t: torch.zeros(n, dtype=torch.bool, device=args.device) for t in TOKENS_V32}
            variants = {'full': dict(b),
                        'masked': {**b, 'intent_keep': none},
                        'permuted': {**b, **{k: v[perm.to(v.device)] for k, v in b.items() if k.startswith('intent.')}}}
            if args.window == 'decision':
                T = cfg.intent_schema['targets']
                sw = {k: torch.stack([torch.as_tensor(permute_targets(b[k][i].cpu().numpy(), slot_partners(hri[e]['spec']), T))
                                      for i, e in enumerate(eps_b)]).to(b[k].device) for k in ('intent.p_target', 'intent.c_target')}
                variants['swapped'] = {**b, **sw}
            groups = [tasks[t] for t in b[TASK_ID].reshape(-1).long().tolist()]
            if args.window == 'decision':
                groups = [grp + ('/agree' if hri[e]['expert_commits'][0]['target'] == hri[e]['spec']['target'] else '/twin_first')
                          for grp, e in zip(groups, eps_b)]
            pad = ~b['action_is_pad']                                              # (B, S)
            out = {}
            for name, vb in variants.items():
                a, _, _ = policy._run_act(vb)
                out[name] = a
                l1 = ((a - b['action']).abs().mean(-1) * pad).sum(1) / pad.sum(1).clamp_min(1)
                for grp, v in zip(groups, l1.tolist()):
                    acc[grp][f'l1_{name}'].append(v)
                    acc['all'][f'l1_{name}'].append(v)
            change = ((out['full'] - out['masked']).abs().mean(-1) * pad).sum(1) / pad.sum(1).clamp_min(1)
            for grp, v in zip(groups, change.tolist()):
                acc[grp]['change_masked'].append(v)
                acc['all']['change_masked'].append(v)

    res = {}
    for task, q in acc.items():
        m = {k: float(np.mean(v)) for k, v in q.items()}
        m.update(n=len(q['l1_full']), reliance_masked=m['l1_masked'] / m['l1_full'],
                 reliance_permuted=m['l1_permuted'] / m['l1_full'])
        if 'l1_swapped' in m:
            m['reliance_swapped'] = m['l1_swapped'] / m['l1_full']
        res[task] = m
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = 'reliance' if (args.split, args.window) == ('val', 'all') else f'reliance_{args.split}_{args.window}'
    (out_dir / f'{name}.json').write_text(json.dumps(dict(checkpoint=str(ckpt), split=args.split, window=args.window,
                                                          episodes=episodes, results=res), indent=1))
    order = sorted(k for k in res if k != 'all') + ['all']
    lines = [f'# Intent reliance: {ckpt}', '',
             f'{args.split} frames, window {args.window}; L1 in normalised action units. Ratio ≈ 1 = the policy ignores its intent.', '',
             '| group | n | L1 full | L1 masked | L1 permuted | L1 swapped | masked / full | permuted / full | swapped / full | mean |Δa| full vs masked |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for t in order:
        r = res[t]
        lines.append(f"| {t} | {r['n']} | {r['l1_full']:.4f} | {r['l1_masked']:.4f} | {r['l1_permuted']:.4f} | "
                     f"{r.get('l1_swapped', float('nan')):.4f} | {r['reliance_masked']:.3f} | {r['reliance_permuted']:.3f} | "
                     f"{r.get('reliance_swapped', float('nan')):.3f} | {r['change_masked']:.4f} |")
    (out_dir / f'{name}.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
