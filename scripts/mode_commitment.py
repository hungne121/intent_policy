"""How much does the policy commit compared with the expert after the early decision? (research_collab.md [C13]-B1)

Mode-averaging check (Behavior Transformers — Shafiullah et al. 2022: a unimodal regression head returns the average
of the demonstrated modes). On the training episodes whose expert's first object decision was the human's object
(`agree`, no redirect), for the frames from that decision to 1 s after it: the end-effector of the predicted 2 s plan
and of the expert's own next 2 s (forward kinematics of the last chunk step), each relative to the end-effector of the
expert's current command (first chunk step), projected on the horizontal unit vector from the alternative candidate
to the human's object. commitment = Σ predicted progress / Σ expert progress over the frames (≈ 1: commits like the
expert; ≈ 0: no lateral commitment, the plan stays between the candidates), with a 95 % interval by episode. Also the
lateral progress itself (m). The intent is given as recorded, and with every token masked.
`--group twin_first` takes instead the episodes whose expert's first decision was the alternative (the device's early
belief pointed there; [X10]: there the intent may carry what the cameras do not), projecting on the direction of
that first decision; with `--window conflict` the frames from that decision to the expert's redirect.
`--window pre` takes instead the frames from the human's cue onset to the expert's decision (anticipation: the 2 s
plan already contains the expert's move, the decision is not yet visible in the robot's own motion).

  ./run.sh -m scripts.mode_commitment --checkpoint <ckpt> --out <dir> [--split train]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from intent_policy.benchmark.counterfactual import SWAP_TASKS, alternative_partners
from intent_policy.intent.places import target_positions
from intent_policy.policies.intent_encoder_v32 import TOKENS_V32
from intent_policy.policies.policy_agent import load_policy
from scripts.train_policy import INSTR_DEST, INSTR_OBJECT, INTENT_COLUMN_PREFIX, TASK_ID, IntentSelector, source_prefixes

WINDOW_S = 1.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--split', choices=['train', 'val'], default='train')
    p.add_argument('--window', choices=['post', 'pre', 'conflict'], default='post')
    p.add_argument('--group', choices=['agree', 'twin_first'], default='agree')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    ckpt = Path(args.checkpoint)
    run = json.loads((ckpt / 'training_metadata.json').read_text())
    policy, pre, _ = load_policy(ckpt, args.device)
    policy.eval()
    cfg = policy.config
    root, repo = run['dataset_root'], run['dataset_repo_id']
    meta = LeRobotDatasetMetadata(repo, root=root)
    episodes = run['train_episodes'] if args.split == 'train' else run['val_episodes']
    label_key = cfg.restricted_label_key
    delta = {k: [i / meta.fps for i in range(cfg.chunk_size)] for k in ('action', label_key)}
    ds = LeRobotDataset(repo, root=root, episodes=episodes, delta_timestamps=delta)
    hri = {m['episode_index']: m for m in map(json.loads, (Path(root) / 'meta/hri_episodes.jsonl').read_text().splitlines())}
    win = {}
    for e in episodes:
        m = hri[e]
        objs = set((m['spec'] or {}).get('layout') or {})
        com = [c for c in m.get('expert_commits') or [] if c['target'] in objs]
        agree = bool(com) and com[0]['target'] == m['spec']['target']
        if m['list_task'] in SWAP_TASKS and com and agree == (args.group == 'agree') \
                and alternative_partners(m['spec']).get(m['spec']['target']):
            onset = int(round(m['human_cue_onset_t'] * meta.fps)) if m.get('human_cue_onset_t') is not None else None
            if args.window == 'post':
                win[e] = (com[0]['frame'], com[0]['frame'] + int(WINDOW_S * meta.fps))
            elif args.window == 'conflict':
                if len(com) > 1:
                    win[e] = (com[0]['frame'], com[1]['frame'] - 1)
            elif onset is not None and onset < com[0]['frame']:
                win[e] = (onset, com[0]['frame'] - 1)
    cols = ds.hf_dataset.with_format('numpy', columns=['episode_index', 'frame_index'])
    ep, fr = np.asarray(cols['episode_index']).ravel(), np.asarray(cols['frame_index']).ravel()
    keep = [i for i in range(len(ep)) if ep[i] in win and win[ep[i]][0] <= fr[i] <= win[ep[i]][1]]
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, keep), batch_size=32, shuffle=False, num_workers=2)
    selector = IntentSelector({cfg.intent_source: source_prefixes(cfg.intent_source, meta)}, {cfg.intent_source: 1.0}, 0) \
        if cfg.use_intent else None
    pos = target_positions(cfg.intent_schema)
    st = meta.stats['action']
    a_mean, a_std = (torch.as_tensor(np.asarray(st[k]), dtype=torch.float32) for k in ('mean', 'std'))
    from intent_policy.scenarios.scenario_registry import make_scenario
    sc = make_scenario('t1_pick_place')
    env, mjd = sc.env, mujoco.MjData(sc.env.model)

    def tcp(a_norm, step):
        q = (a_norm[:, step].cpu() * a_std + a_mean)[:, :6].numpy()
        out = []
        for qi in q:
            mjd.qpos[env.qids] = qi
            mujoco.mj_kinematics(env.model, mjd)
            out.append(mjd.site_xpos[env.tcp][:2].copy())
        return np.array(out)

    rows = defaultdict(list)                      # condition -> (episode, task, predicted progress, expert progress)
    with torch.no_grad():
        for batch in loader:
            eps_b = batch['episode_index'].reshape(-1).tolist()
            extra = {k: batch.pop(k) for k in list(batch)
                     if k in (label_key, TASK_ID, INSTR_OBJECT, INSTR_DEST) or k.startswith(INTENT_COLUMN_PREFIX)}
            b = pre(batch)
            if selector is not None:
                extra.update(selector.select(extra, False))
            b.update({k: v.to(args.device) for k, v in extra.items() if not k.startswith(INTENT_COLUMN_PREFIX)})
            n = len(eps_b)
            now, exp_end = tcp(b['action'], 0), tcp(b['action'], -1)
            u = []
            for e in eps_b:
                spec = hri[e]['spec']
                h, a = spec['layout'][spec['target']], spec['layout'][alternative_partners(spec)[spec['target']]]
                if args.group == 'twin_first':          # towards the expert's first (the alternative) decision
                    h, a = a, h
                v = (pos[h] - pos[a])[:2]
                u.append(v / np.linalg.norm(v))
            u = np.array(u)
            conds = {'with intent': dict(b)}
            if cfg.use_intent:
                conds['intent masked'] = {**b, 'intent_keep': {t: torch.zeros(n, dtype=torch.bool, device=args.device)
                                                                for t in TOKENS_V32}}
            for name, vb in conds.items():
                a, _, _ = policy._run_act(vb)
                pred_end = tcp(a, -1)
                for i, e in enumerate(eps_b):
                    rows[name].append((e, hri[e]['list_task'], float((pred_end[i] - now[i]) @ u[i]),
                                       float((exp_end[i] - now[i]) @ u[i])))
    res, lines = {}, [f'# Plan commitment ({args.window} the early decision, {args.group}): {ckpt} ({args.split})', '',
                      'commitment = Σ predicted lateral progress / Σ expert lateral progress (2 s plan, towards the human\'s '
                      'object, from the expert\'s first decision to +1 s; agree episodes). 95 % interval by episode.', '',
                      '| condition | group | frames / episodes | predicted progress (m) | expert progress (m) | commitment [95 % CI] |',
                      '|---|---|---|---|---|---|']
    rng = np.random.default_rng(0)
    for name, r in rows.items():
        arr = np.array([(x[2], x[3]) for x in r])
        epi = np.array([x[0] for x in r])
        tasks = np.array([x[1] for x in r])
        for grp in sorted(set(tasks)) + ['all']:
            m = np.ones(len(r), bool) if grp == 'all' else tasks == grp
            uniq = np.unique(epi[m])
            idx = {e: np.where(m & (epi == e))[0] for e in uniq}
            boots = []
            for _ in range(2000):
                sel = np.concatenate([idx[e] for e in rng.choice(uniq, len(uniq))])
                boots.append(arr[sel, 0].sum() / arr[sel, 1].sum())
            c = arr[m, 0].sum() / arr[m, 1].sum()
            lo, hi = np.percentile(boots, [2.5, 97.5])
            res[f'{name}/{grp}'] = dict(n=int(m.sum()), episodes=len(uniq), commitment=float(c), ci=[float(lo), float(hi)],
                                        pred=float(arr[m, 0].mean()), expert=float(arr[m, 1].mean()))
            lines.append(f'| {name} | {grp} | {m.sum()} / {len(uniq)} | {arr[m, 0].mean():+.4f} | {arr[m, 1].mean():+.4f} | '
                         f'{c:.3f} [{lo:.3f}, {hi:.3f}] |')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{args.window}_{args.split}' + ('' if args.group == 'agree' else f'_{args.group}')
    (out / f'commitment_{tag}.json').write_text(json.dumps(res, indent=1))
    (out / f'commitment_{tag}.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
