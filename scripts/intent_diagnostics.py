"""Offline diagnostics of the intent pathway of a v3.2 policy (INTENT_ACT_v3.2.md §5.3; research_collab.md [X1]).

On the decision-window frames (from the human's cue onset to the moment the object cue becomes evident, T1-T3;
scripts/intent_reliance.py --window decision), grouped by whether the expert's first decision was the human's object
(`agree`) or the alternative (`twin_first`):
  lag       the intent of frame t + k of the same episode with the observation of frame t: the action L1 per lag k,
            over the chunk and over its first step only. The chunk spans 2 s, so future intent can legitimately
            predict it better; the first step depends on the intent up to t only, so a minimum of the first-step L1
            away from k = 0 means the intent and the frames are out of sync
  token lag the fields of one token only from frame t + k, every other field at t ([X5]-Q16: the lag sweep above
            moves all fields together and cannot tell which token prefers the past): first-step L1 change vs k = 0
  token     one intent token masked at a time: L1(masked) / L1(full) and the mean |Δa| (normalised units)
  grad      the gradient norm of the L1 with respect to each intent token at the encoder input, next to the token's
            input-dependent spread across frames (how much the loss can move each token, and how much it varies)
  pull      the chunk's last end-effector position (forward kinematics of the predicted joint targets, 2 s ahead) with
            the swapped intent minus with the true intent, projected on the horizontal unit vector from the human's
            object to the alternative (m; > 0: the swapped intent pulls the plan towards the alternative); and with
            the true intent minus every token masked, projected on the vector from the other candidate to the object
            the recorded intent ranks first (> 0: the intent pulls the plan where it points)
  keyframe  the dataset only: the expert's decision frame against the frame of the largest change of its commanded
            joint velocity in [onset, evident + 1 s] (action changepoint, Wen et al. ICML 2021)

  ./run.sh -m scripts.intent_diagnostics --checkpoint <ckpt> --out <dir> [--split train]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from intent_policy.benchmark.counterfactual import SWAP_TASKS, permute_targets, slot_partners
from intent_policy.intent.contract import field_shapes_v32
from intent_policy.intent.places import target_positions
from intent_policy.policies.intent_encoder_v32 import TOKENS_V32
from intent_policy.policies.policy_agent import load_policy
from scripts.train_policy import INSTR_DEST, INSTR_OBJECT, INTENT_COLUMN_PREFIX, TASK_ID

LAGS = (-10, -6, -3, -1, 0, 1, 3, 6, 10)
TOKEN_LAGS = (-10, -3, 3, 10)
TOKEN_FIELDS = {'human': ('hand_hist', 'occupancy', 'min_dist', 'point_dir', 'human_present'),     # IntentEncoderV32
                'sem': ('p_who', 'c_who', 'phase', 'p_gesture'), 'target': ('p_target',), 'c_target': ('c_target',),
                'time': ('tte', 'tte_std'), 'motion': ('xi',)}


def windows(hri: dict, episodes, fps) -> dict:
    out = {}
    for e in episodes:
        m = hri[e]
        ev = next((x['frame'] for x in m.get('intention_evident', []) if x['kind'] == 'target_object'), None)
        if m['list_task'] in SWAP_TASKS and ev is not None and m.get('human_cue_onset_t') is not None \
                and m.get('expert_commits'):
            out[e] = (int(round(m['human_cue_onset_t'] * fps)), ev)
    return out


def keyframe_check(ds, hri, win, fps) -> dict:
    cols = ds.hf_dataset.with_format('numpy', columns=['episode_index', 'frame_index', 'action'])
    ep, fr, act = (np.asarray(cols[c]) for c in ('episode_index', 'frame_index', 'action'))
    lags = []
    for e, (on, ev) in win.items():
        sel = np.where(ep.ravel() == e)[0]
        a = act[sel][np.argsort(fr[sel].ravel())][:, :6]
        acc = np.linalg.norm(np.diff(a, n=2, axis=0), axis=1)          # change of the commanded joint velocity
        lo, hi = on, min(len(acc) - 1, ev + fps)
        if hi <= lo:
            continue
        lags.append((int(np.argmax(acc[lo:hi + 1])) + lo + 1 - hri[e]['expert_commits'][0]['frame']) / fps)
    lags = np.array(lags)
    return dict(n=len(lags), median_s=float(np.median(lags)), iqr_s=[float(np.percentile(lags, 25)), float(np.percentile(lags, 75))],
                within_0p25s=float(np.mean(np.abs(lags) <= 0.25)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--split', choices=['train', 'val'], default='train')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    run = json.loads((ckpt / 'training_metadata.json').read_text())
    policy, pre, _ = load_policy(ckpt, args.device)
    policy.eval()
    cfg = policy.config
    S, T = cfg.intent_schema, cfg.intent_schema['targets']
    root, repo = run['dataset_root'], run['dataset_repo_id']
    meta = LeRobotDatasetMetadata(repo, root=root)
    episodes = run['train_episodes'] if args.split == 'train' else run['val_episodes']
    label_key = cfg.restricted_label_key
    delta = {k: [i / meta.fps for i in range(cfg.chunk_size)] for k in ('action', label_key)}
    ds = LeRobotDataset(repo, root=root, episodes=episodes, delta_timestamps=delta)
    hri = {m['episode_index']: m for m in map(json.loads, (Path(root) / 'meta/hri_episodes.jsonl').read_text().splitlines())}
    win = windows(hri, episodes, meta.fps)
    shapes = field_shapes_v32(S)
    prefix = run['intent_tokens']['columns'][cfg.intent_source][0]
    cols = ds.hf_dataset.with_format('numpy', columns=['episode_index', 'frame_index'] + [f'{prefix}.{k}' for k in shapes])
    ep_all, fr_all = np.asarray(cols['episode_index']).ravel(), np.asarray(cols['frame_index']).ravel()
    intent_of = defaultdict(dict)                                  # episode -> field -> (n_frames, ...) in frame order
    for e in win:
        sel = np.where(ep_all == e)[0]
        sel = sel[np.argsort(fr_all[sel])]
        for k, sh in shapes.items():
            intent_of[e][k] = np.stack([np.asarray(x, np.float32) for x in cols[f'{prefix}.{k}'][sel]]).reshape((len(sel),) + tuple(sh))
    keep = [i for i in range(len(ep_all)) if ep_all[i] in win and win[ep_all[i]][0] <= fr_all[i] <= win[ep_all[i]][1]]
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, keep), batch_size=args.batch_size, shuffle=False,
                                         num_workers=2)
    pos = target_positions(S)
    stats = meta.stats['action']
    a_mean, a_std = (torch.as_tensor(np.asarray(stats[k]), dtype=torch.float32) for k in ('mean', 'std'))
    from intent_policy.scenarios.scenario_registry import make_scenario
    sc = make_scenario('t1_pick_place')
    env, mjd = sc.env, mujoco.MjData(sc.env.model)

    def tcp_end(a_norm: torch.Tensor) -> np.ndarray:             # (B, S, 7) normalised -> (B, 3) TCP of the last step
        q = (a_norm[:, -1].cpu() * a_std + a_mean)[:, :6].numpy()
        out = []
        for qi in q:
            mjd.qpos[env.qids] = qi
            mujoco.mj_kinematics(env.model, mjd)
            out.append(mjd.site_xpos[env.tcp].copy())
        return np.array(out)

    grads = defaultdict(list)
    captured = {}
    policy.intent_encoder.register_forward_hook(lambda m, i, o: captured.__setitem__('tok', o[0]))
    acc = defaultdict(lambda: defaultdict(list))
    for batch in loader:
        eps_b = batch['episode_index'].reshape(-1).tolist()
        frs_b = batch['frame_index'].reshape(-1).tolist()
        extra = {k: batch.pop(k) for k in list(batch)
                 if k in (label_key, TASK_ID, INSTR_OBJECT, INSTR_DEST) or k.startswith(INTENT_COLUMN_PREFIX)}
        b = pre(batch)
        b.update({k: v.to(args.device) for k, v in extra.items() if not k.startswith(INTENT_COLUMN_PREFIX)})
        n = len(eps_b)
        groups = [hri[e]['list_task'] + ('/agree' if hri[e]['expert_commits'][0]['target'] == hri[e]['spec']['target']
                                         else '/twin_first') for e in eps_b]
        pad = ~b['action_is_pad']

        def intent_at(lag: int, fields=None) -> dict:
            out = {}
            for k in shapes:
                lk = lag if fields is None or k in fields else 0
                rows = [intent_of[e][k][int(np.clip(f + lk, 0, len(intent_of[e][k]) - 1))] for e, f in zip(eps_b, frs_b)]
                out[f'intent.{k}'] = torch.as_tensor(np.stack(rows), device=args.device)
            return out

        def l1(a):
            return (((a - b['action']).abs().mean(-1) * pad).sum(1) / pad.sum(1).clamp_min(1)).tolist()

        def l1_first(a):
            return (a[:, 0] - b['action'][:, 0]).abs().mean(-1).tolist()

        def put(name, vals):
            for g, v in zip(groups, vals):
                acc[g][name].append(v)
                acc['all'][name].append(v)

        put('episode', eps_b)

        true = intent_at(0)
        with torch.no_grad():
            for lag in LAGS:
                a, _, _ = policy._run_act({**b, **intent_at(lag)})
                put(f'lag{lag:+d}', l1(a))
                put(f'lag1st{lag:+d}', l1_first(a))
                if lag == 0:
                    a_full = a
            for t, fields in TOKEN_FIELDS.items():
                for lag in TOKEN_LAGS:
                    a, _, _ = policy._run_act({**b, **intent_at(lag, fields)})
                    put(f'tlag_{t}{lag:+d}', l1_first(a))
            for t in TOKENS_V32:
                keepd = {u: torch.full((n,), u != t, dtype=torch.bool, device=args.device) for u in TOKENS_V32}
                a, _, _ = policy._run_act({**b, **true, 'intent_keep': keepd})
                put(f'tok_{t}', l1(a))
                put(f'dtok_{t}', (((a - a_full).abs().mean(-1) * pad).sum(1) / pad.sum(1).clamp_min(1)).tolist())
            none = {u: torch.zeros(n, dtype=torch.bool, device=args.device) for u in TOKENS_V32}
            a_mask, _, _ = policy._run_act({**b, **true, 'intent_keep': none})
            sw = {k: torch.as_tensor(np.stack([permute_targets(true[k][i].cpu().numpy(), slot_partners(hri[e]['spec']), T)
                                               for i, e in enumerate(eps_b)]), device=args.device)
                  for k in ('intent.p_target', 'intent.c_target')}
            a_sw, _, _ = policy._run_act({**b, **true, **sw})
            put('l1_masked', l1(a_mask))
            put('l1_swapped', l1(a_sw))
        e_full, e_mask, e_sw = tcp_end(a_full), tcp_end(a_mask), tcp_end(a_sw)
        pull_sw, pull_int = [], []
        for i, e in enumerate(eps_b):
            spec = hri[e]['spec']
            lay = spec['layout']
            human = lay[spec['target']]
            alt = slot_partners(spec).get(human)
            if alt is None:
                pull_sw.append(np.nan), pull_int.append(np.nan)
                continue
            u = (pos[alt] - pos[human])[:2]
            u /= np.linalg.norm(u)
            pull_sw.append(float((e_sw[i] - e_full[i])[:2] @ u))
            pt = true['intent.p_target'][i].cpu().numpy()
            first, other = (human, alt) if pt[T.index(human)] >= pt[T.index(alt)] else (alt, human)
            v = (pos[first] - pos[other])[:2]
            v /= np.linalg.norm(v)
            pull_int.append(float((e_full[i] - e_mask[i])[:2] @ v))
        put('pull_swapped', pull_sw)
        put('pull_intent', pull_int)
        # gradient of the L1 with respect to each intent token at the encoder input
        with torch.enable_grad():
            a, _, _ = policy._run_act({**b, **true})
            tok = captured['tok']
            tok.retain_grad()
            loss = ((a - b['action']).abs().mean(-1) * pad).sum() / pad.sum()
            loss.backward()
            g = tok.grad.norm(dim=-1).mean(0).tolist()
            sig = (tok - policy.intent_encoder.type_emb.weight[None]).detach()
            spread = (sig - sig.mean(0, keepdim=True)).norm(dim=-1).mean(0).tolist()
            for t, gv, sv in zip(TOKENS_V32, g, spread):
                grads[t].append((gv, sv))
            policy.zero_grad(set_to_none=True)

    def cluster_ci(q: dict, stat) -> list[float]:
        """95 % interval of stat(frames) over episode resamples (frames of one episode are not independent, [X2])."""
        ep = np.asarray(q['episode'])
        uniq = np.unique(ep)
        idx = {e: np.where(ep == e)[0] for e in uniq}
        rng = np.random.default_rng(0)
        vals = [stat(np.concatenate([idx[e] for e in rng.choice(uniq, len(uniq))])) for _ in range(2000)]
        return [float(x) for x in np.nanpercentile(vals, [2.5, 97.5])]

    res = {}
    for grp, q in acc.items():
        r = {k: float(np.nanmean(v)) for k, v in q.items() if k != 'episode'}
        r['n'], r['n_episodes'] = len(q['lag+0']), len(set(q['episode']))
        arr = {k: np.asarray(v, float) for k, v in q.items() if k != 'episode'}
        r['pull_swapped_ci'] = cluster_ci(q, lambda i: np.nanmean(arr['pull_swapped'][i]))
        r['swapped_ratio_ci'] = cluster_ci(q, lambda i: arr['l1_swapped'][i].mean() / arr['lag+0'][i].mean())
        r['masked_ratio_ci'] = cluster_ci(q, lambda i: arr['l1_masked'][i].mean() / arr['lag+0'][i].mean())
        for t in TOKEN_FIELDS:
            for lag in TOKEN_LAGS:
                k = f'tlag_{t}{lag:+d}'
                r[f'{k}_ci'] = cluster_ci(q, lambda i, k=k: arr[k][i].mean() - arr['lag1st+0'][i].mean())
        res[grp] = r
    grad = {t: dict(grad_norm=float(np.mean([x[0] for x in v])), content_spread=float(np.mean([x[1] for x in v])),
                    type_norm=float(policy.intent_encoder.type_emb.weight[TOKENS_V32.index(t)].norm()))
            for t, v in grads.items()}
    kf = keyframe_check(LeRobotDataset(repo, root=root, episodes=episodes), hri, win, meta.fps)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f'diagnostics_{args.split}.json').write_text(json.dumps(dict(checkpoint=str(ckpt), groups=res, grad=grad, keyframe=kf), indent=1))
    order = sorted(k for k in res if k != 'all') + ['all']
    L = [f'# Intent pathway diagnostics: {ckpt} ({args.split}, decision windows)', '',
         '## Lag sweep: L1 with the intent of frame t + k (k in frames, 20 Hz)', '',
         '| group | n | ' + ' | '.join(f'k {k:+d}' for k in LAGS) + ' | best k |', '|---|---|' + '---|' * (len(LAGS) + 1)]
    for g in order:
        r = res[g]
        vals = [r[f'lag{k:+d}'] for k in LAGS]
        L.append(f'| {g} | {r["n"]} | ' + ' | '.join(f'{v:.4f}' for v in vals) + f' | {LAGS[int(np.argmin(vals))]:+d} |')
    L += ['', 'First step of the chunk only (depends on the intent up to t):', '',
          '| group | n | ' + ' | '.join(f'k {k:+d}' for k in LAGS) + ' | best k |', '|---|---|' + '---|' * (len(LAGS) + 1)]
    for g in order:
        r = res[g]
        vals = [r[f'lag1st{k:+d}'] for k in LAGS]
        L.append(f'| {g} | {r["n"]} | ' + ' | '.join(f'{v:.4f}' for v in vals) + f' | {LAGS[int(np.argmin(vals))]:+d} |')
    L += ['', '## One token lagged at a time: first-step L1(k) - L1(0), 95 % CI by episode (< 0: that token from t + k fits better)', '',
          '| token | ' + ' | '.join(f'k {k:+d}' for k in TOKEN_LAGS) + ' |', '|---|' + '---|' * len(TOKEN_LAGS)]
    ra = res['all']
    for t in TOKEN_FIELDS:
        L.append(f'| {t} | ' + ' | '.join(
            f"{ra[f'tlag_{t}{k:+d}'] - ra['lag1st+0']:+.4f} [{ra[f'tlag_{t}{k:+d}_ci'][0]:+.4f}, {ra[f'tlag_{t}{k:+d}_ci'][1]:+.4f}]"
            for k in TOKEN_LAGS) + ' |')
    L += ['', '## One token masked: L1 / L1(full) and mean |Δa|', '',
          '| group | ' + ' | '.join(TOKENS_V32) + ' | all masked | target swapped |', '|---|' + '---|' * (len(TOKENS_V32) + 2)]
    for g in order:
        r = res[g]
        L.append(f'| {g} | ' + ' | '.join(f"{r[f'tok_{t}'] / r['lag+0']:.3f} ({r[f'dtok_{t}']:.3f})" for t in TOKENS_V32) +
                 f" | {r['l1_masked'] / r['lag+0']:.3f} | {r['l1_swapped'] / r['lag+0']:.3f} |")
    L += ['', '## Gradient of the L1 at the encoder input, per intent token', '',
          '| token | type emb norm | content spread across frames | grad norm |', '|---|---|---|---|']
    for t in TOKENS_V32:
        L.append(f"| {t} | {grad[t]['type_norm']:.2f} | {grad[t]['content_spread']:.3f} | {grad[t]['grad_norm']:.2e} |")
    L += ['', '## Pull of the 2 s plan (end-effector, forward kinematics), m', '',
          '| group | frames / episodes | swapped - true, towards the alternative [95 % CI by episode] | true - masked, towards the intent\'s first object |',
          '|---|---|---|---|']
    for g in order:
        r = res[g]
        lo, hi = r['pull_swapped_ci']
        L.append(f"| {g} | {r['n']} / {r['n_episodes']} | {r['pull_swapped']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {r['pull_intent']:+.4f} |")
    L += ['', '## Pre-registered F1 checks ([C4]): ratios with 95 % intervals by episode', '',
          '| group | L1(target swapped) / L1(full) | L1(all masked) / L1(full) |', '|---|---|---|']
    for g in order:
        r = res[g]
        L.append(f"| {g} | {r['l1_swapped'] / r['lag+0']:.3f} [{r['swapped_ratio_ci'][0]:.3f}, {r['swapped_ratio_ci'][1]:.3f}] | "
                 f"{r['l1_masked'] / r['lag+0']:.3f} [{r['masked_ratio_ci'][0]:.3f}, {r['masked_ratio_ci'][1]:.3f}] |")
    L += ['', f"## Keyframe check (dataset): action changepoint - expert decision, s: median {kf['median_s']:+.2f}, "
          f"IQR [{kf['iqr_s'][0]:+.2f}, {kf['iqr_s'][1]:+.2f}], |lag| <= 0.25 s in {kf['within_0p25s']:.0%} (n {kf['n']})", '']
    (out / f'diagnostics_{args.split}.md').write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
