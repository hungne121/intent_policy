"""Checks that a Phase-2 demonstration set does not force the result (run before training).

1. Twin labels: in every clean twin pair (same layout/timing, different human choice) the expert's
   labels must be identical until the expert commits, i.e. the demonstrator never acts on information
   that is not yet available.
2. Visual decodability of the target (logistic regression on frozen ImageNet ResNet18 features of
   both cameras, cross-validated over twin groups): at chance BEFORE the cue (no leak through the
   layout) and high AFTER the evidence point (the cue is visible, so a No-Information policy can in
   principle infer it).
3. Motion-oracle leakage: how well the target is decodable from the future-hand features alone,
   before / after the evidence point (Motion and Spatial are not independent).
4. L_demo: expert commitment time w.r.t. cue onset, evidence and cue completion; noise statistics.

  ./run.sh -m scripts.check_dataset --root outputs/datasets/hri_phase2_oracle \
      --output outputs/viz/hri_phase2_oracle/dataset_check
"""
import argparse
import json
from collections import defaultdict
import numpy as np
import torch
import torchvision
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scripts.common import resolve

CUE_COMPLETE = {'instructor_object_to_target': 'instruction_given', 'collaborator_object_handover': 'human_selection_shown',
                'collaborator_bowl_assistance': 'human_picked_object'}
POST_EVIDENCE_FRAMES = 10          # 0.5 s after the target became predictable
PRE_CUE_FRAMES = 2


def target_label(m: dict) -> str | None:
    v = m['variation']
    return v.get('selected_object') or (v.get('requested_object') if m['scenario_id'] == 'instructor_object_to_target' else None)


def target_side(m: dict) -> int | None:
    """Rank (by y) of the object the human's hand goes to: motion reveals WHERE, not the object identity
    (identities are permuted over the layout slots)."""
    obj = target_label(m)
    xy = m['variation'].get('object_xy', {})
    if obj not in xy:
        return None
    return sorted(xy, key=lambda k: xy[k][1]).index(obj)


def first_evidence_frame(m: dict) -> int | None:
    ev = [e for e in m.get('intention_evident', []) if e['kind'] in ('instruction', 'target_object')]
    return ev[0]['frame'] if ev else None


def logistic_cv(x: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = 5, l2: float = 1e-2) -> float:
    """Grouped k-fold accuracy of an L2 logistic regression (torch LBFGS)."""
    classes = sorted(set(y))
    yi = np.array([classes.index(v) for v in y])
    ug = np.unique(groups)
    rng = np.random.default_rng(0)
    rng.shuffle(ug)
    correct = 0
    for f in range(folds):
        test_g = set(ug[f::folds])
        te = np.array([g in test_g for g in groups])
        if te.all() or not te.any():
            continue
        mu, sd = x[~te].mean(0), x[~te].std(0) + 1e-6
        xt = torch.tensor((x[~te] - mu) / sd, dtype=torch.float32)
        yt = torch.tensor(yi[~te])
        w = torch.zeros(x.shape[1], len(classes), requires_grad=True)
        b = torch.zeros(len(classes), requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(xt @ w + b, yt) + l2 * (w ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            pred = (torch.tensor((x[te] - mu) / sd, dtype=torch.float32) @ w + b).argmax(1).numpy()
        correct += int((pred == yi[te]).sum())
    return correct / len(y)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default='outputs/datasets/hri_phase2_oracle')
    p.add_argument('--repo-id')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root = resolve(args.root)
    repo_id = args.repo_id or f'local/{root.name}'
    metas = [json.loads(line) for line in (root / 'meta/hri_episodes.jsonl').read_text().splitlines()]
    ds = LeRobotDataset(repo_id, root=root)
    start = {int(e['episode_index']): int(e['dataset_from_index']) for e in ds.meta.episodes}
    report = {}

    # 1. twin label identity
    by_pair = defaultdict(dict)
    for m in metas:
        if m.get('twin') and m['noise_frames'] == 0:
            by_pair[(m['scenario_id'], m['split'], m['twin']['base_seed'])][m['twin']['member']] = m
    labels = lambda m: [int(ds.hf_dataset[start[m['episode_index']] + i]['restricted_action'].item() if hasattr(
        ds.hf_dataset[start[m['episode_index']] + i]['restricted_action'], 'item') else ds.hf_dataset[start[m['episode_index']] + i]['restricted_action'])
        for i in range(m['n_frames'])]
    twin = defaultdict(lambda: dict(pairs=0, identical_before_commit=0, divergence_minus_commit=[]))
    for (sid, split, _), pair in by_pair.items():
        if len(pair) != 2 or not pair[0]['expert_commits'] or not pair[1]['expert_commits']:
            continue
        a, b = pair[0], pair[1]
        ca, cb = a['expert_commits'][0]['frame'], b['expert_commits'][0]['frame']
        la, lb = labels(a), labels(b)
        div = next((i for i, (u, v) in enumerate(zip(la, lb)) if u != v), min(len(la), len(lb)))
        t = twin[f'{sid}/{split}']
        t['pairs'] += 1
        t['identical_before_commit'] += int(div >= min(ca, cb) and ca == cb)
        t['divergence_minus_commit'].append(div - ca)
    report['twin_labels'] = {k: dict(pairs=v['pairs'], identical_before_commit=v['identical_before_commit'],
                                     divergence_minus_commit_frames=dict(min=int(min(v['divergence_minus_commit'])),
                                                                         max=int(max(v['divergence_minus_commit']))))
                             for k, v in twin.items() if v['pairs']}

    # 2-3. decodability (nominal episodes with a human choice)
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    net = torchvision.models.resnet18(weights=weights).eval()
    net.fc = torch.nn.Identity()
    norm = torchvision.transforms.Normalize(weights.transforms().mean, weights.transforms().std)
    cams = ds.meta.camera_keys

    def image_feat(idx):
        item = ds[idx]
        with torch.no_grad():
            return torch.cat([net(norm(item[c]).unsqueeze(0))[0] for c in cams]).numpy()

    decod = {}
    for sid in CUE_COMPLETE:
        eps = [m for m in metas if m['scenario_id'] == sid and m['split'] == 'nominal' and first_evidence_frame(m) is not None]
        if len(eps) < 10:
            continue
        y = np.array([target_label(m) for m in eps])
        groups = np.array([m['twin']['base_seed'] if m.get('twin') else m['seed'] for m in eps])
        cue_f = [int(round(m['human_cue_onset_t'] * 20)) for m in eps]
        ev_f = [first_evidence_frame(m) for m in eps]
        pre = np.stack([image_feat(start[m['episode_index']] + max(c - PRE_CUE_FRAMES, 0)) for m, c in zip(eps, cue_f)])
        post = np.stack([image_feat(start[m['episode_index']] + min(e + POST_EVIDENCE_FRAMES, m['n_frames'] - 1))
                         for m, e in zip(eps, ev_f)])
        motion = lambda frames: np.stack([np.concatenate([
            np.asarray(ds.hf_dataset[start[m['episode_index']] + f]['observation.oracle.future_hand_rel_0p5']),
            np.asarray(ds.hf_dataset[start[m['episode_index']] + f]['observation.oracle.future_hand_vel_0p5'])])
            for m, f in zip(eps, frames)])
        mid = [c + max((e - c) // 2, 0) for c, e in zip(cue_f, ev_f)]
        side = np.array([target_side(m) for m in eps])
        decod[sid] = dict(episodes=len(eps), classes=sorted(set(y)), chance=round(max(np.mean(y == c) for c in set(y)), 3),
                          image_before_cue=round(logistic_cv(pre, y, groups), 3),
                          image_after_evidence=round(logistic_cv(post, y, groups), 3),
                          side_chance=round(max(np.mean(side == c) for c in set(side)), 3),
                          motion_side_between_cue_and_evidence=round(logistic_cv(motion(mid), side, groups), 3),
                          motion_side_after_evidence=round(logistic_cv(motion([e + 2 for e in ev_f]), side, groups), 3),
                          motion_identity_after_evidence=round(logistic_cv(motion([e + 2 for e in ev_f]), y, groups), 3))
        print(sid, decod[sid], flush=True)
    report['target_decodability'] = decod

    # 4. L_demo and noise
    lat = defaultdict(lambda: defaultdict(list))
    for m in metas:
        if not m['expert_commits'] or m['human_cue_onset_t'] is None:
            continue
        c = m['expert_commits'][0]['t']
        L = lat[f"{m['scenario_id']}/{m['split']}"]
        L['commit_minus_cue_onset_s'].append(c - m['human_cue_onset_t'])
        ev = [e['t'] for e in m.get('intention_evident', []) if e['kind'] in ('instruction', 'target_object')]
        if ev:
            L['commit_minus_evidence_s'].append(c - ev[0])
        done = [s['t'] for s in m['protocol_steps'] if s['step'] == CUE_COMPLETE.get(m['scenario_id'])]
        if done:
            L['cue_complete_minus_commit_s'].append(done[0] - c)
    report['L_demo'] = {k: {q: dict(mean=round(float(np.mean(v)), 3), std=round(float(np.std(v)), 3), n=len(v))
                            for q, v in d.items()} for k, d in lat.items()}
    report['noise'] = dict(noisy_episodes=sum(m['noise_frames'] > 0 for m in metas), episodes=len(metas),
                           noisy_frames=int(sum(m['noise_frames'] for m in metas)), frames=int(sum(m['n_frames'] for m in metas)))

    out = resolve(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps(report, indent=2))
    lines = [f'# Dataset check — {root.name}', '', '## 1. Twin labels (clean pairs)', '',
             '| scenario/split | pairs | identical until commit | first divergence − commit [frames] |', '|---|---|---|---|']
    for k, v in report['twin_labels'].items():
        d = v['divergence_minus_commit_frames']
        lines.append(f"| {k} | {v['pairs']} | {v['identical_before_commit']} | {d['min']} … {d['max']} |")
    lines += ['', '## 2–3. Target decodability (grouped 5-fold logistic regression)', '',
              'Target identity from images; target side (which slot) and identity from the future-hand oracle alone.', '',
              '| scenario | episodes | identity chance | image before cue | image 0.5 s after evidence | side chance | motion→side before evidence | motion→side after evidence | motion→identity after evidence |',
              '|---|---|---|---|---|---|---|---|---|']
    for k, v in decod.items():
        lines.append(f"| {k} | {v['episodes']} | {v['chance']} | {v['image_before_cue']} | {v['image_after_evidence']} | "
                     f"{v['side_chance']} | {v['motion_side_between_cue_and_evidence']} | {v['motion_side_after_evidence']} | "
                     f"{v['motion_identity_after_evidence']} |")
    lines += ['', '## 4. Expert commitment (L_demo)', '', '| scenario/split | commit − cue onset [s] | commit − evidence [s] | cue complete − commit [s] |',
              '|---|---|---|---|']
    f = lambda d, q: f"{d[q]['mean']:.2f} ± {d[q]['std']:.2f}" if q in d else '—'
    for k, d in report['L_demo'].items():
        lines.append(f"| {k} | {f(d, 'commit_minus_cue_onset_s')} | {f(d, 'commit_minus_evidence_s')} | {f(d, 'cue_complete_minus_commit_s')} |")
    n = report['noise']
    lines += ['', f"Noise (DART): {n['noisy_episodes']}/{n['episodes']} episodes, {n['noisy_frames']}/{n['frames']} frames perturbed.", '']
    out.with_suffix('.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
