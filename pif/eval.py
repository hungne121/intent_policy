"""Evaluate a trained PIF on HABIT S8789 (Supervisor: the person points at 1 of 4 identical containers).

  python -m pif.eval <run>/checkpoints/NNNNNN.pt [--split test|train]

PIF runs causally over each whole episode; b_t(k) = sum_r b_t(r, k). Target container token: the release-position
rule (pif/data/habit.py; main session only), mapped to the container tracks ordered left to right (decreasing table y),
kept when it agrees with the container nearest to the bread 1 s after the release (two independent sources).
Accuracy of argmax over the container tokens per 1 s bin of time before the release, against
  chance     1 / number of containers
  pointing   the container with the smallest angle to the index ray (base -> tip) of the more extended hand
  bread      the container nearest to the bread the robot carries (shows when the answer leaks through the robot's
             motion, docs/PIF.md §5)
plus the accuracy on the last frame before the bread leaves the plate (before the robot moves to a container) and
code usage (effective number of relation codes). Writes <run>/eval_<split>.json / .png.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pif.data import load_episodes
from pif.data.dataset import collate
from pif.data.tokens import build_tokens
from pif.model import PIF

BINS = np.arange(-10, 3)          # s relative to the release, [b, b + 1)


def as_batch(ep: dict, stats: dict) -> dict:
    tok = build_tokens(ep, stats)
    T = len(ep['holding'])
    return collate([dict(human=torch.from_numpy(tok['human']), objects=torch.from_numpy(tok['objects']),
                         obj_type=torch.from_numpy(tok['obj_type']), holding=torch.from_numpy(tok['holding']),
                         valid=torch.ones(T, dtype=torch.bool), target_valid=torch.zeros(1, 1, dtype=torch.bool),
                         horizons=torch.zeros(1, dtype=torch.long), n_input=T, episode=0, start=0)])


def pointing_choice(ep: dict, t: int, cont: list[int], names: list[str]) -> int | None:
    J, v, best = {n: i for i, n in enumerate(names)}, ep['human_vis'][t], None
    for s in ('r', 'l'):
        ib, it, sh = J[f'{s}_index_base'], J[f'{s}_index_tip'], J[f'{s}_shoulder']
        if not (v[ib] and v[it]):
            continue
        tip, ray = ep['human_pos'][t, it], ep['human_pos'][t, it] - ep['human_pos'][t, ib]
        if np.linalg.norm(ray) < 1e-6:
            continue
        ext = np.linalg.norm(tip - ep['human_pos'][t, sh]) if v[sh] else 0.0
        ang = [np.arccos(np.clip(ray @ (ep['obj_pos'][t, k] - tip) / (np.linalg.norm(ray) * np.linalg.norm(ep['obj_pos'][t, k] - tip) + 1e-9), -1, 1))
               for k in cont]
        if best is None or ext > best[0]:
            best = (ext, cont[int(np.argmin(ang))])
    return None if best is None else best[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('ckpt')
    p.add_argument('--split', default='test', choices=['test', 'train'])
    args = p.parse_args()
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg, stats = ck['cfg'], ck['stats']
    run = Path(args.ckpt).parents[1]
    want = set(json.loads((run / 'meta.json').read_text())[args.split])
    model = PIF(cfg)
    model.load_state_dict(ck['model'])
    model.eval()
    eps, labs = load_episodes(dict(cfg, data=dict(cfg['data'], tasks=['S8789'])))
    names = cfg['tokens']['keypoints']
    curves = {k: [[] for _ in BINS] for k in ('pif', 'pointing', 'bread', 'chance')}
    before, rows, usage = {k: [] for k in ('pif', 'pointing', 'bread')}, [], []
    for ep, lab in zip(eps, labs):
        i = ep['meta']['episode_index']
        if i not in want:
            continue
        cont = [k for k, n in enumerate(ep['obj_names']) if n == 'airtight container']
        bread = [k for k, n in enumerate(ep['obj_names']) if n == 'bread roll']
        rel, grasp, T = lab.get('release'), lab.get('grasp'), len(ep['holding'])
        row = dict(episode=i, containers=len(cont))
        if not (lab.get('trusted') and lab.get('target') and rel is not None and len(cont) == 4 and bread):
            rows.append(dict(row, skipped='no trusted label / not 4 containers / no bread track'))
            continue
        t_after = min(rel + 10, T - 1)
        order = sorted(cont, key=lambda k: -ep['obj_pos'][t_after, k, 1])
        tgt = order[lab['target'] - 1]
        near = cont[int(np.argmin([np.linalg.norm(ep['obj_pos'][t_after, k, :2] - ep['obj_pos'][t_after, bread[0], :2]) for k in cont]))]
        if near != tgt:
            rows.append(dict(row, skipped='label sources disagree'))
            continue
        log_b, _ = model.infer(as_batch(ep, stats))
        b = log_b[0].exp()
        pk, pr = b.sum(1).numpy(), b.sum(2).numpy()                      # [T, K+1], [T, R]
        usage.append(pr.mean(0))
        b0 = ep['obj_pos'][grasp, bread[0]] if grasp is not None else None
        moved = np.flatnonzero(np.linalg.norm(ep['obj_pos'][grasp:, bread[0]] - b0, axis=1) > 0.05) if b0 is not None else []
        depart = grasp + int(moved[0]) if len(moved) else None
        for t in range(T):
            j = int(np.floor((t - rel) / 10.0)) - BINS[0]
            if not 0 <= j < len(BINS):
                continue
            guess = dict(pif=cont[int(np.argmax(pk[t, [k + 1 for k in cont]]))], pointing=pointing_choice(ep, t, cont, names),
                         bread=cont[int(np.argmin([np.linalg.norm(ep['obj_pos'][t, k, :2] - ep['obj_pos'][t, bread[0], :2]) for k in cont]))]
                         if ep['obj_vis'][:t + 1, bread[0]].any() else None)
            for k, g in guess.items():
                if g is not None:
                    curves[k][j].append(g == tgt)
                    if depart is not None and t == depart - 1:
                        before[k].append(g == tgt)
            curves['chance'][j].append(1 / len(cont))
        rows.append(dict(row, target=int(tgt), depart=depart, release=rel,
                         pif_at_release=round(float(pk[rel, tgt + 1] / pk[rel, 1:].sum()), 3)))
    mean_r = np.mean(usage, 0) if usage else np.zeros(1)
    res = dict(ckpt=str(args.ckpt), split=args.split, scored=sum('skipped' not in r for r in rows), episodes=len(rows),
               bins_s=BINS.tolist(), n_frames=[len(v) for v in curves['pif']],
               accuracy={k: [None if not v else round(float(np.mean(v)), 3) for v in c] for k, c in curves.items()},
               before_robot_moves={k: dict(acc=None if not v else round(float(np.mean(v)), 3), n=len(v)) for k, v in before.items()},
               effective_codes=round(float(np.exp(-(mean_r * np.log(mean_r + 1e-12)).sum())), 2), rows=rows)
    (run / f'eval_{args.split}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, indent=1))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=130)
    for k, style in (('pif', 'o-'), ('pointing', 's--'), ('bread', '^:'), ('chance', '-')):
        ax.plot(BINS + 0.5, [np.nan if a is None else a for a in res['accuracy'][k]], style, label=k, color={'chance': '0.6'}.get(k))
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('time relative to the robot releasing the bread (s)')
    ax.set_ylabel('target container accuracy')
    ax.set_ylim(0, 1.02)
    ax.set_title(f"S8789 {args.split}: {res['scored']} episodes", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run / f'eval_{args.split}.png')


if __name__ == '__main__':
    main()
