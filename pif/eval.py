"""PIF evaluation on HABIT S8789 (Supervisor: the person points at 1 of 4 identical containers; handoff §P4, subset).

For every held-out episode the trained PIF runs causally over the whole episode; b_t(k) = sum_r b_t(r, k).
Target container token: the container nearest to the bread 1 s after the release (pif/data/habit_episode.py), kept only
when it agrees with the release-position rule (pif/data/habit_label_check.py, numbered left to right = decreasing
table y of the container tracks).
Accuracy of argmax over the container tokens, per 1 s bin of time before the release, against:
  chance       1 / number of container tokens
  pointing     the container with the smallest angle to the index ray (base -> tip) of the more extended hand
  bread        the container nearest to the bread (the robot carries it to the target: shows when the answer leaks
               through the robot's motion; HANDOFF_PIF_LOCAL §4.7)
Also at the last frame before the bread leaves the plate (before the robot moves towards a container).
Writes <run>/eval_s8789.json and eval_s8789.png.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from intent_policy.utils import resolve
from pif.data.dataset import collate
from pif.data.habit_episode import s8789_label
from pif.data.tokens import build_tokens
from pif.model.pif import PIF
from pif.train import load_data

BINS = np.arange(-10, 3)                      # seconds relative to the release, [b, b+1)


def episode_batch(ep: dict, stats: dict) -> dict:
    tok = build_tokens(ep, stats)
    T = len(ep['holding'])
    item = dict(human=torch.from_numpy(tok['human']), objects=torch.from_numpy(tok['objects']),
                obj_type=torch.from_numpy(tok['obj_type']), holding=torch.from_numpy(tok['holding']),
                valid=torch.ones(T, dtype=torch.bool), target_valid=torch.zeros(1, 1, dtype=torch.bool),
                horizons=torch.zeros(1, dtype=torch.long), n_input=T, episode=0, start=0)
    return collate([item])


def pointing_choice(ep: dict, t: int, cont: list[int], names: list[str]) -> int | None:
    J = {n: i for i, n in enumerate(names)}
    best = None
    for s in ('r', 'l'):
        ib, it, sh = J[f'{s}_index_base'], J[f'{s}_index_tip'], J[f'{s}_shoulder']
        v = ep['human_vis'][t]
        if not (v[ib] and v[it]):
            continue
        tip, base = ep['human_pos'][t, it], ep['human_pos'][t, ib]
        ext = np.linalg.norm(tip - ep['human_pos'][t, sh]) if v[sh] else 0.0
        ray = tip - base
        if np.linalg.norm(ray) < 1e-6:
            continue
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
    meta = json.loads((run / 'meta.json').read_text())
    model = PIF(cfg)
    model.load_state_dict(ck['model'])
    model.eval()
    eps, _, _, _ = load_data(cfg)
    want = set(meta[args.split])
    eps = [e for e in eps if e['meta']['episode_index'] in want and e['meta']['sid'] == 'S8789']
    rule = {r['episode']: r for r in json.loads(resolve('outputs/habit/label_check/s8789_targets.json').read_text())['episodes']}
    names = cfg['tokens']['keypoints']
    rows, curves = [], {k: [[] for _ in BINS] for k in ('pif', 'pointing', 'bread', 'chance')}
    before_depart = {k: [] for k in ('pif', 'pointing', 'bread')}
    for ep in eps:
        i = ep['meta']['episode_index']
        rr = rule.get(i, {})
        rel, grasp = rr.get('release'), rr.get('grasp')
        cont = [k for k, n in enumerate(ep['obj_names']) if n == 'airtight container']
        tgt = s8789_label(ep, rel)
        row = dict(episode=i, containers=len(cont), release=rel)
        if tgt is None or len(cont) < 2:
            rows.append(dict(row, skipped='no label / containers'))
            continue
        order = sorted(cont, key=lambda k: -ep['obj_pos'][min(rel, len(ep['holding']) - 1), k, 1])     # left to right
        rule_ok = len(cont) == 4 and rr.get('target') is not None and order[rr['target'] - 1] == tgt
        row.update(target=tgt, rule_agrees=bool(rule_ok))
        if not rule_ok:
            rows.append(dict(row, skipped='label sources disagree'))
            continue
        log_b, rho = model.infer(episode_batch(ep, stats))
        pk = log_b[0].exp().sum(1).numpy()                                  # [T, K+1], index 0 = none
        bread = [k for k, n in enumerate(ep['obj_names']) if n == 'bread roll']
        T = len(ep['holding'])
        depart = None
        if bread and grasp is not None:
            b0 = ep['obj_pos'][grasp, bread[0]]
            moved = np.flatnonzero(np.linalg.norm(ep['obj_pos'][grasp:, bread[0]] - b0, axis=1) > 0.05)
            depart = grasp + int(moved[0]) if len(moved) else None
        for t in range(T):
            dt = (t - rel) / 10.0
            j = int(np.floor(dt)) - BINS[0]
            if not 0 <= j < len(BINS):
                continue
            guess = dict(pif=cont[int(np.argmax(pk[t, [k + 1 for k in cont]]))], pointing=pointing_choice(ep, t, cont, names),
                         bread=cont[int(np.argmin([np.linalg.norm(ep['obj_pos'][t, k, :2] - ep['obj_pos'][t, bread[0], :2]) for k in cont]))]
                         if bread and ep['obj_vis'][:t + 1, bread[0]].any() else None)
            for k, g in guess.items():
                if g is not None:
                    curves[k][j].append(g == tgt)
            curves['chance'][j].append(1.0 / len(cont))
            if depart is not None and t == depart - 1:
                for k, g in guess.items():
                    if g is not None:
                        before_depart[k].append(g == tgt)
        row.update(depart=depart, pif_final=float(pk[min(rel, T - 1), tgt + 1] / pk[min(rel, T - 1), 1:].sum()))
        rows.append(row)
    summary = dict(ckpt=str(args.ckpt), split=args.split, episodes_scored=sum('skipped' not in r for r in rows), episodes=len(rows),
                   bins_s=BINS.tolist(),
                   accuracy={k: [None if not v else round(float(np.mean(v)), 3) for v in c] for k, c in curves.items()},
                   n_frames=[len(v) for v in curves['pif']],
                   before_depart={k: dict(acc=None if not v else round(float(np.mean(v)), 3), n=len(v)) for k, v in before_depart.items()},
                   rows=rows)
    (run / f'eval_s8789_{args.split}.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=1))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=130)
    for k, style in (('pif', 'o-'), ('pointing', 's--'), ('bread', '^:'), ('chance', '-')):
        y = [np.nan if a is None else a for a in summary['accuracy'][k]]
        ax.plot(BINS + 0.5, y, style, label=k, color={'chance': '0.6'}.get(k))
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('time relative to the robot releasing the bread (s)')
    ax.set_ylabel('target container accuracy')
    ax.set_ylim(0, 1.02)
    ax.set_title(f"S8789 {args.split}: {summary['episodes_scored']} episodes", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run / f'eval_s8789_{args.split}.png')


if __name__ == '__main__':
    main()
