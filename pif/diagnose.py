"""Why does the v3 belief on the true container fall after the gesture? (docs/reports/PIF_pilot.md §3, option C)

  python -m pif.diagnose <pilot run dir>      e.g. outputs/pilot/2026-10-24/12-29-33_habit

Uses the per-fold parameters fitted by the pilot (results.json) and the same episode selection / folds. For every
episode, between the peak frame (argmax of b(target) in [t_leave - 10 s, t_leave]) and t_leave, the log-odds
lambda_t = log b_t(target) - log b_t(rival) (rival = the best wrong container at t_leave) changes by
  transition part   log b^-_t(target) - log b^-_t(rival) - lambda_{t-1}   (hazard pulls towards the prior)
  emission part     log e_t(target) - log e_t(rival)                      (evidence of frame t)
which add up exactly. Frames with the largest negative emission part are described (hand step, outlier-sized step,
direction towards target / rival). Counterfactuals (diagnosis only, not a claim): the same fitted model with a smaller
hazard. Writes <run>/diagnose.json and contact sheets of a few episodes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pif.config import load_config
from pif.filter import HANDS, Params, batch, forward, log_emission, log_prior
from pif.pilot import habit_cases


def decompose(x: dict, p: Params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """log e [B, T, G], log b^- [B, T, G], log b [B, T, G] of the exact filter."""
    with torch.no_grad():
        le = log_emission(x, p)
        lb, _ = forward(le, x['cmask'], p)
        h = torch.sigmoid(p.logit_h)
        lp = log_prior(x['cmask'], p)
        prev = torch.cat([lp[:, None], lb[:, :-1]], 1)
        pred = torch.logaddexp(torch.log1p(-h) + prev, torch.log(h) + lp[:, None])
    return le.numpy(), pred.numpy(), lb.numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run')
    ap.add_argument('--config', default='configs/pif/pif.yaml')
    ap.add_argument('--sheets', type=int, default=4)
    args = ap.parse_args()
    run = Path(args.run)
    res = json.loads((run / 'results.json').read_text())
    cfg = load_config(args.config)
    kp = cfg['tokens']['keypoints']
    J = {n: i for i, n in enumerate(kp)}
    cases = habit_cases(cfg)
    order = sorted(range(len(cases)), key=lambda i: cases[i]['episode'])
    fold = np.zeros(len(cases), int)
    for r, i in enumerate(order):
        fold[i] = r % cfg['pilot']['folds']
    fps, rows = cfg['fps'], []
    counter = {}
    for f in range(cfg['pilot']['folds']):
        te = [i for i in range(len(cases)) if fold[i] == f]
        vals = next(x['params'] for x in res['fitted'] if x['fold'] == f)
        x = batch([cases[i]['ep'] for i in te], [cases[i]['cand'] for i in te], kp)
        p = Params(vals)
        le, pred, lb = decompose(x, p)
        cf = {}
        for hz in (0.003, 0.0003):                                   # counterfactual hazards (diagnosis only)
            q = Params(dict(vals, hazard=hz))
            with torch.no_grad():
                cf[hz] = forward(log_emission(x, q), x['cmask'], q)[0].exp().numpy()
        for j, i in enumerate(te):
            c, tl = cases[i], cases[i]['t_leave']
            y = 1 + c['target']
            b = np.exp(lb[j])
            lo = max(1, tl - 10 * fps)
            peak = lo + int(np.argmax(b[lo:tl + 1, y]))
            wrong = [k for k in range(1, 5) if k != y]
            rival = wrong[int(np.argmax(b[tl, wrong]))]
            lam = lb[j, :, y] - lb[j, :, rival]
            trans = (pred[j, :, y] - pred[j, :, rival]) - np.r_[0.0, lam[:-1]]
            emis = le[j, :, y] - le[j, :, rival]
            seg = slice(peak + 1, tl + 1)
            ep = c['ep']
            tip = np.stack([ep['human_pos'][:, J[t]] for _, t in HANDS], 1)
            step = np.r_[np.zeros((1, 2)), np.linalg.norm(tip[1:] - tip[:-1], axis=-1)]
            worst = [int(t) for t in peak + 1 + np.argsort(emis[seg])[:5]] if tl > peak else []
            desc = []
            for t in worst:
                cand = ep['obj_pos'][t - 1][c['cand']]
                d = []
                for h in range(2):
                    dp = tip[t, h] - tip[t - 1, h]
                    u = lambda k: (cand[k] - tip[t - 1, h]) / (np.linalg.norm(cand[k] - tip[t - 1, h]) + 1e-9)
                    d.append(dict(hand='rl'[h], step_m=round(float(step[t, h]), 3),
                                  cos_target=round(float(dp @ u(y - 1) / (np.linalg.norm(dp) + 1e-9)), 2),
                                  cos_rival=round(float(dp @ u(rival - 1) / (np.linalg.norm(dp) + 1e-9)), 2)))
                desc.append(dict(t_rel_s=round((t - tl) / fps, 1), emission=round(float(emis[t]), 2), hands=d))
            correct_peak, correct_leave = bool(b[peak, 1:].argmax() == y - 1), bool(b[tl, 1:].argmax() == y - 1)
            key = f"peak {'ok' if correct_peak else 'wrong'} -> t_leave {'ok' if correct_leave else 'wrong'}"
            counter[key] = counter.get(key, 0) + 1
            rows.append(dict(episode=c['episode'], fold=f, target=c['target'], rival=rival - 1, peak_rel_s=round((peak - tl) / fps, 1),
                             b_peak=round(float(b[peak, y]), 3), b_leave=round(float(b[tl, y]), 3), group=key,
                             d_lambda=round(float(lam[tl] - lam[peak]), 2), transition=round(float(trans[seg].sum()), 2),
                             emission=round(float(emis[seg].sum()), 2),
                             emission_neg=round(float(emis[seg][emis[seg] < 0].sum()), 2),
                             emission_pos=round(float(emis[seg][emis[seg] > 0].sum()), 2),
                             big_steps=int((step[seg] > 0.10).any(1).sum()), frames=tl - peak,
                             adjacent=bool(abs((rival - 1) - c['target']) == 1),
                             cf_correct={str(hz): bool(cf[hz][j, tl, 1:].argmax() == y - 1) for hz in cf},
                             worst_frames=desc))
    lost = [r for r in rows if r['group'] == 'peak ok -> t_leave wrong']
    agg = lambda rs, k: round(float(np.median([r[k] for r in rs])), 2) if rs else None
    summary = dict(episodes=len(rows), groups=counter,
                   lost=dict(n=len(lost), median_d_lambda=agg(lost, 'd_lambda'), median_transition=agg(lost, 'transition'),
                             median_emission=agg(lost, 'emission'), median_emission_neg=agg(lost, 'emission_neg'),
                             median_emission_pos=agg(lost, 'emission_pos'), median_frames=agg(lost, 'frames'),
                             median_big_steps=agg(lost, 'big_steps'), rival_adjacent=sum(r['adjacent'] for r in lost)),
                   counterfactual_acc_at_t_leave={hz: round(float(np.mean([r['cf_correct'][hz] for r in rows])), 3)
                                                  for hz in rows[0]['cf_correct']},
                   acc_at_t_leave=round(float(np.mean([r['group'].endswith('t_leave ok') for r in rows])), 3))
    (run / 'diagnose.json').write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
    print(json.dumps(summary, indent=1))
    sheets(cfg, run, sorted(lost, key=lambda r: r['emission'])[:args.sheets], cases)


def sheets(cfg: dict, run: Path, rows: list[dict], cases: list[dict]) -> None:
    """Contact sheet per episode: front-view frames from the belief peak to t_leave, belief on target / rival."""
    from PIL import Image, ImageDraw
    from pif.data.habit import contact_sheet, paths
    by = {c['episode']: c for c in cases}
    for r in rows:
        c = by[r['episode']]
        tl = c['t_leave']
        frames = sorted({max(0, tl + int(round(s * 10))) for s in (r['peak_rel_s'], r['peak_rel_s'] / 2, -1.0, -0.5, 0.0)}
                        | {max(0, tl + int(round(w['t_rel_s'] * 10))) for w in r['worst_frames'][:3]})
        _, video = paths(r['episode'], cfg['habit']['root'])
        out = run / f"diagnose_ep{r['episode']}.jpg"
        ys = c['ep']['obj_pos'][tl, c['cand'], 1]
        rank = {k: int(np.sum(ys > ys[k])) + 1 for k in range(len(ys))}     # 1 = largest table y = robot's left
        contact_sheet(video, frames, f"ep {r['episode']}: target box {rank[r['target']]}, rival box {rank[r['rival']]} "
                                     f"(1-4 from the robot's left), t_leave = {tl / 10:.1f} s", out)
        print('sheet', out)


if __name__ == '__main__':
    main()
