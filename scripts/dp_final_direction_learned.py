"""Learned predictor of the final pointing direction from a partial approach, real people (DP); the learned
counterpart of scripts/dp_final_direction.py (same gestures, ρ grid and error measure).

Principle taken from Mavsar, Simonič & Ude (Front. Robot. AI 2025): a recurrent network reads the motion so far and
is trained on EVERY prefix of the motion with the label of its END (here the final eye -> wrist ray), so it learns
where the motion is going rather than where the arm points now. Not their architecture (no code released):
a small GRU, chosen for 2.3k sequences.
  input per frame  eyes, both shoulders, the pointing elbow and wrist, in a body frame (origin: pointing shoulder;
                   up: mid-hip -> mid-shoulder; lateral: the shoulder line), scaled by the shoulder width; velocities
  output per frame a unit ray in the body frame; loss 1 - cos to the final ray, every frame of the approach, equal
                   weights (Mavsar's logistic weights need an α they do not report)
  training         Adam 1e-3, batch 40 (Mavsar); early stopping on validation people; 4 folds by PERSON (no person in
                   both train and test); errors pooled over the test folds.

  ./run.sh scripts/dp_final_direction_learned.py --out outputs/dp_prediction/2026-10-16
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from scripts.dp_final_direction import EYES, RHOS, SH, WR, ang, gestures, person_ci, unit

EL = {'l': 7, 'r': 8}
HIPS = (11, 12)


def body_frame(P: np.ndarray, lr: str, k0: int):
    """Rotation (rows: lateral, forward, up) and origin from frame k0 (the onset: no future frames used)."""
    other = 'l' if lr == 'r' else 'r'
    sh, so = P[k0, SH[lr]], P[k0, SH[other]]
    up = unit((sh + so) / 2 - P[k0, list(HIPS)].mean(0))
    lat = unit((sh - so) - ((sh - so) @ up) * up)
    fwd = np.cross(up, lat)
    return np.stack([lat, fwd, up]), sh, float(np.linalg.norm(sh - so))


def features(g: dict):
    P, lr, on, pose = g['P'], g['lr'], g['on'], g['pose']
    if not np.isfinite(P[:, list(HIPS) + [EL[lr], SH['l'], SH['r']]]).all():
        return None
    R, o, s = body_frame(P, lr, on)
    other = 'l' if lr == 'r' else 'r'
    J = np.stack([P[:, EYES].mean(1), P[:, SH[lr]], P[:, SH[other]], P[:, EL[lr]], P[:, WR[lr]]], 1)  # (T, 5, 3)
    X = ((J - o) @ R.T) / s
    seq = X[on:pose + 1].reshape(pose + 1 - on, -1)
    vel = np.vstack([np.zeros((1, seq.shape[1])), np.diff(seq, axis=0)]) * 15
    return np.hstack([seq, vel]).astype(np.float32), (R @ g['final']).astype(np.float32)


class Net(nn.Module):
    """residual: the output is the present eye -> wrist ray plus a learned correction (zero-initialised head, so
    training starts from the present ray, which is already accurate late in the approach: dp_final_direction.md)."""

    def __init__(self, d_in, h=64, residual=False):
        super().__init__()
        self.gru = nn.GRU(d_in, h, num_layers=2, batch_first=True)
        self.head = nn.Linear(h, 3)
        self.residual = residual
        if residual:
            nn.init.zeros_(self.head.weight), nn.init.zeros_(self.head.bias)

    def forward(self, x):
        y, _ = self.gru(x)
        d = self.head(y)
        if self.residual:                  # features: eye = x[..., 0:3], wrist = x[..., 12:15] (body frame)
            d = d + nn.functional.normalize(x[..., 12:15] - x[..., 0:3], dim=-1)
        return nn.functional.normalize(d, dim=-1)


def batches(idx, X, Y, bs, rng):
    idx = rng.permutation(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        L = max(len(X[j]) for j in b)
        x = np.zeros((len(b), L, X[b[0]].shape[1]), np.float32)
        m = np.zeros((len(b), L), np.float32)
        for r, j in enumerate(b):
            x[r, :len(X[j])], m[r, :len(X[j])] = X[j], 1
        yield torch.from_numpy(x), torch.from_numpy(np.stack([Y[j] for j in b])), torch.from_numpy(m)


def loss_of(net, x, y, m, dev):
    p = net(x.to(dev))
    cos = (p * y.to(dev)[:, None]).sum(-1)
    m = m.to(dev)
    return ((1 - cos) * m).sum() / m.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dp', default='third_party/dp_meta/DPDataset_public')
    ap.add_argument('--persons', default='third_party/deepoint/src/utils.py')
    ap.add_argument('--half-sep', type=float, default=8.5)
    ap.add_argument('--folds', type=int, default=4)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--max-epochs', type=int, default=400)
    ap.add_argument('--patience', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--residual', action='store_true', help='predict a correction of the present ray')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    src = Path(args.persons).read_text()
    persons = {(m.group(1), m.group(2)): int(m.group(3)) for m in re.finditer(r'\("([^"]+)", "([^"]+)"\): (\d+)', src)}
    G, X, Y = [], [], []
    for g in gestures(Path(args.dp), persons, 0.10):
        f = features(g) if g['person'] is not None else None
        if f is not None:
            G.append(g), X.append(f[0]), Y.append(f[1])
    pid = np.array([g['person'] for g in G])
    people = rng.permutation(np.unique(pid))
    fold_of = {p: i % args.folds for i, p in enumerate(people)}
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    err = np.full((len(G), len(RHOS)), np.nan)
    log = []
    for f in range(args.folds):
        test = np.flatnonzero([fold_of[p] == f for p in pid])
        rest = [p for p in people if fold_of[p] != f]
        val_people = set(rest[:max(2, len(rest) // 6)])
        val = np.flatnonzero([p in val_people for p in pid])
        tr = np.flatnonzero([fold_of[p] != f and p not in val_people for p in pid])
        net = Net(X[0].shape[1], args.hidden, args.residual).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        best, best_state, bad = np.inf, None, 0
        for ep in range(args.max_epochs):
            net.train()
            for x, y, m in batches(tr, X, Y, 40, rng):
                opt.zero_grad()
                loss_of(net, x, y, m, dev).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(np.mean([loss_of(net, x, y, m, dev).item() for x, y, m in batches(val, X, Y, 256, rng)]))
            if v < best - 1e-5:
                best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= args.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        log.append(dict(fold=f, n_train=len(tr), n_val=len(val), n_test=len(test), epochs=ep + 1, val_loss=best))
        print(log[-1], flush=True)
        with torch.no_grad():
            for i in test:
                p = net(torch.from_numpy(X[i])[None].to(dev))[0].cpu().numpy()
                n = len(X[i]) - 1
                for j, rho in enumerate(RHOS):
                    err[i, j] = ang(p[max(int(round(rho * n)), 1)], Y[i])
    table = []
    for j, rho in enumerate(RHOS):
        e = err[:, j]
        ok = (e < args.half_sep).astype(float)
        q1, med, q3 = np.percentile(e, [25, 50, 75])
        lo, hi = person_ci(ok, pid, np.mean, rng)
        table.append(dict(rho=float(rho), median_deg=float(med), q1=float(q1), q3=float(q3), share_below=float(ok.mean()),
                          share_ci=[lo, hi]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = '_residual' if args.residual else ''
    (out / f'dp_final_direction_learned{tag}.json').write_text(json.dumps(dict(args=vars(args), folds=log, table=table), indent=1))
    lines = [f'# Learned final-direction predictor (GRU{" + residual on the present ray" if args.residual else ""}, trained on every prefix), real people (DP), {len(G)} pointings\n',
             '| | ' + ' | '.join(f'ρ {r:.1f}' for r in RHOS) + ' |', '|---|' + '---|' * len(RHOS),
             '| learned: median error [°] | ' + ' | '.join(f"{c['median_deg']:.1f}" for c in table) + ' |',
             f'| learned: share < {args.half_sep}° | ' + ' | '.join(
                 f"{c['share_below']:.2f} [{c['share_ci'][0]:.2f}, {c['share_ci'][1]:.2f}]" for c in table) + ' |']
    (out / f'dp_final_direction_learned{tag}.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
