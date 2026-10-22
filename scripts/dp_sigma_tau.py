"""Time-aware pointing likelihood from REAL people (DP): how far is the present ray from the FINAL ray, τ seconds
after the motion onset, and what does it do to a two-candidate belief? (docs/discussions/de_xuat_intent_thong_nhat.md)

The device belief (FETCH-POMDP form, Whitney et al. 2017) is L(g) = p exp(-θ_g² / 2σ²) + (1 - p), θ_g the angle
between the present pointing ray and candidate g. The simulated device widens σ by the frames since the gesture was
detected (device_folds.json `sigma_deg_by_frames`, measured in simulation). Here, on real people:
  σ_real(τ), p(τ)  maximum likelihood (EM) of the two parts of that likelihood as a density of the angle θ between
                   the present ray and the final ray (median over the hold): p Rayleigh(θ; σ) + (1 - p) sin θ / 2,
                   per frame since the onset (15 fps); fitted without the test person (leave-one-person-out)
  check            a candidate pair `--sep` apart: the true one on the final ray, the other displaced horizontally
                   (`--distractor horizontal`: sideways in the body frame, as two objects on a table) or in a random
                   direction; the belief of the true one under: the simulated device's σ table (p = 1), σ_real(τ)
Rays (`--ray`): eye -> wrist (default; the eye-finger line people aim with), shoulder -> wrist, elbow -> wrist.
Onset: the wrist speed above `--frac` of its approach peak (scripts/measure_dp_pointing.py). Noise floor of the label:
the angle of each hold frame's ray to the hold median.
Reported per τ: accuracy (ties 1/2), share of frames with the belief >= 0.7 / 0.9 / 0.95 on the WRONG candidate,
log score, ECE; per θ: the first crossing of each pointing (Proba Threshold, Renault et al. 2025): share triggered,
median τ, share wrong; 95 % bootstrap over people.

  ./run.sh -m scripts.dp_sigma_tau --out outputs/dp_prediction/2026-10-16 [--ray eye_wrist --frac 0.10
      --distractor horizontal --sep 17 --tag base]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from scripts.dp_final_direction import EYES, FPS, SH, W, WR, gestures, person_ci, unit

EL = {'l': 7, 'r': 8}
HIPS = (11, 12)
MAX_F = 22                                              # frames since the onset reported (1.47 s)
THETAS = (0.7, 0.9, 0.95)


def mixture_fit(theta_deg: np.ndarray, iters: int = 300) -> tuple[float, float]:
    """EM for p Rayleigh(θ; σ) + (1 - p) sin θ / 2 (θ the angle to the final ray). Returns (p, σ [deg])."""
    t = np.radians(np.clip(theta_deg, 1e-3, 180))
    p, s2 = 0.5, np.mean(t ** 2) / 2
    for _ in range(iters):
        fr = t / s2 * np.exp(-t ** 2 / (2 * s2))
        fu = np.sin(t) / 2
        r = p * fr / (p * fr + (1 - p) * fu + 1e-300)
        p = float(np.clip(r.mean(), 1e-3, 1 - 1e-3))
        s2 = float((r * t ** 2).sum() / (2 * r.sum()))
    return p, float(np.degrees(np.sqrt(s2)))


def direction_likelihood(th_deg, p, sigma_deg):
    """Per unit solid angle: p N(θ; σ) / (2π σ²) + (1 - p) / 4π (the density whose angle law mixture_fit fits)."""
    s2, t = np.radians(sigma_deg) ** 2, np.radians(th_deg)
    return p * np.exp(-t ** 2 / (2 * s2)) / (2 * np.pi * s2) + (1 - p) / (4 * np.pi)


def rotate_towards(v: np.ndarray, d: np.ndarray, deg: float) -> np.ndarray:
    """v turned by `deg` towards the direction d (d made orthogonal to v)."""
    d = unit(d - (d @ v) * v)
    a = np.radians(deg)
    return unit(v * np.cos(a) + d * np.sin(a))


def ang(a, b):
    return np.degrees(np.arccos(np.clip((unit(a) * unit(b)).sum(-1), -1.0, 1.0)))


def ece(conf, correct, bins=10):
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        sel = (conf >= lo) & ((conf < lo + 1 / bins) | (lo >= 1 - 1 / bins - 1e-9))
        if sel.any():
            e += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(e)


def ray_of(P, lr, kind):
    wr = P[:, WR[lr]]
    base = {'eye_wrist': P[:, list(EYES)].mean(1), 'shoulder_wrist': P[:, SH[lr]], 'elbow_wrist': P[:, EL[lr]]}[kind]
    return wr - base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dp', default='third_party/dp_meta/DPDataset_public')
    ap.add_argument('--persons', default='third_party/deepoint/src/utils.py')
    ap.add_argument('--device-folds', default='outputs/prelim/2026-10-13_slow/device/device_folds.json')
    ap.add_argument('--ray', default='eye_wrist', choices=['eye_wrist', 'shoulder_wrist', 'elbow_wrist'])
    ap.add_argument('--frac', type=float, default=0.10)
    ap.add_argument('--distractor', default='horizontal', choices=['horizontal', 'random'])
    ap.add_argument('--sep', type=float, default=17.0, help='candidate separation [deg] (17 = the measured twins)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='base')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    src = Path(args.persons).read_text()
    persons = {(m.group(1), m.group(2)): int(m.group(3)) for m in re.finditer(r'\("([^"]+)", "([^"]+)"\): (\d+)', src)}
    sb = json.loads(Path(args.device_folds).read_text())['folds']['0']['sigma_deg_by_frames']
    sim_frames, sim_sigma = np.asarray(sb['frames']), np.asarray(sb['point'])
    rows, floor = [], []                                # (person, frames since onset, θ to final, θ to the other, gesture)
    for gi, g in enumerate(gestures(Path(args.dp), persons, args.frac)):
        P, lr, on = g['P'], g['lr'], g['on']
        if g['person'] is None or not np.isfinite(P[on, list(HIPS) + [SH['l'], SH['r'], EL[lr]]]).all():
            continue
        R = ray_of(P, lr, args.ray)
        hold = R[W:len(R) - W]
        final = unit(np.median(unit(hold), axis=0))
        floor.extend(ang(hold, final).tolist())
        if args.distractor == 'horizontal':
            up = unit((P[on, SH['l']] + P[on, SH['r']]) / 2 - P[on, list(HIPS)].mean(0))
            side = np.cross(up, final) * rng.choice([-1, 1])
        else:
            side = rng.normal(size=3)
        other = rotate_towards(final, side, args.sep)
        for k in range(on, min(len(R) - W, on + MAX_F + 1)):     # up to the end of the hold (retraction excluded)
            rows.append((g['person'], k - on, ang(R[k], final), ang(R[k], other), gi))
    A = np.array(rows, float)
    pid, f, th, to, gix = A[:, 0].astype(int), A[:, 1].astype(int), A[:, 2], A[:, 3], A[:, 4].astype(int)
    people = np.unique(pid)
    fit_all = {int(j): mixture_fit(th[f == j]) for j in np.unique(f)}
    loo = {q: {int(j): mixture_fit(th[(f == j) & (pid != q)]) for j in np.unique(f)} for q in people}
    pf = np.array([loo[q][j][0] for q, j in zip(pid, f)])
    sf = np.array([loo[q][j][1] for q, j in zip(pid, f)])
    lt, lo = direction_likelihood(th, pf, sf), direction_likelihood(to, pf, sf)
    s_sim = sim_sigma[np.searchsorted(sim_frames, f, 'right') - 1]
    et, eo = -th ** 2 / (2 * s_sim ** 2), -to ** 2 / (2 * s_sim ** 2)
    m = np.maximum(et, eo)
    beliefs = {'sim_table': np.exp(et - m) / (np.exp(et - m) + np.exp(eo - m)), 'real_tau': lt / (lt + lo)}
    per_tau = []
    for j in range(0, MAX_F + 1):
        sel = f == j
        if not sel.any():
            continue
        r = dict(frames=j, tau=j / FPS, n=int(sel.sum()), p=fit_all[j][0], sigma_deg=fit_all[j][1],
                 sim_sigma_deg=float(s_sim[sel][0]), median_angle=float(np.median(th[sel])))
        for name, b in beliefs.items():
            x = b[sel]
            r[f'{name}_acc'] = float(np.mean(np.where(np.isclose(x, 0.5), 0.5, x > 0.5)))
            for t in THETAS:
                r[f'{name}_wrong_ge_{t}'] = float((1 - x >= t).mean())
            r[f'{name}_log_score'] = float(np.log(np.clip(x, 1e-12, 1)).mean())
            r[f'{name}_ece'] = ece(np.maximum(x, 1 - x), (x > 0.5).astype(float))
        per_tau.append(r)
    crossing = {}
    for name, b in beliefs.items():
        crossing[name] = []
        for t in THETAS:
            tt, wrong, pp, n = [], [], [], 0
            for gi in np.unique(gix):
                sel = np.flatnonzero(gix == gi)
                n += 1
                x = b[sel]
                hit = np.flatnonzero((x >= t) | (1 - x >= t))
                if len(hit):
                    tt.append(f[sel][hit[0]] / FPS), wrong.append(float(1 - x[hit[0]] >= t)), pp.append(pid[sel][0])
            wrong, pp = np.array(wrong), np.array(pp)
            crossing[name].append(dict(theta=t, triggered=len(tt) / n, median_tau=float(np.median(tt)),
                                       wrong=float(wrong.mean()), wrong_ci=list(person_ci(wrong, pp, np.mean, rng, n=1000))))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(args=vars(args), n_pointings=int(len(np.unique(gix))), n_people=int(len(people)),
                label_noise_floor_deg=dict(median=float(np.median(floor)), p90=float(np.percentile(floor, 90))))
    (out / f'dp_sigma_tau_{args.tag}.json').write_text(json.dumps(dict(meta=meta, per_tau=per_tau, crossing=crossing), indent=1))
    L = [f"# Time-aware pointing likelihood on real people (DP) — {args.tag}: ray {args.ray}, onset {args.frac:.0%} of the "
         f"peak, distractor {args.distractor} at {args.sep:g}°\n",
         f"{meta['n_pointings']} pointings, {meta['n_people']} people. Label noise floor (hold frames vs the hold median): "
         f"median {meta['label_noise_floor_deg']['median']:.1f}°, 90 % {meta['label_noise_floor_deg']['p90']:.1f}°.\n",
         '| τ [s] | p(τ) | σ_real(τ) [°] | σ sim table [°] | median θ to final [°] | accuracy | wrong ≥ 0.9: sim table / real | '
         'wrong ≥ 0.7: sim table / real | log score: sim / real | ECE: sim / real |', '|' + '---|' * 10]
    for r in per_tau:
        if r['frames'] % 2:
            continue
        L.append(f"| {r['tau']:.2f} | {r['p']:.2f} | {r['sigma_deg']:.1f} | {r['sim_sigma_deg']:.1f} | {r['median_angle']:.1f} | "
                 f"{r['real_tau_acc']:.2f} | {r['sim_table_wrong_ge_0.9']:.2f} / {r['real_tau_wrong_ge_0.9']:.2f} | "
                 f"{r['sim_table_wrong_ge_0.7']:.2f} / {r['real_tau_wrong_ge_0.7']:.2f} | "
                 f"{r['sim_table_log_score']:.2f} / {r['real_tau_log_score']:.2f} | {r['sim_table_ece']:.2f} / {r['real_tau_ece']:.2f} |")
    L += ['', '| belief | θ | triggered | median τ [s] | wrong at trigger |', '|---|---|---|---|---|']
    for name, cs in crossing.items():
        for c in cs:
            L.append(f"| {name} | {c['theta']} | {c['triggered']:.2f} | {c['median_tau']:.2f} | "
                     f"{c['wrong']:.3f} [{c['wrong_ci'][0]:.3f}, {c['wrong_ci'][1]:.3f}] |")
    (out / f'dp_sigma_tau_{args.tag}.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
