"""How early is the final pointing direction known on REAL people? (docs/discussions/de_xuat_intent_thong_nhat.md)

DP dataset (DeePoint, Nakamura et al. ICCV 2023): triangulated 3D COCO keypoints at 15 fps, one row per labelled
pointing (target marker, hold start / end, arm). Per pointing, the approach segment [on, pose] is found as in
scripts/measure_dp_pointing.py (wrist speed above / below 10 % of its approach peak). The final direction is the
median eye -> wrist ray over the hold (the eye-finger line people aim with). At ρ = 0.1 .. 1.0 of the approach a
predictor sees only the frames up to on + ρ (pose - on) and gives a ray:
  present   the eye -> wrist ray of the current frame (what a pointing-direction estimator reports now)
  cv        constant velocity (Schöller et al., RA-L 2020): wrist + velocity x remaining time
  minjerk   straight-line minimum jerk from the onset wrist (Flash & Hogan 1985; Morasso 1981): the current
            displacement divided by the minimum-jerk progress s(u), u = elapsed / T
Remaining time / T: the median approach duration of the OTHER people (leave-one-person-out), no tuned parameter.
Reported per ρ: angular error to the final ray (median, IQR) and the share below `--half-sep` degrees (8.5 = half
the measured twin separation, 17°: below it the right twin is chosen wherever the other twin is), with 95 %
bootstrap intervals over people.

  ./run.sh scripts/dp_final_direction.py --out outputs/dp_prediction/2026-10-16
"""
import argparse
import csv
import json
import pickle
import re
from pathlib import Path

import numpy as np
import yaml

SH, WR = {'l': 5, 'r': 6}, {'l': 9, 'r': 10}
EYES = (1, 2)
FPS, W = 15, 22                                   # as scripts/measure_dp_pointing.py
RHOS = np.round(np.arange(0.1, 1.01, 0.1), 1)
METHODS = ('present', 'cv', 'minjerk')


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def ang(a, b):
    return np.degrees(np.arccos(np.clip((unit(a) * unit(b)).sum(-1), -1.0, 1.0)))


def min_jerk(u):
    u = np.clip(u, 0.0, 1.0)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def gestures(dp: Path, persons: dict, frac: float) -> list[dict]:
    coords = pickle.load((dp / 'keypoints/triangulation.pickle').open('rb'))
    out = []
    for lab in sorted((dp / 'labels').iterdir()):
        info = yaml.safe_load((lab / 'timeinfo.yaml').read_text())
        filt, venue = info['filt'], lab.name
        for sess in info['sessions']:
            C = coords.get(venue, {}).get(sess)
            if C is None:
                continue
            r0 = list(csv.reader((lab / f'{sess}.txt').open()))
            v0, v1 = int(r0[0][0]) // filt, int(r0[0][1]) // filt
            for r in r0[1:]:
                if not r or r[0].startswith('#'):
                    continue
                try:
                    s, e, lr = int(r[1]) // filt, int(r[2]) // filt, r[3].strip()
                except (ValueError, IndexError):
                    continue
                if lr not in WR or s - W < max(v0, 1) or e + W >= min(v1, len(C)) or e <= s:
                    continue
                P = np.stack([np.asarray(C[t], float).reshape(3, 17).T for t in range(s - W, e + W + 1)])
                if not np.isfinite(P[:, [SH[lr], WR[lr], *EYES]]).all():
                    continue
                wr = P[:, WR[lr]]
                sp = np.r_[np.nan, np.linalg.norm(np.diff(wr, axis=0), axis=1) * FPS]
                i_s, i_e = W, W + (e - s)
                pa = 1 + int(np.nanargmax(sp[1:i_s + 3]))
                on = next((k for k in range(pa, 0, -1) if sp[k] < frac * sp[pa]), None)
                pose = next((k for k in range(pa, len(sp)) if sp[k] < frac * sp[pa]), None)
                if on is None or pose is None or pose - on < 3:
                    continue
                eye = P[:, EYES].mean(1)
                final = unit(np.median(unit(wr[i_s:i_e + 1] - eye[i_s:i_e + 1]), axis=0))
                out.append(dict(person=persons.get((venue, sess)), wr=wr, eye=eye, on=on, pose=pose, final=final,
                                P=P, lr=lr))
    return out


def predict(g: dict, k: int, T: float) -> dict:
    wr, eye, on = g['wr'], g['eye'], g['on']
    el = (k - on) / FPS
    rays = {'present': wr[k] - eye[k]}
    v = (wr[k] - wr[k - 1]) * FPS
    rays['cv'] = wr[k] + v * max(T - el, 0.0) - eye[k]
    s = min_jerk(el / T)
    end = wr[on] + (wr[k] - wr[on]) / s if s > 0.05 else wr[k]
    rays['minjerk'] = end - eye[k]
    return rays


def person_ci(vals: np.ndarray, pid: np.ndarray, stat, rng, n=2000):
    ps = np.unique(pid)
    idx = {p: np.flatnonzero(pid == p) for p in ps}
    bs = [stat(vals[np.concatenate([idx[p] for p in rng.choice(ps, len(ps))])]) for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dp', default='third_party/dp_meta/DPDataset_public')
    ap.add_argument('--persons', default='third_party/deepoint/src/utils.py', help='VENUESESSION2PERSONID source')
    ap.add_argument('--frac', type=float, default=0.10)
    ap.add_argument('--half-sep', type=float, default=8.5, help='degrees: half the twin separation (17° measured)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    src = Path(args.persons).read_text()
    persons = {(m.group(1), m.group(2)): int(m.group(3)) for m in re.finditer(r'\("([^"]+)", "([^"]+)"\): (\d+)', src)}
    G = [g for g in gestures(Path(args.dp), persons, args.frac) if g['person'] is not None]
    pid_all = np.array([g['person'] for g in G])
    dur = np.array([(g['pose'] - g['on']) / FPS for g in G])
    T_loo = {p: float(np.median(dur[pid_all != p])) for p in np.unique(pid_all)}
    err = {m: np.full((len(G), len(RHOS)), np.nan) for m in METHODS}
    for i, g in enumerate(G):
        for j, rho in enumerate(RHOS):
            k = max(g['on'] + int(round(rho * (g['pose'] - g['on']))), g['on'] + 1)
            for m, ray in predict(g, k, T_loo[g['person']]).items():
                err[m][i, j] = ang(ray, g['final'])
    rng = np.random.default_rng(0)
    rows, table = [], {}
    for m in METHODS:
        table[m] = []
        for j, rho in enumerate(RHOS):
            e = err[m][:, j]
            ok = (e < args.half_sep).astype(float)
            q1, med, q3 = np.percentile(e, [25, 50, 75])
            lo, hi = person_ci(ok, pid_all, np.mean, rng)
            table[m].append(dict(rho=float(rho), median_deg=float(med), q1=float(q1), q3=float(q3),
                                 share_below=float(ok.mean()), share_ci=[lo, hi]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(n_gestures=len(G), n_people=int(len(np.unique(pid_all))), approach_s_median=float(np.median(dur)),
                approach_s_iqr=[float(x) for x in np.percentile(dur, [25, 75])], half_sep_deg=args.half_sep, frac=args.frac)
    (out / 'dp_final_direction.json').write_text(json.dumps(dict(meta=meta, table=table), indent=1))
    lines = [f"# Final pointing direction from a partial approach, real people (DP)\n",
             f"{meta['n_gestures']} pointings, {meta['n_people']} people; approach (10 % peak-speed rule) median "
             f"{meta['approach_s_median']:.2f} s, IQR {meta['approach_s_iqr'][0]:.2f}-{meta['approach_s_iqr'][1]:.2f} s. "
             f"Error = angle to the final eye -> wrist ray (median over the hold). Share = error < {args.half_sep}°, "
             f"95 % bootstrap over people.\n",
             '| method | ' + ' | '.join(f'ρ {r:.1f}' for r in RHOS) + ' |', '|---|' + '---|' * len(RHOS)]
    for m in METHODS:
        lines.append(f'| {m}: median error [°] | ' + ' | '.join(f"{c['median_deg']:.1f}" for c in table[m]) + ' |')
        lines.append(f'| {m}: share < {args.half_sep}° | ' + ' | '.join(
            f"{c['share_below']:.2f} [{c['share_ci'][0]:.2f}, {c['share_ci'][1]:.2f}]" for c in table[m]) + ' |')
    (out / 'dp_final_direction.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
