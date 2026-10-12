"""Real pointing kinematics from the DP dataset (DeePoint, Nakamura et al. ICCV 2023: 33 people, triangulated 3D
keypoints at 15 fps, pointing labels = the hold), written as the sampling distributions of the simulated human
(assets/human/dp_pointing_kinematics.json; docs/requirements/INTENT_ACT_v3.2.md §4.2d).

Per labelled pointing (pointing arm: shoulder / elbow / wrist = COCO 6 / 8 / 10 or 5 / 7 / 9):
  approach_s   the wrist's approach motion before the label: from the speed rising above `frac` of the approach peak
               to it falling below it again (frac = 10 %: the usual 5-10 % of peak velocity convention for movement
               onset / end; 5 % and 20 % are reported for sensitivity)
  hold_s       the label itself (the participant held a button while pointing; measured: it starts as the pose is
               reached and ends as the retraction starts)
  retract_s    the retraction after the label, same threshold rule
  extension    |wrist - shoulder| / (|shoulder - elbow| + |elbow - wrist|), median over the hold (unit-free: the
               triangulation is not in metres)
  elbow_flex_deg  angle between upper arm and forearm over the hold (0 = straight)
Stored: the values within each measure's 5-95 % range (drops triangulation / threshold outliers), the summary, and
`minjerk_fraction_above_frac`: the share of a minimum-jerk motion's duration whose speed is above `frac` of its peak,
so a sampled `approach_s` converts to a minimum-jerk duration T = approach_s / that fraction (the simulated motion
then measures the same way).

  scripts/deepoint.sh scripts/measure_dp_pointing.py --dp third_party/dp_meta/DPDataset_public --out assets/human/dp_pointing_kinematics.json
"""
import argparse
import csv
import json
import pickle
import re
from datetime import date
from pathlib import Path

import numpy as np
import yaml

SH, EL, WR = {'l': 5, 'r': 6}, {'l': 7, 'r': 8}, {'l': 9, 'r': 10}
FPS, W = 15, 22


def minjerk_fraction(frac: float) -> float:
    """Share of a minimum-jerk motion above `frac` of its peak speed (speed ∝ u^2 (1-u)^2, peak at u = 0.5)."""
    c = frac * 0.0625                                  # u (1 - u) = sqrt(c)
    u = (1 - np.sqrt(1 - 4 * np.sqrt(c))) / 2
    return float(1 - 2 * u)


def measure(dp: Path, frac: float, persons: dict) -> list[dict]:
    coords = pickle.load((dp / 'keypoints/triangulation.pickle').open('rb'))
    rows = []
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
                if not np.isfinite(P[:, [SH[lr], EL[lr], WR[lr]]]).all():
                    continue
                wr = P[:, WR[lr]]
                sp = np.r_[np.nan, np.linalg.norm(np.diff(wr, axis=0), axis=1) * FPS]
                i_s, i_e = W, W + (e - s)
                pa = 1 + int(np.nanargmax(sp[1:i_s + 3]))
                on = next((k for k in range(pa, 0, -1) if sp[k] < frac * sp[pa]), None)
                pose = next((k for k in range(pa, len(sp)) if sp[k] < frac * sp[pa]), None)
                pr = i_e - 2 + int(np.nanargmax(sp[i_e - 2:]))
                ron = next((k for k in range(pr, i_s, -1) if sp[k] < frac * sp[pr]), None)
                rend = next((k for k in range(pr, len(sp)) if sp[k] < frac * sp[pr]), None)
                seg = P[i_s:i_e + 1]
                up, fo = seg[:, EL[lr]] - seg[:, SH[lr]], seg[:, WR[lr]] - seg[:, EL[lr]]
                L = np.median(np.linalg.norm(up, axis=1) + np.linalg.norm(fo, axis=1))
                cosang = (up * fo).sum(1) / np.linalg.norm(up, axis=1) / np.linalg.norm(fo, axis=1)
                rows.append(dict(
                    person=persons.get((venue, sess)),
                    approach_s=None if on is None or pose is None else (pose - on) / FPS,
                    hold_s=(e - s) / FPS,
                    retract_s=None if ron is None or rend is None else (rend - ron) / FPS,
                    extension=float(np.median(np.linalg.norm(seg[:, WR[lr]] - seg[:, SH[lr]], axis=1)) / L),
                    elbow_flex_deg=float(np.degrees(np.median(np.arccos(np.clip(cosang, -1, 1)))))))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dp', required=True, help='DPDataset_public folder (labels/ + keypoints/triangulation.pickle)')
    p.add_argument('--persons', default='third_party/deepoint/src/utils.py', help='VENUESESSION2PERSONID source')
    p.add_argument('--frac', type=float, default=0.10)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    src = Path(args.persons).read_text()
    persons = {(m.group(1), m.group(2)): int(m.group(3)) for m in re.finditer(r'\("([^"]+)", "([^"]+)"\): (\d+)', src)}
    out = dict(source='DP dataset (DeePoint, Nakamura et al., ICCV 2023): labels + keypoints/triangulation.pickle',
               script='scripts/measure_dp_pointing.py', date=str(date.today()), threshold_frac=args.frac,
               minjerk_fraction_above_frac=minjerk_fraction(args.frac), summary={}, samples={}, sensitivity={})
    for frac in sorted({args.frac, 0.05, 0.20}):
        rows = measure(Path(args.dp), frac, persons)
        for k in ('approach_s', 'retract_s'):
            v = np.array([r[k] for r in rows if r[k] is not None])
            out['sensitivity'].setdefault(k, {})[f'{frac:.2f}'] = dict(median=float(np.median(v)), n=int(len(v)))
        if frac != args.frac:
            continue
        out['n_pointings'], out['n_people'] = len(rows), len({r['person'] for r in rows})
        for k in ('approach_s', 'hold_s', 'retract_s', 'extension', 'elbow_flex_deg'):
            v = np.array([r[k] for r in rows if r[k] is not None and np.isfinite(r[k])])
            lo, hi = np.percentile(v, [5, 95])
            kept = v[(v >= lo) & (v <= hi)]
            out['samples'][k] = np.round(np.sort(kept), 4).tolist()
            out['summary'][k] = dict(n=int(len(v)), median=float(np.median(v)), q25=float(np.percentile(v, 25)),
                                     q75=float(np.percentile(v, 75)), p5=float(lo), p95=float(hi))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(json.dumps({k: out[k] for k in ('summary', 'sensitivity', 'n_pointings', 'n_people', 'minjerk_fraction_above_frac')}, indent=1))


if __name__ == '__main__':
    main()
