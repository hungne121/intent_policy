"""Measure the intent-perception constants on a recorded dataset (INTENT_ACT_GUIDE_v2.md §9.2, §9.3, M0 answers) and
optionally write them into configs/intent_schema.yaml:

  ray      pointing-ray error (index base -> index tip vs the pointed place) over the pointing holds: mean, std, p95
           -> perfect.cone_sigma_deg; per pointing gesture the angular gap between the pointed place and the nearest
           other place (scenario difficulty variable, meta/intent_scenario_vars.json)
  ramp     median (t_clear - t_onset) of the training episodes -> perfect.ramp_s
  holding  gripper-opening holding detector vs the simulator grasp state: grid over the two opening thresholds and
           the steadiness, precision / recall / grasp and release latency -> holding

  ./run.sh -m scripts.calibrate_intent --dataset-root outputs/datasets/intent_act_early --write-schema
Uses meta/intent_segments.json (run scripts/build_intent_labels.py first; any source).
"""
import argparse
import itertools
import json
import re

import numpy as np

from intent_policy.intent.labels import SCHEMA_PATH, load_schema
from intent_policy.intent.perception import angles_deg
from intent_policy.intent.places import target_matrix
from intent_policy.intent.tracker import HoldingDetector
from intent_policy.utils import resolve
from scripts.build_intent_labels import load_episodes


def ray_stats(episodes, segments, schema) -> tuple[dict, list]:
    targets, names = target_matrix(schema), schema['keypoints']
    errs, gaps = [], []
    for e, segs in segments.items():
        kp = episodes[e]['keypoints']
        for k, s in enumerate(segs):
            if s['gesture'] != 'point':
                continue
            ti = schema['targets'].index(s['target'])
            hand = max(('r', 'l'), key=lambda h: np.linalg.norm(kp[s['t_clear'], names.index(f'{h}_index_tip')]
                                                                - kp[0, names.index(f'{h}_index_tip')]))
            b, tip = names.index(f'{hand}_index_base'), names.index(f'{hand}_index_tip')
            th_all = []
            for t in range(s['t_clear'], s['t_event'] + 1):
                th = angles_deg(kp[t, tip], kp[t, tip] - kp[t, b], targets[1:])
                errs.append(th[ti - 1])
                th_all.append(th)
            th = np.mean(th_all, 0)
            others = np.delete(th, ti - 1)
            gaps.append(dict(episode=int(e), segment=k, target=s['target'], error_deg=round(float(th[ti - 1]), 3),
                             gap_deg=round(float(others.min() - th[ti - 1]), 3),
                             nearest_other=schema['targets'][1:][int(np.argmin(np.where(np.arange(len(th)) == ti - 1, np.inf, th)))]))
    errs = np.asarray(errs)
    return dict(n_frames=int(len(errs)), mean_deg=float(errs.mean()), std_deg=float(errs.std()),
                p95_deg=float(np.percentile(errs, 95)), max_deg=float(errs.max())), gaps


def holding_grid(episodes) -> list[dict]:
    rows = []
    for lo, hi, st, sf in itertools.product([-0.005, 0.0, 0.002, 0.004], [0.05, 0.06, 0.07], [0.001, 0.002, 0.003, 0.004], [1, 2]):
        tp = fp = fn = 0
        lat_on, lat_off, missed = [], [], 0
        for ep in episodes.values():
            gt = ep['holding_gt']
            d = HoldingDetector(lo, hi, st, sf)
            det = np.array([d.step(float(g)) for g in ep['state'][:, 6]])
            tp, fp, fn = tp + int((det & gt).sum()), fp + int((det & ~gt).sum()), fn + int((~det & gt).sum())
            for a, b, lat in ((gt, det, lat_on), (~gt, ~det, lat_off)):
                for t in np.flatnonzero(a[1:] & ~a[:-1]) + 1:
                    hit = np.flatnonzero(b[t:])
                    if len(hit):
                        lat.append(int(hit[0]))
                    else:
                        missed += 1
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        rows.append(dict(open_min_m=lo, open_max_m=hi, steady_m=st, steady_frames=sf, precision=round(p, 4), recall=round(r, 4),
                         f1=round(2 * p * r / max(p + r, 1e-9), 4), grasp_latency_frames=round(float(np.mean(lat_on)), 2) if lat_on else None,
                         grasp_latency_max=int(np.max(lat_on)) if lat_on else None,
                         release_latency_frames=round(float(np.mean(lat_off)), 2) if lat_off else None, missed_edges=missed))
    return sorted(rows, key=lambda r: (-r['f1'], r['grasp_latency_frames'] or 99))


def write_schema(path, cone: float, ramp: float, hold: dict) -> None:
    p = resolve(path)
    s = p.read_text()
    s = re.sub(r'(  cone_sigma_deg: )[0-9.]+(\s+# measured)[^\n]*', rf'\g<1>{cone:.2f}\g<2>: pointing-ray error (index base -> tip), see intent_calibration', s)
    s = re.sub(r'(  ramp_s: )[0-9.]+(\s+# measured)[^\n]*', rf'\g<1>{ramp:.2f}\g<2>: median (t_clear - t_onset) of the training episodes', s)
    s = re.sub(r'holding: \{[^}]*\}', 'holding: {' + ', '.join(f'{k}: {v}' for k, v in hold.items()) + '}', s)
    p.write_text(s)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--schema', default=SCHEMA_PATH)
    p.add_argument('--out', default='outputs/intent_calibration')
    p.add_argument('--write-schema', action='store_true', help='write cone_sigma_deg, ramp_s and holding into the schema')
    args = p.parse_args()
    root, schema = resolve(args.dataset_root), load_schema(args.schema)
    episodes = load_episodes(root)
    segments = {int(k): v for k, v in json.loads((root / 'meta/intent_segments.json').read_text()).items()}
    train = json.loads((root / 'meta/intent_schema.json').read_text()).get('train_episodes') or sorted(segments)
    ray, gaps = ray_stats(episodes, segments, schema)
    dur = [(s['t_clear'] - s['t_onset']) / schema['fps'] for e in train for s in segments.get(e, [])]
    ramp = dict(median_s=float(np.median(dur)), mean_s=float(np.mean(dur)), n=len(dur),
                by_gesture={g: float(np.median([(s['t_clear'] - s['t_onset']) / schema['fps'] for e in train for s in segments.get(e, [])
                                                 if s['gesture'] == g])) for g in sorted({s['gesture'] for e in train for s in segments.get(e, [])})})
    grid = holding_grid(episodes)
    best = grid[0]
    report = dict(dataset=str(root), ray=ray, ramp=ramp, holding_best=best, holding_top=grid[:8],
                  gap_deg=dict(min=float(np.min([g['gap_deg'] for g in gaps])), median=float(np.median([g['gap_deg'] for g in gaps])),
                               p5=float(np.percentile([g['gap_deg'] for g in gaps], 5))))
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{root.name}.json').write_text(json.dumps(report, indent=2))
    (root / 'meta/intent_scenario_vars.json').write_text(json.dumps(gaps, indent=1))
    print(f"ray error (index base -> tip): mean {ray['mean_deg']:.2f} deg, std {ray['std_deg']:.2f}, p95 {ray['p95_deg']:.2f}, "
          f"max {ray['max_deg']:.2f} over {ray['n_frames']} frames")
    g = report['gap_deg']
    print(f"angular gap pointed place -> nearest other place: min {g['min']:.2f} deg, p5 {g['p5']:.2f}, median {g['median']:.2f}")
    print(f"ramp: median t_clear - t_onset {ramp['median_s']:.2f} s (n {ramp['n']}), by gesture {ramp['by_gesture']}")
    print('holding (best F1): ' + json.dumps(best))
    if args.write_schema:
        sigma = max(round(ray['p95_deg'], 2), 0.5)      # cone sigma = p95 of the clean ray error, floor 0.5 deg
        write_schema(args.schema, sigma, ramp['median_s'],
                     dict(open_min_m=best['open_min_m'], open_max_m=best['open_max_m'], steady_m=best['steady_m'],
                          steady_frames=best['steady_frames']))
        print(f'wrote cone_sigma_deg {sigma:.2f}, ramp_s {ramp["median_s"]:.2f} and the holding thresholds to {args.schema}')


if __name__ == '__main__':
    main()
