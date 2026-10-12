"""Calibrate the device's decision hold on the validation fold (docs/requirements/INTENT_ACT_v3.2.md §4.2e): how many
consecutive device frames (15 fps) a gesture decision must last so that false triggers stay rare.

From the multi-gesture DeePoint outputs of the validation-fold episodes (scripts/deepoint_infer.py, suffix
`_deepoint_ft_multi`) and their ground-truth gesture segments (onset labels), per rule:
  late   the gesture's probability >= late_star (0.95)
  early  the gesture's probability >= early_star (0.7; 2026-10-12, before: the device's argmax)
held for k consecutive device frames. For the 'wait' (palm_forward) and withdraw gestures it counts, per k:
  false   episodes with a trigger while the true gesture is something else (a false 'wait' pauses the robot)
  missed  true segments never triggered
  delay   trigger time after the gesture's onset (fraction of its motion / s, median)
The hold k chosen = the smallest k whose false-trigger episode rate <= alpha (5 %, the same alpha as the late level
1 - 0.95). Also reported: the interval from a 'wait' onset to the onset of the next pointing in the change-of-mind
episodes of the training folds (neither the validation nor the test fold; 95th percentile = how long a paused robot
waits for the new target), and the likelihood width vs the device frames since the detection.

One fold model (workstation/job1_device.sh): model k's outputs `<tag>_side_m<k>.npz` on its validation fold k + 1;
`--json` writes its calibration for intent/device.py (device_folds.json, scripts/make_device_folds.py).

  ./run.sh -m scripts.calibrate_device_rules --dir outputs/device/<date>/render --suffix _m0 --test-fold 0 --val-fold 1 \
      --json outputs/device/<date>/fold0/calibration.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from intent_policy.utils import resolve

CLASSES = ['none', 'point', 'reach', 'palm_up', 'palm_forward', 'withdraw']
FPS20 = 20


def segments(meta: dict, motions: list[dict], T: int) -> list[dict]:
    segs = [dict(g=s['gesture'], t0=s['t_onset'], t1=min(s['t_end'], T - 1), tc=s['t_clear']) for s in meta['segments']]
    for m in motions:
        if m['label'] == 'withdraw_change':
            t1 = T - 1 if m['end'] is None else min(m['end'], T - 1)
            segs.append(dict(g='withdraw', t0=m['start'], t1=t1, tc=t1))
    return segs


def first_hold(flags: np.ndarray, k: int, start: int = 0) -> int | None:
    """First index >= start where flags has been True for k consecutive entries (the index of the k-th)."""
    run = 0
    for i in range(start, len(flags)):
        run = run + 1 if flags[i] else 0
        if run >= k:
            return i
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--suffix', default='_deepoint_ft_multi')
    ap.add_argument('--val-fold', type=int, default=1)
    ap.add_argument('--test-fold', type=int, default=0, help='excluded from the pause-timeout data too')
    ap.add_argument('--json', help='write the calibration of this model (hold_frames, pause_timeout_s, sigma tables)')
    ap.add_argument('--late-star', type=float, default=0.95)
    ap.add_argument('--early-star', type=float, default=0.7)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--kmax', type=int, default=8)
    args = ap.parse_args()
    d = resolve(args.dir)
    root = Path(json.loads((d / 'episodes.json').read_text())['dataset'])
    motions = {m['episode_index']: m['human_motions'] for m in map(json.loads, (root / 'meta/hri_episodes.jsonl').read_text().splitlines())}
    eps, gaps = [], []
    for j in sorted(d.glob('episode_*.json')):
        meta = json.loads(j.read_text())
        side = np.load(d / f"{meta['tag']}.npz")
        T, idx20 = len(side['keypoints']), side['idx20_of_15']
        segs = segments(meta, motions.get(meta['episode_index'], []), T)
        if meta['fold'] != args.val_fold:                 # calibration data: the validation fold only
            if meta['fold'] != args.test_fold:            # training folds: wait -> next pointing interval
                for s in segs:
                    if s['g'] == 'palm_forward':
                        nxt = [x['t0'] for x in segs if x['g'] == 'point' and x['t0'] > s['t0']]
                        if nxt:
                            gaps.append((min(nxt) - s['t0']) / FPS20)
            continue
        f = d / f"{meta['tag']}_side{args.suffix}.npz"
        if not f.exists():
            continue
        probs = np.load(f)['probs']                       # (T15, C), NaN before the first full window
        gt15 = np.zeros(len(idx20), int)
        for s in segs:
            gt15[(idx20 >= s['t0']) & (idx20 <= s['t1'])] = CLASSES.index(s['g'])
        eps.append(dict(meta=meta, probs=probs, gt=gt15, idx20=idx20, segs=segs))
    md = [f'# Device decision hold, validation fold {args.val_fold} ({len(eps)} episodes)', '',
          f'alpha = {args.alpha} (false-trigger episodes), late_star = {args.late_star}, early_star = {args.early_star}', '']
    chosen = {}
    for rule in ('late', 'early'):
        for g in ('palm_forward', 'withdraw'):
            c = CLASSES.index(g)
            md += [f'## {rule}: {g}', '', '| k (frames @15 fps) | false-trigger episodes | missed segments | delay after onset: fraction / s |',
                   '|---|---|---|---|']
            for k in range(1, args.kmax + 1):
                false_eps, missed, n_true, dfrac, ds = 0, 0, 0, [], []
                for ep in eps:
                    ok = np.isfinite(ep['probs']).all(1)
                    star = args.late_star if rule == 'late' else args.early_star
                    flag = ok & (np.nan_to_num(ep['probs'][:, c]) >= star)
                    wrong = flag & (ep['gt'] != c)
                    false_eps += first_hold(wrong, k) is not None
                    for s in ep['segs']:
                        if s['g'] != g:
                            continue
                        n_true += 1
                        inside = flag & (ep['idx20'] >= s['t0']) & (ep['idx20'] <= s['t1'])
                        hit = first_hold(inside, k)
                        if hit is None:
                            missed += 1
                        else:
                            t = ep['idx20'][hit]
                            dfrac.append((t - s['t0']) / max(s['tc'] - s['t0'], 1))
                            ds.append((t - s['t0']) / FPS20)
                rate = false_eps / max(len(eps), 1)
                if (rule, g) not in chosen and rate <= args.alpha:
                    chosen[(rule, g)] = k
                md.append(f"| {k}{' ←' if chosen.get((rule, g)) == k else ''} | {false_eps}/{len(eps)} ({rate:.0%}) | {missed}/{n_true} | "
                          f"{np.median(dfrac) if dfrac else float('nan'):.2f} / {np.median(ds) if ds else float('nan'):.2f} |")
            md.append('')
    wait_gaps = np.array(gaps)
    md += ['## Pause timeout', '', f"'wait' onset -> next pointing onset, training folds: n = {len(wait_gaps)}, median "
           f"{np.median(wait_gaps):.2f} s, 95th percentile {np.percentile(wait_gaps, 95):.2f} s, max {wait_gaps.max():.2f} s"
           if len(wait_gaps) else 'no change-of-mind episodes',
           '', 'Chosen hold: ' + ', '.join(f'{r}/{gg}: k = {k}' for (r, gg), k in chosen.items())]
    sig = sigma_by_time(d, args.val_fold, args.suffix)
    md += ['', '## Likelihood width vs device frames since the detection (Rayleigh MLE, validation fold)', '',
           '| class | frames since detection | n | sigma (deg) |', '|---|---|---|---|']
    for g, rows in sig.items():
        md += [f'| {g} | {lo}-{hi - 1 if hi < 10_000 else "..."} | {n} | {s_:.2f} |' for lo, hi, n, s_ in rows]
    out = d / f'device_rules{args.suffix}_fold{args.val_fold}.md'
    out.write_text('\n'.join(md) + '\n')
    (d / f'device_sigma{args.suffix}_fold{args.val_fold}.json').write_text(json.dumps(sig, indent=1))
    print('\n'.join(md))
    if args.json:
        holds = {r: {g: chosen.get((r, g), args.kmax) for g in ('palm_forward', 'withdraw')} for r in ('late', 'early')}
        cal = dict(test_fold=args.test_fold, val_fold=args.val_fold, suffix=args.suffix, alpha=args.alpha,
                   hold_frames=holds, hold_not_reached=[f'{r}/{g}' for (r, g) in
                                                        [(r, g) for r in holds for g in holds[r]] if (r, g) not in chosen],
                   pause_timeout_s=round(float(np.percentile(wait_gaps, 95)), 3) if len(wait_gaps) else None,
                   pause_intervals=len(wait_gaps), **sigma_tables(sig))
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(cal, indent=1))
        print(f'calibration: {args.json}')


SIGMA_BINS = (0, 1, 2, 3, 4, 6, 9, 13, 10_000)


def sigma_tables(sig: dict) -> dict:
    """sigma_by_time rows -> intent/device.py `sigma_deg_by_frames` (bin lower edges; a bin without data takes the
    class's pooled width) and `sigma_deg` (pooled per class)."""
    pooled = {g: float(np.sqrt(sum(n * s_ ** 2 for _, _, n, s_ in rows) / max(sum(n for _, _, n, _ in rows), 1)))
              for g, rows in sig.items()}
    table = {'frames': list(SIGMA_BINS[:-1])}
    for g, rows in sig.items():
        by = {lo: s_ for lo, _, _, s_ in rows}
        table[g] = [round(by.get(lo, pooled[g]), 3) for lo in SIGMA_BINS[:-1]]
    return dict(sigma_deg={g: round(v, 3) for g, v in pooled.items()}, sigma_deg_by_frames=table)


def sigma_by_time(d: Path, val_fold: int, suffix: str, bins=SIGMA_BINS) -> dict:
    """Direction error of each directed class vs the device frames since the class was detected (consecutive argmax),
    on the validation fold: Rayleigh MLE sigma per bin (the belief's likelihood width; a single sigma for the whole
    gesture makes its first frames over-confident). Returns {class: [(lo, hi, n, sigma), ...]}."""
    from scripts.deepoint_gesture_report import load as load_episode, theta
    root = Path(json.loads((d / 'episodes.json').read_text())['dataset'])
    motions = {m['episode_index']: m['human_motions'] for m in map(json.loads, (root / 'meta/hri_episodes.jsonl').read_text().splitlines())}
    errs = defaultdict(lambda: defaultdict(list))
    for j in sorted(d.glob('episode_*.json')):
        meta = json.loads(j.read_text())
        if meta['fold'] != val_fold or not (d / f"{meta['tag']}_side{suffix}.npz").exists():
            continue
        ep = load_episode(d, meta['tag'], 'side', suffix, motions)
        side = np.load(d / f"{meta['tag']}.npz")
        idx20 = side['idx20_of_15']
        for s in ep['segs']:
            if s['gesture'] not in ('point', 'reach', 'palm_up') or s['target'] not in meta['targets']:
                continue
            c = CLASSES.index(s['gesture'])
            run = 0
            for t15, t in enumerate(idx20):                  # device frames inside the segment
                if not (s['t0'] <= t <= s['t1']) or not np.isfinite(ep['probs'][t]).all():
                    continue
                run = run + 1 if ep['probs'][t].argmax() == c else 0
                if run == 0:
                    continue
                th = theta(ep, t, s['target'])
                if th is not None:
                    errs[s['gesture']][run - 1].append(th)
    out = {}
    for g, by in errs.items():
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            th = np.array([x for k, v in by.items() if lo <= k < hi for x in v])
            if len(th):
                rows.append((lo, hi, len(th), float(np.sqrt(np.mean(th ** 2) / 2))))
        out[g] = rows
    return out


if __name__ == '__main__':
    main()
