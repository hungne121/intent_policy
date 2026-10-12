"""Does the multi-gesture DeePoint (scripts/deepoint_finetune.py on `--classes multi` labels) answer the intent
questions of T1-T5? Per test-fold episode, from its per-frame class probabilities and direction:

  class      which gesture is going on (none / point / reach / palm_up / palm_forward ('wait') / withdraw): frame
             confusion against the onset-convention labels (a segment's [t_onset, t_end]; withdraw = the change-of-mind
             withdrawal), per-class precision / recall, and per segment the first frame its class is the argmax
             (as a fraction of the gesture motion: 0 = onset, 1 = goal pose reached)
  target     for the directed gestures, the belief over the task's candidates
               L(g) = p_c * exp(-theta_g^2 / 2 sigma_c^2) + (1 - p_c),  b = L / sum L   (current output only)
             theta_g = angle between the world direction and hand -> g (hand = wrist + 0.2 (wrist - elbow), DeePoint's
             origin), p_c = the probability of the segment's gesture class, sigma_c = Rayleigh MLE of the angle error
             of class c on the validation fold. Reported: the first time the true target's belief is >= 0.95, its value
             at the first detection (the lowest level the device reports) and the choice at the end of the segment.
Candidates (the places a gesture can mean in a task; instruction = the object kind for pointing):
  T1 point: instructed kind's slots + P1 / P2; T2 point: instructed kind's slots; T2 palm_up: H1 / H2;
  T3 reach: U1 / U2 / U_cup; T4 reach: robot_zone / zone_edge / U.

  ./run.sh -m scripts.deepoint_gesture_report --dir outputs/sensor_view/2026-10-11/pilot_side --suffix _deepoint_ft_multi \\
      --val-fold 1 --test-fold 0
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from intent_policy.utils import resolve
from scripts.deepoint_sim_report import FPS, Camera, angle

CLASSES = ['none', 'point', 'reach', 'palm_up', 'palm_forward', 'withdraw']
DIRECTED = ('point', 'reach', 'palm_up')
R_WRIST, R_ELBOW = 10, 8


def candidates(meta: dict, gesture: str) -> list[str]:
    code, spec, desc = meta['task_code'], meta['spec'], meta['object_desc']
    kind = desc.get(spec.get('target'))
    slots = [slot for obj, slot in (spec.get('layout') or {}).items() if desc.get(obj) == kind]
    return {('T1', 'point'): slots + ['P1', 'P2'], ('T2', 'point'): slots, ('T2', 'palm_up'): ['H1', 'H2'],
            ('T3', 'reach'): ['U1', 'U2', 'U_cup'], ('T4', 'reach'): ['robot_zone', 'zone_edge', 'U']}.get((code, gesture), [])


def load(d: Path, tag: str, view: str, suffix: str, motions: dict) -> dict:
    meta = json.loads((d / f'{tag}.json').read_text())
    side, dp = np.load(d / f'{tag}.npz'), np.load(d / f'{tag}_{view}{suffix}.npz')
    T = len(side['keypoints'])
    j_of = np.clip(np.searchsorted(side['idx20_of_15'], np.arange(T), 'right') - 1, 0, len(dp['p_point']) - 1)
    cam = Camera(side[f'{view}_pos'], side[f'{view}_xmat'], side[f'{view}_fy'], side['width'], side['height'])
    coco = side['coco3d']
    gt = np.zeros(T, int)
    segs = [dict(gesture=s['gesture'], target=s['target'], t0=s['t_onset'], t1=min(s['t_end'], T - 1),
                 tc=s['t_clear']) for s in meta['segments']]
    for m in motions.get(meta['episode_index'], []):
        if m['label'] == 'withdraw_change':
            t1 = T - 1 if m['end'] is None else min(m['end'], T - 1)
            segs.append(dict(gesture='withdraw', target='none', t0=m['start'], t1=t1, tc=t1))
    for s in segs:
        gt[s['t0']:s['t1'] + 1] = CLASSES.index(s['gesture'])
    probs = dp['probs'][j_of]
    dirs = [cam.world_direction(dp['direction'][j], dp['keypoints'][j]) for j in j_of]
    hand = coco[:, R_WRIST] + 0.2 * (coco[:, R_WRIST] - coco[:, R_ELBOW])
    return dict(meta=meta, T=T, gt=gt, probs=probs, dirs=dirs, hand=hand, segs=segs)


def theta(ep: dict, t: int, place: str) -> float | None:
    d = ep['dirs'][t]
    return None if d is None else angle(d, np.asarray(ep['meta']['targets'][place]) - ep['hand'][t])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--suffix', default='_deepoint_ft_multi')
    ap.add_argument('--view', default='side')
    ap.add_argument('--val-fold', type=int, default=1)
    ap.add_argument('--test-fold', type=int, default=0)
    args = ap.parse_args()
    d = resolve(args.dir)
    root = Path(json.loads((d / 'episodes.json').read_text())['dataset'])
    motions = {m['episode_index']: m['human_motions'] for m in map(json.loads, (root / 'meta/hri_episodes.jsonl').read_text().splitlines())}
    eps = defaultdict(list)
    for j in sorted(d.glob('episode_*.json')):
        meta = json.loads(j.read_text())
        if meta['fold'] in (args.val_fold, args.test_fold) and (d / f"{meta['tag']}_{args.view}{args.suffix}.npz").exists():
            eps[meta['fold']].append(load(d, meta['tag'], args.view, args.suffix, motions))

    # sigma per directed class on the validation fold (frames of the class, predicted as it)
    sigma = {}
    for c in DIRECTED:
        th = [theta(ep, t, s['target']) for ep in eps[args.val_fold] for s in ep['segs'] if s['gesture'] == c and s['target'] in ep['meta']['targets']
              for t in range(s['t0'], s['t1'] + 1) if ep['probs'][t].argmax() == CLASSES.index(c)]
        th = np.array([x for x in th if x is not None])
        sigma[c] = float(np.sqrt(np.mean(th ** 2) / 2)) if len(th) else float('nan')

    test = eps[args.test_fold]
    conf = np.zeros((len(CLASSES), len(CLASSES)), int)
    for ep in test:
        ok = np.isfinite(ep['probs']).all(1)
        np.add.at(conf, (ep['gt'][ok], ep['probs'][ok].argmax(1)), 1)
    md = [f'# Multi-gesture DeePoint ({args.suffix}), test fold {args.test_fold} ({len(test)} episodes)', '',
          'sigma (validation fold {}): '.format(args.val_fold) + ', '.join(f'{c} {sigma[c]:.2f} deg' for c in DIRECTED), '',
          '## Which gesture (frames, onset labels)', '',
          '| true \\ predicted | ' + ' | '.join(CLASSES) + ' | recall |', '|---|' + '---|' * (len(CLASSES) + 1)]
    for i, c in enumerate(CLASSES):
        md.append(f'| {c} | ' + ' | '.join(str(v) for v in conf[i]) + f' | {conf[i, i] / max(conf[i].sum(), 1):.3f} |')
    md.append('| precision | ' + ' | '.join(f'{conf[i, i] / max(conf[:, i].sum(), 1):.3f}' for i in range(len(CLASSES))) + ' | |')

    rows = defaultdict(lambda: defaultdict(list))
    for ep in test:
        code = ep['meta']['task_code']
        for s in ep['segs']:
            c = CLASSES.index(s['gesture'])
            key = (code, s['gesture'])
            r = rows[key]
            r['n'].append(1)
            span = max(s['tc'] - s['t0'], 1)
            det = next((t for t in range(s['t0'], s['t1'] + 1) if np.isfinite(ep['probs'][t]).all() and ep['probs'][t].argmax() == c), None)
            r['detected'].append(det is not None)
            if det is not None:
                r['det_frac'].append((det - s['t0']) / span)
                r['det_s'].append((det - s['tc']) / FPS)
            cand = candidates(ep['meta'], s['gesture'])
            if s['gesture'] not in DIRECTED or s['target'] not in cand or len(cand) < 2:
                continue
            hit95, at_det, last = None, None, None
            for t in range(s['t0'], s['t1'] + 1):
                if not np.isfinite(ep['probs'][t]).all() or ep['dirs'][t] is None:
                    continue
                p = float(ep['probs'][t][c])
                L = {g: p * np.exp(-0.5 * (theta(ep, t, g) / sigma[s['gesture']]) ** 2) + (1 - p) for g in cand}
                z = sum(L.values())
                b = {g: v / z for g, v in L.items()}
                if t == det:
                    at_det = b[s['target']]
                if hit95 is None and b[s['target']] >= 0.95:
                    hit95 = t
                last = max(b, key=b.get)
            r['b_at_det'] += [at_det] if at_det is not None else []
            r['t95_frac'] += [(hit95 - s['t0']) / span] if hit95 is not None else []
            r['t95_s'] += [(hit95 - s['tc']) / FPS] if hit95 is not None else []
            r['reached95'].append(hit95 is not None)
            r['final_ok'].append(last == s['target'])
    med = lambda x, f='{:.2f}': '—' if not x else f.format(np.median(x))
    md += ['', '## Per task question (segments of the test fold)', '',
           '(fractions of the gesture motion: 0 = onset, 1 = goal pose reached; times relative to the goal pose)', '',
           '| task | gesture | segments | detected | detection: fraction / s | target belief at detection | belief ≥ 0.95: fraction / s (reached) | chosen at the end = true |',
           '|---|---|---|---|---|---|---|---|']
    for (code, g), r in sorted(rows.items()):
        n = len(r['n'])
        t95 = f"{med(r['t95_frac'])} / {med(r['t95_s'], '{:+.2f}')} ({sum(r['reached95'])}/{len(r['reached95'])})" if r['reached95'] else '—'
        final = f"{sum(r['final_ok'])}/{len(r['final_ok'])}" if r['final_ok'] else '—'
        md.append(f"| {code} | {g} | {n} | {sum(r['detected'])}/{n} | {med(r['det_frac'])} / {med(r['det_s'], '{:+.2f}')} | "
                  f"{med(r['b_at_det'])} | {t95} | {final} |")
    out = d / f'gesture_report{args.suffix}_fold{args.test_fold}.md'
    out.write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
