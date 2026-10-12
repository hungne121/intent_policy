"""What the released DeePoint model measures on the simulated human, over time (measurement step 1, docs/requirements/
INTENT_ACT_v3.2.md): annotated videos + a summary, per episode and sensor view.

Inputs: scripts/render_sensor_view.py (<tag>_<view>.mp4 at 20 fps + <tag>.npz: simulator ground truth, camera pose)
and scripts/deepoint_infer.py on the 15 fps resample (<tag>_<view>_deepoint.npz). DeePoint's output at 20 fps frame i
is its window that ends at the last 15 fps frame <= i (what a running system would have at that time).

DeePoint's direction is in its own camera-centred frame (src/dataset.py hand2marker_direction): Ez along the camera ->
hand ray, Ey up in the image, Ex right, then y/z swapped (x right, y depth, z up); the hand = wrist + 0.2 (wrist -
elbow). The frame is rebuilt from DeePoint's own OpenPifPaf keypoints (right wrist / elbow, COCO 10 / 8) and the
camera intrinsics (pinhole, fovy) to get the direction in the world frame. It is compared with
  - the ground-truth pointing ray (index base -> tip, the ray of the v3.2 perfect source), and
  - the direction from the index tip to the true target, and the candidate it selects (smallest angle; candidates =
    the instructed kind's objects (twins included) + the place zones in T1).
Ground-truth gesture timing: the simulator segments (intent/labels.segments_from_sim: t_onset = the pointing motion
starts, t_clear = the pointing pose is reached, t_end = the hold ends).

  ./run.sh -m scripts.deepoint_sim_report --dir outputs/sensor_view/2026-10-11/deepoint_test --dataset-root outputs/datasets/intent_act_late_v5
"""
import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from intent_policy.intent.labels import load_schema, segments_from_sim
from intent_policy.intent.device import Camera, angle, unit  # noqa: F401  (camera geometry, shared with the device)
from intent_policy.intent.places import target_positions
from intent_policy.utils import resolve

FPS, DP_FPS = 20, 15
THRESH = 0.5                       # DeePoint's own decision: argmax of its two classes = p_point >= 0.5
VW, VH, PANEL = 960, 540, 200
FONT = ImageFont.load_default(size=14)
SMALL = ImageFont.load_default(size=12)
R_WRIST, R_ELBOW = 10, 8           # COCO-17 indices (DeePoint's hand_id / elbow_id for the right arm)
COCO_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def candidates(meta: dict, schema: dict, pos: dict) -> list[str]:
    """Places the pointing can select in this episode: slots of the instructed kind's objects (+ P zones in T1)."""
    spec = meta['spec']
    desc = schema['instruction']['object_desc']
    kind = desc.get(spec.get('target'))
    slots = [slot for obj, slot in (spec.get('layout') or {}).items() if desc.get(obj) == kind and slot in pos]
    if meta['task_code'] == 'T1':
        slots += [k for k in ('P1', 'P2') if k in pos]
    return slots


def frame_classes(segs: list[dict], T: int) -> tuple[list[str], np.ndarray]:
    """Ground-truth class per 20 fps frame (the segment's gesture over [t_onset, t_end], with the pointing approach
    [t_onset, t_clear) as `point_approach`; `retract` in a withdrawal window; else `rest`) and the DP-convention
    positives (a pointing segment's hold [t_clear, t_end]) and the onset-convention positives ([t_onset, t_end], the
    labels chosen 2026-10-11; scripts/deepoint_export_dp.py --label-start)."""
    cls, pos, pos_on = ['rest'] * T, np.zeros(T, bool), np.zeros(T, bool)
    for s in segs:
        if s.get('t_retract_start') is not None:
            for t in range(s['t_retract_start'], min(s['t_retract'], T - 1) + 1):
                cls[t] = 'retract'
    for s in segs:
        for t in range(s['t_onset'], min(s['t_end'], T - 1) + 1):
            cls[t] = 'point_approach' if s['gesture'] == 'point' and t < s['t_clear'] else s['gesture']
            pos[t] = s['gesture'] == 'point' and t >= s['t_clear']
            pos_on[t] = s['gesture'] == 'point'
    return cls, pos, pos_on


def analyse(tag: str, view: str, d: Path, meta: dict, schema: dict, pos: dict, suffix: str = '_deepoint') -> dict:
    side = np.load(d / f'{tag}.npz')
    dp = np.load(d / f'{tag}_{view}{suffix}.npz')
    cam = Camera(side[f'{view}_pos'], side[f'{view}_xmat'], side[f'{view}_fy'], side['width'], side['height'])
    names = [str(n) for n in side['keypoint_names']]
    kp = side['keypoints']
    T = len(kp)
    segs = segments_from_sim(meta['human_motions'], meta['human_stages'], meta['spec'], schema, T)
    cand = candidates(meta, schema, pos)
    if 'idx20_of_15' in side:                   # exact: the last 15 fps frame rendered at or before frame t
        j_of = np.clip(np.searchsorted(side['idx20_of_15'], np.arange(T), 'right') - 1, 0, len(dp['p_point']) - 1)
    else:                                       # ffmpeg -r 15 resample (first test)
        j_of = np.minimum((np.arange(T) * DP_FPS) // FPS, len(dp['p_point']) - 1)
    p = dp['p_point'][j_of]
    base, tip = kp[:, names.index('r_index_base')], kp[:, names.index('r_index_tip')]
    seg_at = [next((s for s in reversed(segs) if s['t_onset'] <= t <= s['t_end']), None) for t in range(T)]
    rows = []
    for t in range(T):
        dw = cam.world_direction(dp['direction'][j_of[t]], dp['keypoints'][j_of[t]])
        s = seg_at[t]
        r = dict(t=t, p=float(p[t]), seg=s, dw=dw, ray_err=None, tgt_err=None, pick=None)
        if dw is not None:
            r['ray_err'] = angle(dw, tip[t] - base[t])
            if s is not None and s['target'] in pos:
                r['tgt_err'] = angle(dw, pos[s['target']] - tip[t])
            if cand:
                r['pick'] = min(cand, key=lambda c: angle(dw, pos[c] - tip[t]))
        rows.append(r)
    per_seg = []
    for s in segs:
        if s['gesture'] != 'point' and s['gesture'] != 'palm_forward':
            continue
        win = range(s['t_onset'], s['t_end'] + 1)
        det = next((t for t in win if rows[t]['p'] >= THRESH), None)
        hold = [rows[t] for t in range(s['t_clear'], s['t_end'] + 1)]
        hold_det = [r for r in hold if r['p'] >= THRESH and r['dw'] is not None]
        per_seg.append(dict(
            gesture=s['gesture'], target=s['target'], t_onset_s=s['t_onset'] / FPS, t_clear_s=s['t_clear'] / FPS,
            gesture_s=(s['t_clear'] - s['t_onset']) / FPS,
            detect_after_onset_s=None if det is None else (det - s['t_onset']) / FPS,
            detect_minus_clear_s=None if det is None else (det - s['t_clear']) / FPS,
            p_at_clear=rows[s['t_clear']]['p'], p_max=float(np.nanmax([rows[t]['p'] for t in win])),
            hold_recall=float(np.mean([r['p'] >= THRESH for r in hold])) if hold else None,
            ray_err_deg=float(np.median([r['ray_err'] for r in hold_det])) if hold_det else None,
            target_err_deg=float(np.median([r['tgt_err'] for r in hold_det if r['tgt_err'] is not None])) if hold_det else None,
            pick_acc=float(np.mean([r['pick'] == s['target'] for r in hold_det])) if hold_det and s['gesture'] == 'point' else None))
    near = np.zeros(T, bool)                   # frames within 0.5 s of a pointing / wait gesture
    for s in segs:
        if s['gesture'] in ('point', 'palm_forward'):
            near[max(0, s['t_onset'] - 10):s['t_end'] + 11] = True
    other = [rows[t]['p'] >= THRESH for t in range(T) if not near[t]]
    person = float(np.mean((dp['keypoints'][:, :, 2] > 0).any(1)))
    cls, pos_dp, pos_on = frame_classes(segs, T)
    return dict(tag=tag, view=view, task=meta.get('list_task'), rows=rows, segs=segs, cand=cand, cam=cam, tip=tip,
                cls=cls, pos_dp=pos_dp, pos_on=pos_on,
                per_seg=per_seg, false_pos=float(np.mean(other)) if other else None, person_frames=person,
                kp2d=dp['keypoints'], j_of=j_of, stages=[s['name'] for s in meta['human_stages'] for _ in range(s['end'] - s['start'] + 1)])


def draw_panel(res: dict, t: int) -> Image.Image:
    rows, T = res['rows'], len(res['rows'])
    img = Image.new('RGB', (VW, PANEL), (18, 18, 18))
    d = ImageDraw.Draw(img)
    x0, x1, y0, y1 = 50, VW - 10, 30, PANEL - 30
    sx = lambda k: x0 + (x1 - x0) * k / max(T - 1, 1)
    sy = lambda v: y1 - (y1 - y0) * v
    for s in res['segs']:                      # ground-truth gesture intervals
        col = {'point': (40, 90, 40), 'palm_forward': (90, 60, 20)}.get(s['gesture'], (40, 40, 70))
        d.rectangle([sx(s['t_onset']), y0, sx(s['t_end']), y1], fill=col)
        d.line([sx(s['t_clear']), y0, sx(s['t_clear']), y1], fill=(200, 200, 200), width=1)
        d.text((sx(s['t_onset']) + 2, y0 + 2), f"{s['gesture']} {s['target']}", fill=(220, 220, 220), font=SMALL)
    d.line([x0, sy(THRESH), x1, sy(THRESH)], fill=(110, 110, 110))
    pts = [(sx(k), sy(r['p'])) for k, r in enumerate(rows[:t + 1]) if np.isfinite(r['p'])]
    if len(pts) > 1:
        d.line(pts, fill=(80, 220, 255), width=2)
    d.line([sx(t), y0, sx(t), y1], fill=(255, 255, 255))
    for v in (0.0, 0.5, 1.0):
        d.text((8, sy(v) - 7), f'{v:.1f}', fill=(180, 180, 180), font=SMALL)
    d.text((x0, 6), 'DeePoint p(pointing) (cyan); shaded = ground-truth gesture (onset..hold end, white line = pose '
                    'reached); grey line = 0.5', fill=(200, 200, 200), font=SMALL)
    d.text((x0, PANEL - 22), f'{T / FPS:.1f} s', fill=(150, 150, 150), font=SMALL)
    return img


def export_video(res: dict, d: Path, out: Path) -> None:
    cam, rows = res['cam'], res['rows']
    sxy = np.array([VW / cam.W, VH / cam.H])
    container = av.open(str(d / f"{res['tag']}_{res['view']}.mp4"))
    w = av.open(str(out), 'w')
    s = w.add_stream('libx264', rate=FPS)
    s.width, s.height, s.pix_fmt, s.options = VW, VH + PANEL, 'yuv420p', {'crf': '22', 'preset': 'veryfast'}
    for t, frame in enumerate(container.decode(video=0)):
        if t >= len(rows):
            break
        r = rows[t]
        img = frame.to_image().resize((VW, VH))
        dr = ImageDraw.Draw(img)
        kp = res['kp2d'][res['j_of'][t]]
        for a, b in COCO_EDGES:
            if kp[a, 2] > 0 and kp[b, 2] > 0:
                dr.line([*(kp[a, :2] * sxy), *(kp[b, :2] * sxy)], fill=(255, 255, 0), width=2)
        for c in res['cand']:
            q = cam.project(pos_cache[c])
            if q is not None:
                truth = r['seg'] is not None and r['seg']['target'] == c
                rad = 9 if truth else 6
                dr.ellipse([*(q * sxy - rad), *(q * sxy + rad)], outline=(0, 255, 0) if truth else (255, 255, 255), width=2)
                dr.text(tuple(q * sxy + [8, -8]), c, fill=(255, 255, 255), font=SMALL)
        if r['dw'] is not None:
            a3, b3 = res['tip'][t], res['tip'][t] + 0.45 * r['dw']
            qa, qb = cam.project(a3), cam.project(b3)
            if qa is not None and qb is not None:
                col = (int(255 * (1 - r['p'])), int(255 * r['p']), 60)
                dr.line([*(qa * sxy), *(qb * sxy)], fill=col, width=4)
        seg = r['seg']
        lines = [f"{res['tag']}  view {res['view']}  t={t / FPS:5.2f}s  human: {res['stages'][min(t, len(res['stages']) - 1)]}",
                 f"DeePoint p(pointing) = {r['p']:.2f}" + ('  -> POINTING' if r['p'] >= THRESH else ''),
                 'ground truth: ' + (f"{seg['gesture']} -> {seg['target']}" if seg else 'no gesture')]
        if r['dw'] is not None:
            lines.append(f"direction error vs true ray {r['ray_err']:.0f} deg"
                         + (f", vs true target {r['tgt_err']:.0f} deg" if r['tgt_err'] is not None else '')
                         + (f";  nearest candidate: {r['pick']}" if r['pick'] else ''))
        dr.rectangle([0, 0, VW, 18 * len(lines) + 6], fill=(0, 0, 0))
        for k, line in enumerate(lines):
            dr.text((6, 3 + 18 * k), line, fill=(255, 255, 255), font=FONT)
        canvas = Image.new('RGB', (VW, VH + PANEL))
        canvas.paste(img, (0, 0))
        canvas.paste(draw_panel(res, t), (0, VH))
        for pkt in s.encode(av.VideoFrame.from_image(canvas)):
            w.mux(pkt)
    for pkt in s.encode():
        w.mux(pkt)
    w.close()
    container.close()


def aggregate(results: list[dict]) -> list[str]:
    """Pooled over episodes (20 fps frames with a DeePoint output): frame-level precision / recall against the
    DP-convention labels, 'pointing' rate per ground-truth class, and per pointing segment the detection time, hold
    recall, direction error and the candidate chosen."""
    pred, pos, pos_on, cls = [], [], [], []
    for r in results:
        for k, row in enumerate(r['rows']):
            if np.isfinite(row['p']):
                pred.append(row['p'] >= THRESH)
                pos.append(bool(r['pos_dp'][k]))
                pos_on.append(bool(r['pos_on'][k]))
                cls.append(r['cls'][k])
    pred, pos, pos_on, cls = np.array(pred), np.array(pos), np.array(pos_on), np.array(cls)
    tp, fp, fn = (pred & pos).sum(), (pred & ~pos).sum(), (~pred & pos).sum()
    tq, fq, fm = (pred & pos_on).sum(), (pred & ~pos_on).sum(), (~pred & pos_on).sum()
    segs = [s for r in results for s in r['per_seg'] if s['gesture'] == 'point']
    q = lambda key: [s[key] for s in segs if s[key] is not None]
    med = lambda x, f='{:+.2f}': '—' if not x else (f'{f} (IQR {f} .. {f})').format(np.median(x), *np.percentile(x, [25, 75]))
    lines = [f'{len(results)} episodes, {len(segs)} pointing segments, {len(pred)} frames.', '',
             '| measure | value |', '|---|---|',
             f'| frame precision / recall (DP labels: pointing hold) | {tp / max(tp + fp, 1):.3f} / {tp / max(tp + fn, 1):.3f} |',
             f'| frame precision / recall (onset labels: whole gesture) | {tq / max(tq + fq, 1):.3f} / {tq / max(tq + fm, 1):.3f} |',
             f"| detection − motion onset [s] | {med(q('detect_after_onset_s'))} |",
             f"| detection − pointing pose reached [s] | {med(q('detect_minus_clear_s'))} |",
             f"| pointings never detected | {sum(s['detect_after_onset_s'] is None for s in segs)}/{len(segs)} |",
             f"| hold recall (mean) | {np.mean(q('hold_recall')):.3f} |" if q('hold_recall') else '| hold recall | — |',
             f"| direction error vs target [°] (median of segment medians) | {med(q('target_err_deg'), '{:.1f}')} |",
             f"| nearest candidate = target (mean over segments) | {np.mean(q('pick_acc')):.3f} |" if q('pick_acc') else '| nearest candidate = target | — |',
             '', '| ground-truth class (not the pointing hold) | frames | said "pointing" |', '|---|---|---|']
    for c in ('rest', 'point_approach', 'retract', 'palm_up', 'reach', 'palm_forward'):
        m = (cls == c) & ~pos
        if m.any():
            lines.append(f'| {c} | {int(m.sum())} | {pred[m].mean():.3f} |')
    # when p(pointing) first reaches a level, relative to the pointing pose being reached (t_clear): the device's
    # timing for the late (high level) / early (lowest level) experts
    lines += ['', '| p(pointing) first ≥ | time − pose reached [s], median (IQR) | fraction of the gesture motion (onset = 0, pose = 1) | segments reaching it |',
              '|---|---|---|---|']
    for lev in (0.1, 0.5, 0.9, 0.95):
        dt, fr = [], []
        for r in results:
            for sg in r['segs']:
                if sg['gesture'] != 'point':
                    continue
                hit = next((t for t in range(max(0, sg['t_onset'] - 10), sg['t_end'] + 1)
                            if np.isfinite(r['rows'][t]['p']) and r['rows'][t]['p'] >= lev), None)
                if hit is not None:
                    dt.append((hit - sg['t_clear']) / FPS)
                    fr.append((hit - sg['t_onset']) / max(sg['t_clear'] - sg['t_onset'], 1))
        lines.append(f"| {lev} | {med(dt)} | {med(fr, '{:.2f}')} | {len(dt)}/{len(segs)} |")
    return lines


def fmt(v, f='{:.2f}'):
    return '—' if v is None else f.format(v)


def main():
    global pos_cache
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dir', required=True)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--no-video', action='store_true')
    p.add_argument('--suffix', default='_deepoint', help="DeePoint output files <tag>_<view><suffix>.npz (e.g. '_deepoint_ft')")
    p.add_argument('--fold', type=int, help='only the episodes of this seed fold (<tag>.json from render_sensor_view)')
    p.add_argument('--title', default='DeePoint (released model) on the simulated sensor views')
    args = p.parse_args()
    d, root = resolve(args.dir), resolve(args.dataset_root)
    exp = json.loads((root / 'meta/hri_experiment_config.json').read_text())
    schema = load_schema(exp['intent']['schema'])
    pos_cache = target_positions(schema)
    metas = [json.loads(line) for line in (root / 'meta/hri_episodes.jsonl').read_text().splitlines()]
    results = []
    for f in sorted(d.glob(f'*{args.suffix}.npz')):
        tag, view = f.name.removesuffix(f'{args.suffix}.npz').rsplit('_', 1)
        if args.fold is not None and json.loads((d / f'{tag}.json').read_text())['fold'] != args.fold:
            continue
        ep = int(np.load(d / f'{tag}.npz')['episode_index'])
        res = analyse(tag, view, d, metas[ep], schema, pos_cache, args.suffix)
        results.append(res)
        if not args.no_video:
            export_video(res, d, d / f'{tag}_{view}{args.suffix}_annotated.mp4')
        print(f'{tag} {view}: done', flush=True)
    md = [f'# {args.title}', '', *aggregate(results), '',
          f'Threshold {THRESH} = DeePoint\'s own decision (argmax). Times in s; `det − onset`: first frame with '
          'p ≥ 0.5 after the pointing motion started; `det − clear`: relative to the moment the pointing pose is reached '
          '(negative = before). Errors: median over the hold frames DeePoint detects.', '',
          '| episode | view | task | gesture → target | gesture s | det − onset | det − clear | p at clear | p max | '
          'hold recall | err vs true ray ° | err vs target ° | nearest = target | false pos. elsewhere | person seen |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        segs = r['per_seg'] or [None]
        for s in segs:
            if s is None:
                md.append(f"| {r['tag']} | {r['view']} | {r['task']} | (no pointing) | | | | | | | | | | "
                          f"{fmt(r['false_pos'])} | {fmt(r['person_frames'])} |")
                continue
            md.append(f"| {r['tag']} | {r['view']} | {r['task']} | {s['gesture']} → {s['target']} | {fmt(s['gesture_s'])} | "
                      f"{fmt(s['detect_after_onset_s'])} | {fmt(s['detect_minus_clear_s'])} | {fmt(s['p_at_clear'])} | "
                      f"{fmt(s['p_max'])} | {fmt(s['hold_recall'])} | {fmt(s['ray_err_deg'], '{:.0f}')} | "
                      f"{fmt(s['target_err_deg'], '{:.0f}')} | {fmt(s['pick_acc'])} | {fmt(r['false_pos'])} | "
                      f"{fmt(r['person_frames'])} |")
    name = f"summary{args.suffix}{'' if args.fold is None else f'_fold{args.fold}'}.md"
    (d / name).write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
