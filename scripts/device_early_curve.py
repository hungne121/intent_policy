"""Verification of the measuring device's early belief in simulation (research_collab.md K14; the protocol of
docs/discussions/de_xuat_intent_thong_nhat.md): how early, how reliably and how well calibrated is the belief about
the human's target, per task, on episodes the device never saw in training (device_folds.json `seen_seeds`).

Per directed gesture of meta/intent_segments.json (simulator ground truth: t_onset = motion start, t_clear = the
pose is reached, t_end = the end of the hold; target = what the human wants), frames t_onset .. t_end:
  target   the segment's target; T3 reaches (U1 / U2) are scored on the paired cup's slot (spec target), which is
           where the device puts its belief (intent/device.py reach_map); T5 on T3 reaches are left out
  belief   intent_pd.p_target (recorded online during the collection, what the expert acted on), restricted to the
           true target's kind (intent_schema target_kinds) and to the candidates the device considered in that
           gesture (p > 0 at some frame), renormalised; uniform over them when the device gives the kind no mass
  τ        (t - t_onset) / 20 s;  ρ = (t - t_onset) / (t_clear - t_onset), the share of the approach seen
Reported, per task and pooled, with 95 % bootstrap intervals over episodes:
  accuracy (argmax = target, ties shared), log score, Brier score, the share of frames with the belief >= θ on a
  WRONG candidate (θ = 0.7 / 0.9 / 0.95), ECE (10 bins) per τ bin; reliability P(correct | belief in a band);
  the first frame of each gesture whose belief reaches θ (the Proba Threshold trigger of Renault et al. 2025):
  share triggered before t_end, median τ / ρ, share wrong at the trigger.
Sim human motion vs real people: the angle between the present pointing ray (`intent_pd.point_dir`, simulator
keypoints) and the gesture's final ray (median over the hold), per ρ, to set beside scripts/dp_final_direction.py.

  ./run.sh -m scripts.device_early_curve --run outputs/prelim/2026-10-13_slow --out outputs/dp_prediction/2026-10-16
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 20
DIRECTED = ('point', 'reach', 'palm_up')
TAU_BINS = np.round(np.arange(0.0, 2.51, 0.25), 2)
RHO_BINS = np.round(np.arange(0.1, 1.01, 0.1), 1)
THETAS = (0.7, 0.9, 0.95)


def boot(vals: np.ndarray, ep: np.ndarray, stat=np.mean, n=1000, rng=None):
    rng = rng or np.random.default_rng(0)
    eps = np.unique(ep)
    idx = {e: np.flatnonzero(ep == e) for e in eps}
    bs = [stat(vals[np.concatenate([idx[e] for e in rng.choice(eps, len(eps))])]) for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def ece(conf, correct, bins=10):
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        sel = (conf >= lo) & ((conf < lo + 1 / bins) | (lo >= 1 - 1 / bins - 1e-9))
        if sel.any():
            e += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(e)


def angle(a, b):
    a, b = a / np.linalg.norm(a, axis=-1, keepdims=True), b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def load(root: Path, seen: set):
    schema = json.loads((root / 'meta/intent_schema.json').read_text())
    T = schema['targets']
    kind_of = {t: k for k, ts in schema['target_kinds'].items() for t in ts}
    eps = {e['episode_index']: e for e in map(json.loads, (root / 'meta/hri_episodes.jsonl').read_text().splitlines())}
    segs = json.loads((root / 'meta/intent_segments.json').read_text())
    cols = ['episode_index', 'frame_index', 'intent_pd.p_target', 'intent_pd.point_dir']
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(root / 'data/**/*.parquet'), recursive=True))])
    out, rays = [], []
    for ei, g in df.groupby('episode_index'):
        e = eps[int(ei)]
        if e['seed'] in seen:
            continue
        g = g.sort_values('frame_index')
        P = np.stack(g['intent_pd.p_target'].values)
        R = np.stack(g['intent_pd.point_dir'].values)[:, :3]
        for si, s in enumerate(segs.get(str(ei), [])):
            true_t = s['target']
            if s['gesture'] == 'reach' and true_t in ('U1', 'U2'):     # T3: the device's belief is on the paired cup's slot
                if e['spec'].get('task') != 'T3':                       # (T5 on T3: the cube -> cup pairs are not in the spec)
                    continue
                true_t = e['spec']['layout'][e['spec']['target']]
            if s['gesture'] not in DIRECTED or true_t not in kind_of:
                continue
            t0, tc, te = int(s['t_onset']), int(s['t_clear']), min(int(s['t_end']), len(P) - 1)
            if te <= t0 or tc <= t0:
                continue
            ks = [T.index(t) for t in schema['target_kinds'][kind_of[true_t]] if t in T]
            seg_P = P[t0:te + 1][:, ks]
            cand = [k for j, k in enumerate(ks) if seg_P[:, j].max() > 0]
            if T.index(true_t) not in cand or len(cand) < 2:
                continue
            cj = [ks.index(k) for k in cand]
            ti = cand.index(T.index(true_t))
            for t in range(t0, te + 1):
                p = P[t, [ks[j] for j in cj]]
                b = p / p.sum() if p.sum() > 1e-9 else np.full(len(cand), 1 / len(cand))
                out.append(dict(ep=f"{root.name}:{ei}", task=e['list_task'], seg=f"{root.name}:{ei}:{si}", kind=kind_of[true_t],
                                tau=(t - t0) / FPS, rho=(t - t0) / (tc - t0), b_true=float(b[ti]),
                                b_wrong=float(np.delete(b, ti).max()), K=len(cand),
                                correct=float(np.isclose(b.max(), b[ti]) / max(1, np.isclose(b, b.max()).sum()))))
            if s['gesture'] == 'point' and np.isfinite(R[t0:te + 1]).all() and np.linalg.norm(R[tc:te + 1], axis=1).min() > 0:
                fin = np.median(R[tc:te + 1] / np.linalg.norm(R[tc:te + 1], axis=1, keepdims=True), axis=0)
                for t in range(t0, tc + 1):
                    if np.linalg.norm(R[t]) > 0:
                        rays.append(dict(ep=f"{root.name}:{ei}", task=e['list_task'], rho=(t - t0) / (tc - t0),
                                         angle=float(angle(R[t], fin))))
    return pd.DataFrame(out), pd.DataFrame(rays)


def summarise(D: pd.DataFrame, rng) -> dict:
    res = {'tau': [], 'reliability': [], 'first_crossing': []}
    ep = D['ep'].values
    for lo in TAU_BINS:
        sel = (D['tau'] >= lo) & (D['tau'] < lo + 0.25)
        if sel.sum() < 20:
            continue
        x = D[sel]
        conf = np.maximum(x['b_true'].values, x['b_wrong'].values)
        row = dict(tau=float(lo), n_frames=int(sel.sum()), n_eps=int(x['ep'].nunique()),
                   acc=float(x['correct'].mean()), acc_ci=boot(x['correct'].values, x['ep'].values, rng=rng),
                   log_score=float(np.log(np.clip(x['b_true'], 1e-12, 1)).mean()),
                   brier=float(((1 - x['b_true']) ** 2 + x['b_wrong'] ** 2).mean()),
                   ece=ece(conf, (x['b_true'].values >= x['b_wrong'].values).astype(float)))
        for th in THETAS:
            fh = (x['b_wrong'].values >= th).astype(float)
            row[f'wrong_ge_{th}'] = float(fh.mean())
            row[f'wrong_ge_{th}_ci'] = boot(fh, x['ep'].values, rng=rng)
        res['tau'].append(row)
    for lo, hi in ((0.5, 0.7), (0.7, 0.9), (0.9, 0.95), (0.95, 1.01)):
        conf = np.maximum(D['b_true'].values, D['b_wrong'].values)
        sel = (conf >= lo) & (conf < hi)
        if sel.sum():
            ok = (D['b_true'].values[sel] >= D['b_wrong'].values[sel]).astype(float)
            res['reliability'].append(dict(band=[lo, min(hi, 1.0)], n=int(sel.sum()), mean_conf=float(conf[sel].mean()),
                                           accuracy=float(ok.mean()), ci=boot(ok, ep[sel], rng=rng)))
    for th in (0.7, 0.8, 0.9, 0.95):
        tt, rr, wr, ee, n = [], [], [], [], 0
        for sg, x in D.groupby('seg'):
            n += 1
            hit = np.flatnonzero((x['b_true'].values >= th) | (x['b_wrong'].values >= th))
            if len(hit):
                h = hit[0]
                tt.append(x['tau'].values[h]), rr.append(x['rho'].values[h])
                wr.append(float(x['b_wrong'].values[h] >= th)), ee.append(x['ep'].values[0])
        wr, ee = np.array(wr), np.array(ee)
        res['first_crossing'].append(dict(theta=th, n_gestures=n, triggered=len(tt) / max(n, 1),
                                          median_tau=float(np.median(tt)) if tt else None,
                                          median_rho=float(np.median(rr)) if rr else None,
                                          wrong=float(wr.mean()) if len(wr) else None,
                                          wrong_ci=boot(wr, ee, rng=rng) if len(wr) else None))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='prelim run folder (datasets/, device/device_folds.json)')
    ap.add_argument('--datasets', nargs='+', default=['intent_act_device_early_prelim', 'intent_act_device_late_prelim'])
    ap.add_argument('--exclude-list', default='lists/pilot.jsonl',
                    help="also leave out this list's seeds (relative to --run; the device's training / validation pool, "
                         "where its σ table was fitted); '' keeps them")
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    run = Path(args.run)
    seen = set(json.loads((run / 'device/device_folds.json').read_text()).get('seen_seeds', []))
    if args.exclude_list:
        seen |= {json.loads(l)['seed'] for l in (run / args.exclude_list).read_text().splitlines() if l.strip()}
    parts = [load(run / 'datasets' / d, seen) for d in args.datasets]
    D = pd.concat([p[0] for p in parts], ignore_index=True)
    Rays = pd.concat([p[1] for p in parts], ignore_index=True)
    report = {'n_gestures': int(D['seg'].nunique()), 'n_episodes': int(D['ep'].nunique()),
              'by_task': {}, 'pooled': summarise(D, rng)}
    for t in sorted(D['task'].unique()):
        report['by_task'][t] = summarise(D[D['task'] == t], rng)
    sim_ray = []
    for r in RHO_BINS:
        sel = np.isclose(Rays['rho'].round(1), r)
        if sel.any():
            a = Rays['angle'].values[sel]
            sim_ray.append(dict(rho=float(r), median_deg=float(np.median(a)), share_below_8_5=float((a < 8.5).mean())))
    report['sim_present_vs_final_ray'] = sim_ray
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f'device_early_curve{args.tag}.json').write_text(json.dumps(report, indent=1))
    L = [f"# Device early belief in simulation (episodes outside the device's training / validation pool: {len(seen)} seeds left out): {report['n_gestures']} directed gestures, "
         f"{report['n_episodes']} episodes\n"]

    def block(name, res):
        L.append(f'## {name}\n')
        L.append('| τ [s] | episodes | accuracy | log score | Brier | ECE | wrong ≥ 0.7 | wrong ≥ 0.9 | wrong ≥ 0.95 |')
        L.append('|---|---|---|---|---|---|---|---|---|')
        for r in res['tau']:
            L.append(f"| {r['tau']:.2f} | {r['n_eps']} | {r['acc']:.2f} [{r['acc_ci'][0]:.2f}, {r['acc_ci'][1]:.2f}] | "
                     f"{r['log_score']:.2f} | {r['brier']:.3f} | {r['ece']:.3f} | {r['wrong_ge_0.7']:.3f} | "
                     f"{r['wrong_ge_0.9']:.3f} [{r['wrong_ge_0.9_ci'][0]:.3f}, {r['wrong_ge_0.9_ci'][1]:.3f}] | "
                     f"{r['wrong_ge_0.95']:.3f} |")
        L.append('\n| confidence band | frames | mean confidence | accuracy |\n|---|---|---|---|')
        for r in res['reliability']:
            L.append(f"| {r['band'][0]:.2f}-{r['band'][1]:.2f} | {r['n']} | {r['mean_conf']:.3f} | "
                     f"{r['accuracy']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] |")
        L.append('\n| θ | gestures | triggered before the hold ends | median τ [s] | median ρ | wrong at trigger |\n|---|---|---|---|---|---|')
        for r in res['first_crossing']:
            w = f"{r['wrong']:.3f} [{r['wrong_ci'][0]:.3f}, {r['wrong_ci'][1]:.3f}]" if r['wrong'] is not None else '—'
            L.append(f"| {r['theta']} | {r['n_gestures']} | {r['triggered']:.2f} | {r['median_tau']:.2f} | "
                     f"{r['median_rho']:.2f} | {w} |")
        L.append('')
    block('Pooled (T1, T2, T3, T5)', report['pooled'])
    for t, res in report['by_task'].items():
        block(f'Task {t}', res)
    L.append('## Sim human: present pointing ray vs the final ray (simulator keypoints), per share of the approach\n')
    L.append('| ρ | ' + ' | '.join(f"{r['rho']:.1f}" for r in sim_ray) + ' |')
    L.append('|---|' + '---|' * len(sim_ray))
    L.append('| median angle [°] | ' + ' | '.join(f"{r['median_deg']:.1f}" for r in sim_ray) + ' |')
    L.append('| share < 8.5° | ' + ' | '.join(f"{r['share_below_8_5']:.2f}" for r in sim_ray) + ' |')
    (out / f'device_early_curve{args.tag}.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
