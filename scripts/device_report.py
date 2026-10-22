"""Report of the measuring device (session 2026-10-18): accuracy, timing and recovery of its belief about the human's
target, in simulation, on episodes the device never saw (device_folds.json `seen_seeds` and the pilot cells left out).

Data: the belief recorded online during dataset collection (intent_pd.p_target / p_gesture: exactly what the expert
acted on) and the simulator's gesture segments (meta/intent_segments.json: t_onset = motion start, t_clear = pose
reached, t_end = end of the hold; target = what the human wants). Scored like scripts/device_early_curve.py: the
belief restricted to the true target's kind and to the candidates the device considered, renormalised. T3 reaches are
scored on the cube spot U1 / U2 when the device puts its belief there (contract of 2026-10-18), else on the paired
cup's slot (older datasets).

  1. gesture length        approach (t_clear - t_onset) and whole gesture (t_end - t_onset) [s], per task
  2. accuracy              argmax = target, by the share of the approach seen (ρ) and by time since the onset (τ)
  3. thresholds            first frame whose leading belief reaches 0.85 (approach) / 0.9 (commit) / 0.95 (late):
                           reached before t_end, τ mean ± sd and median, ρ, wrong at that frame
  4. recovery in a gesture  gestures whose first 0.85 / 0.9 crossing led with the wrong candidate: recovered = the
                           true target later leads with belief >= 0.9 within the gesture; time from the wrong crossing
                           to that; wrong 0.9 = the wrong candidate itself reached 0.9 (an expert would have committed)
  5. change of mind (T5)   t_sw = the human's change (meta intention_changes): the device's 'wait' / withdraw signal
                           (probability >= 0.7 for the early rule's hold frames, device_folds.json), then the new
                           target's belief >= 0.9; latencies from t_sw and from the onset of the new gesture
Means with 95 % bootstrap intervals over episodes (1000 resamples).

  ./run.sh -m scripts.device_report --datasets <root> ... --folds outputs/device/current/device_folds.json \
      --exclude-list outputs/prelim/2026-10-13_slow/lists/pilot.jsonl --out outputs/device_report/<date>
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.device_early_curve import boot

FPS = 20
DIRECTED = ('point', 'reach', 'palm_up')
LEVELS = (0.85, 0.9, 0.95)
RHOS = np.round(np.arange(0.1, 1.01, 0.1), 1)
TAUS = np.round(np.arange(0.0, 2.01, 0.25), 2)
CLASSES = ['none', 'point', 'reach', 'palm_up', 'palm_forward', 'withdraw']


def load(root: Path, seen: set):
    """Per-frame rows of the directed gestures, per-gesture metadata, and the T5 change records of one dataset."""
    schema = json.loads((root / 'meta/intent_schema.json').read_text())
    T = schema['targets']
    kind_of = {t: k for k, ts in schema['target_kinds'].items() for t in ts}
    eps = {e['episode_index']: e for e in map(json.loads, (root / 'meta/hri_episodes.jsonl').read_text().splitlines())}
    segs = json.loads((root / 'meta/intent_segments.json').read_text())
    cols = ['episode_index', 'frame_index', 'intent_pd.p_target', 'intent_pd.p_gesture']
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(root / 'data/**/*.parquet'), recursive=True))])
    frames, gestures, changes, changes_signal = [], [], [], []
    for ei, g in df.groupby('episode_index'):
        e = eps[int(ei)]
        if int(e['seed']) in seen:
            continue
        g = g.sort_values('frame_index')
        P = np.stack(g['intent_pd.p_target'].values)
        G = np.stack(g['intent_pd.p_gesture'].values)
        epk = f"{root.name}:{ei}"
        u_mass = P[:, [T.index('U1'), T.index('U2')]].sum(1)
        for si, s in enumerate(segs.get(str(ei), [])):
            true_t = s['target']
            if s['gesture'] == 'reach' and true_t in ('U1', 'U2'):
                t0, te = int(s['t_onset']), min(int(s['t_end']), len(P) - 1)
                if u_mass[t0:te + 1].max() <= 0:                      # older contract: belief on the paired cup's slot
                    if e['spec'].get('task') != 'T3' or 'change' in e['spec']:
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
            sid = f"{epk}:{si}"
            gestures.append(dict(seg=sid, ep=epk, task=e['list_task'], gesture=s['gesture'], t_onset=t0,
                                 approach_s=(tc - t0) / FPS, gesture_s=(te - t0) / FPS, K=len(cand)))
            for t in range(t0, te + 1):
                p = P[t, [ks[j] for j in cj]]
                b = p / p.sum() if p.sum() > 1e-9 else np.full(len(cand), 1 / len(cand))
                frames.append(dict(seg=sid, ep=epk, task=e['list_task'], tau=(t - t0) / FPS, rho=(t - t0) / (tc - t0),
                                   end=t == te, b_true=float(b[ti]), b_wrong=float(np.delete(b, ti).max()),
                                   correct=float(np.isclose(b.max(), b[ti]) / max(1, np.isclose(b, b.max()).sum()))))
        stages = [(h['name'], int(h['start']), int(h['end'])) for h in e.get('human_stages') or []]
        changes_signal.append(dict(ep=epk, task=e['list_task'], G=G, stages=stages))
        if e['list_task'] == 'T5' and e.get('intention_changes'):
            ch = e['intention_changes'][0]
            t_sw = int(round(float(ch['t']) * FPS)) if 't' in ch else int(round(float(ch['timestamp']) * FPS))
            new_t = None
            for s in segs.get(str(ei), []):                          # the first directed gesture after the change
                if s['gesture'] in DIRECTED and int(s['t_onset']) >= t_sw - 2:
                    new_t = s
                    break
            changes.append(dict(ep=epk, base=e['spec'].get('task'), t_sw=t_sw, new=new_t, P=P, G=G, T=T,
                                kind_of=kind_of, schema=schema, spec=e['spec']))
    return pd.DataFrame(frames), pd.DataFrame(gestures), changes, changes_signal


SIGNAL_STAGE = {'palm_forward': 'stop_signal', 'withdraw': 'withdraw_change'}   # the human's true change signals


def alarm_rows(eps, hold_frames: dict) -> list[dict]:
    """Per episode run: the device's confirmed change signals (class probability >= 0.7 for the early rule's hold
    frames), each matched to a true signal stage of the human (from 0.5 s before its start to 1 s after its end) or
    counted as a false alarm."""
    rows = []
    for e in eps:
        G = e['G'][:, :len(CLASSES)]
        r = dict(ep=e['ep'], task=e['task'], false=0, true_signals=0, detected=0)
        for cls, stage in SIGNAL_STAGE.items():
            on = G[:, CLASSES.index(cls)] >= 0.7
            n, run, starts = int(hold_frames.get(cls, 1)), 0, []
            for i, f in enumerate(on):
                run = run + 1 if f else 0
                if run == n:
                    starts.append(i - n + 1)
            win = [(s0 - 10, s1 + FPS) for name, s0, s1 in e['stages'] if name == stage]
            r['true_signals'] += len(win)
            r['detected'] += sum(any(a <= t <= b for t in starts) for a, b in win)
            r['false'] += sum(not any(a <= t <= b for a, b in win) for t in starts)
        rows.append(r)
    return rows


def first_run(flags: np.ndarray, n: int) -> int | None:
    """First index at which `flags` has been true for n consecutive frames."""
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        if run >= n:
            return i
    return None


def change_rows(changes, hold_frames: dict) -> list[dict]:
    rows = []
    for c in changes:
        P, G, T, t_sw = c['P'], c['G'], c['T'], c['t_sw']
        G = G[:, :len(CLASSES)] if G.shape[1] >= len(CLASSES) else G
        idx = {k: CLASSES.index(k) for k in ('palm_forward', 'withdraw')}
        sig = None
        for k in idx:
            i = first_run(G[t_sw:, idx[k]] >= 0.7, int(hold_frames.get(k, 1)))
            if i is not None and (sig is None or i < sig[1]):
                sig = (k, i)
        r = dict(ep=c['ep'], base=c['base'], signal=sig[0] if sig else None,
                 signal_s=sig[1] / FPS if sig else np.nan)
        s = c['new']
        if s is not None:
            true_t = s['target']
            if s['gesture'] == 'reach' and true_t in ('U1', 'U2') and P[:, [T.index('U1'), T.index('U2')]].sum(1).max() <= 0:
                true_t = None                                       # older contract (cup slots): not scored
            if true_t is not None and true_t in c['kind_of']:
                ks = [T.index(t) for t in c['schema']['target_kinds'][c['kind_of'][true_t]] if t in T]
                t0 = int(s['t_onset'])
                hit = None
                for t in range(t_sw, len(P)):
                    p = P[t, ks]
                    if p.sum() > 1e-9 and p[ks.index(T.index(true_t))] / p.sum() >= 0.9:
                        hit = t
                        break
                r.update(new_onset_after_sw_s=(t0 - t_sw) / FPS,
                         retarget_s=(hit - t_sw) / FPS if hit is not None else np.nan,
                         retarget_from_onset_s=(hit - t0) / FPS if hit is not None else np.nan,
                         retargeted=hit is not None, new_gesture=s['gesture'])
        rows.append(r)
    return rows


def mean_ci(x: np.ndarray, ep: np.ndarray, rng) -> str:
    x, ep = np.asarray(x, float), np.asarray(ep)
    ok = np.isfinite(x)
    if not ok.any():
        return '—'
    lo, hi = boot(x[ok], ep[ok], rng=rng)
    return f'{x[ok].mean():.2f} [{lo:.2f}; {hi:.2f}]'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', required=True)
    ap.add_argument('--folds', default='outputs/device/current/device_folds.json')
    ap.add_argument('--exclude-list', nargs='*', default=[])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    folds = json.loads(Path(args.folds).read_text())
    seen = set(folds.get('seen_seeds') or [])
    for f in args.exclude_list:
        seen |= {json.loads(l)['seed'] for l in Path(f).read_text().splitlines() if l.strip()}
    hold = next(iter(folds['folds'].values()))['hold_frames']['early']
    parts = [load(Path(d), seen) for d in args.datasets]
    F = pd.concat([p[0] for p in parts], ignore_index=True)
    Gs = pd.concat([p[1] for p in parts], ignore_index=True)
    # one row per unique episode seed: the late and early runs of a cell show the same human motion; keep the first
    Gs['key'] = Gs['seg'].str.split(':').str[1:].str.join(':')
    rng = np.random.default_rng(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = dict(n_gestures=int(len(Gs)), n_episodes=int(Gs['ep'].nunique()))
    L = [f"# Measuring device report ({len(Gs)} directed gestures, {Gs['ep'].nunique()} episode runs, simulation, "
         f"episodes the device never saw)\n"]

    # 1. gesture length
    L += ['## 1. Gesture length [s]\n', '| task | gesture | n | approach mean ± sd (median) | whole gesture mean ± sd (median) |',
          '|---|---|---|---|---|']
    res['length'] = []
    for (task, gest), g in Gs.groupby(['task', 'gesture']):
        a, w = g['approach_s'], g['gesture_s']
        L.append(f'| {task} | {gest} | {len(g)} | {a.mean():.2f} ± {a.std():.2f} ({a.median():.2f}) | '
                 f'{w.mean():.2f} ± {w.std():.2f} ({w.median():.2f}) |')
        res['length'].append(dict(task=task, gesture=gest, n=int(len(g)), approach_mean=a.mean(), approach_sd=a.std(),
                                  gesture_mean=w.mean(), gesture_sd=w.std()))
    a = Gs['approach_s']
    L.append(f'| all | all | {len(Gs)} | {a.mean():.2f} ± {a.std():.2f} ({a.median():.2f}) | '
             f"{Gs['gesture_s'].mean():.2f} ± {Gs['gesture_s'].std():.2f} ({Gs['gesture_s'].median():.2f}) |\n")

    # 2. accuracy by ρ and τ
    L += ['## 2. Accuracy (argmax = target)\n', '| share of the approach seen ρ | ' + ' | '.join(f'{r:.0%}' for r in RHOS) + ' |',
          '|---|' + '---|' * len(RHOS)]
    row, res['acc_rho'] = [], []
    for r in RHOS:
        sel = F.groupby('seg').apply(lambda x: x.iloc[np.argmin(np.abs(x['rho'].values - r))], include_groups=False)
        acc, ci = sel['correct'].mean(), boot(sel['correct'].values, sel['ep'].values, rng=rng)
        row.append(f'{acc:.2f} [{ci[0]:.2f}; {ci[1]:.2f}]')
        res['acc_rho'].append(dict(rho=float(r), acc=float(acc), ci=ci))
    L.append('| accuracy | ' + ' | '.join(row) + ' |')
    endf = F[F['end']]
    L.append(f"\nAt the end of the gesture: {endf['correct'].mean():.3f} "
             f"[{boot(endf['correct'].values, endf['ep'].values, rng=rng)[0]:.3f}; "
             f"{boot(endf['correct'].values, endf['ep'].values, rng=rng)[1]:.3f}] ({len(endf)} gestures).\n")
    L += ['| time since the onset τ [s] | ' + ' | '.join(f'{t:.2f}' for t in TAUS) + ' |', '|---|' + '---|' * len(TAUS)]
    row, res['acc_tau'] = [], []
    for t in TAUS:
        x = F[np.isclose(F['tau'], t)]
        if len(x) < 20:
            row.append('—')
            continue
        ci = boot(x['correct'].values, x['ep'].values, rng=rng)
        row.append(f"{x['correct'].mean():.2f} [{ci[0]:.2f}; {ci[1]:.2f}]")
        res['acc_tau'].append(dict(tau=float(t), acc=float(x['correct'].mean()), ci=ci, n=int(len(x))))
    L.append('| accuracy | ' + ' | '.join(row) + ' |\n')

    # 3. thresholds + 4. recovery within the gesture
    L += ['## 3. Belief levels: when, and how often wrong\n',
          '| level | reached in the gesture | τ mean ± sd [s] | τ median [s] | ρ median | wrong at the crossing |',
          '|---|---|---|---|---|---|']
    rec_rows, res['levels'] = [], []
    for th in LEVELS:
        cr = []
        for sid, x in F.groupby('seg', sort=False):
            bt, bw = x['b_true'].values, x['b_wrong'].values
            lead = np.maximum(bt, bw)
            hit = np.flatnonzero(lead >= th)
            r = dict(seg=sid, ep=x['ep'].iloc[0], task=x['task'].iloc[0], reached=bool(len(hit)))
            if len(hit):
                i = hit[0]
                wrong = bool(bw[i] > bt[i])
                r.update(tau=float(x['tau'].values[i]), rho=float(x['rho'].values[i]), wrong=wrong)
                if wrong:                               # recovery: the true target later leads with >= 0.9
                    j = np.flatnonzero((bt[i:] >= 0.9) & (bt[i:] > bw[i:]))
                    r.update(recovered=bool(len(j)), recover_s=float(j[0] / FPS) if len(j) else np.nan,
                             wrong_reached_09=bool((bw[i:] >= 0.9).any()))
            cr.append(r)
        C = pd.DataFrame(cr)
        c = C[C['reached']]
        w = c['wrong'].astype(float).values
        ci = boot(w, c['ep'].values, rng=rng)
        L.append(f"| {th:.2f} | {C['reached'].mean():.2f} | {c['tau'].mean():.2f} ± {c['tau'].std():.2f} | "
                 f"{c['tau'].median():.2f} | {c['rho'].median():.0%} | {w.mean():.1%} [{ci[0]:.1%}; {ci[1]:.1%}] |")
        res['levels'].append(dict(level=th, reached=float(C['reached'].mean()), tau_mean=float(c['tau'].mean()),
                                  tau_sd=float(c['tau'].std()), tau_median=float(c['tau'].median()),
                                  rho_median=float(c['rho'].median()), wrong=float(w.mean()), wrong_ci=ci,
                                  by_task={t: dict(n=int(len(g)), tau_mean=float(g['tau'].mean()), wrong=float(g['wrong'].mean()))
                                           for t, g in c.groupby('task')}))
        cw = c[c['wrong']]
        if th in (0.85, 0.9) and len(cw):
            rec_rows.append((th, len(c), cw))
    L += ['\nBy task (τ mean [s] / wrong at the crossing):\n', '| level | ' + ' | '.join(sorted(Gs['task'].unique())) + ' |',
          '|---|' + '---|' * Gs['task'].nunique()]
    for r in res['levels']:
        L.append(f"| {r['level']:.2f} | " + ' | '.join(
            f"{r['by_task'][t]['tau_mean']:.2f} s / {r['by_task'][t]['wrong']:.1%} (n {r['by_task'][t]['n']})" if t in r['by_task'] else '—'
            for t in sorted(Gs['task'].unique())) + ' |')
    L += ['\n## 4. Recovery after a wrong lead, within the gesture\n',
          '| level | gestures | wrong at the crossing | recovered in the gesture | time to recover, median (max) [s] | '
          'the wrong candidate reached 0.9 |', '|---|---|---|---|---|---|']
    res['recovery'] = []
    for th, n, cw in rec_rows:
        rec = cw['recovered'].astype(float)
        L.append(f"| {th:.2f} | {n} | {len(cw)} | {int(rec.sum())}/{len(cw)} | {cw['recover_s'].median():.2f} "
                 f"({cw['recover_s'].max():.2f}) | {int(cw['wrong_reached_09'].sum())}/{len(cw)} |")
        res['recovery'].append(dict(level=th, gestures=n, wrong=int(len(cw)), recovered=int(rec.sum()),
                                    recover_median_s=float(cw['recover_s'].median()),
                                    wrong_reached_09=int(cw['wrong_reached_09'].sum())))

    # 5. change of mind
    CH = pd.DataFrame([r for p in parts for r in change_rows(p[2], hold)])
    if len(CH):
        L += ['\n## 5. Change of mind (T5): from the human\'s change to the device\n',
              f'Hold frames of the early rule (device_folds.json): {hold}.\n',
              '| base | episodes | change signal seen | signal latency mean [CI] [s] | new target >= 0.9 | '
              'from the change, mean [CI] [s] | from the new gesture onset, mean [CI] [s] |', '|---|---|---|---|---|---|---|']
        res['change'] = []
        for base, g in CH.groupby('base'):
            sig = g['signal'].notna()
            rt = g.get('retargeted', pd.Series(False, index=g.index)).fillna(False).astype(bool)
            L.append(f"| {base} | {len(g)} | {int(sig.sum())}/{len(g)} ({', '.join(f'{k} {v}' for k, v in g['signal'].value_counts().items())}) | "
                     f"{mean_ci(g['signal_s'].values, g['ep'].values, rng)} | {int(rt.sum())}/{len(g)} | "
                     f"{mean_ci(g.get('retarget_s', pd.Series(np.nan, index=g.index)).values, g['ep'].values, rng)} | "
                     f"{mean_ci(g.get('retarget_from_onset_s', pd.Series(np.nan, index=g.index)).values, g['ep'].values, rng)} |")
            res['change'].append(dict(base=base, n=int(len(g)), signal_seen=int(sig.sum()),
                                      signal_mean_s=float(g['signal_s'].mean()), retargeted=int(rt.sum()),
                                      retarget_mean_s=float(g['retarget_s'].mean()) if 'retarget_s' in g else None))
    A = pd.DataFrame([r for p in parts for r in alarm_rows(p[3], hold)])
    if len(A):
        L += ['\n## 6. Change signals: detected and false alarms\n',
              "Confirmed 'wait' (palm_forward) / withdraw detections of the early rule; true = within a signal stage of the "
              'human (0.5 s before its start to 1 s after its end).\n',
              '| task | episode runs | true signals detected | runs with >= 1 false alarm | false alarms per run |',
              '|---|---|---|---|---|']
        res['alarms'] = []
        for task, g in A.groupby('task'):
            fa = (g['false'] > 0).astype(float)
            ci = boot(fa.values, g['ep'].values, rng=rng)
            L.append(f"| {task} | {len(g)} | {int(g['detected'].sum())}/{int(g['true_signals'].sum())} | "
                     f"{fa.mean():.0%} [{ci[0]:.0%}; {ci[1]:.0%}] | {g['false'].mean():.2f} |")
            res['alarms'].append(dict(task=task, runs=int(len(g)), detected=int(g['detected'].sum()),
                                      true_signals=int(g['true_signals'].sum()), runs_with_false=float(fa.mean()),
                                      false_per_run=float(g['false'].mean())))
    (out / 'device_report.md').write_text('\n'.join(L) + '\n')
    (out / 'device_report.json').write_text(json.dumps(res, indent=1, default=float))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
