"""PIF v3 pilot (docs/PIF.md §6, criteria pre-registered in commit 7ad563e). Settings: configs/pif/pif.yaml `v3`, `pilot`.

  python -m pif.pilot habit      stage B (decides): HABIT S8789, 5-fold CV by episode
  python -m pif.pilot sim        stage A (debugging only): sim tokens, no go / no-go

Stage B. Episodes with a trusted label, 4 container tracks and a bread track; target container = release-position rule
mapped to the tracks left to right and kept when the container nearest to the bread 1 s after the release agrees
(same selection as pif/eval.py). t_leave = last frame before the bread is > `pilot.depart_m` from its grasp position.
Per fold: the v3 core is fitted on the training episodes WITHOUT labels (pif/filter.py), the GRU baseline III is
trained WITH labels (3 seeds), the pointing-ray rule needs no training. Scored on the held-out fold at t_leave + offsets.
  G1  v3 accuracy: lower 95% bootstrap bound > 1/4
  G2  v3 - III accuracy: upper bound >= 0, against every III seed
  G3  ECE(III) - ECE(v3): lower bound >= 0 or the interval contains 0, against every III seed
Writes <run>/results.json, <run>/report.md, <run>/accuracy.png.
"""
from __future__ import annotations
import argparse
import json
import time

import numpy as np
import torch

from pif.config import load_config
from pif.data import load_episodes
from pif.eval import pointing_choice
from pif.filter import batch, belief, fit
from pif.utils import log_run, run_dir


# ------------------------------------------------------------------ data
def habit_cases(cfg: dict) -> list[dict]:
    eps, labs = load_episodes(dict(cfg, data=dict(cfg['data'], domain='habit', tasks=['S8789'])))
    names = cfg['v3']['candidates']['habit']['names']
    out, skipped = [], {}
    for ep, lab in zip(eps, labs):
        cont = [k for k, n in enumerate(ep['obj_names']) if n in names]
        bread = [k for k, n in enumerate(ep['obj_names']) if n == 'bread roll']
        rel, grasp, T = lab.get('release'), lab.get('grasp'), len(ep['holding'])
        why = None
        if not (lab.get('trusted') and lab.get('target') and rel is not None and grasp is not None and len(cont) == 4 and bread):
            why = 'no trusted label / not 4 containers / no bread track'
        else:
            t_after = min(rel + 10, T - 1)
            order = sorted(cont, key=lambda k: -ep['obj_pos'][t_after, k, 1])
            tgt = order[lab['target'] - 1]
            near = cont[int(np.argmin([np.linalg.norm(ep['obj_pos'][t_after, k, :2] - ep['obj_pos'][t_after, bread[0], :2]) for k in cont]))]
            moved = np.flatnonzero(np.linalg.norm(ep['obj_pos'][grasp:, bread[0]] - ep['obj_pos'][grasp, bread[0]], axis=1) > cfg['pilot']['depart_m'])
            if near != tgt:
                why = 'label sources disagree'
            elif not len(moved) or grasp + int(moved[0]) < 1:
                why = 'bread never leaves the plate'
        if why:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        out.append(dict(ep=ep, cand=cont, target=cont.index(tgt), t_leave=grasp + int(moved[0]) - 1,
                        episode=ep['meta']['episode_index']))
    print(f'HABIT S8789: {len(out)} scored episodes, skipped {skipped}')
    return out


# ------------------------------------------------------------------ baseline III (supervised GRU)
def gru_features(ep: dict, cand: list[int], keypoints: list[str]) -> np.ndarray:
    """[T, K, F] per candidate: index tip / base of both hands relative to the candidate, tip velocity, visibility."""
    J = {n: i for i, n in enumerate(keypoints)}
    hp, hv = ep['human_pos'], ep['human_vis']
    feats = []
    for s in ('r', 'l'):
        tip, base = hp[:, J[f'{s}_index_tip']], hp[:, J[f'{s}_index_base']]
        v = (hv[:, J[f'{s}_index_tip']] & hv[:, J[f'{s}_index_base']]).astype(float)
        vel = np.zeros_like(tip)
        vel[1:] = (tip[1:] - tip[:-1]) * ep['fps'] * (v[1:] * v[:-1])[:, None]
        rel_tip = (tip[:, None] - ep['obj_pos'][:, cand]) * v[:, None, None]
        rel_base = (base[:, None] - ep['obj_pos'][:, cand]) * v[:, None, None]
        feats += [rel_tip, rel_base, np.repeat(vel[:, None], len(cand), 1), np.repeat(v[:, None, None], len(cand), 1)]
    return np.concatenate(feats, -1).astype(np.float32)


class GRUBaseline(torch.nn.Module):
    """Shared per-candidate encoder + GRU over time; softmax over candidates (permutation-equivariant)."""

    def __init__(self, f: int, hidden: int):
        super().__init__()
        self.inp = torch.nn.Sequential(torch.nn.Linear(f, hidden), torch.nn.ReLU())
        self.gru = torch.nn.GRU(hidden, hidden, batch_first=True)
        self.out = torch.nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x [B, T, K, F] -> logits [B, T, K]
        B, T, K, F = x.shape
        h, _ = self.gru(self.inp(x.permute(0, 2, 1, 3).reshape(B * K, T, F)))
        return self.out(h).reshape(B, K, T).permute(0, 2, 1)


def train_gru(train: list[dict], feats: dict, g: dict, seed: int) -> GRUBaseline:
    torch.manual_seed(seed)
    T = max(c['t_leave'] + 1 for c in train)
    x = np.zeros((len(train), T, 4, next(iter(feats.values())).shape[-1]), np.float32)
    m = np.zeros((len(train), T), bool)
    for i, c in enumerate(train):
        n = c['t_leave'] + 1                                     # train on the frames the evaluation uses
        x[i, :n], m[i, 1:n] = feats[c['episode']][:n], True
    y = torch.as_tensor([c['target'] for c in train])
    x, m = torch.as_tensor(x), torch.as_tensor(m)
    net = GRUBaseline(x.shape[-1], g['hidden'])
    opt = torch.optim.Adam(net.parameters(), lr=g['lr'])
    for _ in range(g['epochs']):
        for idx in torch.randperm(len(train)).split(16):
            logit = net(x[idx])
            ce = torch.nn.functional.cross_entropy(logit.reshape(-1, 4), y[idx, None].expand(-1, logit.shape[1]).reshape(-1),
                                                   reduction='none')
            loss = (ce * m[idx].reshape(-1)).sum() / m[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


# ------------------------------------------------------------------ statistics
def ece(conf: np.ndarray, correct: np.ndarray, bins: int) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def boot(stat, n: int, reps: int, rng) -> tuple[float, float, float]:
    vals = np.array([stat(rng.integers(0, n, n)) for _ in range(reps)])
    return float(stat(np.arange(n))), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ------------------------------------------------------------------ stage B
def run_habit(cfg: dict, out) -> dict:
    v3, pc = cfg['v3'], cfg['pilot']
    kp = cfg['tokens']['keypoints']
    cases = habit_cases(cfg)
    order = sorted(range(len(cases)), key=lambda i: cases[i]['episode'])
    fold = np.zeros(len(cases), int)
    for r, i in enumerate(order):
        fold[i] = r % pc['folds']
    offs = [int(round(o * cfg['fps'])) for o in pc['offsets_s']]
    feats = {c['episode']: gru_features(c['ep'], c['cand'], kp) for c in cases}
    seeds = pc['gru']['seeds']
    P = {m: np.zeros((len(cases), len(offs), 4)) for m in ['v3'] + [f'III_s{s}' for s in seeds]}
    point = np.full((len(cases), len(offs)), np.nan)
    curve_rel = np.arange(-100, 11)                                  # frames relative to t_leave for the curve
    curve = {m: np.full((len(cases), len(curve_rel)), np.nan) for m in ('v3', 'III_s0')}
    fitted, p4 = [], []
    for f in range(pc['folds']):
        tr, te = [i for i in range(len(cases)) if fold[i] != f], [i for i in range(len(cases)) if fold[i] == f]
        t0 = time.time()
        x_tr = batch([cases[i]['ep'] for i in tr], [cases[i]['cand'] for i in tr], kp)
        p, hist = fit(x_tr, v3['init'], **v3['fit'])
        fitted.append(dict(fold=f, params=p.values(), nll_first=hist[0], nll_last=hist[-1], seconds=round(time.time() - t0, 1)))
        print(f'fold {f}: v3 fit {fitted[-1]}', flush=True)
        x_te = batch([cases[i]['ep'] for i in te], [cases[i]['cand'] for i in te], kp)
        b = belief(x_te, p)[..., 1:]
        # P4 (report only): belief forced onto a wrong container with confidence c, `inject_s` before t_leave
        c4, d4 = pc['p4']['confidence'], int(round(pc['p4']['inject_s'] * cfg['fps']))
        t_inj, log_inj = [], []
        for i in te:
            c = cases[i]
            wrong = (c['target'] + 1) % 4
            v = np.full(5, (1 - c4) / 4)
            v[1 + wrong] = c4
            t_inj.append(c['t_leave'] - d4 if c['t_leave'] - d4 >= 1 else -1)
            log_inj.append(np.log(v))
        b4 = belief(x_te, p, (torch.as_tensor(t_inj), torch.as_tensor(np.array(log_inj))))[..., 1:]
        for j, i in enumerate(te):
            c, t0_ = cases[i], t_inj[j]
            if t0_ < 0:
                continue
            T = len(c['ep']['holding'])
            ok = np.flatnonzero(b4[j, t0_ + 1:T].argmax(-1) == c['target'])
            p4.append(dict(episode=c['episode'], recovered=bool(len(ok)), frames=int(ok[0]) + 1 if len(ok) else None,
                           by_t_leave=bool(len(ok) and t0_ + 1 + ok[0] <= c['t_leave'])))
        nets = {s: train_gru([cases[i] for i in tr], feats, pc['gru'], s) for s in seeds}
        for j, i in enumerate(te):
            c, T = cases[i], len(cases[i]['ep']['holding'])
            bk = b[j, :T] / b[j, :T].sum(-1, keepdims=True)
            with torch.no_grad():
                pg = {s: torch.softmax(nets[s](torch.as_tensor(feats[c['episode']][None])), -1)[0].numpy() for s in seeds}
            for o, d in enumerate(offs):
                t = c['t_leave'] + d
                if 0 <= t < T:
                    P['v3'][i, o] = bk[t]
                    for s in seeds:
                        P[f'III_s{s}'][i, o] = pg[s][t]
                    ch = pointing_choice(c['ep'], t, c['cand'], kp)
                    point[i, o] = np.nan if ch is None else float(c['cand'].index(ch) == c['target'])
            for q, d in enumerate(curve_rel):
                t = c['t_leave'] + d
                if 0 <= t < T:
                    curve['v3'][i, q] = bk[t, c['target']]
                    curve['III_s0'][i, q] = pg[seeds[0]][t, c['target']]
        print(f'fold {f} done ({time.time() - t0:.0f} s)', flush=True)
    rng = np.random.default_rng(0)
    y = np.array([c['target'] for c in cases])
    n, reps, bins = len(cases), pc['bootstrap'], pc['ece_bins']
    res = dict(n_episodes=n, folds=pc['folds'], fitted=fitted, offsets_s=pc['offsets_s'], per_offset=[])
    for o, off in enumerate(pc['offsets_s']):
        corr = {m: (P[m][:, o].argmax(-1) == y).astype(float) for m in P}
        conf = {m: P[m][:, o].max(-1) for m in P}
        row = dict(offset_s=off, accuracy={}, ece={}, pointing=None, diff_v3_minus_III={}, diff_ece_III_minus_v3={})
        for m in P:
            row['accuracy'][m] = boot(lambda s, m=m: corr[m][s].mean(), n, reps, rng)
            row['ece'][m] = ece(conf[m], corr[m], bins)
        okp = ~np.isnan(point[:, o])
        row['pointing'] = boot(lambda s: point[:, o][okp][s].mean(), int(okp.sum()), reps, rng) + (int(okp.sum()),)
        for s in seeds:
            m = f'III_s{s}'
            row['diff_v3_minus_III'][m] = boot(lambda i, m=m: corr['v3'][i].mean() - corr[m][i].mean(), n, reps, rng)
            row['diff_ece_III_minus_v3'][m] = boot(lambda i, m=m: ece(conf[m][i], corr[m][i], bins) - ece(conf['v3'][i], corr['v3'][i], bins),
                                                   n, reps, rng)
        res['per_offset'].append(row)
    at0 = next(r for r in res['per_offset'] if r['offset_s'] == 0.0)
    g1 = at0['accuracy']['v3'][1] > 0.25
    g2 = all(v[2] >= 0 for v in at0['diff_v3_minus_III'].values())
    g3 = all(v[1] >= 0 or v[1] <= 0 <= v[2] for v in at0['diff_ece_III_minus_v3'].values())
    res['criteria'] = dict(G1=g1, G2=g2, G3=g3, verdict='go' if g1 and g2 and g3 else ('partial' if g1 else 'no-go'))
    fr = [r['frames'] for r in p4 if r['recovered']]
    res['p4'] = dict(confidence=pc['p4']['confidence'], inject_s_before_t_leave=pc['p4']['inject_s'], n=len(p4),
                     recovered=sum(r['recovered'] for r in p4), recovered_by_t_leave=sum(r['by_t_leave'] for r in p4),
                     frames_median=float(np.median(fr)) if fr else None,
                     frames_p90=float(np.percentile(fr, 90)) if fr else None, rows=p4)
    res['curve'] = dict(frames_rel=curve_rel.tolist(), **{m: np.nanmean(v, 0).round(4).tolist() for m, v in curve.items()},
                        n=np.sum(~np.isnan(curve['v3']), 0).tolist())
    np.savez_compressed(out / 'predictions.npz', episodes=[c['episode'] for c in cases], target=y, fold=fold, point=point,
                        **{k: v for k, v in P.items()})
    return res


def plot_curve(res: dict, out) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c = res['curve']
    t = np.array(c['frames_rel']) / 10
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, c['v3'], label='v3 belief on the target (no labels)')
    ax.plot(t, c['III_s0'], label='III GRU p(target) (labels, seed 0)')
    ax.axhline(0.25, color='0.6', ls='--', label='chance')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('time relative to t_leave (s)')
    ax.set_ylabel('mean probability of the true container')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'accuracy.png', dpi=120)


def report(res: dict) -> str:
    f = lambda v: f'{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]'
    lines = [f"# PIF v3 pilot, stage B (HABIT S8789)\n", f"Episodes: {res['n_episodes']}, {res['folds']}-fold CV by episode. "
             f"95% bootstrap intervals. Criteria pre-registered in docs/PIF.md §6 (commit 7ad563e).\n",
             f"**Verdict: {res['criteria']['verdict']}** (G1 {res['criteria']['G1']}, G2 {res['criteria']['G2']}, G3 {res['criteria']['G3']})\n"]
    for r in res['per_offset']:
        lines.append(f"\n## t_leave {r['offset_s']:+.1f} s\n\n| model | accuracy | ECE |\n|---|---|---|")
        for m, v in r['accuracy'].items():
            lines.append(f"| {m} | {f(v)} | {r['ece'][m]:.3f} |")
        lines.append(f"| pointing rule | {f(r['pointing'][:3])} (n = {r['pointing'][3]}) | |")
        lines.append('\n| vs | v3 - III accuracy | ECE(III) - ECE(v3) |\n|---|---|---|')
        for m in r['diff_v3_minus_III']:
            lines.append(f"| {m} | {f(r['diff_v3_minus_III'][m])} | {f(r['diff_ece_III_minus_v3'][m])} |")
    q = res['p4']
    lines.append(f"\n## P4 self-correction (report only)\n\nBelief forced onto a wrong container with b = {q['confidence']}, "
                 f"{q['inject_s_before_t_leave']} s before t_leave: recovered in {q['recovered']}/{q['n']} episodes, "
                 f"{q['recovered_by_t_leave']}/{q['n']} by t_leave; frames to recover median {q['frames_median']}, p90 {q['frames_p90']} (10 Hz).")
    lines.append('\n## Fitted v3 parameters per fold\n')
    for x in res['fitted']:
        lines.append(f"- fold {x['fold']}: " + ', '.join(f'{k} {v:.4g}' for k, v in x['params'].items()) + f" (NLL {x['nll_first']:.3f} -> {x['nll_last']:.3f})")
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------ stage A
def run_sim(cfg: dict, out, tasks: list[str], max_eps: int, noise_m: float) -> dict:
    """Debugging only (scripted human): fit on sim episodes, accuracy of argmax at each gesture's 'clear' frame and
    1 / 2 s later. `noise_m`: Gaussian noise added to the keypoints (sim keypoints are exact, HABIT's are not)."""
    from pif.data.sim import read_dataset
    types = cfg['v3']['candidates']['sim']['types']
    tid = [cfg['tokens']['object_types'].index(t) for t in types]
    eps, labs = read_dataset(cfg, tasks)
    eps, labs = eps[:max_eps], labs[:max_eps]
    rng = np.random.default_rng(0)
    for e in eps:
        e['human_pos'] = e['human_pos'] + rng.normal(0, noise_m, e['human_pos'].shape)
    cands = [[k for k in range(len(e['obj_type'])) if e['obj_type'][k] in tid] for e in eps]
    x = batch(eps, cands, cfg['tokens']['keypoints'])
    p, hist = fit(x, cfg['v3']['init'], **cfg['v3']['fit'])
    b = belief(x, p)
    hits, n_none = [], 0
    for i, (e, lab) in enumerate(zip(eps, labs)):
        for g in lab['gestures']:
            if g['target'] < 0 or g['clear'] is None:
                n_none += 1
                continue
            if g['target'] in cands[i]:
                k = cands[i].index(g['target'])
                row = dict(task=lab['task'], label=g['label'], chance=1 / len(cands[i]))
                for name, d in (('clear', 0), ('clear+1s', 10), ('clear+2s', 20)):   # memory: intent still held per label
                    t = g['clear'] + d
                    if t < len(e['holding']) and lab['intent_object'][t] == g['target']:
                        row[name] = bool(int(np.argmax(b[i, t, 1:len(cands[i]) + 1])) == k)
                hits.append(row)
    acc = {}
    for h in hits:
        for name in ('clear', 'clear+1s', 'clear+2s'):
            if name in h:
                acc.setdefault(f"{h['task']} {name}", []).append(h[name])
    return dict(noise_m=noise_m, params=p.values(), nll_first=hist[0], nll_last=hist[-1], episodes=len(eps), gestures_scored=len(hits),
                gestures_without_object_target=n_none,
                accuracy={k: dict(acc=round(float(np.mean(v)), 3), n=len(v)) for k, v in acc.items()},
                chance_mean=round(float(np.mean([h['chance'] for h in hits])), 3) if hits else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['habit', 'sim'])
    ap.add_argument('--config', default='configs/pif/pif.yaml')
    ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--sim-tasks', nargs='*', default=['T1'])
    ap.add_argument('--sim-max-episodes', type=int, default=120)
    ap.add_argument('--sim-noise-m', type=float, default=0.0)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    out = run_dir('pilot', args.stage)
    log_run(out, 'pif.pilot')
    torch.set_num_threads(4)
    if args.stage == 'habit':
        res = run_habit(cfg, out)
        (out / 'report.md').write_text(report(res))
        plot_curve(res, out)
        print(report(res))
    else:
        res = run_sim(cfg, out, args.sim_tasks, args.sim_max_episodes, args.sim_noise_m)
        print(json.dumps(res, indent=1))
    (out / 'results.json').write_text(json.dumps(res, indent=1))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
