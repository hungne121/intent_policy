"""Fit the reach library of the belief (Pérez-D'Arpino & Shah, ICRA 2015; docs/requirements/INTENT_ACT_v3.2.md §3).

Every reach_block / reach_out segment of the dataset (meta/intent_segments.json) gives one demonstration of its target
(U1 / U2 for T3, H1 / H2 for T2): the active hand's wrist and index tip relative to their first-frame position over
[t_onset, t_clear]. Out-of-fold: fold f is fitted on the episodes of the other folds (belief.fold_of: by list cell),
and an episode uses the fold it belongs to. The late and the early dataset share seeds, so the early expert never uses
a library that saw its own late twin's reach.

Reaches whose start no library cluster covers fall back to MaxEnt IOC goal inference (belief.GoalInference); its
temperature beta is calibrated the same way. The posterior's temperature is calibrated per target family on the out-of-fold predictions of every prefix (minimum
NLL; the per-step diagonal Gaussians make the raw posterior over-confident). Reports, out-of-fold, the accuracy of the
calibrated posterior vs the fraction of the reach observed (the curve of the paper's Fig. 5 / 7; here from the
ground-truth onset) and the fraction already right with p >= b*.

  ./run.sh -m scripts.fit_reach_library --dataset outputs/datasets/intent_act_late_v5 --output assets/intent/reach_library_v5.npz
"""
import argparse
import json
from collections import defaultdict
import numpy as np
from intent_policy.intent.belief import (REACH_LABELS, GoalInference, OnlineDTW, ReachLibrary, ReachPredictor, fold_of,
                                         reach_features, save_libraries)
from intent_policy.intent.labels import hand_indices, load_schema
from intent_policy.intent.perception import fit_temperature
from intent_policy.utils import resolve
from scripts.build_intent_labels import load_episodes

FRACTIONS = np.linspace(0.1, 1.0, 10)


def demos_of(root, schema) -> list[dict]:
    """[{episode, seed, target, x (T, 6)}] of every reach segment."""
    segs = {int(k): v for k, v in json.loads((root / 'meta/intent_segments.json').read_text()).items()}
    seeds = {}
    with open(root / 'meta/hri_episodes.jsonl') as f:
        for line in f:
            m = json.loads(line)
            seeds[int(m['episode_index'])] = int(m['seed'])
    eps = load_episodes(root)
    names, hands = list(schema['keypoints']), hand_indices(schema)
    out = []
    for e, ss in segs.items():
        kp = eps[e]['keypoints']
        rest = kp[0]
        for s in ss:
            if not s['source'].split('+')[0] in REACH_LABELS or s['t_clear'] <= s['t_onset']:
                continue
            far = lambda h: float(np.linalg.norm(kp[s['t_clear'], hands[h]].mean(0) - rest[hands[h]].mean(0)))
            idx = hands[max(hands, key=far)]
            x = reach_features(kp[s['t_onset']:s['t_clear'] + 1], rest, idx, names)
            wrist = names.index(f"{names[idx[0]].split('_', 1)[0]}_wrist")
            out.append(dict(episode=e, seed=seeds[e], target=s['target'], x=x, w=kp[s['t_onset']:s['t_clear'] + 1, wrist]))
    return out


def oof_logits(lib: ReachLibrary, demos: list[dict], classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Uncalibrated logits of every prefix of every demo and its true class index."""
    Z, y = [], []
    for d in demos:
        dtw = OnlineDTW(lib, classes, temperature=1.0)
        for x in d['x']:
            dtw.step(x)
            Z.append(dtw.logits.copy())
            y.append(classes.index(d['target']))
    return np.asarray(Z), np.asarray(y)


def fit_beta(Z: np.ndarray, y: np.ndarray) -> float:
    """IOC temperature beta [m] minimising the NLL of the true goals (grid 0.1 mm - 1 m; the logits are metres)."""
    best, best_b = np.inf, 1.0
    for b in np.exp(np.linspace(np.log(1e-4), np.log(1.0), 80)):
        z = Z / b
        z = z - z.max(1, keepdims=True)
        nll = -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean()
        if nll < best:
            best, best_b = nll, float(b)
    return best_b


def ioc_logits(demos: list[dict], classes: list[str], libs: dict, folds: int) -> tuple[np.ndarray, np.ndarray]:
    """IOC logits of every prefix (each demo with the goals of the library that did not see it)."""
    Z, y = [], []
    for d in demos:
        gi = GoalInference(libs[fold_of(d['seed'], folds)].goal, classes, 1.0)
        for w in d['w']:
            gi.step(w)
            Z.append(gi.logits.copy())
            y.append(classes.index(d['target']))
    return np.asarray(Z), np.asarray(y)


def posteriors(mode: str, lib: ReachLibrary, d: dict, classes: list[str]) -> list[np.ndarray]:
    """Posterior after every frame of demo d with ReachPredictor method `mode` (library / ioc / library_then_ioc)."""
    rp = ReachPredictor(lib, classes, mode)
    return [rp.step(x, w) for x, w in zip(d['x'], d['w'])]


def accuracy_curve(lib: ReachLibrary, demos: list[dict], classes: list[str], b_star: float,
                   mode: str = 'library') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """After observing each fraction of the reach: accuracy (ties within 0.01 split the point, e.g. 0.5 / 0.5 counts
    1/2), the fraction of demos already right with p_true >= b_star (what a commit needs) and the fraction confidently
    wrong (another class >= b_star: a wrong commit)."""
    acc, sure, bad = np.zeros(len(FRACTIONS)), np.zeros(len(FRACTIONS)), np.zeros(len(FRACTIONS))
    for d in demos:
        post = posteriors(mode, lib, d, classes)
        T, true = len(post), classes.index(d['target'])
        for i, f in enumerate(FRACTIONS):
            p = post[max(int(np.ceil(f * T)) - 1, 0)]
            top = np.flatnonzero(p >= p.max() - 0.01)
            acc[i] += (true in top) / len(top)
            sure[i] += p[true] >= b_star
            bad[i] += np.delete(p, true).max() >= b_star
    n = max(len(demos), 1)
    return acc / n, sure / n, bad / n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='outputs/datasets/intent_act_late_v5')
    p.add_argument('--schema', default='configs/intent_schema.yaml')
    p.add_argument('--output', default='assets/intent/reach_library_v5.npz')
    args = p.parse_args()
    schema = load_schema(args.schema)
    cfg = schema['belief']['reach']
    folds, n_steps, floor = int(cfg['folds']), int(cfg['n_steps']), float(cfg['var_floor_m'])
    radius, min_demos = float(cfg['start_radius_m']), int(cfg['min_demos'])
    root = resolve(args.dataset)
    demos = demos_of(root, schema)
    by_target = defaultdict(int)
    for d in demos:
        by_target[d['target']] += 1
    print(f'{len(demos)} reach demonstrations: {dict(sorted(by_target.items()))}')
    families = [['U1', 'U2'], ['H1', 'H2']]
    libs = {}
    for f in range(folds):
        train, ends = defaultdict(list), defaultdict(list)
        for d in demos:
            if fold_of(d['seed'], folds) != f:
                train[d['target']].append(d['x'])
                ends[d['target']].append(d['w'][-1])
        libs[f] = ReachLibrary.fit(train, n_steps, floor, families, radius, min_demos, ends)
        print(f"fold {f}: start-cluster libraries {sorted(k for k in libs[f].mean if '@' in k)}")
    report = {}
    b_star = float(schema['belief']['b_star'])
    for fam in families:
        held = [d for d in demos if d['target'] in fam]
        Zi, yi = ioc_logits(held, fam, libs, folds)
        beta = max(fit_beta(Zi, yi), float(cfg['ioc_beta_min_m']))
        for lib in libs.values():
            lib.temperature[f"ioc:{'+'.join(sorted(fam))}"] = beta
        print(f"{'/'.join(fam)}: MaxEnt IOC beta = {1000 * beta:.2f} mm")
        parts = [oof_logits(libs[f], [d for d in held if fold_of(d['seed'], folds) == f], fam) for f in range(folds)]
        parts = [(Z, y) for Z, y in parts if len(y)]
        T = fit_temperature(np.concatenate([Z for Z, _ in parts]), np.concatenate([y for _, y in parts]))
        for lib in libs.values():
            lib.temperature['+'.join(sorted(fam))] = T
        print(f"{'/'.join(fam)}: calibrated temperature T = {T:.2f}")
        uncovered = [d for d in held if libs[fold_of(d['seed'], folds)].cluster_for(fam, d['x'][0]) is None]
        rep = dict(n=len(held), n_without_start_cluster=len(uncovered), temperature=round(T, 3), ioc_beta=round(beta, 5))
        print(f"{'/'.join(fam)} (n={len(held)}, {len(uncovered)} without a start cluster), % of the reach observed: "
              + ' '.join(f'{int(100 * fr):>4d}' for fr in FRACTIONS))
        for mode, subset, label in (('library', held, 'library'), ('ioc', held, 'ioc'), ('library_then_ioc', held, 'library_then_ioc'),
                                    ('library', uncovered, 'library, no cluster'), ('ioc', uncovered, 'ioc, no cluster')):
            if not subset:
                continue
            acc, sure, bad = np.zeros(len(FRACTIONS)), np.zeros(len(FRACTIONS)), np.zeros(len(FRACTIONS))
            for f in range(folds):
                part = [d for d in subset if fold_of(d['seed'], folds) == f]
                a, s_, b_ = accuracy_curve(libs[f], part, fam, b_star, mode)
                acc, sure, bad = acc + a * len(part), sure + s_ * len(part), bad + b_ * len(part)
            acc, sure, bad = acc / len(subset), sure / len(subset), bad / len(subset)
            rep[label] = dict(accuracy=[round(float(a), 3) for a in acc], right_with_p_ge_b_star=[round(float(a), 3) for a in sure],
                              wrong_with_p_ge_b_star=[round(float(a), 3) for a in bad])
            print(f'   {label:<20s} acc     ' + ' '.join(f'{a:4.2f}' for a in acc))
            print(f'   {"":<20s} right>=b* ' + ' '.join(f'{a:4.2f}' for a in sure))
            print(f'   {"":<20s} wrong>=b* ' + ' '.join(f'{a:4.2f}' for a in bad))
        report['/'.join(fam)] = rep
    out = resolve(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(dataset=str(args.dataset), folds=folds, n_steps=n_steps, var_floor_m=floor, start_radius_m=radius,
                min_demos=min_demos, demos=dict(by_target), fractions=FRACTIONS.round(2).tolist(), oof_accuracy=report)
    save_libraries(out, libs, meta)
    out.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print(f'library: {out}')


if __name__ == '__main__':
    main()
