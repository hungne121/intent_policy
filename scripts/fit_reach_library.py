"""Fit the reach library of the belief (Pérez-D'Arpino & Shah, ICRA 2015; docs/requirements/INTENT_ACT_v3.2.md §3).

Every reach_block / reach_out segment of the dataset (meta/intent_segments.json) gives one demonstration of its target
(U1 / U2 for T3, H1 / H2 for T2): the active hand's wrist and index tip relative to their first-frame position over
[t_onset, t_clear]. Out-of-fold: fold f is fitted on the episodes of the other folds (belief.fold_of: by list cell),
and an episode uses the fold it belongs to. The late and the early dataset share seeds, so the early expert never uses
a library that saw its own late twin's reach.

The posterior's temperature is calibrated per target family on the out-of-fold predictions of every prefix (minimum
NLL; the per-step diagonal Gaussians make the raw posterior over-confident). Reports, out-of-fold, the accuracy of the
calibrated posterior vs the fraction of the reach observed (the curve of the paper's Fig. 5 / 7; here from the
ground-truth onset) and the fraction already right with p >= b*.

  ./run.sh -m scripts.fit_reach_library --dataset outputs/datasets/intent_act_late_v5 --output assets/intent/reach_library_v5.npz
"""
import argparse
import json
from collections import defaultdict
import numpy as np
from intent_policy.intent.belief import REACH_LABELS, OnlineDTW, ReachLibrary, fold_of, reach_features, save_libraries
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
            out.append(dict(episode=e, seed=seeds[e], target=s['target'], x=x))
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


def accuracy_curve(lib: ReachLibrary, demos: list[dict], classes: list[str], b_star: float) -> tuple[np.ndarray, np.ndarray]:
    """After observing each fraction of the reach: accuracy (ties within 0.01 split the point, e.g. 0.5 / 0.5 counts
    1/2) and the fraction of demos already right with p_true >= b_star (what a commit needs)."""
    acc, sure = np.zeros(len(FRACTIONS)), np.zeros(len(FRACTIONS))
    for d in demos:
        dtw = OnlineDTW(lib, classes)
        post = [dtw.step(x) for x in d['x']]
        T, true = len(post), classes.index(d['target'])
        for i, f in enumerate(FRACTIONS):
            p = post[max(int(np.ceil(f * T)) - 1, 0)]
            top = np.flatnonzero(p >= p.max() - 0.01)
            acc[i] += (true in top) / len(top)
            sure[i] += p[true] >= b_star
    n = max(len(demos), 1)
    return acc / n, sure / n


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
        train = defaultdict(list)
        for d in demos:
            if fold_of(d['seed'], folds) != f:
                train[d['target']].append(d['x'])
        libs[f] = ReachLibrary.fit(train, n_steps, floor, families, radius, min_demos)
        print(f"fold {f}: start-cluster libraries {sorted(k for k in libs[f].mean if '@' in k)}")
    report = {}
    b_star = float(schema['belief']['b_star'])
    for fam in families:
        held = [d for d in demos if d['target'] in fam]
        parts = [oof_logits(libs[f], [d for d in held if fold_of(d['seed'], folds) == f], fam) for f in range(folds)]
        parts = [(Z, y) for Z, y in parts if len(y)]
        T = fit_temperature(np.concatenate([Z for Z, _ in parts]), np.concatenate([y for _, y in parts]))
        for lib in libs.values():
            lib.temperature['+'.join(sorted(fam))] = T
        print(f"{'/'.join(fam)}: calibrated temperature T = {T:.2f}")
        acc, sure = np.zeros(len(FRACTIONS)), np.zeros(len(FRACTIONS))
        for f in range(folds):
            part = [d for d in held if fold_of(d['seed'], folds) == f]
            a, s_ = accuracy_curve(libs[f], part, fam, b_star)
            acc, sure = acc + a * len(part), sure + s_ * len(part)
        acc, sure = acc / max(len(held), 1), sure / max(len(held), 1)
        report['/'.join(fam)] = dict(n=len(held), temperature=round(T, 3), accuracy=[round(float(a), 3) for a in acc],
                                     right_with_p_ge_b_star=[round(float(a), 3) for a in sure])
        print(f"{'/'.join(fam)} (n={len(held)}), % of the reach observed: " + ' '.join(f'{int(100 * fr):>4d}' for fr in FRACTIONS))
        print('   accuracy (ties split)        ' + ' '.join(f'{a:4.2f}' for a in acc))
        print(f'   right with p >= b* = {b_star:.2f}     ' + ' '.join(f'{a:4.2f}' for a in sure))
    out = resolve(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(dataset=str(args.dataset), folds=folds, n_steps=n_steps, var_floor_m=floor, start_radius_m=radius,
                min_demos=min_demos, demos=dict(by_target), fractions=FRACTIONS.round(2).tolist(), oof_accuracy=report)
    save_libraries(out, libs, meta)
    out.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print(f'library: {out}')


if __name__ == '__main__':
    main()
