"""Leak test of intent labels (docs/requirements/INTENT_ACT_v3.2.md §5, check before training C / D).

Before the human's first selecting hand motion (frames < its ground-truth start), no causal intent can know which of
the candidates (the instructed kind's twins in T1 / T2, the two cups in T3) the human will choose:
  (a) excess: mean share of p_target on the eventual target among the candidates minus 1 / n_candidates (~0;
      a source with no mass on the candidates yet counts as uniform)
  (b) probe: a logistic regression on every intent field of those frames, cross-validated by episode, predicting
      which candidate it will be (should be at chance, ~1 / n_candidates)
Red flag: excess > 0.02 or probe accuracy above chance by more than 2 binomial standard errors.

  ./run.sh -m scripts.intent_leak_test --dataset-root outputs/datasets/intent_act_early_v5 --sources pb hs pp
"""
import argparse
import json
import numpy as np
import pyarrow.parquet as pq
import torch
from intent_policy.intent.labels import load_schema
from intent_policy.utils import resolve
from scripts.build_intent_labels import column_numpy, data_files

SELECTING = ('point_object', 'reach_block')


def first_choice(m: dict, schema: dict):
    """(start frame of the first selecting motion, true target slot, candidate slots) or None (T4)."""
    spec = m['spec']
    layout = spec.get('layout') or {}
    start = next((mo['start'] for mo in m['human_motions'] if mo['label'] in SELECTING), None)
    if start is None or m['task_code'] == 'T4':
        return None
    first = (spec.get('change') or {}).get('old') or spec['target']
    desc = schema['instruction']['object_desc']
    if m['task_code'] == 'T3':
        cands = list(layout)
    else:
        cands = [k for k in layout if desc.get(k) == desc[first]]
    return int(start), layout[first], [layout[k] for k in cands]


def logistic_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_cls: int, folds: int = 5, seed: int = 0) -> float:
    """Episode-grouped cross-validated accuracy of a multinomial logistic regression (torch, L2, standardised X)."""
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold = dict(zip(ug, rng.permutation(len(ug)) % folds))
    f = np.array([fold[g] for g in groups])
    hits = 0
    for k in range(folds):
        tr, te = f != k, f == k
        if not te.any() or len(np.unique(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xt, Xe = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32), torch.tensor((X[te] - mu) / sd, dtype=torch.float32)
        W = torch.zeros(X.shape[1], n_cls, requires_grad=True)
        b = torch.zeros(n_cls, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=200)
        yt = torch.tensor(y[tr])

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt) + 1e-2 * (W ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        hits += int(((Xe @ W + b).argmax(1).numpy() == y[te]).sum())
    return hits / len(y)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--sources', nargs='+', default=['pb'], help='column prefixes intent_<src> (pb hs pp pr0 ...)')
    p.add_argument('--schema', default='configs/intent_schema.yaml')
    args = p.parse_args()
    schema, root = load_schema(args.schema), resolve(args.dataset_root)
    metas = {m['episode_index']: m for m in map(json.loads, (root / 'meta/hri_episodes.jsonl').read_text().splitlines())}
    choice = {e: first_choice(m, schema) for e, m in metas.items()}
    T = schema['targets']
    for src in args.sources:
        prefix = f'intent_{src}'
        cols = None
        rows = {'ep': [], 'fr': []}
        data: dict[str, list] = {}
        for f in data_files(root):
            names = pq.read_schema(f).names
            cols = [c for c in names if c.startswith(prefix + '.')]
            if not cols:
                raise SystemExit(f'{prefix} missing (scripts/build_intent_labels.py --sources ...)')
            t = pq.read_table(f, columns=['episode_index', 'frame_index'] + cols)
            rows['ep'].append(t.column('episode_index').to_numpy())
            rows['fr'].append(t.column('frame_index').to_numpy())
            for c in cols:
                data.setdefault(c, []).append(column_numpy(t, c).reshape(t.num_rows, -1))
        ep, fr = np.concatenate(rows['ep']), np.concatenate(rows['fr'])
        X_all = np.concatenate([np.concatenate(data[c]) for c in sorted(data)], 1)
        pt = np.concatenate(data[f'{prefix}.p_target'])
        excess, X, y, g, n_cls = [], [], [], [], 2
        for i in np.flatnonzero(np.array([choice.get(int(e)) is not None and fr_ < choice[int(e)][0] for e, fr_ in zip(ep, fr)])):
            start, true, cands = choice[int(ep[i])]
            mass = sum(pt[i, T.index(c)] for c in cands)            # the decision's share among its candidates
            excess.append((pt[i, T.index(true)] / mass if mass > 1e-6 else 1.0 / len(cands)) - 1.0 / len(cands))
            X.append(X_all[i])
            y.append(sorted(cands).index(true))
            g.append(int(ep[i]))
            n_cls = max(n_cls, len(cands))
        X, y, g = np.asarray(X), np.asarray(y), np.asarray(g)
        acc = logistic_cv(X, y, g, n_cls)
        chance = float(np.bincount(y).max() / len(y))                 # majority class of the pre-onset frames
        se = np.sqrt(chance * (1 - chance) / len(np.unique(g)))         # frames of an episode are not independent
        ex = float(np.mean(excess))
        flag = ex > 0.02 or acc > chance + 2 * se
        print(f'{prefix}: {len(y)} pre-onset frames of {len(np.unique(g))} episodes | excess p_target on the true '
              f'target {ex:+.4f} | probe accuracy {acc:.3f} vs chance {chance:.3f} (+2 SE {chance + 2 * se:.3f}) '
              f'-> {"LEAK" if flag else "ok"}')


if __name__ == '__main__':
    main()
