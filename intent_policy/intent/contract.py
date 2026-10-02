"""Intent data contract (docs/requirements/INTENT_ACT_GUIDE_v2.md §2, §9): field names, shapes, value ranges.

Every intent source writes these fields per frame (dataset columns `intent_<src>.<field>`, src = hs | pp | pr0..);
the policy reads them as `intent.<field>` and never knows the source. `validate_intent` is called when a dataset is
written and in the tests.
"""
from __future__ import annotations
import numpy as np

SOURCES = {'hindsight': 'hs', 'perfect': 'pp', 'predicted': 'pr'}
DIST_TOL = 1e-4


def field_shapes(schema: dict) -> dict[str, tuple]:
    K, W, P = len(schema['targets']), len(schema['who']), len(schema['phases'])
    M, J, D = schema['n_waypoints'], len(schema['keypoints']), schema['keypoint_dim']
    return {'p_who': (W,), 'p_target': (K,), 'c_who': (W,), 'c_target': (K,), 'occupancy': (7,), 'tte': (1,),
            'tte_std': (1,), 'phase': (P,), 'xi': (M, J, D), 'confidence': (2,)}


DISTRIBUTIONS = ('p_who', 'p_target', 'c_who', 'c_target', 'phase')
FIELDS = ('p_who', 'p_target', 'c_who', 'c_target', 'occupancy', 'tte', 'tte_std', 'phase', 'xi', 'confidence')


def dataset_features(schema: dict, prefix: str) -> dict:
    """LeRobot feature specs of one source's columns `<prefix>.<field>`."""
    return {f'{prefix}.{k}': {'dtype': 'float32', 'shape': shape, 'names': None} for k, shape in field_shapes(schema).items()}


def confidence(p_target: np.ndarray, p_who: np.ndarray) -> np.ndarray:
    """[1 - H(p_target) / log K, max(p_who)] (last axis)."""
    p = np.clip(p_target, 1e-12, 1.0)
    h = -(p * np.log(p)).sum(-1) / np.log(p_target.shape[-1])
    return np.stack([np.clip(1.0 - h, 0.0, 1.0), p_who.max(-1)], -1).astype(np.float32)


def one_hot(i: int, n: int) -> np.ndarray:
    v = np.zeros(n, np.float32)
    v[int(i)] = 1.0
    return v


def validate_intent(d: dict, schema: dict, batched: bool = False) -> None:
    """Raise ValueError unless `d` holds every contract field with the right shape (after an optional leading
    batch / time axis), float32, finite, distributions >= 0 summing to 1 (tolerance 1e-4) and values in range."""
    shapes, tmax = field_shapes(schema), float(schema['tte_max_s'])
    missing = [k for k in FIELDS if k not in d]
    if missing:
        raise ValueError(f'intent fields missing: {missing}')
    for k, shape in shapes.items():
        v = np.asarray(d[k])
        got = v.shape[1:] if batched else v.shape
        if tuple(got) != tuple(shape):
            raise ValueError(f'{k}: shape {v.shape}, expected {"(N, " if batched else "("}{", ".join(map(str, shape))})')
        if v.dtype != np.float32:
            raise ValueError(f'{k}: dtype {v.dtype}, expected float32')
        if not np.isfinite(v).all():
            raise ValueError(f'{k}: NaN / inf')
    for k in DISTRIBUTIONS:
        v = np.asarray(d[k])
        if (v < -DIST_TOL).any() or (np.abs(v.sum(-1) - 1.0) > DIST_TOL).any():
            raise ValueError(f'{k}: not a distribution (min {v.min():.4g}, sums {np.unique(v.sum(-1).round(5))[:5]})')
    for k in ('tte', 'tte_std'):
        v = np.asarray(d[k])
        if (v < 0).any() or (v > tmax + 1e-6).any():
            raise ValueError(f'{k} outside [0, {tmax}]')
    occ = np.asarray(d['occupancy'])
    if (occ[..., 3:6] < 0).any() or (occ[..., 6] < 0).any() or (occ[..., 6] > 1).any():
        raise ValueError('occupancy: negative size or conf outside [0, 1]')
    conf = np.asarray(d['confidence'])
    if (conf < -1e-6).any() or (conf > 1 + 1e-6).any():
        raise ValueError('confidence outside [0, 1]')
