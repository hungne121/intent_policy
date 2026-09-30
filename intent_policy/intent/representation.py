"""Oracle record -> numeric policy features (`observation.oracle.*`), relative to the current TCP.

Spatial/target: one-hot target identity over a fixed vocabulary, validity flag and the target
position relative to the TCP, separately for the object slot and the interaction-region slot
(the slot declares object versus region). Motion: current and future hand position (relative
to the TCP) and velocity. World-frame copies are stored for analysis only. The same function is
used for recording demonstrations, training and closed-loop evaluation.
"""
from __future__ import annotations
import numpy as np
from intent_policy.intent.oracle import HORIZONS, OracleRecord, horizon_tag

PREFIX = 'observation.oracle.'
OBJECT_VOCAB = ('B1', 'B2', 'B3', 'B1p', 'C1', 'C2', 'C3')              # objects_catalog (scence_construct.md §3.1)
REGION_VOCAB = ('P1', 'P2', 'H1', 'H2', 'U_cup', 'intrusion_point')     # configs/layout.yaml zones + T4 intrusion


def feature_shapes(horizons=HORIZONS) -> dict[str, int]:
    shapes = {'object_id': len(OBJECT_VOCAB), 'object_valid': 1, 'object_rel': 3, 'object_pos': 3,
              'region_id': len(REGION_VOCAB), 'region_valid': 1, 'region_rel': 3, 'region_pos': 3,
              'hand_rel': 3, 'hand_vel': 3, 'hand_pos': 3}
    for h in horizons:
        tag = horizon_tag(h)
        shapes.update({f'future_hand_rel_{tag}': 3, f'future_hand_vel_{tag}': 3, f'future_hand_pos_{tag}': 3})
    return {PREFIX + k: n for k, n in shapes.items()}


def dataset_features(horizons=HORIZONS) -> dict:
    """LeRobot feature specs for the oracle fields."""
    return {k: {'dtype': 'float32', 'shape': (n,), 'names': [f'{k.removeprefix(PREFIX)}_{i}' for i in range(n)]}
            for k, n in feature_shapes(horizons).items()}


def _one_hot(key: str | None, vocab) -> np.ndarray:
    v = np.zeros(len(vocab), np.float32)
    if key is not None:
        v[vocab.index(key)] = 1.0
    return v


def features(rec: OracleRecord, tcp) -> dict[str, np.ndarray]:
    tcp = np.asarray(tcp, float)
    f32 = lambda a: np.asarray(a, np.float32)
    zero = np.zeros(3, np.float32)
    out = {}
    for slot, vocab in (('object', OBJECT_VOCAB), ('region', REGION_VOCAB)):
        key, pos = getattr(rec, f'{slot}_key'), getattr(rec, f'{slot}_pos')
        valid = key is not None and pos is not None
        out[f'{slot}_id'] = _one_hot(key if valid else None, vocab)
        out[f'{slot}_valid'] = f32([float(valid)])
        out[f'{slot}_rel'] = f32(pos - tcp) if valid else zero
        out[f'{slot}_pos'] = f32(pos) if valid else zero
    out['hand_rel'], out['hand_vel'], out['hand_pos'] = f32(rec.hand_pos - tcp), f32(rec.hand_vel), f32(rec.hand_pos)
    for h, p in rec.future_pos.items():
        tag = horizon_tag(h)
        out[f'future_hand_rel_{tag}'] = f32(p - tcp)
        out[f'future_hand_vel_{tag}'] = f32(rec.future_vel[h])
        out[f'future_hand_pos_{tag}'] = f32(p)
    return {PREFIX + k: v for k, v in out.items()}


def record_from_features(f: dict, t: float = 0.0, horizons=HORIZONS) -> OracleRecord:
    """Inverse of `features` (from the stored world-frame fields), e.g. to rebuild oracle inputs offline."""
    g = lambda k: np.asarray(f[PREFIX + k], float).reshape(-1)
    slot = lambda name, vocab: ((vocab[int(np.argmax(g(f'{name}_id')))], g(f'{name}_pos'))
                                if g(f'{name}_valid')[0] > 0.5 else (None, None))
    (ok, op), (rk, rp) = slot('object', OBJECT_VOCAB), slot('region', REGION_VOCAB)
    return OracleRecord(t, ok, op, rk, rp, g('hand_pos'), g('hand_vel'),
                        {h: g(f'future_hand_pos_{horizon_tag(h)}') for h in horizons},
                        {h: g(f'future_hand_vel_{horizon_tag(h)}') for h in horizons})
