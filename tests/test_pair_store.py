import numpy as np
import pytest
from intent_policy.benchmark.pair_store import PairStore


def records():
    frame = {'action': np.zeros(7, np.float32), 'observation.state': np.zeros(10, np.float32),
             'human.keypoints': np.zeros((6, 3), np.float32),
             'observation.images.high': np.zeros((8, 8, 3), np.uint8)}
    return {mode: dict(success=True, collection_quality={'passed': True},
        exogenous_condition_sha256='matched', frames=[frame], metadata={'split': 'train'})
        for mode in ('early', 'observed')}


def test_roundtrip_preserves_rgb_and_both_sides(tmp_path):
    store = PairStore(tmp_path)
    pair = store.write_pair({'pair_id': 'I1_0'}, records())
    manifest, = store.verified_pairs()
    assert manifest['sides']['early']['frames'] == 1
    for mode in ('early', 'observed'):
        with np.load(pair / f'{mode}.npz', allow_pickle=False) as data:
            assert data['observation.images.high'].dtype == np.uint8
            assert data['action'].shape == (1, 7)
    with pytest.raises(FileExistsError):
        store.write_pair({'pair_id': 'I1_0'}, records())


def test_failure_in_second_side_does_not_publish_first(tmp_path):
    store = PairStore(tmp_path)
    r = records()
    r['observed']['frames'] = []
    with pytest.raises(ValueError, match='Empty'):
        store.write_pair({'pair_id': 'I1_0'}, r)
    assert list(store.verified_pairs()) == []
    assert list(tmp_path.iterdir()) == []


def test_resume_ignores_partial_and_detects_corruption(tmp_path):
    store = PairStore(tmp_path)
    (tmp_path / '.partial-crashed').mkdir()
    pair = store.write_pair({'pair_id': 'I1_0'}, records())
    assert len(list(store.verified_pairs())) == 1
    with (pair / 'observed.npz').open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='Checksum'):
        list(store.verified_pairs())


def test_unchecked_or_unmatched_pair_cannot_be_saved(tmp_path):
    store = PairStore(tmp_path)
    r = records()
    r['observed']['collection_quality'] = None
    with pytest.raises(ValueError, match='quality'):
        store.write_pair({'pair_id': 'I1_0'}, r)
    r = records()
    r['observed']['exogenous_condition_sha256'] = 'different'
    with pytest.raises(ValueError, match='Unmatched'):
        store.write_pair({'pair_id': 'I1_0'}, r)
