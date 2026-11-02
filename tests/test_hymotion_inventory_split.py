import numpy as np
import pytest

from scripts.audit_hymotion_corpus import inventory, split_manifest


def test_manifest_uses_sample_assignments_and_rejects_cross_split_duplicates(tmp_path):
    clips = np.zeros((2, 3, 52, 3), dtype=np.float32)
    clips[1] += 1
    path = tmp_path / 'samples.npz'
    np.savez(path, keypoints3d=clips, seeds=[1, 2], sample_splits=['train', 'test'])
    assert sorted(split_manifest(inventory(tmp_path))['clip_splits'].values()) == ['test', 'train']
    clips[1] = clips[0]
    np.savez(path, keypoints3d=clips, seeds=[1, 2], sample_splits=['train', 'test'])
    with pytest.raises(ValueError, match='both train and test'):
        split_manifest(inventory(tmp_path))


def test_inventory_rejects_incomplete_sample_metadata(tmp_path):
    np.savez(tmp_path / 'samples.npz', keypoints3d=np.zeros((2, 3, 52, 3)), seeds=[1])
    with pytest.raises(ValueError, match='metadata must match'):
        inventory(tmp_path)
