import json
import numpy as np
import pytest
from intent_policy.sim.motion_bank import MotionBank, clip_id, palm_path, segments, segment_ranges


def clips():
    x = np.zeros((2, 16, 52, 3), np.float32)
    x[:, :, 15, 1] = 1.5
    x[:, :, 1, 0] = -.1
    x[:, :, 2, 0] = .1
    for i in range(2):
        x[i, :, [21, 40, 43, 46, 49], 2] = np.linspace(0, .2 + i * .1, 16)
    return x


def test_crops_keep_original_frames():
    p = palm_path(clips()[0])
    assert len(segment_ranges(p)) > 0
    for (a, b), crop in zip(segment_ranges(p), segments(p)):
        np.testing.assert_array_equal(p[a:b], crop)


def test_split_is_on_original_clip_not_crop(tmp_path):
    x = clips()
    np.savez(tmp_path / 'reach.npz', keypoints3d=x, fps=30, seeds=[1, 2])
    ids = [clip_id(v) for v in x]
    assert ids[0] != ids[1]
    manifest = tmp_path / 'splits.json'
    manifest.write_text(json.dumps({'clip_splits': dict(zip(ids, ['train', 'test']))}))
    kw = dict(folder=tmp_path, pelvis_world=[0, 0, 1], facing_deg=180, floor_z=0, height=1.5)
    banks = [MotionBank(**kw, split_manifest=manifest, split=s) for s in ('train', 'test')]
    for b, cid in zip(banks, ids):
        assert b.pool_records(['reach'])
        assert {r.source['clip_id'] for r in b.pool_records(['reach'])} == {cid}
    assert clip_id(x[0].copy()) == ids[0]
    manifest.write_text(json.dumps({'clip_splits': {ids[0]: 'train'}}))
    with pytest.raises(ValueError, match='absent'):
        MotionBank(**kw, split_manifest=manifest, split='train')


def test_left_is_not_replaced_by_right():
    x = clips()[0]
    assert np.ptp(palm_path(x, 'left')[:, 2]) == 0
    assert np.ptp(palm_path(x, 'right')[:, 2]) > .1
