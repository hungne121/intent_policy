import pytest
from intent_policy.intent.data_split import provenance_split


def row(split, clip):
    return dict(split=split, human_motion_sources=[dict(source=dict(kind='hymotion', clip_id=clip, split=split))])


def test_test_episodes_cannot_enter_training():
    meta = {0: row('train', 'a'), 1: row('val', 'b'), 2: row('test', 'c')}
    with pytest.raises(ValueError, match='separate evaluation dataset'):
        provenance_split(meta, list(meta))
    assert provenance_split(meta, [0, 1]) == ([0], [1])


def test_duplicate_source_clip_in_val_is_rejected():
    meta = {0: row('train', 'a'), 1: row('val', 'a')}
    with pytest.raises(ValueError, match='leaks'):
        provenance_split(meta, list(meta))


def test_missing_source_is_not_silently_index_split():
    with pytest.raises(ValueError, match='traceable'):
        provenance_split({0: dict(split='train')}, [0])
