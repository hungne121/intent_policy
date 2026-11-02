import pytest
from intent_policy.benchmark.paired_collection import collect_pairs, IncompleteQuota
from intent_policy.scenarios.config import InfeasibleVariation


def record(valid=True, condition='same'):
    return dict(success=True, collection_quality=dict(passed=valid), exogenous_condition_sha256=condition)


def test_one_side_rejection_discards_whole_pair_and_refills():
    calls, accepted, log = [], [], []
    def run(mode, candidate):
        calls.append((mode, candidate['pair_id']))
        return record(not (mode == 'early' and candidate['pair_id'] == 0))
    result = collect_pairs([{'pair_id': i} for i in range(3)], 2, run,
                           lambda c, r: accepted.append(c['pair_id']), log.append)
    assert accepted == [1, 2]
    assert ('observed', 0) in calls  # report the rejected pair's other side too
    assert result == dict(complete=True, accepted_pairs=2, attempts=3, rejected_pairs=1)
    assert not log[0]['accepted']


def test_same_seed_without_matched_motion_is_not_accepted():
    with pytest.raises(IncompleteQuota) as e:
        collect_pairs([{'pair_id': 1}], 1, lambda mode, c: record(condition=mode),
                      lambda *a: pytest.fail('unmatched pair written'), lambda x: None)
    assert e.value.accepted == 0


def test_missing_quality_report_never_counts_as_success():
    with pytest.raises(IncompleteQuota):
        collect_pairs([{'pair_id': 1}], 1, lambda *a: dict(success=True, exogenous_condition_sha256='x'),
                      lambda *a: pytest.fail('unchecked pair written'), lambda x: None)


def test_infeasible_variation_is_a_rejected_attempt():
    accepted, log = [], []
    def run(mode, candidate):
        if candidate['pair_id'] == 0:
            raise InfeasibleVariation('No decoy spot found')
        return record()
    result = collect_pairs([{'pair_id': i} for i in range(2)], 1, run,
                           lambda c, r: accepted.append(c['pair_id']), log.append)
    assert accepted == [1]
    assert log[0]['reasons'] == ['infeasible_variation:No decoy spot found'] and not log[0]['accepted']
    assert result == dict(complete=True, accepted_pairs=1, attempts=2, rejected_pairs=1)
