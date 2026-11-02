"""Quota accounting for paired early/observed demonstrations, not policy evaluation.

The runner must supply a shared exogenous condition hash and a completed quality
report. Only accepted pairs reach the writer. The journal includes both sides of
every attempt, including failures. Exhaustion is explicit, never called complete.
A seed whose variation cannot be sampled (InfeasibleVariation) is journaled as a rejected attempt.
"""
from intent_policy.scenarios.config import InfeasibleVariation


class IncompleteQuota(RuntimeError):
    def __init__(self, accepted, requested, attempts):
        self.accepted, self.requested, self.attempts = accepted, requested, attempts
        super().__init__(f'Only {accepted}/{requested} valid pairs after {attempts} attempts')


def collect_pairs(candidates, quota, run, write_pair, write_attempt, *, require_motion_match=False):
    if not isinstance(quota, int) or quota <= 0:
        raise ValueError('quota must be a positive integer')
    accepted = attempts = 0
    seen = set()
    for candidate in candidates:
        key = candidate['pair_id']
        if key in seen:
            raise ValueError(f'Duplicate candidate pair_id {key}')
        seen.add(key)
        try:
            records = {mode: run(mode, candidate) for mode in ('early', 'observed')}
        except InfeasibleVariation as exc:      # the seed defines no episode (I1 seed 3020060, 2026-10-31)
            attempts += 1
            write_attempt(dict(pair_id=key, accepted=False, reasons=[f'infeasible_variation:{exc}'], sides={}))
            continue
        attempts += 1
        reasons = []
        for mode, r in records.items():
            if not r.get('success'):
                reasons.append(f'{mode}:task_failed')
            if (r.get('collection_quality') or {}).get('passed') is not True:
                reasons.append(f'{mode}:quality_not_passed')
        hashes = [r.get('exogenous_condition_sha256') for r in records.values()]
        if not all(hashes) or hashes[0] != hashes[1]:
            reasons.append('unmatched_exogenous_conditions')
        if require_motion_match:
            from intent_policy.benchmark.collection_integrity import motion_source_hash
            sources = [motion_source_hash(r) for r in records.values()]
            if not all(sources) or sources[0] != sources[1]:
                reasons.append('unmatched_motion_sources')
            for r, h in zip(records.values(), sources):
                r['motion_source_sha256'] = h
        entry = dict(pair_id=key, accepted=False, reasons=reasons,
            sides={mode: {k: r.get(k) for k in ('success', 'failure', 'collection_quality', 'exogenous_condition_sha256', 'motion_source_sha256')}
                   for mode, r in records.items()})
        if reasons:
            write_attempt(entry)
            continue
        try:
            write_pair(candidate, records)
        except Exception as exc:
            entry['reasons'] = ['pair_write_failed:' + type(exc).__name__]
            write_attempt(entry)
            raise
        entry['accepted'] = True
        write_attempt(entry)
        accepted += 1
        if accepted == quota:
            return dict(complete=True, accepted_pairs=accepted, attempts=attempts, rejected_pairs=attempts-accepted)
    raise IncompleteQuota(accepted, quota, attempts)
