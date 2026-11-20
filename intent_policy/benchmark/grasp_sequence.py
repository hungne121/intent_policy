"""I3 grasp-sequence rule with optional retry tolerance (31/QĐ-HL). Pure functions, unit-tested (tests/test_grasp_sequence.py).

v1 (old): the sequence of `robot_grasp` events must equal [target] (nominal, switch_before_commit) or be [A, target] / [target]
(switch_after_commit). v2 (retry_ok=True): consecutive repeats of the same object are collapsed first. Only the sequence rule
changes; everything else of scenarios.i3.check is untouched."""


def collapse_retries(grasps):
    out = []
    for g in grasps:
        if not out or out[-1] != g:
            out.append(g)
    return out


def sequence_reason(variant, grasps, a, target, retry_ok=False):
    """None if the sequence is acceptable, else the reason text (identical to the v1 wording when retry_ok is False)."""
    seq = collapse_retries(grasps) if retry_ok else list(grasps)
    if variant == 'nominal' and seq != [target]:
        return f'grasp sequence {grasps}'
    if variant == 'switch_before_commit' and seq != [target]:
        return f'grasp sequence {grasps} (A must not be grasped)'
    if variant == 'switch_after_commit' and seq not in ([a, target], [target]):
        return f'grasp sequence {grasps} (A then B, or only B, expected)'
    return None
