"""QD64 / 65: strict A; uncertain intended-contact failures can only be reviewed as B, never auto-trained.

65/QĐ-DL item 7: an intended (allowed) contact is NOT reviewable when its force is abnormal - above the limit of its
class (hand: 140 N, ISO/TS 15066:2016 annex A; gripper: actuator range) - or its penetration is severe
(> B_MAX_DEPTH_M, design value pending approval, conservative). Such episodes are C.
"""
import yaml

from intent_policy.data.episode_validator import validate
from intent_policy.shared.sim_env import ROOT

_C = yaml.safe_load((ROOT / 'configs/hrc/contacts.yaml').read_text())
B_MAX_DEPTH_M = float(_C.get('b_max_depth_m', .005))
HAND_FORCE_N = float(_C['human_force_n'])
GRIP_FORCE_N = 40.


def severe(v):
    depth, force, pair = v.get('max_depth_m', 0.), v.get('max_force_n', 0.), v.get('pair', '')
    if depth > B_MAX_DEPTH_M:
        return True
    if pair.startswith('human_seg_') and force > HAND_FORCE_N:
        return True
    return v.get('rule') == 'robot_grasp' and force > GRIP_FORCE_N


IGNORED = set(_C.get('policy_unobservable_rules', []))


def observable(v):
    """66/QĐ-DL: a violation that reaches the policy / the recorded data (not in the ignored classes)."""
    if v.get('rule') == 'robot_grasp':
        return v.get('max_force_n', 0.) > GRIP_FORCE_N
    return v.get('rule') not in IGNORED


def classify(summary, frames, ticks, review_allowed_contacts=False):
    ok, reasons = validate(summary, frames, ticks)
    if ok:
        return 'A', reasons
    data_summary = dict(summary, success=True, physics_valid=True)
    data_ok, _ = validate(data_summary, frames, ticks)
    failures = [v for v in summary.get('contact_classes', {}).values() if v.get('violation')]
    if (IGNORED and summary['success'] and data_ok and summary.get('robot_limits_ok', False)
            and not any(observable(v) for v in failures)):
        return 'A', reasons + ['66/QĐ-DL: only policy-unobservable contact violations']
    data_summary = dict(summary, success=True, physics_valid=True)
    data_ok, _ = validate(data_summary, frames, ticks)
    failures = [v for v in summary.get('contact_classes', {}).values() if v.get('violation')]
    reviewable = (review_allowed_contacts and summary['success'] and data_ok and failures
                  and summary.get('robot_limits_ok', False)
                  and not summary.get('human_self_violations')
                  and all(v['category'] == 'allowed_contact' and not severe(v) for v in failures))
    return ('B' if reviewable else 'C'), reasons
