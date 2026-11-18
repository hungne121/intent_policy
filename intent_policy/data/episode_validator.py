"""Accept / reject rule (QĐ 53/QĐ-DL §8). The same rule for observed and forecast episodes.

Accepted only if: the task succeeded; physics valid (no human penetration beyond tolerance, no human
self-intersection, robot within joint limits / speed, no grasp drop); every tick produced exactly one frame;
in forecast mode the forecast handed to the expert was valid whenever used; in observed mode no forecast was
handed to the expert. Episodes with a wrong forecast that the expert recovered from are NOT rejected for that.

59/QĐ-DL N5: a fault episode is kept only if the fault was really applied to a forecast the expert received, and
its effective forecast passed the continuity / joint-geometry / start-error checks. N1: the varied robot start
pose must be reached by the IK (error <= 5 mm).

"""


def validate(summary, frames_recorded, ticks):
    reasons = []
    if frames_recorded <= 0 or ticks <= 0:
        reasons.append('empty episode')
    if not summary['success']:
        reasons.append(f"task: {summary.get('failure_reason')}")
    if not summary['physics_valid']:
        reasons.append('physics_valid=false')
        if summary['contact_violations']:
            reasons.append('contacts: ' + ', '.join(sorted(summary['contact_violations'])[:4]))
        if summary['human_self_violations']:
            reasons.append(f"human self-intersection ({len(summary['human_self_violations'])} pairs)")
        if not summary['robot_limits_ok']:
            reasons.append(f"robot limits {summary['robot_limits']}")
    if frames_recorded != ticks:
        reasons.append(f'frames {frames_recorded} != ticks {ticks}')
    fv = summary['forecast_validity_summary']
    if summary['mode'] == 'observed' and fv['used_by_expert'] != 0:
        reasons.append('forecast handed to the observed expert')
    fault = summary.get('fault')
    if fault is not None and summary['mode'] == 'forecast':
        if not fault['fault_applied']:
            reasons.append('fault_not_applied (trigger onset never reached while a forecast was valid)')
        if fault['violations']:
            reasons.append(f"fault_checks {fault['violations'][0]['checks']} at t={fault['violations'][0]['t']}")
    start = (summary.get('variation_log') or {}).get('robot_start')
    if start is not None and (start.get('failed') or start['ik_error_m'] > .005):
        reasons.append(f"robot start pose invalid after {len(start.get('attempts', [1]))} draws "
                       f"(IK error {start['ik_error_m']:.4f} m)")
    return not reasons, reasons
