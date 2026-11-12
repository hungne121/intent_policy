"""Fail-closed decision contract for V2 dataset episodes.

This is a validator for completed measurements, not a geometry checker. Producers
must identify and audit their geometry pairs and swept motion before calling it.
"""
from math import isfinite

REQUIRED_CHECKS = frozenset({
    'human_table', 'human_object', 'human_robot', 'human_self',
    'robot_table', 'robot_object', 'swept_motion', 'task_success',
    'human_motion', 'camera_visibility', 'data_integrity',
})


def evaluate(report):
    errors = []
    if not isinstance(report, dict):
        return ['report must be an object']
    for key in ('episode_id', 'scene_hash', 'motion_seed', 'case_id', 'task_id'):
        if key not in report or report[key] is None or report[key] == '':
            errors.append(f'missing {key}')
    checks = report.get('checks')
    if not isinstance(checks, dict):
        return errors + ['checks must be an object']
    for name in sorted(REQUIRED_CHECKS):
        entry = checks.get(name)
        if not isinstance(entry, dict):
            errors.append(f'{name}: missing check')
            continue
        if entry.get('passed') is not True:
            errors.append(f'{name}: not passed')
        if not isinstance(entry.get('method'), str) or not entry['method'].strip():
            errors.append(f'{name}: missing method')
        if not isinstance(entry.get('evidence'), str) or not entry['evidence'].strip():
            errors.append(f'{name}: missing evidence')
        if name not in {'task_success', 'camera_visibility', 'data_integrity'}:
            n = entry.get('samples')
            if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
                errors.append(f'{name}: missing samples')
    if report.get('geometry_coverage_reviewed') is not True:
        errors.append('geometry pair coverage not reviewed')
    if report.get('unresolved_violations') != []:
        errors.append('unresolved geometry or motion violations')
    return errors
