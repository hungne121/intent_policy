"""Diagnostic contact allowances based only on the current interaction state."""

def overlap_tolerance(*, robot_body, other_body, other_geom, attached, selection,
                      phase, static_robot, human_pair, tolerances):
    if human_pair:
        return tolerances['human']
    if other_geom == 'table':
        return None if static_robot else tolerances['robot_table']
    intended = other_body == attached or (
        other_body == selection and phase in {'close', 'open', 'return_open'})
    if 'finger' in robot_body and intended:
        return tolerances['finger_grasp']
    return tolerances['robot_other']


def record_overlap(entry, depth, tolerance, time_s):
    """Keep raw depth and contextual violations separately; later grasps cannot erase them."""
    entry['max_depth_m'] = max(entry.get('max_depth_m', 0.), depth)
    entry['samples'] = entry.get('samples', 0) + 1
    entry.setdefault('context_violations', 0)
    if tolerance is not None and depth > tolerance:
        entry['context_violations'] += 1
        entry.setdefault('first_context_violation', {'t_s': time_s, 'depth_m': depth,
                                                    'tolerance_m': tolerance})
