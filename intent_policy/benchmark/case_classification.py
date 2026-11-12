"""Checks of the claimed case, before a paired demonstration can be admitted."""


def c3_after_lift(record):
    """The decoy tray's bowl must have been lifted strictly before the person's change of intention."""
    events = record.get('events') or []
    changes = [e for e in events if e.get('event_type') == 'human_intention_change']
    if len(changes) != 1:
        return f'expected_one_change_found_{len(changes)}'
    change = changes[0]
    tray = (change.get('payload') or {}).get('previous')
    timestamp = change.get('timestamp')
    if not tray or timestamp is None:
        return 'missing_decoy_or_change_time'
    target = 'bowl_lifted_' + str(tray)
    lifted = [e.get('timestamp') for e in events
              if e.get('event_type') == 'protocol_step_complete'
              and (e.get('payload') or {}).get('step') == target and e.get('timestamp') is not None]
    if not any(t < timestamp for t in lifted):
        return f'{target}_not_before_change'
    return None


VALIDATORS = {'c3_after_lift': c3_after_lift}
