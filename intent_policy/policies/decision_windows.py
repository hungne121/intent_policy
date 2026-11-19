"""Decision windows of M* (19/QĐ-HL III.3): m_t = 1 for the frames inside a window around a decision event.

Events come from the episode sidecar (meta/hrc_episodes.jsonl `events`, `fault_start`, `fault_end`), never from the
policy input. Windows are configured in configs/policy/m_star.yaml `decision_loss.event_windows`:
    {<kind>: {events: [event names], t_before: s, t_after: s}}   a window [t_event - t_before, t_event + t_after]
    {fault_recovery: {t_after: s}}                                [fault_start, fault_end + t_after] of N5 episodes
Event names / spans follow the decisive windows of 17/QĐ-HL P1 (docs/reports/M1_P1_OVERFIT_2026-11-19.md §1) and the
span of 19/QĐ-HL III.3 (onset +1 s, switch +-1 s).
"""
import numpy as np


def episode_windows(meta: dict, windows: dict) -> list[tuple[float, float, str]]:
    out = []
    for kind, w in windows.items():
        if kind == 'fault_recovery':
            if meta.get('fault_start') is not None and meta.get('fault_end') is not None:
                out.append((float(meta['fault_start']), float(meta['fault_end']) + float(w.get('t_after', 1.0)), kind))
            continue
        names = set(w['events'])
        for ev in meta.get('events', []):
            if ev['event'] in names or (ev['event'] == 'phase' and 'phase_basis_forecast' in names and ev.get('basis') == 'forecast'):
                out.append((float(ev['t']) - float(w.get('t_before', 0.0)), float(ev['t']) + float(w.get('t_after', 0.0)), kind))
    return out


def decision_mask(ep_index: np.ndarray, timestamp: np.ndarray, ep_meta: dict, windows: dict) -> tuple[np.ndarray, dict]:
    """(N,) bool mask over dataset rows (episode index, time [s]) and the share of masked frames per task."""
    mask = np.zeros(len(ep_index), bool)
    for e in np.unique(ep_index):
        rows = ep_index == e
        t = timestamp[rows]
        m = np.zeros(rows.sum(), bool)
        for lo, hi, _ in episode_windows(ep_meta[int(e)], windows):
            m |= (t >= lo - 1e-6) & (t <= hi + 1e-6)
        mask[rows] = m
    task = np.array([ep_meta[int(e)]['task_name'] for e in ep_index])
    share = {str(k): round(float(mask[task == k].mean()), 4) for k in np.unique(task)}
    share['all'] = round(float(mask.mean()), 4)
    return mask, share
