"""Ground-truth intent timeline of simulated episodes and keypoint helpers (INTENT_ACT_GUIDE_v2.md §3.1).

A segment is {t_onset, t_clear, t_event, t_end, target, gesture, who} in frame indices (configs/intent_schema.yaml
`sim_segments`), plus t_evident (the sim's evidence point, prepare -> stroke) and t_retract (end of the withdrawal
that follows, if any). Segments come from simulator ground truth (`segments_from_sim`) or an annotation file in the
same format. Pure functions (numpy in, numpy out).
"""
from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter
from intent_policy.utils import load_yaml

SCHEMA_PATH = 'configs/intent_schema.yaml'


def load_schema(path=SCHEMA_PATH) -> dict:
    schema = load_yaml(path)
    for vocab in ('targets', 'who'):
        if schema[vocab][0] != 'none':
            raise ValueError(f'{vocab} must start with `none`')
    return schema


def segment_end(seg: dict) -> int:
    return int(seg.get('t_end', seg['t_event']))


def hand_indices(schema: dict) -> dict[str, list[int]]:
    """Keypoint indices of each hand: {'r': [...], 'l': [...]}."""
    out = {}
    for i, k in enumerate(schema['keypoints']):
        out.setdefault(k.split('_', 1)[0], []).append(i)
    return out


def smooth_keypoints(kp: np.ndarray, window: int = 7, order: int = 2) -> np.ndarray:
    """Savitzky-Golay along time (window shrunk to the episode length if needed)."""
    T = kp.shape[0]
    w = min(window, T if T % 2 else T - 1)
    if w <= order:
        return kp.astype(np.float32)
    return savgol_filter(kp, w, order, axis=0).astype(np.float32)


def waypoint_offsets(fps: float, horizon_s: float, M: int) -> np.ndarray:
    return np.array([round((m + 1) / M * horizon_s * fps) for m in range(M)], int)


def future_waypoints(kp: np.ndarray, fps: float, horizon_s: float, M: int, window: int = 7, order: int = 2) -> np.ndarray:
    """kp (T, J, D) -> xi (T, M, J, D): xi[t, m] = kp[min(t + k_m, T-1)] - kp[t] on the smoothed keypoints,
    k_m = round((m + 1) / M * horizon_s * fps). Uses the future: hindsight only."""
    kp = smooth_keypoints(np.asarray(kp, np.float32), window, order)
    T = kp.shape[0]
    idx = np.minimum(np.arange(T)[:, None] + waypoint_offsets(fps, horizon_s, M)[None], T - 1)   # (T, M)
    return (kp[idx] - kp[:, None]).astype(np.float32)


# ---------------------------------------------------------------------- simulator ground truth -> segments
def event_frame(t: float, fps: float) -> int:
    """First recorded frame whose observation reflects a human update at time t (frame k is observed before the
    update at k / fps, scripts/collect_demos.py)."""
    return int(round(t * fps)) + 1


def motions_from_events(events: list[dict], fps: float) -> list[dict]:
    """human_motion_start / _end events -> [{label, key, start, end, stopped}] (frames). A motion replaced by a
    new one before it ended ends at the new one's start."""
    out, open_ = [], None
    for e in events:
        if e['event_type'] == 'human_motion_start':
            if open_ is not None:
                open_['end'] = event_frame(e['timestamp'], fps)
                out.append(open_)
            p = e['payload']
            open_ = dict(label=p['label'], key=p.get('target_key'), start=event_frame(e['timestamp'], fps), end=None,
                         stopped=False)
        elif e['event_type'] == 'human_motion_end' and open_ is not None:
            open_['end'] = event_frame(e['timestamp'], fps)
            open_['stopped'] = bool(e['payload'].get('stopped_for_clearance', False))
            out.append(open_)
            open_ = None
    if open_ is not None:
        out.append(open_)
    return out


def location_of(key: str | None, spec: dict, schema: dict) -> str:
    """Target key of a motion -> target place: an object on a slot -> its slot, a T3 cube -> U1/U2 (spot order), a
    zone -> itself, a T4-neg target -> `neg_target_obj`."""
    if key is None:
        return 'none'
    if key in (spec.get('layout') or {}):
        return spec['layout'][key]
    if key in (spec.get('u_blocks') or []):
        return f"U{spec['u_blocks'].index(key) + 1}"
    return (schema.get('neg_target_obj') or {}).get(key, key)


def segments_from_sim(motions: list[dict], stages: list[dict], spec: dict, schema: dict, T: int) -> list[dict]:
    """Intent segments of one simulated episode (configs/intent_schema.yaml `sim_segments`).

    motions: motions_from_events(...); stages: per-frame human stage run-lengths [{name, start, end}] (inclusive).
    """
    rules, frac = schema['sim_segments'], float(schema.get('evidence_fraction', 0.4))
    segs = []
    for mo in motions:
        rule = rules.get(mo['label'])
        if rule is None:
            continue
        i = next((k for k, s in enumerate(stages) if s['name'] == mo['label'] and s['start'] <= mo['start'] + 1
                  and s['end'] >= mo['start']), None)
        if i is None:                                   # the motion never showed up in a recorded frame
            continue
        on = max(min(mo['start'], stages[i]['end']), stages[i]['start'])
        event = stages[i]['end']
        clear = event if mo['end'] is None else int(np.clip(mo['end'], on, event))
        end = event
        for s in stages[i + 1:]:
            if s['name'] not in (rule.get('hold') or []):
                break
            end = s['end']
        seg = dict(t_onset=int(on), t_clear=int(clear), t_event=int(event), t_end=int(min(end, T - 1)),
                   target=location_of(mo['key'], spec, schema), gesture=rule['gesture'], who=rule['who'],
                   source=mo['label'])
        last = segs[-1] if segs else None
        if last and (last['target'], last['gesture']) == (seg['target'], seg['gesture']) and seg['t_onset'] <= last['t_end'] + 1:
            last.update(t_clear=seg['t_clear'], t_event=seg['t_event'], t_end=seg['t_end'],
                        source=f"{last['source']}+{seg['source']}")
        else:
            segs.append(seg)
    retract = [m for m in motions if m['label'] in (schema.get('retract_labels') or [])]
    for k, s in enumerate(segs):
        s['t_event'] = min(s['t_event'], T - 1)
        s['t_clear'] = min(s['t_clear'], s['t_event'])
        s['t_evident'] = int(min(s['t_onset'] + round(frac * (s['t_clear'] - s['t_onset'])), s['t_clear']))
        until = segs[k + 1]['t_onset'] if k + 1 < len(segs) else T      # withdrawal before the next gesture
        nxt = next((m for m in retract if s['t_end'] <= m['start'] < until), None)
        if nxt is None:
            s['t_retract_start'] = s['t_retract'] = None
        else:
            end = T - 1 if nxt['end'] is None else nxt['end']
            s['t_retract_start'], s['t_retract'] = int(nxt['start']), int(min(end, until - 1, T - 1))
    return segs
