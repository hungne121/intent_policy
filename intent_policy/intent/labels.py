"""Per-timestep intent labels (docs/requirements/INTENT_ACT_GUIDE.md §1): obj / act / tau / xi.

A segment is {t_onset, t_clear, t_event, obj, act[, t_end]} in timestep (frame) indices, as exported by an
annotation tool (ELAN / CVAT / Label Studio) or derived from simulator ground truth (`segments_from_sim`). `t_end`
(optional, default t_event) extends the obj/act labels over a hold that follows the gesture; phase/tte always run
to t_event. These functions are pure (numpy in, numpy out) and are used by scripts/build_intent_labels.py.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter
from intent_policy.utils import load_yaml

SCHEMA_PATH = 'configs/intent_schema.yaml'
COMPONENTS = ('obj', 'act', 'tau', 'xi')


def load_schema(path=SCHEMA_PATH) -> dict:
    schema = load_yaml(path)
    for vocab in ('obj_vocab', 'act_vocab'):
        if schema[vocab][0] != 'none':
            raise ValueError(f'{vocab} must start with the data label `none`')
    return schema


def label_index(value, vocab) -> int:
    return int(value) if isinstance(value, (int, np.integer)) else vocab.index(value)


def segment_end(seg: dict) -> int:
    return int(seg.get('t_end', seg['t_event']))


def per_timestep_labels(T: int, segments: list[dict], obj_vocab, act_vocab) -> tuple[np.ndarray, np.ndarray]:
    """'none' (index 0) outside every segment; the segment's labels in [t_onset, t_end] (t_end = t_event by
    default). Later segments win where spans touch."""
    obj, act = np.zeros(T, np.int64), np.zeros(T, np.int64)
    for seg in sorted(segments, key=lambda s: s['t_onset']):
        a, b = max(int(seg['t_onset']), 0), min(segment_end(seg), T - 1)
        obj[a:b + 1] = label_index(seg['obj'], obj_vocab)
        act[a:b + 1] = label_index(seg['act'], act_vocab)
    return obj, act


def phase_tte(T: int, segments: list[dict], fps: float, tte_max_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Inside a segment: phase = clip((t - t_onset) / (t_event - t_onset), 0, 1), tte = (t_event - t) / fps.
    Before the first segment: phase = 0, tte = tte_max_s. After t_event (until the next segment): phase = 1,
    tte = 0. tte is capped at tte_max_s."""
    phase, tte = np.zeros(T, np.float32), np.full(T, tte_max_s, np.float32)
    for seg in sorted(segments, key=lambda s: s['t_onset']):
        on, ev = int(seg['t_onset']), int(seg['t_event'])
        t = np.arange(max(on, 0), T)
        phase[t] = np.clip((t - on) / max(ev - on, 1e-9), 0.0, 1.0) if ev > on else 1.0
        tte[t] = np.clip((ev - t) / fps, 0.0, tte_max_s)
    return phase, tte


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
    k_m = round((m + 1) / M * horizon_s * fps)."""
    kp = smooth_keypoints(np.asarray(kp, np.float32), window, order)
    T = kp.shape[0]
    idx = np.minimum(np.arange(T)[:, None] + waypoint_offsets(fps, horizon_s, M)[None], T - 1)   # (T, M)
    return (kp[idx] - kp[:, None]).astype(np.float32)


def episode_labels(T: int, segments: list[dict], kp: np.ndarray, schema: dict) -> dict[str, np.ndarray]:
    """All per-timestep intent arrays of one episode (xi unnormalised)."""
    obj, act = per_timestep_labels(T, segments, schema['obj_vocab'], schema['act_vocab'])
    phase, tte = phase_tte(T, segments, schema['fps'], schema['tte_max_s'])
    sg = schema.get('savgol') or {}
    xi = future_waypoints(kp, schema['fps'], schema['horizon_s'], schema['n_waypoints'], sg.get('window', 7), sg.get('order', 2))
    return {'obj': obj, 'act': act, 'phase': phase, 'tte': tte, 'xi': xi}


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
    """Target key of a motion -> obj label: an object on a slot -> its slot, a T3 cube -> U1/U2 (spot order), a
    zone -> itself, a T4-neg target -> `neg_target_obj`."""
    if key is None:
        return 'none'
    if key in (spec.get('layout') or {}):
        return spec['layout'][key]
    if key in (spec.get('u_blocks') or []):
        return f"U{spec['u_blocks'].index(key) + 1}"
    return (schema.get('neg_target_obj') or {}).get(key, key)


def segments_from_sim(motions: list[dict], stages: list[dict], spec: dict, schema: dict, T: int) -> list[dict]:
    """Intent segments of one simulated episode (see configs/intent_schema.yaml `sim_segments`).

    motions: motions_from_events(...); stages: per-frame human stage run-lengths [{name, start, end}] (inclusive).
    """
    rules = schema['sim_segments']
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
                   obj=location_of(mo['key'], spec, schema), act=rule['act'], source=mo['label'])
        last = segs[-1] if segs else None
        if last and (last['obj'], last['act']) == (seg['obj'], seg['act']) and seg['t_onset'] <= last['t_end'] + 1:
            last.update(t_clear=seg['t_clear'], t_event=seg['t_event'], t_end=seg['t_end'],
                        source=f"{last['source']}+{seg['source']}")
        else:
            segs.append(seg)
    for s in segs:
        s['t_event'] = min(s['t_event'], T - 1)
        s['t_clear'] = min(s['t_clear'], s['t_event'])
    return segs
