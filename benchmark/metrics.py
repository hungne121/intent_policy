"""HRIBench-style metrics computed only from logged benchmark events.

These are *HRIBench-style* definitions written for this project; the exact formulas of the
HRIBench paper have not been verified, so results must not be reported as an exact
reproduction. Metrics never import or inspect policy code.

Each metric produces an episode value (float/bool or None = not observable in that episode)
and aggregates episode values by mean over the non-None values. There is no composite score.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

ALL_METRICS = ('CSR', 'CT', 'IR', 'TSync', 'Rsp', 'OC', 'CFR', 'HCS', 'CIR', 'DSR',
               'AM', 'WCR', 'WCD', 'CRsp')      # Phase 2 anticipation metrics (robot_target_commit events)

# Default per-role trigger/response definitions; scenario configs may override via `metric_params`.
DEFAULT_PARAMS = {
    'rsp_trigger': {'event_type': 'human_cue_onset'},
    'rsp_response': 'robot_motion_start',
    'tsync_human': {'event_type': 'interaction_window_start'},
    'tsync_robot': None,
    'ir_start': {'event_type': 'human_cue_onset'},
    'am_cue_complete': None,          # event that ends the human cue (e.g. a protocol step)
    'commit_stage': None,             # robot_target_commit stage the cue is about (e.g. 'object')
}


def _t(e: dict) -> float:
    return float(e['timestamp'])


def find_first(events: list[dict], spec: dict | None, after: float = -np.inf) -> dict | None:
    """First event matching {'event_type': ..., optional 'step': ..., optional payload keys}."""
    if spec is None:
        return None
    for e in events:
        if e['event_type'] != spec['event_type'] or _t(e) < after:
            continue
        if all(e['payload'].get(k) == v for k, v in spec.items() if k != 'event_type'):
            return e
    return None


def motion_intervals(events: list[dict]) -> list[tuple[float, float]]:
    """Robot motion intervals from robot_motion_start/end (end uses the `stopped_at` payload)."""
    intervals, start = [], None
    for e in events:
        if e['event_type'] == 'robot_motion_start' and start is None:
            start = _t(e)
        elif e['event_type'] == 'robot_motion_end' and start is not None:
            intervals.append((start, float(e['payload'].get('stopped_at', _t(e)))))
            start = None
    if start is not None:
        end = [e for e in events if e['event_type'] == 'episode_end']
        intervals.append((start, _t(end[-1]) if end else start))
    return intervals


def moving_at(intervals, t: float) -> bool:
    return any(a <= t < b for a, b in intervals)


def episode_end_time(events):
    for e in events:
        if e['event_type'] == 'task_success':
            return _t(e)
    return _t(events[-1])


@dataclass
class EpisodeContext:
    protocol_steps: list[str]
    params: dict = field(default_factory=dict)

    def param(self, key):
        return self.params.get(key, DEFAULT_PARAMS.get(key))


class BenchmarkMetric:
    name = ''
    higher_is_better = True

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.values: list = []

    def episode_value(self, events: list[dict], ctx: EpisodeContext):
        raise NotImplementedError

    def update(self, events: list[dict], ctx: EpisodeContext):
        v = self.episode_value(events, ctx)
        self.values.append(v)
        return v

    def compute(self) -> dict:
        vals = [float(v) for v in self.values if v is not None]
        return dict(metric=self.name, value=float(np.mean(vals)) if vals else None,
                    std=float(np.std(vals)) if vals else None, n=len(vals), n_episodes=len(self.values))


class CSR(BenchmarkMetric):
    """Collaboration Success Rate: fraction of episodes with task_success."""
    name = 'CSR'

    def episode_value(self, events, ctx):
        return float(any(e['event_type'] == 'task_success' for e in events))


class CT(BenchmarkMetric):
    """Completion Time [s]: episode_start -> task_success, successful episodes only."""
    name, higher_is_better = 'CT', False

    def episode_value(self, events, ctx):
        s = find_first(events, {'event_type': 'task_success'})
        return None if s is None else _t(s) - _t(events[0])


class IR(BenchmarkMetric):
    """Idle Ratio: fraction of [task start, task end] in which the robot is not moving.

    Task start = first `ir_start` event (default human_cue_onset; episode_start if absent),
    task end = task_success or episode end.
    """
    name, higher_is_better = 'IR', False

    def episode_value(self, events, ctx):
        start_e = find_first(events, ctx.param('ir_start'))
        t0 = _t(start_e) if start_e else _t(events[0])
        t1 = episode_end_time(events)
        if t1 <= t0:
            return None
        busy = sum(max(0.0, min(b, t1) - max(a, t0)) for a, b in motion_intervals(events))
        return 1.0 - busy / (t1 - t0)


class TSync(BenchmarkMetric):
    """Temporal Synchronisation [s]: |t(robot ready) - t(human ready)| for the interaction."""
    name, higher_is_better = 'TSync', False

    def episode_value(self, events, ctx):
        h = find_first(events, ctx.param('tsync_human'))
        r = find_first(events, ctx.param('tsync_robot'))
        return None if h is None or r is None else abs(_t(r) - _t(h))


class Rsp(BenchmarkMetric):
    """Response latency [s]: trigger event -> first response event of the robot.

    response 'robot_motion_start': 0 if already moving at the trigger.
    response 'robot_motion_end': robot stop time (stopped_at); 0 if already stationary.
    """
    name, higher_is_better = 'Rsp', False

    def episode_value(self, events, ctx):
        trig = find_first(events, ctx.param('rsp_trigger'))
        if trig is None:
            return None
        t0 = _t(trig)
        intervals = motion_intervals(events)
        response = ctx.param('rsp_response')
        if response == 'robot_motion_start':
            if moving_at(intervals, t0):
                return 0.0
            nxt = [a for a, _ in intervals if a >= t0]
        elif response == 'robot_motion_end':
            if not moving_at(intervals, t0):
                return 0.0
            nxt = [b for a, b in intervals if a <= t0 < b]
        else:
            raise ValueError(response)
        return (min(nxt) - t0) if nxt else None


class OC(BenchmarkMetric):
    """Order Compliance: fraction of required protocol steps completed in the required order."""
    name = 'OC'

    def episode_value(self, events, ctx):
        required = ctx.protocol_steps
        if not required:
            return None
        done = [e['payload'].get('step') for e in events if e['event_type'] == 'protocol_step_complete']
        done = [s for s in done if s in required]
        k = 0
        for s in done:
            if k < len(required) and s == required[k]:
                k += 1
            else:
                break     # out-of-order step: compliance stops counting here
        return k / len(required)


class CFR(BenchmarkMetric):
    """Collision-Free Rate: episode has no human_robot_contact."""
    name = 'CFR'

    def episode_value(self, events, ctx):
        return float(not any(e['event_type'] == 'human_robot_contact' for e in events))


class HCS(BenchmarkMetric):
    """Human Contact Safety: no contact AND no safety_distance_violation in the episode."""
    name = 'HCS'

    def episode_value(self, events, ctx):
        bad = {'human_robot_contact', 'safety_distance_violation'}
        return float(not any(e['event_type'] in bad for e in events))


class CIR(BenchmarkMetric):
    """Contradictory Instruction Recognition: contradictory human_intention_change events followed
    by a protocol step `contradiction_recognized`. None when no contradiction occurred."""
    name = 'CIR'

    def episode_value(self, events, ctx):
        contradictions = [e for e in events if e['event_type'] == 'human_intention_change'
                          and e['payload'].get('contradictory')]
        if not contradictions:
            return None
        hits = sum(find_first(events, {'event_type': 'protocol_step_complete', 'step': 'contradiction_recognized'},
                              after=_t(c)) is not None for c in contradictions)
        return hits / len(contradictions)


class DSR(BenchmarkMetric):
    """Disruption Success Rate: disrupted episodes that recover and succeed without contact."""
    name = 'DSR'

    def episode_value(self, events, ctx):
        types = [e['event_type'] for e in events]
        if 'disruption_start' not in types:
            return None
        return float('recovery_complete' in types and 'task_success' in types and 'human_robot_contact' not in types)


def commits(events: list[dict], stage: str | None = None) -> list[dict]:
    return [e for e in events if e['event_type'] == 'robot_target_commit'
            and (stage is None or e['payload'].get('stage') == stage)]


class AM(BenchmarkMetric):
    """Anticipation Margin [s]: t(cue complete) - t(first correct robot_target_commit of the cue's stage).

    Positive = the robot headed for the correct target before the human finished the cue. None when
    the robot never committed correctly or the cue never completed.
    """
    name = 'AM'

    def episode_value(self, events, ctx):
        cue = find_first(events, ctx.param('am_cue_complete'))
        first = next((e for e in commits(events, ctx.param('commit_stage')) if e['payload'].get('correct')), None)
        return None if cue is None or first is None else _t(cue) - _t(first)


def wrong_commit_intervals(events: list[dict]) -> list[tuple[float, float]]:
    """Intervals during which the robot was committed to a target that was wrong when it committed.

    An interval ends at the next commit, the next robot object_grasp or the end of the episode.
    """
    ends = sorted(_t(e) for e in events if e['event_type'] in ('robot_target_commit', 'episode_end')
                  or (e['event_type'] == 'object_grasp' and e['entity_id'] == 'robot'))
    out = []
    for e in commits(events):
        if not e['payload'].get('correct'):
            t0 = _t(e)
            out.append((t0, next((t for t in ends if t > t0), t0)))
    return out


class WCR(BenchmarkMetric):
    """Wrong-Commitment Rate: 1 if the robot committed to a wrong target at least once."""
    name, higher_is_better = 'WCR', False

    def episode_value(self, events, ctx):
        return float(bool(wrong_commit_intervals(events)))


class WCD(BenchmarkMetric):
    """Wrong-Commitment Duration [s]: total time spent committed to wrong targets (0 if never)."""
    name, higher_is_better = 'WCD', False

    def episode_value(self, events, ctx):
        return float(sum(b - a for a, b in wrong_commit_intervals(events)))


class CRsp(BenchmarkMetric):
    """Change Response [s]: human_intention_change -> first correct robot_target_commit afterwards.

    None when the human did not change their mind (or the robot never re-committed).
    """
    name, higher_is_better = 'CRsp', False

    def episode_value(self, events, ctx):
        change = find_first(events, {'event_type': 'human_intention_change'})
        if change is None:
            return None
        t0 = _t(change)
        nxt = [e for e in commits(events) if _t(e) >= t0 and e['payload'].get('correct')]   # correct w.r.t. the new target
        return (_t(nxt[0]) - t0) if nxt else None


METRIC_CLASSES = {c.name: c for c in (CSR, CT, IR, TSync, Rsp, OC, CFR, HCS, CIR, DSR, AM, WCR, WCD, CRsp)}


def compute_episode_metrics(events: list[dict], applicable: list[str], protocol_steps: list[str],
                            params: dict | None = None) -> dict:
    """Episode values for applicable metrics; non-applicable metrics are reported as 'n/a'."""
    ctx = EpisodeContext(list(protocol_steps), dict(params or {}))
    return {name: (METRIC_CLASSES[name]().update(events, ctx) if name in applicable else 'n/a') for name in ALL_METRICS}


def aggregate(episode_metrics: list[dict]) -> dict:
    """Mean/std/n per metric over episodes; 'n/a' entries are ignored, all-'n/a' stays 'n/a'."""
    out = {}
    for name in ALL_METRICS:
        vals = [m[name] for m in episode_metrics if m.get(name) != 'n/a']
        if not vals:
            out[name] = 'n/a'
            continue
        valid = [float(v) for v in vals if v is not None]
        out[name] = dict(value=float(np.mean(valid)) if valid else None, std=float(np.std(valid)) if valid else None,
                         n=len(valid), n_episodes=len(vals))
    return out
