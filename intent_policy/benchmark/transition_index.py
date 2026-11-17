"""Transition index for transition-aware importance sampling of ACT training samples (docs/PIF.md §8.13).

ManiCast (Kedia et al. 2023, §4.1, "Strategy 1: Importance Sampling") trains its forecaster on the mixture
Q = 0.5 P + 0.5 P_T, P the data distribution and P_T the data restricted to transitions. Here the same mixture over
ACT training samples (one sample = the action chunk starting at a frame); the loss is unchanged. This transfers their
principle to imitation learning; its effect on ACT is what D0 / D1 measure, not an established result.

Transitions, from the episode metadata the simulator logs (meta/hri_episodes.jsonl; never an input of ACT):
  cue       the human starts showing a need: human_cue_onset_t, and the starts of the scenario's cue motions
            (settings cue_labels: one cue per round of the multi-round tasks)
  change    the human changes goal: intention_changes
  fault     a controlled forecast fault starts (forecast_fault.start_s; early mode only)
  commit    the expert commits to a target (expert_commits)
  recovery  an expert commit to another target after a change / fault since the previous commit
A sample starting at frame f is a transition sample when some transition k lies in [f - post, f + pre]: the chunk
starts at most `pre_s` before it (= the forecast horizon: the forecast at f covers k) and at most `post_s` after.
Leak check: a transition sample is dropped when its chunk ([f, f + chunk_s]) holds a robot transition (commit /
recovery) whose cause (the latest human transition: cue / change / fault, at or before it) lies beyond f + pre_s:
neither the observation nor the forecast at f shows it.
"""
from __future__ import annotations
import numpy as np

HUMAN, ROBOT = ('cue', 'change', 'fault'), ('commit', 'recovery')


def load_settings(path: str) -> dict:
    from intent_policy.utils import load_yaml
    s = dict(load_yaml(path)['transition_sampling'])
    s['cue_labels'] = {k: list(v) for k, v in (s.get('cue_labels') or {}).items()}
    return s


def transitions(m: dict, fps: float, s: dict) -> list[tuple[str, int]]:
    """(kind, frame) of the episode's transitions, sorted by frame."""
    out = []
    if m.get('human_cue_onset_t') is not None:
        out.append(('cue', int(round(float(m['human_cue_onset_t']) * fps))))
    cue = set(s['cue_labels'].get(m['scenario_id'], []))
    out += [('cue', int(x['start'])) for x in m.get('human_motions') or [] if x['label'] in cue]
    out += [('change', int(x['frame'])) for x in m.get('intention_changes') or [] if x.get('frame') is not None]
    ff = m.get('forecast_fault')
    if ff and m.get('mode') != 'observed' and ff.get('start_s') is not None:
        out.append(('fault', int(round(float(ff['start_s']) * fps))))
    prev = None
    for c in sorted((c for c in m.get('expert_commits') or [] if c.get('frame') is not None), key=lambda c: c['frame']):
        f = int(c['frame'])
        upset = prev is not None and any(k in ('change', 'fault') and prev[1] < g <= f for k, g in out)
        out.append(('recovery' if upset and c['target'] != prev[0] else 'commit', f))
        prev = (c['target'], f)
    return sorted(set(out), key=lambda x: (x[1], x[0]))


def index_episode(m: dict, fps: float, s: dict) -> dict:
    """Transition-sample mask over the episode's frames (sample = chunk start), with the leak check."""
    n = int(m['n_frames'])
    pre, post, chunk = (int(round(float(s[k]) * fps)) for k in ('pre_s', 'post_s', 'chunk_s'))
    tr = transitions(m, fps, s)
    mask = np.zeros(n, bool)
    for _, k in tr:
        mask[max(0, k - pre):min(n, k + post + 1)] = True
    human = [k for kind, k in tr if kind in HUMAN]
    robot = [k for kind, k in tr if kind in ROBOT]
    dropped = 0
    for f in np.flatnonzero(mask):
        for k in robot:
            if f < k <= f + chunk:
                cause = max((h for h in human if h <= k), default=None)
                if cause is not None and cause > f + pre:
                    mask[f] = False
                    dropped += 1
                    break
    return dict(transitions=tr, mask=mask, dropped_leak=dropped)


def mixture_weights(in_t: np.ndarray, mix: float) -> np.ndarray:
    """Per-sample probabilities of Q = (1 - mix) P + mix P_T over the training samples (P uniform, P_T uniform over
    the transition samples); mix = 0 gives uniform sampling (D0)."""
    n, t = len(in_t), int(np.sum(in_t))
    w = np.full(n, (1.0 - mix) / n)
    if t and mix > 0:
        w[np.asarray(in_t, bool)] += mix / t
    return w / w.sum()
