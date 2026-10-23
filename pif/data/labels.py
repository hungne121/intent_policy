"""Evaluation-only labels (handoff §3, §0.4): never returned by the training dataset unless
`cfg.supervised_warmstart` is set (pif/data/dataset.py).

Sim labels come from meta/hri_episodes.jsonl; a target key is mapped to an object token index of the episode
(pif/data/episode_reader.py order) or to a code when it has no token (CORRECTIONS #14):
  TARGET_NONE   -1  no target (return_rest, withdraw, ...)
  TARGET_HUMAN  -2  the human hand (H1 / H2 handover zones: the target comes from the human tokens)
  TARGET_REGION -3  a table region without a visible marker (T4 robot_zone / zone_edge)
Frame indices are at the PIF rate: an event at source frame f is at the first PIF frame i with i * stride >= f.
"""
from __future__ import annotations
import math

import numpy as np

TARGET_NONE, TARGET_HUMAN, TARGET_REGION = -1, -2, -3


def key_to_target(key, obj_names: list[str], cfg: dict) -> int:
    if key is None:
        return TARGET_NONE
    key = cfg['sim']['surface_alias'].get(key, key)
    if key in obj_names:
        return obj_names.index(key)
    if key in cfg['sim']['human_targets']:
        return TARGET_HUMAN
    if key in cfg['sim']['region_targets']:
        return TARGET_REGION
    raise KeyError(f'target {key!r} has no token or code')


def sim_labels(meta: dict, ep: dict, cfg: dict) -> dict:
    """gestures: [{label, key, target, onset, clear}] (clear = None when the motion runs to the episode end);
    intent_object / intent_dest [T]: the human's intended object / destination per frame (human_target_stages, T5
    changes included); steps: [{step, frame, by}]; task, role."""
    stride = ep['meta']['stride']
    T = len(ep['holding'])
    at = lambda f: None if f is None else min(math.ceil(f / stride), T - 1)
    names = ep['obj_names']
    gestures = [dict(label=m['label'], key=m['key'], target=key_to_target(m['key'], names, cfg),
                     onset=at(m['start']), clear=at(m['end'])) for m in meta['human_motions']]
    obj = np.full(T, TARGET_NONE)
    dest = np.full(T, TARGET_NONE)
    for s in meta['human_target_stages']:
        o, _, d = s['name'].partition(' -> ')
        a, b = at(s['start']), at(s['end'])
        obj[a:b + 1] = key_to_target(o if o != 'none' else None, names, cfg)
        dest[a:b + 1] = key_to_target(d or None, names, cfg)
    steps = [dict(step=s['step'], frame=at(s['frame']), by=s['by']) for s in meta['protocol_steps']]
    return dict(gestures=gestures, intent_object=obj, intent_dest=dest, steps=steps, task=meta['list_task'],
                task_code=meta['task_code'], role=meta['role'])
