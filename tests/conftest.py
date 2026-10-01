"""Shared fixtures. Run with:  ./run.sh -m pytest tests -q"""
import json

import numpy as np
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from intent_policy.scenarios.scenario_registry import make_scenario, TASK_SCENARIOS


@pytest.fixture(scope='session')
def scenarios():
    """One instance per task scenario T1-T4 (MuJoCo model built once per session)."""
    made = {sid: make_scenario(sid) for sid in TASK_SCENARIOS}
    yield made
    for sc in made.values():
        sc.close()


def ev(t, event_type, **payload):
    """Synthetic event dict (same schema as Event.to_dict)."""
    return dict(episode_id='e', timestamp=float(t), event_type=event_type, scenario_id='s', role='r',
                entity_id='x', payload=payload, seq=0)


@pytest.fixture
def make_event():
    return ev


def hold_agent_steps(scenario, n):
    """Advance a scenario n ticks with a HOLD command (robot does nothing)."""
    for _ in range(n):
        if scenario.step(scenario.env.hold()):
            break
    return scenario


def assert_monotonic(events):
    ts = [e['timestamp'] for e in events]
    assert all(b >= a for a, b in zip(ts, ts[1:])), 'event timestamps must be non-decreasing'
    assert np.isfinite(ts).all()


@pytest.fixture(scope='session')
def labelled_dataset(tmp_path_factory, scenarios):
    """One T1 and one T2 expert episode recorded like scripts/collect_demos.py (keypoints, meta), then labelled by
    scripts/build_intent_labels.py. Returns (root, segments, labels, episode meta)."""
    from intent_policy.benchmark.runner import ExpertAgent, ObservationConfig
    from intent_policy.intent.labels import load_schema
    from scripts.build_intent_labels import build_labels
    from scripts.collect_demos import episode_meta, features, record_episode
    OBS = ObservationConfig(('high', 'wrist'), 48, 36)
    SCHEMA = load_schema()
    root = tmp_path_factory.mktemp('ds') / 'intent'
    kps = SCHEMA['keypoints']
    ds = LeRobotDataset.create(repo_id='local/intent', fps=20, features=features(OBS, None, kps), root=root, use_videos=True)
    meta = []
    for sid, seed, spec in (('t1_pick_place', 2, None),
                            ('t2_handover', 5, {'task': 'T2', 'target': 'C1', 'layout': {'C1': 'S5', 'B2': 'S1', 'B3': 'S3'},
                                                'pair': False, 'hand': 'H1', 'timing': 'early'})):
        sc = scenarios[sid]
        rec, frames, info = record_episode(sc, ExpertAgent(), seed, None, OBS, None, 0, spec, kps)
        assert rec['success']
        for fr in frames:
            ds.add_frame({**fr, 'task': sc.cfg.task})
        ds.save_episode()
        meta.append(episode_meta(len(meta), sid, 'test', seed, sc, rec, frames, info))
    ds.finalize()
    (root / 'meta/hri_episodes.jsonl').write_text('\n'.join(json.dumps(m, default=str) for m in meta) + '\n')
    segments, labels, _ = build_labels(root)
    return root, segments, labels, meta
