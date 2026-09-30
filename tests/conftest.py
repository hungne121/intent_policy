"""Shared fixtures. Run with:  ./run.sh -m pytest tests -q"""
import numpy as np
import pytest

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
