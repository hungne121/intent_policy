"""Map scenario ids to environment implementations. Selects environments, never policies.

Tasks of docs/requirements/scence_construct.md §4.4: T1-T4 are scenarios; T5 (change of mind) is T1-T3 with a
`change` spec and T4-neg is T4 with `negative: true` (scenario lists, scripts/generate_scenarios.py).
"""
from pathlib import Path
from intent_policy.scenarios.config import ScenarioConfig
from intent_policy.scenarios.t1_pick_place import PickPlaceTask
from intent_policy.scenarios.t2_handover import HandoverTask
from intent_policy.scenarios.t3_assist import AssistTask
from intent_policy.scenarios.t4_interrupt import InterruptTask

SCENARIOS = {
    't1_pick_place': PickPlaceTask,
    't2_handover': HandoverTask,
    't3_assist': AssistTask,
    't4_interrupt': InterruptTask,
}
TASK_SCENARIOS = tuple(SCENARIOS)
SCENARIO_OF_TASK = {'T1': 't1_pick_place', 'T2': 't2_handover', 'T3': 't3_assist', 'T4': 't4_interrupt'}


def make_scenario(config: str | Path | ScenarioConfig | dict, overrides: dict | None = None, **kwargs):
    """`overrides` are deep-merged into a scenario loaded by id/path (the merged config is what records store)."""
    if isinstance(config, dict):
        config = ScenarioConfig.from_dict(config)
    elif not isinstance(config, ScenarioConfig):
        config = ScenarioConfig.load(config, overrides)
    return SCENARIOS[config.id](config, **kwargs)


def scenario_overrides(exp: dict, scenario_id: str, extra: dict | None = None) -> dict:
    """Experiment-level scenario overrides: `scenario_overrides.all`, then `scenario_overrides.<id>`, then `extra`."""
    from intent_policy.scenarios.config import _deep_merge
    so = exp.get('scenario_overrides') or {}
    out = _deep_merge(so.get('all') or {}, so.get(scenario_id) or {})
    return _deep_merge(out, extra or {})
