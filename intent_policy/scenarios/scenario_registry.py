"""Map scenario ids to environment implementations. Selects environments, never policies."""
from pathlib import Path
from intent_policy.scenarios.config import ScenarioConfig
from intent_policy.scenarios.instructor_object_to_target import InstructorObjectToTarget
from intent_policy.scenarios.collaborator_object_handover import CollaboratorObjectHandover
from intent_policy.scenarios.collaborator_bowl_assistance import CollaboratorBowlAssistance
from intent_policy.scenarios.intruder_pick_place_interruption import IntruderPickPlaceInterruption

SCENARIOS = {
    'instructor_object_to_target': InstructorObjectToTarget,
    'collaborator_object_handover': CollaboratorObjectHandover,
    'collaborator_bowl_assistance': CollaboratorBowlAssistance,
    'intruder_pick_place_interruption': IntruderPickPlaceInterruption,
}
PHASE1_SCENARIOS = tuple(SCENARIOS)


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
