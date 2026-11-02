"""Map scenario ids to environment implementations. Selects environments, never policies.

Tasks of docs/sim/scene_construct.md §4.4: T1-T4 are scenarios; T5 (change of mind) is T1-T3 with a
`change` spec and T4-neg is T4 with `negative: true` (scenario lists, scripts/generate_scenarios.py).
"""
from pathlib import Path
from intent_policy.scenarios.config import ScenarioConfig
from intent_policy.scenarios.t1_pick_place import PickPlaceTask
from intent_policy.scenarios.t2_handover import HandoverTask
from intent_policy.scenarios.t3_assist import AssistTask
from intent_policy.scenarios.t4_interrupt import InterruptTask
from intent_policy.scenarios.t3_coaster_cup import CoasterCupTask
from intent_policy.scenarios.t2_fetch_return import FetchReturnTask
from intent_policy.scenarios.t4_coworker import CoworkerTask
from intent_policy.scenarios.i1_coaster_sequence import CoasterSequenceTask
from intent_policy.scenarios.c3_cloth_bowl import ClothBowlTask
from intent_policy.scenarios.w1_sorting import SortingTask
from intent_policy.scenarios.w2_tray_carry import TrayCarryTask
from intent_policy.scenarios.i2_side_request import SideRequestTask
from intent_policy.scenarios.c4_waste_sort import WasteSortTask
from intent_policy.scenarios.c2_shelf_wipe import ShelfWipeTask
from intent_policy.scenarios.w3_lint_roll import LintRollTask
from intent_policy.scenarios.i3_direct_sort import DirectSortTask

SCENARIOS = {
    'i1_coaster_sequence': CoasterSequenceTask,
    't1_pick_place': PickPlaceTask,
    't2_handover': HandoverTask,
    't3_assist': AssistTask,
    't4_interrupt': InterruptTask,
    't1_choose_item': PickPlaceTask,        # v6 (docs/sim/scenarios_v6_proposal.md): donuts on slots, two trays
    't3_coaster_cup': CoasterCupTask,       # v6: the cup onto the coaster the human puts down
    't2_fetch_return': FetchReturnTask,     # v6: hand the can over when the work is done, take it back
    't4_coworker': CoworkerTask,            # v6: rolls into boxes by itself, the human closes lids / fetches an item
    'c1_handover': FetchReturnTask,         # v7 C1: T2 hand-over in the v7 pipeline (configs/scenarios/c1_handover.yaml)
    'c3_cloth_bowl': ClothBowlTask,         # v7 C3: lift the bowl of the tray the cloth goes to, set it on the cloth
    'w1_sorting': SortingTask,              # v7 W1: each side sorts its own items from a shared area (cowork)
    'w2_tray_carry': TrayCarryTask,         # v7 W2: the human carries a tray across, the robot yields its swept area
    'i2_side_request': SideRequestTask,     # v7 I2: the hand held out chooses the side the robot takes an item from
    'c2_shelf_wipe': ShelfWipeTask,         # v10 C2: wipe the section whose item the human lifts and holds (HABIT S6218)
    'c4_waste_sort': WasteSortTask,
    'i3_direct_sort': DirectSortTask,       # v10 I3: the item held up into the basket of the side the arm sweeps to (HABIT S5334)
    'w3_lint_roll': LintRollTask,           # v10 W3: clean where the trays the human takes away stood (HABIT S7236)         # v10 C4: the bin of the item the human takes, held out for it (HABIT S7025)
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
