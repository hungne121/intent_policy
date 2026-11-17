"""Task scenarios (QĐ 53/QĐ-DL): scene layout + scripted human per task. A scenario module defines
TASK, MAX_TICKS, `build(seed)` -> (scene, positions, xml, info) and `human(ep, info)` (frame generator) and
`human_done(ep)`. The info dict holds the human plan; it is evaluator-only and never reaches the expert."""
import importlib


def scenario(task):
    return importlib.import_module(f'intent_policy.scenarios.{task.lower()}')
