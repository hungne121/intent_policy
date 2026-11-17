"""Task experts (QĐ 53/QĐ-DL): one FSM per task, two modes (observed / forecast) in the same class.

`EXPERTS[task]` -> expert class; each task module also names its scenario in `intent_policy.scenarios`."""
import importlib

TASKS = ('I1', 'I2', 'I3', 'C1', 'C2', 'C3', 'C4', 'W1', 'W2', 'W3')


def expert_class(task):
    return importlib.import_module(f'intent_policy.experts.{task.lower()}_expert').Expert
