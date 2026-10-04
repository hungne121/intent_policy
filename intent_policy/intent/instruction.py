"""Simulated task instruction (configs/intent_schema.yaml `instruction`): (task id, object kind, destination kind).

The spoken command of T1 / T2 names the kind of object ("the red cube") and of destination ("a place zone", "my hand")
but never which one; T3 / T4 have none (object and destination = `none`). Fixed for the whole episode.
"""
from __future__ import annotations

INSTR_OBJECT, INSTR_DEST = 'hri.instr_object', 'hri.instr_dest'


def instruction_of(spec: dict, scenario_id: str, schema: dict) -> tuple[int, int, int]:
    """Episode spec -> (task index, object-kind index, destination-kind index); index 0 = none."""
    task = schema['task_of_scenario'][scenario_id]
    ins = schema['instruction']
    t = schema['tasks'].index(task)
    if task not in ins['dest_of_task']:
        return t, 0, 0
    return t, ins['objects'].index(ins['object_desc'][spec['target']]), ins['dests'].index(ins['dest_of_task'][task])
