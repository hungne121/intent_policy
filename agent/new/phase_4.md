# Phase 4 — Transfer Selected Information to Continuous Policy

## Objective and execution contract

Use exactly selected_intention_information from the new Phase 3 selection.
Do not choose semantic, search another representation or change backbone family.
Reuse encoders, normalization and timing convention. No skill routing or
continuous-action discretization.

Restore original continuous head, absolute joint target/gripper schema,
H=16, prefix=2 and [B,16,7]. Match baseline and selected oracle demonstrations,
training budget, seeds and evaluation protocol. Start by freezing most backbone
and train encoders/fusion/necessary head layers. Only unfreeze more layers after
failure evidence. Record frozen/trainable names and trainable parameter counts.

Run Continuous No Information versus Continuous Selected Oracle. Reuse all
three role scenario families, with handover coverage described honestly. Compare
to restricted results using exactly the same metric protocol/config and seeds.
Keep horizon unchanged; any later horizon ablation must be separately labeled.

Change target/motion during ongoing execution; log information change time,
next inference time, first changed command, Cartesian trajectory deviation and
recovery completion. Rsp uses matched event timestamps, not final state.

Continuous counterfactual must use fixed observation and different numeric
oracle inputs. Report chunk difference AND Cartesian effect (toward/away/yield/
approach/timing) from matched simulator states. Norm alone does not pass.

Report action chunks, executed commands, joint/gripper/Cartesian trajectories,
timing and individual HRI metrics. Determine whether restricted gain transfers.
If not, isolate fusion, chunk/replanning and control bottlenecks with controlled
experiments, not speculation. Phase 5 remains paused by user.

# Acceptance Criteria

```text
[ ] Original continuous action schema, H=16 and prefix=2 are preserved.

[ ] The same shared backbone architecture is used without skill routing.

[ ] The new Phase-3 selected information and encoders are reused without semantic input.

[ ] Continuous No Information and Oracle are evaluated under matched conditions.

[ ] Instructor, Collaborator and Intruder scenario families are evaluated.

[ ] Continuous counterfactual includes meaningful Cartesian effects, not only L2.

[ ] Change-of-information timing and first changed command are measured.

[ ] Chunks, commands, trajectories, deviation and recovery are logged.

[ ] Applicable HRI metrics and denominators are reported per scenario and role.

[ ] Frozen/trainable parameter names and trainable counts are recorded.

[ ] Restricted and continuous benefit are compared on the same versioned protocol.

[ ] Transfer outcome is established and non-transfer bottlenecks are experimentally isolated.

[ ] No learned predictor runs; Phase 5 remains paused.
```
