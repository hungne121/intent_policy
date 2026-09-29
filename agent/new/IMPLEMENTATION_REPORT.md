# Implementation Report — Foundation and HRI Scenario Setup

## 1. Purpose

This document is the implementation handoff for the project rebuild.

The project is being rebuilt from scratch, so the previous distinction between a legacy Phase 0 refactor and Phase 1 implementation is removed.

The implementation should establish one clean LeRobot-ACT-based codebase and the required HRI scenarios before running the intention experiments.

The research architecture is:

```text
Observation
    ↓
LeRobot ACT
    ↓
policy latent
    +
oracle/predicted intention information
    ↓
action head
    ↓
robot action
```

There must be one shared ACT policy, not one policy per task.

---

# 2. Required File Structure

The following files/modules should be added.

The exact Python package root may follow the existing repository layout, but the responsibilities below are mandatory.

```text
README.md

docs/
├── PHASE_2_ORACLE_INTENTION.md
├── PHASE_3_INTENTION_INFORMATION_ABLATION.md
└── PHASE_4_CONTINUOUS_ACT.md

configs/
├── policy/
│   ├── act_restricted.yaml
│   └── act_continuous.yaml
│
└── scenarios/
    ├── instructor_object_to_target.yaml
    ├── collaborator_object_handover.yaml
    ├── collaborator_bowl_assistance.yaml
    └── intruder_pick_place_interruption.yaml

src/
├── policies/
│   ├── restricted_action_head.py
│   └── intention_fusion.py
│
├── scenarios/
│   ├── base_scenario.py
│   ├── scenario_registry.py
│   ├── instructor_object_to_target.py
│   ├── collaborator_object_handover.py
│   ├── collaborator_bowl_assistance.py
│   └── intruder_pick_place_interruption.py
│
├── human/
│   ├── scripted_human.py
│   └── human_state.py
│
├── benchmark/
│   ├── events.py
│   ├── logger.py
│   └── metrics.py
│
└── intent/
    ├── oracle.py
    └── representations.py

tests/
├── test_act_policy.py
├── test_restricted_action_head.py
├── test_scenarios.py
├── test_event_logging.py
└── test_metrics.py
```

If the repository already has an equivalent directory organization, reuse it instead of creating duplicate modules.

---

# 3. `README.md`

## Purpose

Replace the old project-level research contract with the new one.

It defines:

- research goal;
- LeRobot ACT requirement;
- shared-policy architecture;
- Phase 0/1 foundation;
- Phase 2 oracle utility;
- Phase 3 information ablation;
- Phase 4 continuous transfer;
- Phase 5 learned prediction;
- HRIBench-style roles;
- exact scenario semantics;
- metric/event requirements.

The README is the project source of truth.

---

# 4. `docs/PHASE_2_ORACLE_INTENTION.md`

## Purpose

Define the exact Phase-2 experiment.

It must specify:

```text
No Intent
vs
Correct Oracle Spatial + Motion
vs
Wrong
vs
Shuffled
vs
Noisy
vs
Delayed
```

The oracle information comes from simulator ground truth.

It must NOT use a learned predictor.

It must specify the same-data/same-seed/same-backbone protocol.

It must include the counterfactual:

```text
same observation
different oracle intention information
```

and define how action changes are logged.

---

# 5. `docs/PHASE_3_INTENTION_INFORMATION_ABLATION.md`

## Purpose

Define which intention-related information should be supplied to the policy.

Main comparison:

```text
No Intent
Spatial / Target
Motion / Kinematic
Spatial + Motion
```

Motion sub-ablation:

```text
current hand
future hand
upper-body/skeleton
```

Important terminology:

```text
Spatial and Motion = information families

gaze / hand tracking / skeleton = possible cue sources
```

Do not compare raw sensors directly.

The oracle representation should approximate the information a future predictor would provide.

---

# 6. `docs/PHASE_4_CONTINUOUS_ACT.md`

## Purpose

Restore the original continuous LeRobot ACT action path.

The selected Phase-3 information is injected into the ACT policy.

Comparison:

```text
Continuous ACT without intention information
vs
Continuous ACT with selected oracle information
```

The document must preserve the original ACT action chunk representation.

It must also define mid-episode intention-information changes and response measurement.

---

# 7. LeRobot ACT Integration

## Requirement

Use the real LeRobot ACT structure.

Relevant LeRobot files are:

```text
src/lerobot/policies/act/configuration_act.py
src/lerobot/policies/act/modeling_act.py
```

ACT's documented architecture uses visual features, robot state, Transformer encoder/decoder processing and action chunks.

Do not write a custom unrelated transformer and call it ACT.

If customization is necessary, preserve the LeRobot ACT interface and subclass/wrap the appropriate policy components.

The project should continue to use LeRobot-compatible:

```text
PolicyConfig
PreTrainedPolicy
dataset feature definitions
preprocessor/postprocessor
action feature schema
```

---

# 8. Restricted Action Head

## File

```text
src/policies/restricted_action_head.py
```

## Purpose

Provide the Phase-2/3 diagnostic action space:

```text
0 HOLD
1 MOVE_FORWARD
2 MOVE_BACKWARD
3 MOVE_LEFT
4 MOVE_RIGHT
5 MOVE_UP
6 MOVE_DOWN
7 OPEN_GRIPPER
8 CLOSE_GRIPPER
```

The mapping from discrete action to low-level controller command must be deterministic and configurable.

Example:

```text
MOVE_FORWARD
    → +Δx in robot/base frame
```

The head must not become a semantic skill classifier.

Do not create:

```text
PICK
PLACE
HANDOVER
RETRACT
```

as restricted labels.

---

# 9. Intention Fusion

## File

```text
src/policies/intention_fusion.py
```

## Purpose

Encode oracle intention information and fuse it with the ACT policy representation.

Minimum conceptual API:

```text
spatial_encoder(spatial_info)
motion_encoder(motion_info)
fuse(policy_latent, intent_embeddings)
```

The fusion implementation should be configurable.

Do not hard-code a particular fusion mechanism into the scenario logic.

---

# 10. Oracle Intention Provider

## File

```text
src/intent/oracle.py
```

## Purpose

Read simulator ground truth and expose it as controlled oracle information.

Required outputs:

### Spatial

```text
target_object
target_position / interaction_region
```

### Motion

```text
current_hand_state
future_hand_position
future_hand_velocity
```

Optional:

```text
upper_body / skeleton future state
```

The provider must support controlled corruption:

```text
correct
wrong
shuffled
noisy
delayed
```

The oracle provider must never silently access fields that are not part of the selected representation.

This is important because oracle experiments are intended to isolate information value.

---

# 11. Intention Representation Definitions

## File

```text
src/intent/representations.py
```

## Purpose

Define stable schemas for:

```text
SpatialInformation
MotionInformation
```

Example conceptual structures:

```text
SpatialInformation:
    target_object_id
    target_relative_position
    target_region
    valid
```

```text
MotionInformation:
    hand_position
    hand_velocity
    future_hand_position
    future_hand_velocity
    valid
```

All coordinates must have a documented frame and normalization.

---

# 12. Scenario Base Interface

## File

```text
src/scenarios/base_scenario.py
```

## Purpose

Provide a common scenario interface.

Each scenario must expose:

```text
reset(seed)
step(action)
get_observation()
get_human_state()
get_ground_truth_intention_information()
get_events()
is_success()
is_failure()
```

The exact API may follow the selected simulator, but the responsibilities must remain separate.

Scenario code must not contain ACT model code.

---

# 13. Scenario Registry

## File

```text
src/scenarios/scenario_registry.py
```

## Purpose

Map configuration IDs to scenario implementations.

Example:

```text
instructor_object_to_target
collaborator_object_handover
collaborator_bowl_assistance
intruder_pick_place_interruption
```

No learned model selection is allowed here.

Scenario registry selects environments, not policies.

---

# 14. Instructor Scenario

## File

```text
src/scenarios/instructor_object_to_target.py
configs/scenarios/instructor_object_to_target.yaml
```

## Scene

Three objects:

```text
object_a
object_b
object_c
```

They must differ by:

```text
color
shape
```

Two square target regions:

```text
target_region_a
target_region_b
```

with different colors.

## Episode generation

Each episode specifies:

```text
requested_object
requested_target_region
```

The human instruction/cue communicates both.

## Robot task

```text
select requested object
→ grasp
→ transport
→ release in requested target
```

## Failure

```text
wrong object
wrong target
failed grasp
release outside target
timeout
```

## Ground truth

The scenario must expose:

```text
requested_object
requested_target_region
target_position
```

for oracle experiments.

---

# 15. Collaborator Scenario — Handover

## Files

```text
src/scenarios/collaborator_object_handover.py
configs/scenarios/collaborator_object_handover.yaml
```

## Scene

```text
object_a
object_b
```

The human selects one.

The selection changes between episodes.

## Robot behavior

```text
identify selected object
→ grasp selected object
→ approach handover region
→ coordinate with human
→ release
```

## Ground truth

Expose:

```text
selected_object
human_hand_position
human_hand_velocity
future_hand_position
future_hand_velocity
```

These are used only by the oracle-intention infrastructure when enabled.

The baseline policy must not receive privileged fields.

---

# 16. Collaborator Scenario — Bowl Assistance

## Files

```text
src/scenarios/collaborator_bowl_assistance.py
configs/scenarios/collaborator_bowl_assistance.yaml
```

## Scene

```text
object_a
object_b

bowl_a
bowl_b
```

Mapping:

```text
object_a ↔ bowl_a
object_b ↔ bowl_b
```

## Episode

Human picks one object.

Robot must:

```text
identify corresponding bowl
→ pick bowl
→ place bowl in front of human
→ human puts selected object into bowl
```

## Important causal property

The two cases must be sufficiently similar before the human selection so that the robot's correct action depends on human information.

Example:

```text
human selects object_a
→ bowl_a

human selects object_b
→ bowl_b
```

This scenario is explicitly intended for counterfactual intention testing.

---

# 17. Intruder Scenario

## Files

```text
src/scenarios/intruder_pick_place_interruption.py
configs/scenarios/intruder_pick_place_interruption.yaml
```

## Nominal robot task

```text
pick object
→ transport
→ place at target
```

## Intrusion

A human hand/body part enters a predefined unsafe workspace region.

The intrusion time must be randomized but reproducible.

## Required response

```text
intrusion detected
→ HOLD / safe retract
→ wait
→ verify human clearance
→ resume
→ complete pick-and-place
```

Do not allow the robot to continue through an occupied safety region.

## Ground truth events

Log:

```text
disruption_start
safety_distance_violation
robot_response
recovery_start
human_clear
recovery_complete
task_success
```

---

# 18. Human Module

## Files

```text
src/human/human_state.py
src/human/scripted_human.py
```

## Purpose

Represent scripted human behavior and expose deterministic ground truth.

The human module should support:

```text
trajectory
speed
start time
selection
cue timing
intrusion timing
withdrawal
```

No learned human predictor is required in the foundation phase.

---

# 19. Event Logger

## Files

```text
src/benchmark/events.py
src/benchmark/logger.py
```

## Purpose

Create a structured event stream for all scenarios.

Minimum events:

```text
episode_start/end
human_motion_start/end
human_cue_onset
human_intention_change
robot_motion_start/end
robot_action_change
interaction_window_start/end
object_grasp/release
human_robot_contact
safety_distance_violation
protocol_step_complete
disruption_start/end
recovery_start/end
task_success/failure
```

Every event must include enough metadata to reconstruct timing.

Recommended fields:

```text
episode_id
timestamp
event_type
scenario_id
role
entity_id
payload
```

---

# 20. Metrics

## File

```text
src/benchmark/metrics.py
```

## Purpose

Calculate HRIBench-style metrics from logged events.

Metrics:

```text
CSR
CT
IR
TSync
Rsp
OC
CFR
HCS
CIR
DSR
```

Metric applicability must be scenario-specific.

Do not make the policy code aware of metric formulas.

---

# 21. Tests

## `tests/test_act_policy.py`

Verify:

```text
ACT initializes
dataset features map correctly
forward pass works
action chunk shape is correct
checkpoint save/load works
```

## `tests/test_restricted_action_head.py`

Verify:

```text
all 9 actions exist
mapping is deterministic
invalid action IDs fail
controller scaling follows configuration
```

## `tests/test_scenarios.py`

Verify for each scenario:

```text
reset works
seed is deterministic
success condition works
failure condition works
ground truth fields are available
```

## `tests/test_event_logging.py`

Verify:

```text
required events are emitted
timestamps are monotonic
interaction windows close
episode boundaries are recorded
```

## `tests/test_metrics.py`

Use synthetic event streams to verify metric calculations.

---

# 22. Foundation Experiment

After the files above are implemented, run:

```text
FOUNDATION-BASELINE
```

with:

```text
policy = LeRobot ACT
action_mode = restricted
intent_provider = none
```

Run all required scenarios with fixed seeds.

This produces the baseline:

```text
No Intent Information
```

This baseline becomes the reference for Phase 2.

---

# 23. Do Not Implement Yet

Do NOT implement the following as part of this foundation handoff:

```text
learned gaze predictor
learned skeleton predictor
learned hand-motion predictor
semantic intent classifier
real-world sensor fusion
one-policy-per-skill
skill selector
```

Those are outside the current foundation and Phase 2/3 oracle work.

---

# 24. Required Implementation Report After Agent Execution

The coding agent must produce:

```text
reports/FOUNDATION_IMPLEMENTATION_REPORT.md
```

It must state:

```text
file created/modified
purpose
test command
test result
scenario implemented
known limitation
```

It must distinguish:

```text
IMPLEMENTED
TESTED
VALIDATED
```

Do not claim benchmark validity merely because unit tests pass.

---

# 25. Definition of Done

Foundation is accepted only when:

```text
[ ] LeRobot ACT runs
[ ] ACT structure follows LeRobot
[ ] restricted head works
[ ] continuous ACT path remains available
[ ] Instructor scenario works
[ ] Collaborator handover works
[ ] Collaborator bowl assistance works
[ ] Intruder interruption works
[ ] deterministic seeds work
[ ] ground-truth intention information is available
[ ] event logs are complete
[ ] metrics run
[ ] No-Intent baseline runs
[ ] tests pass
[ ] implementation report exists
```

The next scientific experiment after this foundation is Phase 2:

```text
No Intent
vs
Oracle Spatial + Motion
```
