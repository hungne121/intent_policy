# Phase 1 — Foundation: Shared ACT Policy + Restricted Motor Head + HRI Benchmark

## Objective

Phase 1 combines the former **Phase 0 (architecture refactor)** and **Phase 1 (restricted motor head)** into one foundation phase.

The goal is to establish a clean, reproducible experimental platform before introducing any human-intention information.

Phase 1 must prove that:

1. the project uses **one shared ACT policy**;
2. the ACT implementation follows the **LeRobot ACT architecture/interface**;
3. the original continuous ACT action-chunk path remains available;
4. a restricted 9-action motor head can be used as a diagnostic testbed;
5. the robot controller correctly executes every restricted action;
6. all HRI scenarios are configuration-driven;
7. the benchmark/event infrastructure is independent of the policy;
8. the complete baseline can run **without any intention input**.

Do **not** implement oracle intention, noisy/delayed intention, or a learned intention predictor in this phase.

---

# 1. Core Architecture

## 1.1 One shared policy

There must be exactly **one shared learned ACT policy/backbone**.

The architecture must NOT become:

```text
PICK       -> separate ACT model
PLACE      -> separate ACT model
HANDOVER   -> separate ACT model
RETRACT    -> separate ACT model
```

and must NOT become:

```text
Intent
   ↓
Skill Selector
   ↓
Separate policy per skill
```

Instead:

```text
                 Observation
                     │
                     ▼
            ┌─────────────────┐
            │   Shared ACT    │
            │    Policy       │
            └─────────────────┘
                     │
             shared representation
                     │
             ┌───────┴────────┐
             ▼                ▼
      Restricted Head    Continuous ACT Head
          Phase 1           preserved
             │                │
             ▼                ▼
       Motor command      Action chunk
             │                │
             └───────┬────────┘
                     ▼
                 Controller
                     │
                     ▼
                   Robot
```

Scenario/task identity may remain as metadata:

```text
scenario_id
episode_id
task_id
role
seed
```

but it must never select a different learned policy.

---

# 2. ACT Must Follow LeRobot

The project must use the actual **LeRobot ACT implementation structure**, rather than creating an unrelated custom transformer policy.

Use the corresponding LeRobot components/interfaces:

```text
ACTConfig
ACTPolicy
LeRobot dataset features
LeRobot observation/action processing
ACT action chunking
ACT vision/state inputs
```

The implementation should be based on the official LeRobot ACT architecture and API.

Relevant source of truth:

```text
LeRobot ACT documentation:
https://huggingface.co/docs/lerobot/act

LeRobot ACT configuration:
src/lerobot/policies/act/configuration_act.py

LeRobot ACT implementation:
src/lerobot/policies/act/modeling_act.py
```

Do not rewrite the ACT encoder/decoder architecture unless a later phase explicitly requires a research modification.

Project-specific components such as the restricted action head or later intention fusion must be implemented **around/integration with the LeRobot ACT policy**, not as a replacement for ACT.

---

# 3. Preserve the Continuous ACT Path

The original continuous ACT path must remain functional throughout Phase 1.

ACT's normal action representation is an action chunk:

```text
observation
    ↓
ACT
    ↓
action chunk
    ↓
robot controller
```

Do not delete this path merely because Phase 1 introduces a restricted action head.

Configuration should allow:

```yaml
policy:
  action_mode: restricted
```

and later:

```yaml
policy:
  action_mode: continuous
```

Both modes must use the **same shared ACT backbone**.

The continuous mode is not the main research condition in Phase 1, but it must remain available because Phase 4 will return to the original continuous action representation.

---

# 4. Restricted Motor Action Space

Phase 1 introduces a deliberately small action space:

```text
0  HOLD
1  MOVE_FORWARD
2  MOVE_BACKWARD
3  MOVE_LEFT
4  MOVE_RIGHT
5  MOVE_UP
6  MOVE_DOWN
7  OPEN_GRIPPER
8  CLOSE_GRIPPER
```

Exactly **9 actions** are required.

These actions are generic short-horizon motor primitives.

They are NOT semantic skills.

Do not define:

```text
PICK
PLACE
HANDOVER
RETRACT
```

as learned actions.

For example, a pick-and-place behavior should emerge from a sequence such as:

```text
MOVE_* → OPEN/CLOSE_GRIPPER → MOVE_* → ...
```

rather than from a `PICK` action class.

The purpose of this restricted head is to make policy decisions easier to inspect and to provide a controlled testbed for later intention-information experiments.

---

# 5. Restricted Action Head

The restricted head should be intentionally simple.

Conceptually:

```python
restricted_action_head = nn.Linear(
    latent_dim,
    9,
)
```

A small MLP is acceptable if required by the actual ACT integration.

Do not redesign the ACT backbone.

The head must output:

```text
9 logits
```

which are converted into one selected motor action.

The implementation must expose:

```text
action_logits
action_probabilities
selected_action
mapped_controller_command
```

in the episode trace.

---

# 6. Action-to-Controller Mapping

Every restricted action must map deterministically to the robot controller.

A Cartesian convention may be:

```text
MOVE_FORWARD  -> +X
MOVE_BACKWARD -> -X

MOVE_LEFT     -> +Y
MOVE_RIGHT    -> -Y

MOVE_UP       -> +Z
MOVE_DOWN     -> -Z
```

However, the actual mapping must follow the robot/simulator coordinate convention.

The mapping must be documented explicitly.

Example configuration:

```yaml
restricted_action:
  translation_step:
    x: 0.02
    y: 0.02
    z: 0.02

  control_frame: robot_base
```

The following must be configurable:

```text
translation magnitude
control frame
action execution duration / step
gripper command
controller limits
```

`HOLD` must not generate a Cartesian translation command.

Gripper actions must use the existing gripper controller.

---

# 7. Benchmark Architecture

The benchmark infrastructure must be independent of ACT.

Recommended structure:

```text
benchmark/
├── scenarios/
├── events/
├── metrics/
└── runner/
```

Minimum conceptual interfaces:

```python
class ScenarioConfig:
    id
    role
    goal
    human_behavior
    constraints
    success_conditions
    applicable_metrics
```

```python
class EpisodeEventLogger:
    def log(event_type, timestamp, **metadata):
        ...
```

```python
class BenchmarkMetric:
    def reset(self):
        ...

    def update(self, ...):
        ...

    def compute(self):
        ...
```

Metrics must not import or depend on ACT internals.

---

# 8. HRI Roles

The project follows the HRIBench-style interaction roles:

```text
INSTRUCTOR
COLLABORATOR
INTRUDER
```

Each scenario must declare exactly one primary role.

The project does not need to reproduce every HRIBench asset or task. It must preserve the interaction-centric evaluation structure.

---

# 9. Phase 1 Scenarios

Phase 1 uses four concrete scenarios.

These scenarios are the foundation for all later intention experiments.

---

## 9.1 Instructor — Object to Target

### Scene

Use:

```text
3 different objects
```

The objects must differ by:

```text
color
shape
```

There are:

```text
2 target regions
```

The target regions are:

```text
square
different colors
```

### Human instruction

The human specifies:

```text
which object to pick
which target region to use
```

Example:

```text
Pick the red triangle
and place it in the blue square.
```

### Robot goal

The robot must:

1. identify the requested object;
2. pick the exact requested object;
3. move it to the exact requested target;
4. place it in the requested target.

The scenario must support controlled variation in:

```text
object color
object shape
target color
object position
target position
initial robot pose
scene seed
```

The instruction must determine the required object-target pairing.

---

## 9.2 Collaborator — Object Handover

### Scene

Use:

```text
2 different objects
```

### Human behavior

The human selects one object.

### Robot goal

The robot must:

1. determine which object the human selected;
2. select the same object;
3. pick it;
4. hand it over to the human.

The scenario should vary:

```text
selected object
object positions
human approach trajectory
human speed
interaction timing
initial robot pose
scene seed
```

The purpose is to test collaborative response and coordination rather than only object manipulation.

---

## 9.3 Collaborator — Bowl Assistance

### Scene

Use:

```text
2 different objects
2 different bowls
```

The correct pairings are:

```text
Object A ↔ Bowl A
Object B ↔ Bowl B
```

### Human behavior

The human picks one object.

### Robot goal

The robot must:

1. determine which object the human selected;
2. infer the corresponding bowl;
3. place the corresponding bowl in front of the human;
4. allow the human to successfully put the object into that bowl.

Example:

```text
Human picks Object A
        ↓
Robot identifies Object A
        ↓
Robot moves Bowl A
        ↓
Bowl A is placed in front of human
        ↓
Human places Object A into Bowl A
```

The robot must not place the wrong bowl.

Controlled variations should include:

```text
selected object
object positions
bowl positions
human trajectory
human speed
interaction timing
scene seed
```

---

## 9.4 Intruder — Pick-and-Place Interruption

### Default robot task

The robot performs a normal pick-and-place task.

### Human intrusion

During robot execution, a human hand or body part enters the robot workspace/safety zone.

The intrusion must occur while the robot is actively executing the task.

### Robot goal

When the human enters the danger region, the robot must:

```text
stop / retract / yield
        ↓
wait while the human remains in the danger region
        ↓
resume after the region becomes safe
        ↓
complete the original pick-and-place task
```

The robot must not continue blindly through the human intrusion.

The scenario should vary:

```text
intrusion timing
intrusion location
human trajectory
human speed
duration inside safety region
initial robot state
scene seed
```

This scenario is essential for testing responsiveness, yielding, safety, and recovery.

---

# 10. Scenario Configuration

Scenarios must be loaded from configuration files.

Recommended structure:

```text
configs/
└── scenarios/
    ├── instructor_object_to_target.yaml
    ├── collaborator_object_handover.yaml
    ├── collaborator_bowl_assistance.yaml
    └── intruder_pick_place_interruption.yaml
```

Do not hard-code scenario timing, object selection, human trajectories, or scene variation directly inside training scripts.

A scenario configuration should contain at least:

```yaml
scenario_id:
role:
seed:
scene_variation:
human_behavior:
timing:
success_conditions:
safety_constraints:
applicable_metrics:
```

---

# 11. Human Simulation

The human should initially be scripted/deterministic.

The purpose is reproducibility, not realistic human modeling yet.

The human simulator must expose enough state for later oracle intention experiments.

At minimum:

```text
human position
human velocity
hand position
hand velocity
target/object selected by human
human interaction state
```

Later phases may derive intention-related information from these states.

Phase 1 must not expose any explicit intention vector to the ACT policy.

---

# 12. Event Logging

Every episode must use a common event stream.

At minimum support:

```text
episode_start
episode_end

human_motion_start
human_motion_end
human_cue_onset
human_intention_change

robot_motion_start
robot_motion_end
robot_action_change

interaction_window_start
interaction_window_end

object_grasp
object_release

human_robot_contact
safety_distance_violation

protocol_step_complete

disruption_start
disruption_end
recovery_start
recovery_complete

task_success
task_failure
```

Not every event is required to be active in every scenario.

The interface must exist now so later phases can use the same event definitions.

---

# 13. Required Episode Record

Every episode must save:

```text
scenario_id
role
seed
scene variation
human trajectory variant
policy configuration

timestamps

robot state
human state
object state

action logits
action probabilities
selected action
mapped controller command

all benchmark events

task success/failure

all applicable metrics
```

The result must be sufficient to reconstruct:

```text
what happened
when it happened
what the policy selected
what the controller executed
why the episode was successful/unsuccessful
```

---

# 14. Reproducibility

Same configuration + same seed should reproduce the same scenario and human behavior as closely as the simulator permits.

Different seeds should provide controlled variation.

At minimum verify:

```text
same seed
    ↓
same scene
same object arrangement
same human trajectory
same timing
```

and:

```text
different seed
    ↓
controlled variation
```

Every result must contain the seed and complete scenario configuration.

---

# 15. HRIBench-Style Metrics

The benchmark infrastructure must support the following metric families:

### Effectiveness

```text
CSR — Collaboration Success Rate
CT  — Completion Time
IR  — Idle Ratio
```

### Coordination

```text
TSync — Temporal Synchronization
Rsp   — Response Latency
OC    — Order Compliance
```

### Safety / Robustness

```text
CFR — Collision-Free Rate
HCS — Human Contact Safety
CIR — Contradictory Instruction Recognition
DSR — Disruption Success Rate
```

Not every metric applies to every scenario.

Do not collapse the evaluation into task success rate.

Recommended reporting:

```text
Instructor
    CSR
    CT
    IR
    Rsp
    OC

Collaborator
    CSR
    CT
    IR
    TSync
    Rsp
    OC

Intruder
    CSR
    CT
    Rsp
    CFR
    HCS
    DSR
    CIR when applicable
```

The exact formal HRIBench metric formulas must be verified against the source paper before claiming exact reproduction. Until then, the implementation should be described as HRIBench-style where appropriate.

---

# 16. Baseline Condition

Phase 1 baseline:

```yaml
policy:
  action_mode: restricted
  use_intent: false

intent:
  provider: none

benchmark:
  scenario_set: phase1
  seed: 0
```

The baseline must receive only the normal policy observations.

There must be:

```text
NO oracle intention
NO predicted intention
NO semantic intent label
NO noisy intent
NO delayed intent
```

This establishes the no-intention reference condition for Phase 2.

---

# 17. Required Tests

At minimum implement tests for:

```text
shared policy architecture
no skill-to-model routing

LeRobot ACT construction
ACT observation/action interface

restricted head output dimension
9-action enum

all 9 action mappings
HOLD behavior
gripper commands

continuous ACT path still works

scenario configuration loading
scenario seed reproducibility

Instructor scenario
Collaborator handover scenario
Collaborator bowl-assistance scenario
Intruder scenario

event ordering
event timestamps

episode termination
success/failure predicates

metric interface
metric applicability

result serialization
```

---

# 18. What Must NOT Be Implemented

Do not implement in Phase 1:

```text
Oracle Intent
Intent Encoder
Intent Fusion
Noisy Intent
Delayed Intent
Intent Dropout
Semantic Intent Classification
Learned Intent Predictor
Skill Selector
Separate ACT per skill
```

The following is also prohibited:

```text
one model for Instructor
one model for Collaborator
one model for Intruder
```

There must remain one shared ACT policy.

---

# 19. Acceptance Criteria

Phase 1 is complete only when all of the following pass:

```text
[ ] One shared ACT policy/backbone is used.

[ ] ACT follows the LeRobot ACT architecture/interface.

[ ] No runtime skill-to-model routing remains.

[ ] Original continuous ACT action-chunk path still runs.

[ ] Restricted head outputs exactly 9 actions.

[ ] The 9 actions are:
    HOLD
    MOVE_FORWARD
    MOVE_BACKWARD
    MOVE_LEFT
    MOVE_RIGHT
    MOVE_UP
    MOVE_DOWN
    OPEN_GRIPPER
    CLOSE_GRIPPER

[ ] Every restricted action maps correctly to the controller.

[ ] Motion magnitude/control frame are configurable.

[ ] Scenario configuration is independent of policy code.

[ ] Instructor scenario runs.

[ ] Collaborator handover scenario runs.

[ ] Collaborator bowl-assistance scenario runs.

[ ] Intruder interruption scenario runs.

[ ] Human behavior is seed-controlled.

[ ] Episode event logging works.

[ ] Required state/action/event data is saved.

[ ] HRIBench-style metric interfaces work.

[ ] Results are reported separately by HRI role.

[ ] Baseline runs with no intention information.

[ ] No intention-conditioning code is required for the Phase 1 baseline.

[ ] Automated tests pass.

[ ] A complete baseline episode can be reproduced from its saved configuration + seed.
```

---

# 20. Stop Condition

Stop after Phase 1 acceptance criteria pass.

Do **not** begin Phase 2 automatically.

Phase 2 starts only after the foundation is stable and the no-intention baseline is reproducible.

Phase 2 will introduce:

```text
Oracle intention
        ↓
policy conditioning
        ↓
intention utility evaluation
```

The scientific question of Phase 2 is whether intention-related information actually provides actionable information to the robot policy.
