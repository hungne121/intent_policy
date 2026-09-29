Human-Intention-Aware Robot Action Policy

1. Research Goal

Study whether human-intention information provides actionable information to a robot action policy and improves human–robot collaborative manipulation.

The central comparison is:

Policy(observation)

vs.

Policy(observation, intention_information)

The research target is the robot action policy.

This project is NOT:

a hierarchy of separate skill policies;

one ACT model per task or skill;

primarily an intention-classification project.

Human-intention prediction is treated as an information-enrichment module for the robot policy:

Human cue
    ↓
Intent / future-behavior inference
    ↓
Intention-related information
    ↓
Robot action policy
    ↓
Robot action

The research question is therefore:

Which type of human-intention information provides the most useful additional information to the robot action policy, and does that benefit transfer from a diagnostic restricted action head to the original continuous ACT policy?

2. Policy Architecture — LeRobot ACT

The robot policy must be implemented around the ACT implementation and conventions used by Hugging Face LeRobot.

Reference:

LeRobot ACT documentation: https://huggingface.co/docs/lerobot/act

LeRobot ACT configuration: src/lerobot/policies/act/configuration_act.py

LeRobot ACT model: src/lerobot/policies/act/modeling_act.py

LeRobot ACT uses a vision backbone, Transformer encoder/decoder and action chunking. It consumes robot observations such as images and robot state and predicts a chunk of future actions.

Do NOT reimplement a separate custom ACT architecture from scratch.

The implementation should preserve the LeRobot ACT organization:

policy
├── configuration
├── modeling
├── processor / preprocessor
└── dataset feature interface

If the project vendors or subclasses LeRobot ACT, keep the structure recognizable and isolate project-specific modifications.

The intended conceptual architecture is:

Robot Observation
       │
       ▼
LeRobot ACT observation processing
       │
       ▼
ACT vision/state encoding
       │
       ▼
ACT Transformer
       │
       ▼
policy latent h
       │
       ├─────────────────────────────┐
       │                             │
       │                    Intent Information
       │                             │
       │                       Intent Encoder
       │                             │
       └─────────────── Fusion ──────┘
                       │
                       ▼
                 Action Head
                       │
                       ▼
                Robot action chunk

There must be one shared policy backbone.

Never implement:

ACT_PICK
ACT_PLACE
ACT_HANDOVER
ACT_RETRACT

as separate learned models.

Never implement:

Intent
   ↓
Skill Selector
   ↓
Separate ACT models

Tasks/scenarios are represented by data and scenario configuration, not by separate learned policies.

3. Project Phases

Because this project is being rebuilt from scratch, the old Phase-0/Phase-1 distinction is simplified.

PHASE 0/1 — FOUNDATION
    LeRobot ACT + simulation + scenarios + restricted diagnostic head
        ↓
PHASE 2 — ORACLE INTENTION UTILITY
    Does intention-related information help?
        ↓
PHASE 3 — INFORMATION ABLATION
    Spatial vs Motion vs Spatial+Motion
        ↓
PHASE 4 — CONTINUOUS ACT TRANSFER
    Does the selected information help the original ACT head?
        ↓
PHASE 5 — LEARNED INTENTION PREDICTION
    Replace oracle information with a real predictor

The project is currently focused on Phases 0/1 through 4.

4. PHASE 0/1 — Foundation

4.1 Objective

Build the project from scratch around LeRobot ACT and establish a reliable HRI simulation benchmark.

There is no requirement to preserve an old multi-policy implementation.

This phase must produce:

a working LeRobot ACT policy;

a simulation environment;

the six required HRI scenarios;

a restricted diagnostic action interface;

event logging;

HRIBench-style metrics;

deterministic scenario variation;

a clean No-Intent baseline.

The continuous ACT path must remain available for Phase 4.

5. ACT Action Interface

5.1 Continuous ACT path

The original LeRobot ACT path must remain the canonical manipulation interface.

ACT predicts a future action chunk:

A_t = [a_t, a_{t+1}, ..., a_{t+H-1}]

The exact action dimension, chunk size, observation features and normalization must follow the actual robot/simulator dataset configuration.

Do not invent a second incompatible ACT action representation.

5.2 Restricted diagnostic path

For Phases 0/1 through 3, expose a small generic motor-action interface:

HOLD

MOVE_FORWARD
MOVE_BACKWARD

MOVE_LEFT
MOVE_RIGHT

MOVE_UP
MOVE_DOWN

OPEN_GRIPPER
CLOSE_GRIPPER

These are short-horizon motor primitives, not semantic skills.

Each action must map deterministically to a configurable low-level controller command.

The restricted head exists only to make policy decisions easier to inspect and perform counterfactual intention experiments.

The final manipulation policy remains continuous ACT.

6. Scenario Benchmark

Use an HRIBench-style interaction-centric organization with three roles:

Instructor
Collaborator
Intruder

The project does not need to reproduce every HRIBench asset or task. It must preserve the interaction-centric idea:

role
goal
interaction phases
human behavior
temporal/causal constraints
success/failure conditions
applicable metrics

Scenarios must be configuration-driven.

7. Scenario Set

The initial benchmark contains exactly five concrete scenarios under the three roles.

Instructor
├── instructor_object_to_target
│
Collaborator
├── collaborator_object_handover
└── collaborator_bowl_assistance
│
Intruder
└── intruder_pick_place_interruption

The fifth item is not a separate task: the Instructor scenario must contain two target regions, and the collaborator bowl scenario contains two object-bowl pairings. These are task variants, not separate learned policies.

8. Instructor Scenario

8.1 Task

The human specifies which object the robot must manipulate and which target region it must use.

The scene contains:

Object A
Object B
Object C

The three objects must differ clearly in:

color;

shape.

Example:

red cube
blue cylinder
green sphere

The exact shapes/colors must be configurable.

There are two destination regions:

Target Region 1 — color X
Target Region 2 — color Y

Both are square regions on the workspace/table.

The human instruction identifies:

which object
+
which target region

The robot must:

select the correct object
→ grasp it
→ move to the requested target region
→ release it inside the correct region

8.2 Purpose

This scenario tests whether the policy can correctly ground human-provided information into:

object selection
+
spatial target selection
+
manipulation execution

This scenario is especially important for Phase 3 Spatial/Target intention information.

8.3 Success

Success requires all of:

correct object selected
correct grasp
object transported to correct target region
object released within target region

Picking the wrong object or using the wrong target region is failure.

9. Collaborator Scenario 1 — Object Handover

9.1 Task

The scene contains:

Object A
Object B

The human chooses one object for handover.

The robot must:

identify the selected object
→ grasp the correct object
→ move to a safe handover pose
→ coordinate transfer
→ release at the appropriate time

The human's selected object is not fixed across episodes.

The scenario generator must randomize which object is selected.

9.2 Purpose

Test collaborative object selection and handover timing.

Important variables:

selected object
human approach speed
human hand position
handover timing
robot approach timing

This scenario is a primary testbed for:

Spatial / Target information
Motion / Kinematic information
Spatial + Motion information

10. Collaborator Scenario 2 — Bowl Assistance

10.1 Task

The scene contains:

Object A
Object B

Bowl A
Bowl B

There are two object-bowl pairings:

Object A ↔ Bowl A
Object B ↔ Bowl B

The human picks up one of the two objects.

The robot must infer which corresponding bowl is required and place that bowl in front of the human.

The robot does NOT need to manipulate the object itself.

Required sequence:

human picks Object A
        ↓
robot selects Bowl A
        ↓
robot grasps Bowl A
        ↓
robot places Bowl A in front of human
        ↓
human places Object A into Bowl A
        ↓
task success

or equivalently for Object B/Bowl B.

10.2 Purpose

This is a particularly important intention-aware collaboration scenario because the human's action provides information that changes the robot's required action.

The current human observation can be similar across the two cases:

human picks an object

but the appropriate robot action differs:

Object A → Bowl A
Object B → Bowl B

This makes it suitable for counterfactual intention experiments.

10.3 Success

Success requires:

correct bowl selected
correct bowl placed in front of human
human can successfully place the corresponding object into the bowl

The robot does not receive credit merely for placing any bowl.

11. Intruder Scenario — Pick-and-Place Interruption

11.1 Nominal task

The robot starts with a normal pick-and-place task:

pick object
→ transport
→ place at target

No intention information is required for the nominal trajectory.

11.2 Intrusion

During execution, a human hand or another defined human body part enters the robot workspace/safety region.

The intrusion must be generated at controlled times.

The robot must:

detect unsafe human intrusion
→ stop / retract / yield
→ wait while the human remains in the unsafe region
→ resume the original task only after the workspace becomes safe
→ complete pick-and-place

The robot must NOT continue the nominal trajectory through the human.

11.3 Purpose

Test:

safety
yielding
response latency
recovery
task continuation after interruption

This scenario is also important for future intention-information experiments involving predicted human motion.

11.4 Success

Success requires:

no unsafe collision/contact
robot responds to intrusion
robot remains safe while human occupies the danger region
robot resumes after clearance
original pick-and-place eventually succeeds

If the human remains in the danger region, the robot must not force completion.

12. Scenario Configuration

Do not hard-code scenario behavior in the policy.

Use configuration files.

Example:

scenario:
  id: collaborator_object_handover
  role: collaborator

scene:
  objects:
    - id: object_a
    - id: object_b

human:
  behavior: select_one_object
  selection_randomization: true
  speed_scale: 1.0

interaction:
  type: handover

success:
  correct_object: true
  successful_transfer: true

metrics:
  csr: true
  ct: true
  ir: true
  tsync: true
  rsp: true
  oc: true
  cfr: true
  hcs: true

The exact fields may be expanded as implementation requires.

13. Scenario Variation

Every scenario must support deterministic variation.

At minimum:

object positions
target positions
robot initial pose
human initial pose
human trajectory
human speed
interaction timing
cue timing
intrusion timing

For intention experiments additionally support:

selected object
selected target
human motion variant
intention-change timing
information delay
information noise

Every episode must record:

scenario_id
episode_id
seed
role
variation parameters

14. Human Motion Representation

Do not implement a complex learned human-motion generator at this stage.

Use scripted/procedural human behavior first.

The important requirement is that the human behavior has:

repeatability
controlled variation
known ground truth

because the oracle intention experiments need exact simulator ground truth.

Later, recorded or learned human motion may replace the scripted generator without changing the scenario contract.

15. Event Logging

The environment must log explicit events.

Minimum event set:

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

Do not reconstruct temporal metrics only from the final episode state.

16. Metrics

Use HRIBench terminology where applicable.

Effectiveness

CSR — Collaboration Success Rate
CT  — Completion Time
IR  — Idle Ratio

Coordination

TSync — Temporal Synchronization
Rsp   — Response Latency
OC    — Order Compliance

Safety / Robustness

CFR — Collision-Free Rate
HCS — Human Contact Safety
CIR — Contradictory Instruction Recognition
DSR — Disruption Success Rate

Not every metric applies to every scenario.

Report metrics individually.

Do not invent a single composite score.

17. Phase 0/1 Acceptance

The foundation phase is complete only when:

[ ] LeRobot ACT runs on the simulator
[ ] ACT dataset/feature interface is functional
[ ] continuous ACT action path is preserved
[ ] restricted action head is functional
[ ] all required scenarios instantiate
[ ] all scenario success/failure conditions work
[ ] human behavior is deterministic under seed
[ ] event logging is complete
[ ] HRIBench-style metrics can be computed
[ ] No-Intent restricted policy can be trained
[ ] No-Intent restricted policy can be evaluated

The output of this phase is the No-Intent baseline.

18. Phase 2 — Oracle Intention-Information Utility

Objective

Determine whether additional human-intention information provides actionable information to the robot policy.

Use oracle information directly from simulator ground truth.

Do NOT use a learned predictor.

The primary oracle bundle is:

Spatial / Target information
+
Future Human Motion information

Conceptually:

Observation
     +
Oracle Spatial Information
     +
Oracle Future-Motion Information
     ↓
Restricted ACT Policy
     ↓
9 motor actions

Compare:

No Intent Information
vs
Correct Oracle Information

Also test:

Wrong Oracle
Shuffled Oracle
Noisy Oracle
Delayed Oracle

Use the same demonstrations, seeds, backbone and training budget.

The key counterfactual is:

same observation
+
different intention information
→
different appropriate robot action

Phase 2 acceptance requires downstream HRI evidence that the oracle information changes useful behavior, not merely that the policy embedding changes.

19. Phase 3 — Intention-Information Ablation

Objective

Determine which type of intention-related information provides the most useful enrichment.

Do NOT compare raw sensors.

Compare the information delivered to the policy.

Primary conditions:

No Intent
Spatial / Target Information
Motion / Kinematic Information
Spatial + Motion Information

Spatial / Target Information

Represents information that future predictors may derive from:

gaze
pointing
head orientation
target estimation

Oracle representation may contain:

target object
target position
interaction region

Interpretation:

WHAT / WHERE the human is likely to interact

Motion / Kinematic Information

Represents information that future predictors may derive from:

hand tracking
pose tracking
skeleton
human-motion prediction

Primary oracle:

future hand position
future hand velocity

Interpretation:

HOW / WHERE the human is likely to move next

Skeleton is a richer source of kinematic information, not a separate information family.

Motion sub-ablation

If simulator body-state quality permits:

current hand state
future hand state
upper-body / skeleton motion

Question:

Does richer human motion information provide additional policy value?

Fusion

Test:

Spatial
+
Motion

using separate encoders followed by fusion with the ACT policy latent.

The experiment must use the same demonstrations and evaluation protocol for every condition.

Phase 3 selects the intention-information representation for Phase 4.

20. Phase 4 — Continuous ACT Transfer

Objective

Determine whether the benefit discovered with the restricted diagnostic head transfers to the original continuous ACT manipulation policy.

Restore the original LeRobot ACT action head.

Compare:

Continuous ACT(observation)

vs

Continuous ACT(observation, selected_intention_information)

Use exactly the information selected by Phase 3.

Do not introduce semantic intent as a new variable at this stage.

Do not change the continuous action representation merely to make the experiment easier.

Keep the original ACT action chunking.

The experiment must include:

same scenario set
same evaluation seeds
same task demonstrations
same observation features
same ACT configuration
same action chunk configuration

Also test intention-information changes during execution:

initial intention information
        ↓
human changes target / motion
        ↓
oracle information changes
        ↓
ACT replans

Measure response latency, coordination, synchronization, recovery and task success.

21. Phase 5 — Learned Intention Prediction

Phase 5 is intentionally downstream of the oracle experiments.

Only after Phase 2–4 establish useful intention-information should the project introduce learned predictors.

The pipeline becomes:

Human cue
    ↓
Predictor
    ↓
Predicted spatial / motion information
    ↓
Intent encoder
    ↓
ACT

Candidate predictors may use:

gaze
hand motion
skeleton
multimodal cues

The predictor should be chosen according to the information representation selected in Phase 3.

Phase 5 compares:

Oracle information
vs
Predicted information
vs
No information

The main question becomes:

How much of the oracle benefit survives realistic prediction error,
noise and latency?

22. Coding Rules

Build from scratch around LeRobot ACT.

Follow LeRobot ACT configuration/modeling conventions.

Use one shared ACT backbone.

Never create one policy per skill.

Keep continuous ACT intact.

Restricted action mode must be selectable by configuration.

Keep scenarios configuration-driven.

Keep scenario logic outside the policy.

Keep benchmark metrics independent of policy implementation.

Use explicit event logging.

Do not train a learned intent predictor before oracle verification.

Do not use semantic intent as a primary Phase-3 representation.

Treat Spatial and Motion as the current primary intention-information families.

Treat skeleton as a richer kinematic source, not a separate family.

All experiments must be reproducible from configuration + seed.

Save experiment configuration and checkpoint metadata with every result.