# Phase 2 — Verify Oracle Intention-Information Utility

## Objective and execution contract

Use the existing shared restricted 9-action policy. Test a perfect oracle bundle
against No Intent Information. The bundle has separate spatial/target and motion
branches: target object/interaction-region identity and relative target position;
future hand position and velocity at a recorded prediction horizon. No semantic
RECEIVE/WAIT/WITHDRAW label is a policy input. Such labels may remain internal
scenario/teacher metadata. No gaze, skeleton or learned intent predictor.

Save every timestep: robot/environment/human observations, robot action target,
target identity (declare object versus region), target position, current and
future hand position/velocity, scenario_id, role, timestamp and seed. Reconstruct
all representations from the same demonstrations. Match train/validation split,
backbone, optimizer, batch size, seed, budget and evaluation seeds.

Run No Information, Correct Oracle, Wrong Oracle, Shuffled Oracle, Noisy Oracle
and delays 100/200/500 ms. Wrong information must change numeric target/motion,
not merely flip a semantic label. Keep observations unchanged during corruption.
Counterfactual holds observation fixed and swaps target A/approaching versus
target B/withdrawing; log logits/probabilities and check meaningful action changes.

Use Instructor, Collaborator and Intruder, including target/motion changes.
Retain the physical handover objective in addition to coordinated reaching;
report which was actually evaluated. Log all applicable HRI metrics per role,
scenario and overall on one versioned protocol. IR increase alone is not gain.

Pass requires downstream Oracle benefit and corruption reducing that benefit,
not only sensitivity. If unsupported, diagnose and do not advance to Phase 3.
Conclusion: predictive human-intention-related information has actionable value.
Do not select Spatial versus Motion in this phase.

# Acceptance Criteria

```text
[ ] Oracle spatial plus future-motion bundle is implemented without semantic policy input.

[ ] Demonstrations contain all oracle fields, observations, actions and episode metadata.

[ ] Baseline and oracle use the same data, split, seeds, budget and architecture.

[ ] Correct, wrong, shuffled, noisy and 100/200/500 ms delayed oracle runs exist.

[ ] Fixed-observation counterfactual logs distributions and meaningful behavior.

[ ] Target or motion changes and applicable role-based scenarios are evaluated.

[ ] HRIBench metrics use one versioned protocol and are reported per scenario and role.

[ ] Oracle improves an important downstream interaction metric over baseline.

[ ] Corrupted oracle reduces the observed benefit.

[ ] No learned predictor is used; limitations and handover coverage are explicit.
```
