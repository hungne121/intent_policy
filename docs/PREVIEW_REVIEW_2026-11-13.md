# Preview review — 13/11/2026 (Vietnam time)

Canonical repository: `/home/hungdao/ur_ws/src/intent_policy`.

The videos are scripted expert rollouts with actual ManiCast inference for Intent. ACT has not been trained. Only the overview camera is published for review.

## Current results

| Task/case | Seed | Observed completion | Intent completion | Gate |
|---|---:|---:|---:|---|
| I2 middle plate | 300106 | 13.50 s | 11.15 s | Task success; contact gate rejected (~3–4 mm finger/cake overlap) |
| I3 nominal | 0 | 13.60 s | 12.20 s | Both pilot gates pass |
| I3 switch_before_commit | 0 | 13.60 s | 13.85 s | Both pilot gates pass; Intent is 0.25 s slower |
| I3 switch_after_commit | 0 | 19.25 s | 17.95 s | Both pilot gates pass |
| I1 Intent regression | 300100 | Not rerun | 30.30 s | Three rounds complete; dataset geometry acceptance pending |
| C1 nominal draft | 300100 | 20.80 s | 20.85 s | Task completes; geometry blocked |

These are preview pilots, not evidence of a general learned-policy improvement.

## What changed

I3 formerly required the pointing ray to disappear before approach and withdrew the hand on a fixed timeline. It now prepares centrally, follows the predicted wrist path from observed motion onset, and uses a bounded continuously updated Cartesian command. Nominal pointing lasts 2.1 s. The person withdraws when pointing is complete and the robot has visibly advanced toward the intended cake.

In the nominal Intent seed-0 pilot: human motion onset 1.70 s; forecast-follow command starts 1.75 s; measured TCP progress of 2 mm toward the intended cake by 1.85 s; completed pointing 3.70 s; hand withdrawal 3.80 s; physical grasp 5.30 s. Observed physical grasp is 7.15 s. The median actual robot/hand speed ratio during the moving approach is approximately 0.59 in this pilot; the controller's target ratio is 0.8 of a conservatively filtered hand-speed estimate.

The shared `HumanPace` retains recent work pace after the human stops giving the cue. `CartesianFollower` integrates the command continuously with a 0.65 m/s² acceleration limit and a tracking-error bound. This avoids the doubled slowdown caused by restarting each target increment from the lagging measured TCP.

I2 uses the centre plate, a 2.2 s reaching gesture, and diagonal transport during Intent lift after grasp. Observed waits for stable slowing pointing. The human withdraws as soon as the held cake visibly moves toward the intended plate, rather than waiting for an arbitrary timer.

HY-Motion supplies I2/I3 palm paths; they are filtered and retargeted. Body/arm poses are reconstructed by IK, and finger shape blends from relaxed to pointing. This is not full-body HY-Motion replay. I1 and the C1 draft use scripted human motion as identified in their summaries.

## Pairing and checks

I3 saves `human_trace.json` per expert. Source clips, seed and interaction rules are shared; nominal withdrawal timing is contingent on visible robot motion and can differ. New records include the realized-human hash and interaction events. Switch schedules remain identical between the experts; the late-switch pilot was calibrated once to occur after both actual grasps and before transfer, and semantics are checked rather than forced to PASS.

The switch-event extractor now records observed A-to-B pointing even when Observed has not picked A. It does not require a robot correction action just to establish that the person visibly changed their mind.

Four automated tests cover conservative forecast pace, continuation after the cue, acceleration/speed bounds and continuous target changes. I3 retains contact, swept-motion, human-self, success, case-semantic and data-integrity checks.

## Remaining blockers

- I2: measured finger/cake overlap remains around 3–4 mm. This is collision-geometry overlap, not simulated cake deformation. Preview is usable for review; rejected by the current dataset gate.
- C1: both cup/cloth rounds now complete using physical assisted grasps. The draft still reports up to 30 mm human-hand/world overlap and approximately 15 mm gripper/mug-geometry overlap. It uses a rigid cloth proxy and a scripted walking/root trajectory. Seed 300100 and full state traces are saved; not accepted as data and not a completed realistic cloth simulation.
- C2, C3, C4, W1, W2, W3 remain at scene/layout stage in V2. No interaction videos or dataset completion are claimed for them.

## Reproduce

From the canonical repo, use:

```bash
export MUJOCO_GL=egl PYTHONPATH=.:scripts:.deps OMP_NUM_THREADS=1
PY=/home/hungdao/miniforge3/envs/ur_bullet312/bin/python
$PY -m pytest -q tests/test_human_paced_v2.py
$PY scripts/i2_v2_preview.py --seed 300106 --out outputs/i2_center_paced_20261113_v4
$PY scripts/i3_collect.py --case nominal --seeds 0 --out outputs/i3_paced_20261113_v4
$PY scripts/i3_collect.py --case switch_before_commit --seeds 0 --out outputs/i3_paced_20261113_v4
$PY scripts/i3_collect.py --case switch_after_commit --seeds 0 --out outputs/i3_paced_20261113_v4
$PY scripts/i1_v2_preview.py --mode intent --out outputs/i1_paced_20261113
$PY scripts/c1_v2_preview.py --out outputs/c1_interactive_paced_20261113_v2
```

Use a fresh output directory after controller or scene changes; do not resume incompatible records. Final videos are copied into `previews/review_20261113/` with an artifact manifest identifying source logs.
