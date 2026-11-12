# I2/I3 interaction timing update (user direction, 2026-11-12)

This newer user direction governs the I2/I3 previews where it refines the V2 documents. It does not change task success predicates or authorize task IDs/future labels in a learned policy.

## I2

- One cake starts on its central source plate; the person points to one of three destination plates.
- Both robot experts first stage with the gripper **above** the cake, without grasping.
- Observed waits for a stable current pointing ray before starting the pick. The person keeps pointing until the robot visibly carries the cake toward the chosen plate, then withdraws the hand.
- Intent uses the actual ManiCast trajectory as an early, reversible cue to start the pick while the pointing arm is moving. It updates the destination from current observation before placement; the robot never receives the script's target label.
- Record pick onset, pointing confirmation, transport onset, hand withdrawal, placement, contacts, and forecast timestamps. The same human motion source/seed is used for the paired comparison; any contingent timing difference caused by the robot's transport must be logged.

## I3

- Four distinct cakes in two trays. Both experts initially stage at a shared high point between the trays.
- Observed waits for a stable current pointing ray before choosing a cake.
- Intent continually scores the time series of ManiCast future wrist positions to route the gripper high toward a likely cake as soon as motion is informative. This is reversible preparation, not an early grasp based solely on a future label.
- Once the current pointing ray confirms a cake, the gripper approaches and grasps it. Switch cases retain the same human clip for both experts; a late switch must occur after the first physical grasp and before transfer completes.

These are expert-preview rules. ACT training and closed-loop policy validation are separate gates.


## Refinement 2026-11-13: continuous pace and contingent withdrawal

This user refinement supersedes the earlier restriction that I3 Intent can only prepare high and must always wait for completed pointing before starting its pick approach.

- All preview experts share `HumanPace` and `CartesianFollower`: target approach pace is approximately 80% of the filtered observed hand speed, conservatively informed by actual ManiCast future positions; Cartesian speed/acceleration are bounded. This is a simulation controller setting, not a certified hardware safety limit.
- Do not latch the episode's maximum hand speed or use one fixed-duration robot trajectory throughout a changing cue. Recompute the following command each control tick. Keep a bounded recent work pace after the cue ends so a stationary human does not freeze the robot's remaining work.
- I3: first reach the neutral point between the cake trays. As soon as observed hand motion starts, use the full time-indexed ManiCast wrist trajectory for continuous reversible motion. A stable predicted candidate can initiate the pick approach before completed pointing; observed contrary evidence redirects or triggers a return after grasp.
- I3 nominal human: finish the pointing reach, keep pointing only until visible robot progress toward the chosen cake, then withdraw smoothly. The human and expert do not share a hidden target variable. Log the realized human trajectory and all contingent timing.
- I2: the requested middle-plate preview uses seed 300106. Its reach is 2.2 s; Observed begins picking after a stable, slowing pointing cue. Intent may pick during the reach and start moving toward the predicted/observed plate while lifting. Human withdrawal follows actual transport toward the intended plate.
- Keep the same source motion, seed and interaction rule in paired runs. Realized contingent withdrawal can differ; save its hash and timestamps instead of describing the two realized videos as identical fixed human clips.
- Preserve all contact measurements and quality thresholds. A visual preview with small grasp penetration may be shown for review, but it is not thereby accepted into the dataset.
