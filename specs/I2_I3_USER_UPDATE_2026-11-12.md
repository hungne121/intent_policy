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
