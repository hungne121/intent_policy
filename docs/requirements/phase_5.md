# Phase 5 — Learned Human-Intention Prediction

## Objective

Replace oracle intention with a real human-intention predictor and determine how much of the oracle-intention benefit survives under realistic prediction error and latency.

This phase answers:

> Can a practical intention predictor provide information of sufficient quality and timeliness to improve the continuous robot action policy?

Do NOT change the action-policy architecture while integrating the predictor.

---

# 1. Fixed Policy

Freeze the research formulation established in Phase 4.

Keep fixed:

```text
shared policy backbone
continuous action head
intent representation
intent encoder interface
fusion mechanism
HRIBench scenarios
benchmark metrics
```

Only replace:

```text
OracleIntentProvider
```

with:

```text
PredictedIntentProvider
```

---

# 2. Predictor Selection Rule

Select a predictor whose output can be mapped directly to the intention representation chosen in Phase 3.

Examples:

```text
semantic representation
→ semantic intention classifier

kinematic representation
→ future human-motion predictor

spatial representation
→ gaze / target / interaction-point predictor
```

Do NOT choose a predictor first and then redesign the intention representation around it.

---

# 3. Predictor Interface

Implement:

```python
class PredictedIntentProvider:
    def reset(self):
        ...

    def update(self, human_observation, timestamp):
        ...

    def get_intent(self):
        ...
```

Return the same policy-facing format used previously:

```python
IntentContext(
    representation_type=...,
    value=...,
    confidence=...,
    timestamp=...,
)
```

The action policy must not know whether intent came from:

```text
oracle
noisy oracle
delayed oracle
learned predictor
```

---

# 4. Predictor Inputs

Use only signals available or realistically obtainable on the final system.

Examples:

```text
RGB
human pose
hand trajectory
gaze
interaction history
```

Do not use simulator-privileged ground truth as predictor input.

Simulator ground truth may still be used for:

```text
labels
evaluation
oracle comparison
```

---

# 5. Primary Experimental Conditions

Required:

```text
A. No Intent

B. Oracle Intent

C. Predicted Intent
```

Also include, when useful:

```text
D. Delayed Oracle matched to predictor latency

E. Noisy Oracle matched to predictor error
```

This allows separation of:

```text
prediction error
```

from:

```text
prediction latency
```

and from:

```text
policy inability to use intent.
```

---

# 6. Predictor Evaluation

Measure predictor quality, but do not stop at predictor metrics.

Depending on representation, report appropriate metrics.

Examples:

## Semantic

```text
accuracy
F1
confusion matrix
confidence calibration
```

## Spatial

```text
position error
target accuracy
```

## Kinematic

```text
future-position error
trajectory error
velocity error
```

Also measure:

```text
inference latency
update frequency
missing-prediction rate
```

---

# 7. Downstream Evaluation

The primary evaluation remains robot behavior.

Run the same HRIBench-style scenario set used in Phase 4.

Compare:

```text
No Intent
Oracle Intent
Predicted Intent
```

using:

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

where applicable.

Do not report predictor accuracy as evidence that the robot policy improved.

---

# 8. Oracle Gap

For each metric compute the gap:

```text
No Intent
        -> Predicted Intent
        -> Oracle Intent
```

Interpretation:

```text
Predicted ≈ Oracle
→ predictor preserves most useful intention information.

Predicted > No Intent but < Oracle
→ useful predictor with remaining headroom.

Predicted ≈ No Intent while Oracle is better
→ predictor is the bottleneck.

Predicted worse than No Intent
→ predictor errors actively harm control.
```

---

# 9. Latency Analysis

Prediction latency is part of the system.

Measure:

```text
human evidence timestamp
predictor output timestamp
policy consumption timestamp
robot response timestamp
```

Total response delay may be decomposed conceptually as:

```text
human event
    ↓
observation acquisition
    ↓
intent inference
    ↓
policy inference
    ↓
controller execution
```

Log each component where possible.

---

# 10. Prediction Frequency

Document:

```text
predictor frequency
policy frequency
control frequency
```

The predictor may update more slowly than the policy.

Define behavior when no new prediction is available:

```text
hold previous intent
use confidence decay
fallback to no-intent
```

The selected behavior must be explicit and configurable.

---

# 11. Confidence Handling

If the predictor outputs confidence or probability distributions, preserve them where possible.

Do not automatically collapse:

```text
P(intent)
```

into:

```text
argmax intent
```

unless the selected Phase-3 representation requires a hard label.

Support uncertainty-aware input when compatible with the policy.

---

# 12. Prediction Failure / Uncertainty

Define configurable fallback behavior for:

```text
low confidence
missing observation
human occlusion
predictor timeout
invalid output
```

Example:

```yaml
intent:
  predicted:
    confidence_threshold: 0.6
    fallback: no_intent
```

Do not silently reuse stale predictions indefinitely.

---

# 13. Distribution Shift

Evaluate predictor robustness under the same HRIBench-style scenario variation:

```text
human speed variation
trajectory variation
initial pose variation
object position variation
timing variation
scene variation
```

This checks whether predictor gains survive beyond a single scripted trajectory.

---

# 14. Change-of-Intent Evaluation

This is mandatory.

Example:

```text
RECEIVE -> WITHDRAW
```

Measure:

```text
predictor detection delay
policy response delay
robot recovery time
wrong commitment duration
safety outcome
```

Compare against:

```text
Oracle Intent
```

to isolate predictor delay.

---

# 15. Error Attribution

For failed episodes classify whether the dominant failure came from:

```text
predictor error
predictor delay
policy response
continuous action execution
scenario ambiguity
safety/controller limitation
```

Do not label every failure as an intention-prediction failure.

---

# 16. End-to-End Reporting

For each representation/predictor configuration report:

```text
predictor metrics
predictor latency
policy metrics
HRIBench metrics
oracle gap
failure categories
```

Keep predictor and downstream metrics separate.

---

# 17. Final Research Evidence

The strongest end-to-end evidence chain is:

```text
No Intent
    ↓
Oracle Intent improves policy
    ↓
Selected intent representation is identified
    ↓
Oracle Intent improves continuous policy
    ↓
Predicted Intent retains part or most of that gain
```

This supports the claim that:

```text
predicted human intention
```

provides actionable information to the robot action policy.

---

# 18. Do NOT Change at This Stage

Do not:

```text
add separate skill policies
replace the policy backbone
change HRIBench scenario definitions
change action representation
change the selected intention representation
add a high-level planner
```

unless a separate follow-up experiment explicitly requires it.

Phase 5 is predictor integration and end-to-end verification.

---

# Acceptance Criteria

Phase 5 is complete when:

```text
[ ] A real intention predictor is integrated.

[ ] Predictor output uses the existing IntentContext interface.

[ ] No simulator-privileged state is used as predictor input.

[ ] No-Intent, Oracle-Intent, and Predicted-Intent conditions are evaluated.

[ ] Predictor accuracy/error is reported.

[ ] Predictor latency is reported.

[ ] HRIBench downstream metrics are reported.

[ ] Oracle-to-predicted performance gap is quantified.

[ ] Change-of-intent detection and robot response are evaluated.

[ ] Low-confidence / missing-prediction behavior is defined.

[ ] Failure attribution distinguishes predictor and policy failures.

[ ] The shared continuous action policy remains unchanged except for intention source.
```

After this phase, the complete research pipeline has been evaluated end-to-end.