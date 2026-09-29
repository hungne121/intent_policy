# Phase 3 — Intention Information Ablation

## Objective and execution contract

Begin only after Phase 2 supports bundle utility. Restricted policy only.
Primary conditions are exactly No Information, Spatial/Target, Future Hand Motion,
and Spatial + Future Hand Motion. No semantic representation in primary selection.
Add Current Hand State as motion sub-ablation. Upper-body/skeleton is optional:
only evaluate if simulator supplies realistic shoulder/elbow/wrist/hand motion;
otherwise explicitly report unavailable, never fabricate joints from hand data.

Spatial encodes target identity and relative interaction point. Motion encodes
future hand position/velocity. Use separate normalized encoders with equal output
embedding dimension, then fuse embeddings with policy latent. Do not concatenate
unscaled raw signals into one encoder. Keep architecture capacity comparable;
mask unused branches. Fit normalization on train only.

Keep demonstrations, scenario set, train/validation split, training and evaluation
seeds, backbone, restricted head, optimizer, steps and batch size fixed. Reconstruct
all fields from the same timestep. Train separate experimental checkpoints with
the same architecture; no runtime skill routing.

Test target noise/wrong target, hand position/velocity and trajectory corruption,
100/200/500 ms delays. Combined branch diagnostics require spatial correct/motion
corrupt, spatial corrupt/motion correct and both corrupt. Log exact parameters.
Counterfactual is mandatory per information family.

Report None/Spatial/Current Hand/Future Hand/Skeleton/Spatial+Motion with CSR,
Rsp, TSync, DSR, safety, noise, delay and sensitivity, plus per-role results.
Select selected_intention_information using coordination, latency, robustness,
input complexity and dimension, not CSR alone. No predictor or continuous run
until selection is reviewed. Legacy semantic artifacts are excluded.

# Acceptance Criteria

```text
[ ] No Information, Spatial, Future Hand and Spatial plus Motion are evaluated.

[ ] Current Hand is evaluated and realistic skeleton availability is documented.

[ ] Only simulator oracle information is used; no semantic input or learned predictor.

[ ] Spatial and motion have separately normalized encoders with equal embedding size.

[ ] Data, splits, seeds, backbone, head, optimizer, batch size and budget are matched.

[ ] Downstream metrics and denominators are reported by role and scenario.

[ ] Counterfactual sensitivity and action meaning are evaluated for each family.

[ ] Spatial and motion noise and 100/200/500 ms delay robustness are reported.

[ ] Fusion branch corruption diagnostics cover each branch and both together.

[ ] One selected_intention_information is recorded without semantic candidates.

[ ] Selection rationale includes coordination, latency, robustness and input complexity.
```
