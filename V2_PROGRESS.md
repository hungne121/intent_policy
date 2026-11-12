# V2 current status — 13/11/2026

Canonical repo: `/home/hungdao/ur_ws/src/intent_policy`. Latest review videos: `previews/review_20261113/`. Full details and repeatable commands: `docs/PREVIEW_REVIEW_2026-11-13.md`.

- I3 now has continuous human-paced forecast following and contingent hand withdrawal. Seed 0 passes the existing gates in all three cases for both modes. Nominal 13.60/12.20 s, switch-before 13.60/13.85 s, switch-after 19.25/17.95 s (Observed/Intent). Intent is slower in the early-switch pilot; no general improvement claim.
- I2 centre-plate preview (seed 300106): 13.50/11.15 s, both task successes; contact gate remains rejected. Pointing reach is 2.2 s and hand withdrawal follows actual transport.
- I1 Intent uses the shared pacing helper and still completes all three rounds (30.30 s). Dataset gate not established.
- C1: new interactive draft completes two cloth/cup rounds (20.80/20.85 s), but has unresolved human/world and gripper/mug overlaps and a rigid cloth proxy. It is blocked for dataset use.
- C2, C3, C4, W1, W2, W3: V2 scene/layout stage; interaction rollout remains to be implemented. No full dataset or ACT training has been completed.
- Four shared-pacing tests pass. See the review report for pilot logs, pair semantics and limitations.

## Historical implementation notes (superseded timings)

# V2 implementation status (2026-11-12)

The source of truth is `specs/v2/00_COMMON/04_TASK_CASE_MANIFEST.json` and the paired task documents under `specs/v2`. These V2 preview files are integrated in this existing `src/intent_policy` repository; legacy modules with unrelated in-progress edits are left intact.

| Task | V2 cases | Executable interaction | Paired overview MP4 | Geometry/data status |
| --- | --- | --- | --- | --- |
| I1 | nominal | yes, three coaster/cup handoffs | yes | Pilot preview only; not accepted as dataset |
| I2 | nominal | yes, one cake and three possible plates; updated wait/point/withdraw sequence | yes | Task succeeds at seed 300100; finger/cake penetration exceeds provisional 2 mm gate |
| I3 | nominal, switch_before_commit, switch_after_commit | yes; shared midpoint preparation and forecast high route | yes, all six MP4s | Revised controller: one paired seed per case passes pilot gate; train split only |
| C1 | nominal, switch_before_commit, switch_after_commit | scene layout only | no | Not collected |
| C2 | nominal | scene layout only, dirt visible | no | Not collected |
| C3 | nominal | scene layout only, dirt visible | no | Not collected |
| C4 | nominal, switch_before_commit, switch_after_commit | scene layout only | no | Not collected |
| W1 | nominal | scene layout only | no | Not collected |
| W2 | nominal | scene layout only | no | Not collected |
| W3 | nominal | scene layout only | no | Not collected |

`previews/` contains the actual I1–I3 MP4s. `outputs/layout_review/` contains static design renders for all ten tasks; these are **not** interaction previews or successful rollouts. `outputs/i3_v2_target_verified/` contains the source-repo I3 nominal pilot and `outputs/i3_v2_target_pilot/` contains the two switch-case pilots; all six pass their seed-0 gates. The paired overview videos are in `previews/`. No V2 ACT policy has been trained; the videos show expert controllers. Do not infer the full-dataset claim or a forecast benefit from these one-seed pilots.

The 2026-11-12 I2 timing update starts Observed pick at 3.45 s after a stable ray and Intent pick at 2.70 s from ManiCast in seed 300100. The person withdraws after robot transport starts. I2 still fails the provisional geometry gate with approximately 3 mm finger/cake overlap. See `specs/I2_I3_USER_UPDATE_2026-11-12.md` for the updated choreography.

Speed revision: ManiCast supplies future **positions/displacements**, not a dedicated velocity output. The preview expert derives predicted wrist speed from adjacent 0.1 s forecast horizons and compares it with recent observed wrist speed. Horizontal robot travel pace is bounded using the human speed; vertical contact motion remains slower where the faster trial caused a 6 mm finger/cake overlap in I3. In the I2 seed-300100 gesture, median predicted speed was 0.022 m/s versus 0.246 m/s observed, so measured speed dominates. The revised I2 seed-300100 pair finishes at 10.15/9.40 s and still fails its geometry gate; left/centre/right destination seeds 300100/300106/300101 each complete but all remain geometry-rejected. The centre seed's early forecast candidate was plate2 although the observed target was plate1; the expert safely corrected before transport. The median 1-second wrist forecast error during the I2 gesture is 0.194/0.151/0.107 m on these three seeds. These are diagnostics of this checkpoint and retargeted motion, not validated ManiCast accuracy. Full derivation and safety limits: `docs/MANICAST_SPEED_CHECK.md`.

The speed-revised I3 pilot at seed 0 passes the geometry and task gate for all three cases: nominal 14.55/14.35 s, switch-before 14.45/14.30 s, switch-after 21.45/21.75 s (Observed/Intent). The late-switch clip was regenerated once, identically for both modes, so its observable switch occurs after first grasp and before transfer. Intent is **0.30 s slower** in this late-switch pilot. These are expert rollouts at one seed; they do not establish a learned-policy benefit.

Reproduce scene renders with:

```bash
MUJOCO_GL=egl PYTHONPATH=.:scripts:.deps OMP_NUM_THREADS=1 /home/hungdao/miniforge3/envs/ur_bullet312/bin/python scripts/layout_gallery.py
```

Reproduce I2 paired pilot with the same environment and `scripts/i2_v2_preview.py`. Reproduce I3 cases using `scripts/i3_collect.py --case CASE --seeds 0 --out outputs/i3_v2_target_pilot`, where `CASE` is each of the three manifest case names. The I3 pilot writes `record.json`, `frames.npz`, and videos; it must be rerun after any scene or controller change.
