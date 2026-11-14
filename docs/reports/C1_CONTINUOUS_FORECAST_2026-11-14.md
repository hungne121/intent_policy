# C1 continuous forecast preview — 2026-11-14

The intent expert follows the ManiCast wrist forecast continuously as the hand starts moving; it does not wait for the forecast to converge on either towel. The current preview uses a rolling +0.5 s forecast, updated each control step. The actual lifted towel is observed before selecting a cup. Forecast errors and changes of intention remain future work. The observed expert waits for the actual towel lift and the hand's return.

One paired seed: both modes placed 2/2 cups. Simulated completion times: observed 23.90 s, intent 20.70 s. Reach onset to forecast-follow command: 0.05 s and 0.15 s for the two rounds; onset to 2 mm actual robot movement: 0.15 s and 0.25 s. These are preview measurements, not a general performance claim. This is a scripted expert preview, not a trained ACT policy.

The existing contact audit reports serious human/robot and human self-overlaps, including head/wrist approximately 38.6 mm. This preview is not approved for dataset collection; contact geometry and motion must be fixed and checked. No claim of real-world safety is made.

Reproduce from the repository root:

```sh
export MUJOCO_GL=egl PYTHONPATH=.:scripts:.deps OMP_NUM_THREADS=1
PY=/home/hungdao/miniforge3/envs/ur_bullet312/bin/python
choom -n 500 -- "$PY" scripts/c1_v2_preview.py --out outputs/c1_follow_repeat
"$PY" scripts/check_c1_preview.py outputs/c1_follow_repeat
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest -q -p no:cacheprovider tests/test_human_paced_v2.py tests/test_human_continuous_skin.py
```

The paired temporal checks passed; the two existing test files passed 7 tests. Videos and summaries are in `previews/c1_follow_20261114/`; detailed traces are in `outputs/c1_follow_20261114_v2/`.
