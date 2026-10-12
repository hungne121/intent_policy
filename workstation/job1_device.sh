#!/usr/bin/env bash
# Job 1: the measuring device, 5-fold (docs/requirements/INTENT_ACT_v3.2.md §4.2c-g). On every non-DART episode of
# intent_act_late_v5 (286; the sim human with the 2026-10-12 hand fixes):
#   render the sensor view -> export DeePoint training data (6 gesture classes, labels from the motion onset)
#   -> fine-tune model k = 0..4 (test fold k, validation fold k + 1, trained on the other 3)
#   -> each model on its test fold (out-of-fold reports) and its validation fold (calibration)
#   -> calibration per model (holds, pause timeout, likelihood width) -> device_folds.json
#   -> outputs/device/current -> this run; upload to $HF_REPO runs/device/<date>
# Resumable: finished steps are skipped. RUN=outputs/device/<date> to continue an earlier run.
set -euo pipefail
cd "$(dirname "$0")/.."
source workstation/env.sh
RUN="${RUN:-outputs/device/$(date +%F)}"
R=$RUN/render
DP=$RUN/dp_multi
mkdir -p "$RUN"
exec > >(tee -a "$RUN/job1.log") 2>&1
log "job 1 -> $RUN (git $(git rev-parse --short HEAD))"

log "1/6 render the sensor view (side camera) of the non-DART episodes"
EPS=$(./run.sh -c "
import json
from scripts.render_sensor_view import select
metas = [json.loads(l) for l in open('outputs/datasets/intent_act_late_v5/meta/hri_episodes.jsonl')]
print(' '.join(map(str, select(metas, 10_000))))")
set -- $EPS
log "  $# episodes"
while [ $# -gt 0 ]; do                  # one process per 15 episodes: the renderer's memory grows ~0.1 GB per episode
  B="${*:1:15}"; shift $(( $# < 15 ? $# : 15 ))
  ./run.sh -m scripts.render_sensor_view --dataset-root outputs/datasets/intent_act_late_v5 --episodes $B --views side \
      --out "$R" --skip-existing | grep -a "^\[" || true
done
./run.sh -c "
import json
from scripts.render_sensor_view import select
metas = [json.loads(l) for l in open('outputs/datasets/intent_act_late_v5/meta/hri_episodes.jsonl')]
json.dump(dict(dataset='$(pwd)/outputs/datasets/intent_act_late_v5', episodes=select(metas, 10_000), views=['side']),
          open('$R/episodes.json', 'w'), indent=1)"
log "  rendered $(ls "$R"/episode_*.json | wc -l)"

log "2/6 export DeePoint data (classes multi, labels from the motion onset)"
[ -f "$DP/export.json" ] || scripts/deepoint.sh scripts/deepoint_export_dp.py --src "$R" --view side --out "$DP" \
    --label-start onset --classes multi 2>&1 | tr '\r' '\n' | grep -a "^\[\|Error\|Traceback" | tail -3

log "3/6 fine-tune the 5 fold models"
for k in 0 1 2 3 4; do
  [ -f "$RUN/fold$k/train/metrics.json" ] && continue
  log "  fold $k"
  scripts/deepoint.sh scripts/deepoint_finetune.py --data "$DP" --test-fold $k --out "$RUN/fold$k/train" \
      > "$RUN/fold$k.train.log" 2>&1
  tail -c 1500 "$RUN/fold$k.train.log" | grep -a -v "it/s" | tail -5
done

log "4/6 inference: model k on folds k (test) and k + 1 (validation)"
CK=$(for k in 0 1 2 3 4; do ./run.sh -c "import json; print(json.load(open('$RUN/fold$k/train/metrics.json'))['best_checkpoint'])"; done)
./run.sh -c "
import json, glob
lines = []
for j in sorted(glob.glob('$R/episode_*.json')):
    m = json.load(open(j)); f = m['fold']; tag = m['tag']
    lines.append(f\"$R/{tag}_side_15fps.mp4 {f}=$R/{tag}_side_m{f}.npz {(f - 1) % 5}=$R/{tag}_side_m{(f - 1) % 5}.npz\")
open('$RUN/infer_jobs.txt', 'w').write('\n'.join(lines) + '\n')"
scripts/deepoint.sh scripts/deepoint_infer.py --ckpt $CK --list "$RUN/infer_jobs.txt" 2>&1 | tr '\r' '\n' | grep -a "^\[" | tail -2

log "5/6 calibration and reports per model"
for k in 0 1 2 3 4; do
  v=$(( (k + 1) % 5 ))
  ./run.sh -m scripts.calibrate_device_rules --dir "$R" --suffix _m$k --test-fold $k --val-fold $v \
      --json "$RUN/fold$k/calibration.json" > "$RUN/fold$k.calibration.log" 2>&1
  ./run.sh -m scripts.deepoint_gesture_report --dir "$R" --suffix _m$k --test-fold $k --val-fold $v \
      > "$RUN/fold$k.report.log" 2>&1
done
./run.sh -m scripts.make_device_folds --run "$RUN"
ln -sfn "$(basename "$RUN")" outputs/device/current

log "6/6 upload to $HF_REPO runs/device/$(basename "$RUN") (without the exported features)"
hfx push "$RUN" "runs/device/$(basename "$RUN")" --exclude "dp_multi/**"
log "job 1 done: next workstation/job2_check.sh, and on the laptop: python workstation/hf.py pull runs/device/$(basename "$RUN") $RUN"
