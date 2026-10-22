#!/bin/bash
# D conditioning study (docs/requirements/INTENT_ACT_v3.2.md §5.3): variants of D (early + intent) trained on the
# prelim early dataset with D's step count, each checked offline (scripts/intent_reliance.py) right after training,
# then evaluated closed-loop with its intent and with every intent token masked, then compared
# (scripts/compare_conditioning.py) with D as trained in the 2x2 (D0) and against B.
#   film        FiLM of every ResNet stage by the intent tokens (BC-Z, RT-1, MT-ACT)
#   vdrop       modality dropout: all image tokens of a training sample masked with IMG_P (ModDrop)
#   film_vdrop  both
# Usage: RUN=outputs/prelim/2026-10-13_slow bash scripts/d_conditioning.sh [variant ...]   (default: all three)
# Finished steps are skipped when re-run.
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-outputs/prelim/2026-10-13_slow}
STEPS=${STEPS:-$(grep -o -- '--steps [0-9]*' "$RUN/train/D/command.txt" | awk '{print $2}')}
IMG_P=${IMG_P:-0.1}                     # ModDrop's per-modality rate (Neverova et al., TPAMI 2016)
EVAL_WORKERS=${EVAL_WORKERS:-1}
DS=$RUN/datasets/intent_act_device_early_prelim
CFG=configs/experiments/intent_act_device_early.yaml
OUT=$RUN/conditioning
VARIANTS=("$@"); [ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(film vdrop film_vdrop)
mkdir -p "$OUT" "$RUN/train" "$RUN/eval" "$RUN/reliance"
py() { choom -n 1000 -- ./run.sh "$@"; }
log() { echo "[$(date '+%F %T')] $*"; }
flags_of() {
  case $1 in
    film) echo --intent-film ;;
    vdrop) echo --image-p-drop "$IMG_P" ;;
    film_vdrop) echo --intent-film --image-p-drop "$IMG_P" ;;
    *) echo "unknown variant $1" >&2; return 1 ;;
  esac
}
ckpt() { echo "$RUN/train/$1/checkpoints/last/pretrained_model"; }

log "variants ${VARIANTS[*]}; steps $STEPS; IMG_P $IMG_P"
[ -f "$RUN/reliance/D/reliance.json" ] || py -m scripts.intent_reliance --checkpoint "$(ckpt D)" --out "$RUN/reliance/D" > /dev/null
for v in "${VARIANTS[@]}"; do
  M=D_$v
  f=$(flags_of "$v") || exit 1
  if [ ! -f "$(ckpt $M)/model.safetensors" ] || ! grep -q "step=$STEPS " "$RUN/train/$M.log" 2>/dev/null; then
    log "train $M ($f)"
    py -m scripts.train_policy --config $CFG --dataset-root "$DS" --repo-id local/intent_act_device_early_prelim \
      --use-intent --intent-source device $f --steps "$STEPS" --save-every $((STEPS / 2)) \
      --output-dir "$RUN/train/$M" $( [ -d "$RUN/train/$M/checkpoints" ] && echo --resume ) >> "$RUN/train/$M.log" 2>&1 \
      || { log "train $M failed: $RUN/train/$M.log"; exit 1; }
    grep -a "val_loss" "$RUN/train/$M.log" | tail -n 1 | cut -c1-160
  fi
  [ -f "$RUN/reliance/$M/reliance.json" ] || py -m scripts.intent_reliance --checkpoint "$(ckpt $M)" --out "$RUN/reliance/$M" > /dev/null
  log "offline reliance $M"; sed -n '5,20p' "$RUN/reliance/$M/reliance.md"
done
for v in "${VARIANTS[@]}"; do
  M=D_$v
  for mode in with masked; do
    E=$RUN/eval/$M; extra=(--workers "$EVAL_WORKERS")
    [ $mode = masked ] && { E=$RUN/eval/${M}_masked; extra=(--intent-keep none --workers 2); }
    [ -f "$E/results.json" ] && continue
    log "eval $M ($mode)"
    py -m scripts.evaluate --config $CFG --checkpoint "$(ckpt $M)" --list "$RUN/lists/eval.jsonl" \
      --temporal-ensemble-coeff=0.01 "${extra[@]}" --videos 1 --output-dir "$E" > "$E.log" 2>&1 \
      || { log "eval $M ($mode) failed: $E.log"; exit 1; }
  done
done
args=(--variant "D0=$RUN/eval/D,$RUN/eval/D_masked,$RUN/reliance/D")
for v in "${VARIANTS[@]}"; do
  M=D_$v; [ -f "$RUN/eval/${M}_masked/results.json" ] && args+=(--variant "$M=$RUN/eval/$M,$RUN/eval/${M}_masked,$RUN/reliance/$M")
done
py -m scripts.compare_conditioning "${args[@]}" --base "B=$RUN/eval/B" --output "$OUT/compare.md" > /dev/null
log "done: $OUT/compare.md"
