#!/usr/bin/env bash
# On the pod (64/QĐ-DL, 17/11/2026): unattended HRC collection. One shard = one seed of one task (all modes / variants of
# the group) -> its own root. Order: round-robin over seeds across the 10 tasks, so a partial run stays balanced.
# Finished shards are uploaded to a private HF dataset repo (uploader.py); after T_MAX_H hours no new shard starts;
# when all started shards end: final upload, then the pod removes itself (runpodctl, pod-scoped key).
#   P=30 T_MAX_H=1.6 HF_REPO=Hungdd88/hrc-collect-20261117 nohup bash workstation/rent/run_collect_shards.sh &
set -uo pipefail
cd "${CODE:-/root/hrc}"
P="${P:-30}"; VER="${VER:-}"; R=/root/out/hrc_collect${VER:+_$VER}; mkdir -p "$R" /root/out/logs; export OUT_ROOT="$R" HF_PREFIX="shards${VER:+_$VER}"
T0=$(date +%s); export T0 T_MAX_S=$(python3 -c "print(int(float('${T_MAX_H:-1.6}')*3600))")
TASKS="${TASKS_N1:-W1 W2 W3 C1 C2 C3 C4 I1 I2 I3}"; TASKS6="${TASKS_N6-C4 I3}"; TRAIN0="${TRAIN0:-300110}"; NTRAIN="${NTRAIN:-40}"; VAL0="${VAL0:-300800}"
SH=/root/out/shards${VER:+_$VER}.txt; PROG=/root/out/progress${VER:+_$VER}.txt; export PROG; : > $SH
for i in $(seq 0 $((NTRAIN - 1))); do
  s=$((TRAIN0 + i))
  for t in $TASKS; do echo "N1 $t train $s" >> $SH; done
  for t in $TASKS6; do echo "N6 $t train $s" >> $SH; done
  if [ $((i % 4)) -eq 3 ]; then v=$((VAL0 + i / 4))
    for t in $TASKS; do echo "N1 $t val $v" >> $SH; done
    for t in $TASKS6; do echo "N6 $t val $v" >> $SH; done
  fi
done
shard() {
  g=$1; t=$2; sp=$3; s0=$4; name="${g}_${t}_${sp}_${s0}"; root="$R/$name"
  [ -f "$root/DONE" ] && exit 0
  if [ $(( $(date +%s) - T0 )) -gt "$T_MAX_S" ]; then echo "$(date '+%F %T') $name skipped_time" >> $PROG; exit 0; fi
  LP_NUM_THREADS=1 MUJOCO_GL=egl INTENT_PYTHON=/root/venv/bin/python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    ./run.sh -m tools.collect --group "$g" --tasks "$t" --split "$sp" --n-seeds 1 --first-seed "$s0" --root "$root" --pilot-abc \
    > /root/out/logs/$name.log 2>&1 && touch "$root/DONE"
  echo "$(date '+%F %T') $name exit $?" >> $PROG
}
export -f shard; export R T0 T_MAX_S
/root/venv/bin/python workstation/rent/uploader.py loop > /root/out/uploader${VER:+_$VER}.log 2>&1 &
UP=$!
xargs -P "$P" -L 1 bash -c 'shard "$@"' _ < $SH
echo "$(date '+%F %T') ALL_SHARDS_FINISHED" >> $PROG
kill $UP 2>/dev/null; /root/venv/bin/python workstation/rent/uploader.py final >> /root/out/uploader${VER:+_$VER}.log 2>&1
echo "$(date '+%F %T') FINAL_UPLOAD_DONE" >> $PROG
# the pod-scoped key cannot remove the pod (Unauthorized, 17/11): the laptop deletes it after FINAL_UPLOAD_DONE
