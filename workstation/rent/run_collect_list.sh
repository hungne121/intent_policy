#!/usr/bin/env bash
# On the pod (68/QĐ-DL đợt 1, 18/11/2026): unattended collection from a shard list (make_shards_d1.py):
# line = `group task split seed [fault_type]`; one shard = one seed of one task (all modes / variants of the group) in
# its own root. Finished shards go to the private HF dataset repo (uploader.py, one commit / 3 min). After T_MAX_H no
# new shard starts (cost stop, set by the lead from the RunPod cap); then final upload. The laptop deletes the pod.
#   LIST=workstation/rent/shards_d1.txt VER=d1 P=30 T_MAX_H=3.5 HF_REPO=... bash workstation/rent/run_collect_list.sh
set -uo pipefail
cd "${CODE:-/root/hrc}"
P="${P:-30}"; VER="${VER:-d1}"; R=/root/out/hrc_collect_$VER; mkdir -p "$R" /root/out/logs; export OUT_ROOT="$R" HF_PREFIX="shards_$VER"
T0=$(date +%s); export T0 T_MAX_S=$(python3 -c "print(int(float('${T_MAX_H:-3.5}')*3600))")
PROG=/root/out/progress_$VER.txt; export PROG; cp "${LIST:-workstation/rent/shards_d1.txt}" /root/out/shards_$VER.txt
shard() {
  g=$1; t=$2; sp=$3; s0=$4; f=${5:-}; name="${g}_${t}_${sp}_${s0}${f:+_$f}"; root="$R/$name"
  [ -f "$root/DONE" ] && exit 0
  if [ $(( $(date +%s) - T0 )) -gt "$T_MAX_S" ]; then echo "$(date '+%F %T') $name skipped_time" >> $PROG; exit 0; fi
  LP_NUM_THREADS=1 MUJOCO_GL=egl INTENT_PYTHON=/root/venv/bin/python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    ./run.sh -m tools.collect --group "$g" --tasks "$t" --split "$sp" --n-seeds 1 --first-seed "$s0" --root "$root" --pilot-abc \
    ${f:+--fault-types $f} > /root/out/logs/$name.log 2>&1 && touch "$root/DONE"
  echo "$(date '+%F %T') $name exit $?" >> $PROG
}
export -f shard; export R T0 T_MAX_S
/root/venv/bin/python workstation/rent/uploader.py loop > /root/out/uploader_$VER.log 2>&1 &
UP=$!
xargs -P "$P" -L 1 bash -c 'shard "$@"' _ < /root/out/shards_$VER.txt
echo "$(date '+%F %T') ALL_SHARDS_FINISHED" >> $PROG
kill $UP 2>/dev/null; /root/venv/bin/python workstation/rent/uploader.py final >> /root/out/uploader_$VER.log 2>&1
echo "$(date '+%F %T') FINAL_UPLOAD_DONE" >> $PROG
