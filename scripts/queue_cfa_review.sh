#!/bin/bash
# Queue for the CFA/HEF review simulations.
#   (a) wait until the taVNS job (PID 1230, 'vagal_population') has exited (poll 60 s; safe if already gone)
#   (b) then run simulations 1-5 at nice 15 with at most 4 worker processes
# Log: outputs/cfa_review/run.log   Marker on completion: outputs/cfa_review/DONE (per-sim exit codes)
# Start detached:  perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' nohup scripts/queue_cfa_review.sh </dev/null >/dev/null 2>&1 &
WAIT_PID=${WAIT_PID:-1230}
WAIT_TAG=${WAIT_TAG:-vagal_population}
PY=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3     # the repo has no .venv; this one has numpy/scipy/h5py/trimesh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${QUEUE_OUT:-$ROOT/outputs/cfa_review}"      # QUEUE_OUT / NREP / PROCS only for plumbing tests
NREP=${NREP:-16}
PROCS=${PROCS:-4}                                     # never more than 4
[ "$PROCS" -gt 4 ] && PROCS=4
LOG="$OUT/run.log"
mkdir -p "$OUT"
rm -f "$OUT/DONE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

log "queue started (waiter PID $$); waiting for PID $WAIT_PID ($WAIT_TAG)"
while kill -0 "$WAIT_PID" 2>/dev/null && ps -p "$WAIT_PID" -o command= 2>/dev/null | grep -q "$WAIT_TAG"; do
  sleep 60
done
log "PID $WAIT_PID no longer running; starting simulations (nice 15, 4 processes)"

cd "$ROOT/scripts" || exit 1
codes=""
nice -n 15 "$PY" cfa_review_regress.py head > "$OUT/regression_head.txt" 2>&1; codes="$codes regression=$?"
for s in 1 2 3 4 5; do
  log "sim $s start"
  nice -n 15 "$PY" cfa_review_sims.py --sim $s --nrep "$NREP" --procs "$PROCS" --outdir "$OUT" >> "$LOG" 2>&1
  rc=$?; codes="$codes sim$s=$rc"; log "sim $s exit $rc"
done
nice -n 15 "$PY" cfa_review_summary.py --dir "$OUT" >> "$LOG" 2>&1; codes="$codes summary=$?"
echo "finished $(date '+%F %T')  exit codes:$codes" > "$OUT/DONE"
log "done:$codes"
