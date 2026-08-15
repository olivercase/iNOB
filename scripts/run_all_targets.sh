#!/usr/bin/env bash
# Solve and draw all three worked examples — vagus, spine, muscle — back to back.
#
#   bash scripts/run_all_targets.sh
#
# Sequential on purpose. Each forward solve already splits the coil array across
# every core (forward.local_workers = 0), so running two targets at once would
# halve each one's cores and finish no sooner.
#
# It waits first for any forward solve already in flight, so it can be queued
# behind a running job rather than fighting it for the machine.
#
# Needs duneuropy. On this Mac that lives in the UCL-volume build, so run it as:
#
#   PYTHONPATH=$PWD/src /Volumes/UCL/duneuro_build/venv/bin/python -V   # check
#   INOB_PY=/Volumes/UCL/duneuro_build/venv/bin/python \
#       bash scripts/run_all_targets.sh
#
# Per-target skips let the queue re-enter a target whose expensive half is done:
#   SKIP_VAGUS_MEG=1   the vagus MEG leadfield is already solved
set -uo pipefail
cd "$(dirname "$0")/.."

INOB_PY="${INOB_PY:-python}"
export PATH="$(dirname "$(command -v "${INOB_PY}")"):${PATH}"
export PYTHONPATH="${PYTHONPATH:-}:$PWD/src"

LOGDIR="outputs/logs"
mkdir -p "${LOGDIR}"

# Wait out a solve that is already running, so this can be queued behind one.
while pgrep -f "inob.cli.main forward" >/dev/null 2>&1 \
   || pgrep -f "inob.forward.chunk" >/dev/null 2>&1; do
    echo "[queue] a forward solve is running — waiting…"
    sleep 60
done

started="$(date -u +%Y%m%dT%H%M%SZ)"
for target in vagus spine muscle; do
    log="${LOGDIR}/queue_${target}_${started}.log"
    echo
    echo "############ ${target} — $(date -u +%H:%M:%SZ) — log: ${log}"
    # Skip the MEG solve when its leadfield is already on disk, so the queue is
    # resumable: a target that finished before an interruption is not re-solved
    # for hours. FORCE_MEG=1 re-solves everything regardless.
    meg="outputs/forward/duneuro_leadfield_${target}.npz"
    if [[ -f "${meg}" && "${FORCE_MEG:-0}" != "1" ]]; then
        echo "[queue] ${meg} exists — skipping the MEG solve (FORCE_MEG=1 to redo)"
        SKIP_MEG=1 bash scripts/run_target.sh "${target}" 2>&1 | tee "${log}"
    else
        bash scripts/run_target.sh "${target}" 2>&1 | tee "${log}"
    fi
    echo "############ ${target} done — $(date -u +%H:%M:%SZ)"
done

echo
echo "############ all three targets attempted. Leadfields on disk:"
ls -la outputs/forward/*.npz 2>/dev/null || echo "  (none)"
