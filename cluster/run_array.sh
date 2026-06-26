#!/bin/bash -l
# Body script for the forward-solve array job. Submitted by cluster/submit.sh.
# Profile (myriad/kathleen) chosen via CLUSTER_PROFILE — forwarded by qsub -V.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Under SGE the body runs from a spool copy; fall back to the staged dir.
[[ -f "${HERE}/lib.sh" ]] || HERE="${INOB_REMOTE_BASE:-${HOME}/Scratch/inob}/code/cluster"
# shellcheck disable=SC1091
source "${HERE}/lib.sh"
inob__init
inob__load_modules

BASE="${HOME}/Scratch/inob"
inob__activate_venv "${BASE}"

# BLAS thread fan-out (so 16 chunks × N threads matches the reservation).
export OMP_NUM_THREADS="${BLAS_THREADS}"
export OPENBLAS_NUM_THREADS="${BLAS_THREADS}"
export MKL_NUM_THREADS="${BLAS_THREADS}"

CONFIG="${BASE}/configs/default.yaml"
CHUNKS_DIR="${BASE}/out"

# Make ``python -m inob.*`` work both from a pip-install and from the staged tree.
export PYTHONPATH="${BASE}/code/src:${PYTHONPATH:-}"

if [[ "${CHUNKS_PER_TASK}" -le 1 ]]; then
    # One chunk per array task (Myriad-style).
    CHUNK_ID=$(( SGE_TASK_ID - 1 ))
    inob__log "task ${SGE_TASK_ID} → chunk ${CHUNK_ID}/${N_CHUNKS}"
    python -u -m inob.forward.chunk \
        --config "${CONFIG}" \
        --chunk-id "${CHUNK_ID}" \
        --n-chunks "${N_CHUNKS}" \
        --set "outputs.forward_chunks_dir=${CHUNKS_DIR}"
else
    # N chunks per task in parallel (Kathleen full-node).
    START=$(( (SGE_TASK_ID - 1) * CHUNKS_PER_TASK ))
    END=$(( START + CHUNKS_PER_TASK - 1 ))
    inob__log "task ${SGE_TASK_ID} → chunks ${START}..${END} on $(hostname)"
    seq "${START}" "${END}" | xargs -P "${CHUNKS_PER_TASK}" -I {} \
        bash -c "python -u -m inob.forward.chunk \
            --config '${CONFIG}' \
            --chunk-id {} --n-chunks ${N_CHUNKS} \
            --set outputs.forward_chunks_dir='${CHUNKS_DIR}' \
            > '${BASE}/logs/chunk_{}.log' 2>&1"
fi

inob__log "task ${SGE_TASK_ID} complete"
