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
inob__source_target        # SOURCE_TARGET (vagus|spine|spine_vagus) -> TISSUES + TARGET_TAG
inob__load_modules

BASE="${HOME}/Scratch/inob"
inob__activate_venv "${BASE}"

# BLAS thread fan-out (so 16 chunks × N threads matches the reservation).
export OMP_NUM_THREADS="${BLAS_THREADS}"
export OPENBLAS_NUM_THREADS="${BLAS_THREADS}"
export MKL_NUM_THREADS="${BLAS_THREADS}"

CONFIG="${BASE}/configs/default.yaml"
# Per-target chunk dir so vagus / spine / spine_vagus runs never clobber each other.
CHUNKS_DIR="${BASE}/out_${TARGET_TAG}"
inob__log "source target=${SOURCE_TARGET} tissues=${TISSUES} chunks=${CHUNKS_DIR}"

# Make ``python -m inob.*`` work both from a pip-install and from the staged tree.
export PYTHONPATH="${BASE}/code/src:${PYTHONPATH:-}"

# Optional per-target dipole spacing (e.g. muscle → 15 mm); empty ⇒ config default.
SPACING_SET=""
[[ -n "${SOURCE_SPACING:-}" ]] && SPACING_SET="--set forward.source_spacing_mm=${SOURCE_SPACING}"

# Optional explicit FEM mesh. All targets normally share the one staged mesh at
# cfg.outputs.fem_mat, which means restaging it would corrupt any run already in
# flight (tasks that start after the rsync would read a different mesh than the
# ones before it). Point a new target at its own file to stage and queue it
# while another target is still running:
#   FEM_MAT=$HOME/Scratch/inob/outputs/fem/fem_cordfix.mat
FEM_SET=""
[[ -n "${FEM_MAT:-}" ]] && FEM_SET="--set outputs.fem_mat=${FEM_MAT}"

# Fibre-aligned muscle conductivity. The config default (mode="auto") already
# enables it iff `muscle` is a source tissue, so we only emit an explicit
# mode=on/off when MUSCLE_ANISOTROPY_OVERRIDE forces an A/B.
ANISO_SET=""
if [[ -n "${MUSCLE_ANISOTROPY_OVERRIDE:-}" ]]; then
    if [[ "${MUSCLE_ANISOTROPY:-}" == "1" ]]; then
        ANISO_SET="--set forward.muscle_anisotropy.mode=on"
    else
        ANISO_SET="--set forward.muscle_anisotropy.mode=off"
    fi
fi

if [[ "${CHUNKS_PER_TASK}" -le 1 ]]; then
    # One chunk per array task (Myriad-style).
    CHUNK_ID=$(( SGE_TASK_ID - 1 ))
    inob__log "task ${SGE_TASK_ID} → chunk ${CHUNK_ID}/${N_CHUNKS}"
    python -u -m inob.forward.chunk \
        --config "${CONFIG}" \
        --chunk-id "${CHUNK_ID}" \
        --n-chunks "${N_CHUNKS}" \
        --set "forward.source_tissue=${TISSUES}" \
        ${SPACING_SET} \
        ${ANISO_SET} \
        ${FEM_SET} \
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
            --set forward.source_tissue='${TISSUES}' \
            ${SPACING_SET} \
            ${ANISO_SET} \
            ${FEM_SET} \
            --set outputs.forward_chunks_dir='${CHUNKS_DIR}' \
            > '${BASE}/logs/chunk_${TARGET_TAG}_{}.log' 2>&1"
fi

inob__log "task ${SGE_TASK_ID} complete"
