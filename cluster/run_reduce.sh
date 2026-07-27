#!/bin/bash -l
# Body script for the reduce step (stitches per-chunk leadfields).
# Submitted by cluster/submit.sh after the array job finishes.
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

CONFIG="${BASE}/configs/default.yaml"
# Must match run_array.sh's per-target layout so reduce stitches the right chunks.
CHUNKS_DIR="${BASE}/out_${TARGET_TAG}"
FINAL="${BASE}/duneuro_leadfield_${TARGET_TAG}.npz"

export PYTHONPATH="${BASE}/code/src:${PYTHONPATH:-}"

# Optional per-target dipole spacing (e.g. muscle → 15 mm); empty ⇒ config default.
# MUST match run_array.sh's spacing: reduce re-derives the source count to check
# it against the chunks, so a mismatch (5 mm here vs 15 mm in the array) aborts.
SPACING_SET=""
[[ -n "${SOURCE_SPACING:-}" ]] && SPACING_SET="--set forward.source_spacing_mm=${SOURCE_SPACING}"

inob__log "reducing chunks from ${CHUNKS_DIR} → ${FINAL} (target=${SOURCE_TARGET}, tissues=${TISSUES})"
python -u -m inob.forward.reduce \
    --config "${CONFIG}" \
    --set "forward.source_tissue=${TISSUES}" \
    ${SPACING_SET} \
    --set "outputs.forward_chunks_dir=${CHUNKS_DIR}" \
    --set "outputs.forward_npz=${FINAL}"

inob__log "reduce complete. Pull back with:"
inob__log "  scp ${REMOTE_HOST:-<host>}:${FINAL} <local>/outputs/forward/"
