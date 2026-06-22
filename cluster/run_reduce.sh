#!/bin/bash -l
# Body script for the reduce step (stitches per-chunk leadfields).
# Submitted by cluster/submit.sh after the array job finishes.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/lib.sh"
vagus_fm__init
vagus_fm__load_modules

BASE="${HOME}/Scratch/vagus_fm"
vagus_fm__activate_venv "${BASE}"

CONFIG="${BASE}/configs/default.yaml"
CHUNKS_DIR="${BASE}/out"
FINAL="${BASE}/duneuro_leadfield_vagus.npz"

export PYTHONPATH="${BASE}/code/src:${PYTHONPATH:-}"

vagus_fm__log "reducing chunks from ${CHUNKS_DIR} → ${FINAL}"
python -u -m vagus_fm.forward.reduce \
    --config "${CONFIG}" \
    --set "outputs.forward_chunks_dir=${CHUNKS_DIR}" \
    --set "outputs.forward_npz=${FINAL}"

vagus_fm__log "reduce complete. Pull back with:"
vagus_fm__log "  scp ${REMOTE_HOST:-<host>}:${FINAL} <local>/outputs/forward/"
