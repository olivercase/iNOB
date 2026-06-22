#!/bin/bash -l
# Body script for the reduce step (stitches per-chunk leadfields).
# Submitted by cluster/submit.sh after the array job finishes.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/lib.sh"
inob__init
inob__load_modules

BASE="${HOME}/Scratch/inob"
inob__activate_venv "${BASE}"

CONFIG="${BASE}/configs/default.yaml"
CHUNKS_DIR="${BASE}/out"
FINAL="${BASE}/duneuro_leadfield_vagus.npz"

export PYTHONPATH="${BASE}/code/src:${PYTHONPATH:-}"

inob__log "reducing chunks from ${CHUNKS_DIR} → ${FINAL}"
python -u -m inob.forward.reduce \
    --config "${CONFIG}" \
    --set "outputs.forward_chunks_dir=${CHUNKS_DIR}" \
    --set "outputs.forward_npz=${FINAL}"

inob__log "reduce complete. Pull back with:"
inob__log "  scp ${REMOTE_HOST:-<host>}:${FINAL} <local>/outputs/forward/"
