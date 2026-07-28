#!/bin/bash -l
# Body script for the EEG forward solve (DUNEuro surface-potential leadfield).
# Submitted non-interactively by cluster/submit.sh:
#
#   SOURCE_TARGET=spine CLUSTER_PROFILE=myriad bash cluster/submit.sh eeg
#
# Unlike the MEG forward (a 32-task array + reduce), the EEG solve is a single
# transfer-matrix solve over all electrodes, so it runs as one plain qsub job.
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
# Match the MEG reduce convention: final leadfield at the base, tagged by target.
FINAL="${BASE}/duneuro_eeg_leadfield_${TARGET_TAG}.npz"

export PYTHONPATH="${BASE}/code/src:${PYTHONPATH:-}"

# Optional per-target dipole spacing (e.g. muscle → 15 mm); empty ⇒ config default.
# MUST match the MEG run's spacing so EEG/MEG share source positions.
SPACING_SET=""
[[ -n "${SOURCE_SPACING:-}" ]] && SPACING_SET="--set forward.source_spacing_mm=${SOURCE_SPACING}"

# MUST match the MEG run's conductivity model (muscle target → anisotropic).
# mode="auto" in the config handles this; only force it for an explicit A/B.
ANISO_SET=""
if [[ -n "${MUSCLE_ANISOTROPY_OVERRIDE:-}" ]]; then
    if [[ "${MUSCLE_ANISOTROPY:-}" == "1" ]]; then
        ANISO_SET="--set forward.muscle_anisotropy.mode=on"
    else
        ANISO_SET="--set forward.muscle_anisotropy.mode=off"
    fi
fi

# The HD patch is sited over the target tissue, so each target has its own
# array file (electrode_array_<tag>.mat). Fall back to the untagged name for
# runs staged before per-target patches existed.
#
# WHOLEBODY=1 swaps this for the target-independent 1,000-contact whole-skin
# net (outputs/sensors/electrode_array_wholebody.mat, generated locally via
# `inob electrodes --set electrodes.shape=whole_body --set
# outputs.electrodes_mat=outputs/sensors/electrode_array_wholebody.mat` and
# staged like any other input) — used by `inob location` to compare the
# cervical paddle against a full-body sweep.
if [[ "${WHOLEBODY:-0}" == "1" ]]; then
    ELEC="${BASE}/outputs/sensors/electrode_array_wholebody.mat"
    FINAL="${BASE}/duneuro_eeg_leadfield_wholebody_${TARGET_TAG}.npz"
else
    ELEC="${BASE}/outputs/sensors/electrode_array_${TARGET_TAG}.mat"
    if [[ ! -f "${ELEC}" ]]; then
        ELEC="${BASE}/outputs/sensors/electrode_array.mat"
        inob__log "WARNING: no electrode_array_${TARGET_TAG}.mat — falling back to"
        inob__log "  ${ELEC}, which may be sited over a different tissue. Regenerate"
        inob__log "  with 'inob-electrodes --source-target ${SOURCE_TARGET}' and restage."
    fi
fi

# Optional explicit FEM mesh — see run_array.sh. MUST match the MEG run's mesh.
FEM_SET=""
[[ -n "${FEM_MAT:-}" ]] && FEM_SET="--set outputs.fem_mat=${FEM_MAT}"

inob__log "EEG forward solve (target=${SOURCE_TARGET}, tissues=${TISSUES}) → ${FINAL}"
inob__log "  electrodes: ${ELEC}"
# source_tissue MUST match the MEG run's target so EEG/MEG share source positions.
python -u -m inob.cli.run_eeg \
    --config "${CONFIG}" \
    --set "forward.source_tissue=${TISSUES}" \
    ${SPACING_SET} \
    ${ANISO_SET} \
    ${FEM_SET} \
    --set "outputs.electrodes_mat=${ELEC}" \
    --set "outputs.forward_eeg_npz=${FINAL}"

inob__log "EEG solve complete. Pull back with:"
inob__log "  scp ${REMOTE_HOST:-<host>}:${FINAL} <local>/outputs/forward/"
