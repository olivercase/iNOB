#!/usr/bin/env bash
# Stage inputs (FEM + sensors) and the inob package to the cluster.
# Profile-driven: select with CLUSTER_PROFILE=myriad|kathleen.
#
#   CLUSTER_PROFILE=myriad bash cluster/stage.sh
#   CLUSTER_PROFILE=kathleen bash cluster/stage.sh --dry-run

set -euo pipefail

# rsync >=3.4 protects remote args by default ("secluded args"), so the remote
# shell no longer expands the literal ``${HOME}`` in REMOTE_BASE (below). Restore
# the old pass-through-the-remote-shell behaviour so that expansion still works.
export RSYNC_OLD_ARGS=1

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/lib.sh"
inob__init
# TARGET_TAG selects which per-target electrode array to stage (see below).
inob__source_target

DRY_RUN=""
for arg in "$@"; do
    case "${arg}" in
        --dry-run|-n) DRY_RUN="--dry-run" ;;
        -h|--help) sed -n '2,9p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown arg: ${arg}" >&2; exit 2 ;;
    esac
done

# REMOTE_BASE is already the literal ``${HOME}/...`` (sourced with \$), so the
# remote shell expands ${HOME}. No local rewrite needed.
REMOTE_BASE_LIT="${REMOTE_BASE}"

# Exclude macOS AppleDouble / Finder junk (the repo lives on an external
# /Volumes mount that sprays ._* and .DS_Store files) from every transfer.
EXCL=(--exclude '._*' --exclude '.DS_Store')

inob__log "ensuring remote dirs on ${REMOTE_HOST}: ${REMOTE_BASE_LIT}"
# Pre-create nested leaf dirs too: rsync only mkdirs one missing level, and the
# remote rsync (3.1.2) is too old for --mkpath, so code/scripts/patches must exist.
ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_BASE_LIT}/{out,logs,configs,outputs/fem,outputs/sensors,outputs/forward,code/src,code/cluster,code/scripts/patches,data/muscle}"

# Inputs must land where the config expects them (cfg.outputs.fem_mat =
# outputs/fem/fem.mat, cfg.outputs.sensors_mat = outputs/sensors/...),
# not a flat inputs/ dir, or inob.forward.chunk can't find them.
inob__log "rsyncing inputs (FEM + sensors)"
rsync -avh ${DRY_RUN} "${EXCL[@]}" --progress \
    "${LOCAL_DIR}/outputs/fem/fem.mat" \
    "${REMOTE_HOST}:${REMOTE_BASE_LIT}/outputs/fem/"
rsync -avh ${DRY_RUN} "${EXCL[@]}" --progress \
    "${LOCAL_DIR}/outputs/sensors/sensor_array.mat" \
    "${REMOTE_HOST}:${REMOTE_BASE_LIT}/outputs/sensors/"
# HD electrode array (cfg.outputs.electrodes_mat) — needed by the EEG forward
# solve (inob.cli.run_eeg). The patch is sited over the target tissue, so
# --source-target writes a per-target file (electrode_array_<tag>.mat); the
# untagged name is still staged for older runs. Optional: skip quietly if not
# generated.
ELEC_STAGED=0
for elec in "electrode_array_${TARGET_TAG}.mat" "electrode_array.mat" "electrode_array_wholebody.mat"; do
    if [[ -f "${LOCAL_DIR}/outputs/sensors/${elec}" ]]; then
        rsync -avh ${DRY_RUN} "${EXCL[@]}" --progress \
            "${LOCAL_DIR}/outputs/sensors/${elec}" \
            "${REMOTE_HOST}:${REMOTE_BASE_LIT}/outputs/sensors/"
        ELEC_STAGED=1
    fi
done
if [[ "${ELEC_STAGED}" -eq 0 ]]; then
    inob__log "no electrode array locally — skipping (run 'inob-electrodes --source-target ${TARGET_TAG}' for EEG)"
fi

# Muscle STLs (data/muscle/*.stl). The muscle target reads these directly at
# solve time — muscle source sampling and the fibre-aligned anisotropy tensors
# both need them on the cluster; the FEM .mat does NOT carry them. Without this,
# SOURCE_TARGET=muscle array tasks die with "no muscle STLs in .../data/muscle".
# Cheap + harmless for other targets, so always stage.
if [[ -d "${LOCAL_DIR}/data/muscle" ]]; then
    inob__log "rsyncing muscle STLs (data/muscle/)"
    rsync -avh ${DRY_RUN} "${EXCL[@]}" \
        "${LOCAL_DIR}/data/muscle/" \
        "${REMOTE_HOST}:${REMOTE_BASE_LIT}/data/muscle/"
fi

inob__log "rsyncing package source (src/ + cluster/ + configs/)"
rsync -avh ${DRY_RUN} "${EXCL[@]}" \
    --exclude '__pycache__' --exclude '*.egg-info' \
    "${LOCAL_DIR}/src/" \
    "${REMOTE_HOST}:${REMOTE_BASE_LIT}/code/src/"

rsync -avh ${DRY_RUN} "${EXCL[@]}" \
    --exclude '*.md' \
    "${LOCAL_DIR}/cluster/" \
    "${REMOTE_HOST}:${REMOTE_BASE_LIT}/code/cluster/"

rsync -avh ${DRY_RUN} "${EXCL[@]}" \
    "${LOCAL_DIR}/configs/" \
    "${REMOTE_HOST}:${REMOTE_BASE_LIT}/configs/"

rsync -avh ${DRY_RUN} "${EXCL[@]}" \
    "${LOCAL_DIR}/pyproject.toml" \
    "${LOCAL_DIR}/requirements-cluster.txt" \
    "${REMOTE_HOST}:${REMOTE_BASE_LIT}/code/"

inob__log "marking scripts executable"
ssh "${REMOTE_HOST}" "chmod +x ${REMOTE_BASE_LIT}/code/cluster/*.sh 2>/dev/null || true"

inob__log "remote layout:"
ssh "${REMOTE_HOST}" "ls -lh ${REMOTE_BASE_LIT}/inputs ${REMOTE_BASE_LIT}/configs"

cat <<EOF

Next steps on ${REMOTE_HOST} (profile=${PROFILE_NAME}):

  ssh ${REMOTE_HOST}
  cd ${REMOTE_BASE_LIT}/code

  # One-time DUNEuro build (interactive compute node):
  qrsh -pe ${PE_DIRECTIVE} -l mem=${MEM_PER_TASK},h_rt=${WALLTIME_BUILD} -now no
  cd ${REMOTE_BASE_LIT}/code
  CLUSTER_PROFILE=${PROFILE_NAME} bash cluster/build_duneuro.sh

  # Then submit the array + reduce:
  CLUSTER_PROFILE=${PROFILE_NAME} bash cluster/submit.sh array
  CLUSTER_PROFILE=${PROFILE_NAME} bash cluster/submit.sh reduce   # holds on vagus_fwd

EOF
