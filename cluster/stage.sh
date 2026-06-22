#!/usr/bin/env bash
# Stage inputs (FEM + sensors) and the vagus_fm package to the cluster.
# Profile-driven: select with CLUSTER_PROFILE=myriad|kathleen.
#
#   CLUSTER_PROFILE=myriad bash cluster/stage.sh
#   CLUSTER_PROFILE=kathleen bash cluster/stage.sh --dry-run

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/lib.sh"
vagus_fm__init

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

vagus_fm__log "ensuring remote dirs on ${REMOTE_HOST}: ${REMOTE_BASE_LIT}"
ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_BASE_LIT}/{inputs,code,out,logs,configs}"

vagus_fm__log "rsyncing inputs (FEM + sensors)"
rsync -avh ${DRY_RUN} --progress \
    "${LOCAL_DIR}/outputs/fem/fem_vagus.mat" \
    "${LOCAL_DIR}/outputs/sensors/sensor_array.mat" \
    "${REMOTE_HOST}:${REMOTE_BASE_LIT}/inputs/"

vagus_fm__log "rsyncing package source (src/ + cluster/ + configs/)"
rsync -avh ${DRY_RUN} \
    --exclude '__pycache__' --exclude '*.egg-info' \
    "${LOCAL_DIR}/src/" \
    "${REMOTE_HOST}:${REMOTE_BASE_LIT}/code/src/"

rsync -avh ${DRY_RUN} \
    --exclude '*.md' \
    "${LOCAL_DIR}/cluster/" \
    "${REMOTE_HOST}:${REMOTE_BASE_LIT}/code/cluster/"

rsync -avh ${DRY_RUN} \
    "${LOCAL_DIR}/configs/" \
    "${REMOTE_HOST}:${REMOTE_BASE_LIT}/configs/"

rsync -avh ${DRY_RUN} \
    "${LOCAL_DIR}/pyproject.toml" \
    "${LOCAL_DIR}/requirements-cluster.txt" \
    "${REMOTE_HOST}:${REMOTE_BASE_LIT}/code/"

vagus_fm__log "marking scripts executable"
ssh "${REMOTE_HOST}" "chmod +x ${REMOTE_BASE_LIT}/code/cluster/*.sh 2>/dev/null || true"

vagus_fm__log "remote layout:"
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
