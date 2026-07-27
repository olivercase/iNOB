#!/usr/bin/env bash
# Single qsub wrapper. Usage:
#
#   CLUSTER_PROFILE=myriad bash cluster/submit.sh array
#   CLUSTER_PROFILE=kathleen bash cluster/submit.sh build
#   CLUSTER_PROFILE=myriad bash cluster/submit.sh reduce
#
# Reads the profile in cluster/profiles/<name>.env and translates its values
# into SGE flags (queue, parallel-environment, walltime, memory, array shape).
# Submits the matching cluster/run_<task>.sh body script.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/lib.sh"
inob__init
inob__source_target        # SOURCE_TARGET (vagus|spine|spine_vagus) -> TISSUES + TARGET_TAG

WHICH="${1:-}"
case "${WHICH}" in
    array|build|reduce|eeg) ;;
    -h|--help|"") echo "usage: CLUSTER_PROFILE=<myriad|kathleen> bash cluster/submit.sh array|build|reduce|eeg" >&2; exit 2 ;;
    *) echo "unknown task: ${WHICH}" >&2; exit 2 ;;
esac

# submit.sh runs ON the cluster, so expand the literal ``${HOME}`` to a real
# path for the local mkdir and qsub -wd.
eval "REMOTE_BASE_LIT=\"${REMOTE_BASE}\""
LOG_DIR="${REMOTE_BASE_LIT}/logs"
mkdir -p "${LOG_DIR}" 2>/dev/null || true

# Common qsub flags built from the profile.
COMMON_FLAGS=(
    -S /bin/bash
    -wd "${LOG_DIR}"
    -j y
    -V                                # forward env (so CLUSTER_PROFILE propagates)
)

case "${WHICH}" in
    array)
        JOB_NAME="fwd_${TARGET_TAG}"
        FLAGS=(
            -N "${JOB_NAME}"
            -pe ${PE_DIRECTIVE}
            -l "h_rt=${WALLTIME_ARRAY},mem=${MEM_PER_TASK}"
            -t "1-${TASKS}"
            -o '$JOB_NAME.$JOB_ID.$TASK_ID.log'
        )
        # Optional: chain this target behind another one, e.g. queue spine to
        # start only once the muscle run has finished:
        #   HOLD_JID=reduce_muscle SOURCE_TARGET=spine bash cluster/submit.sh array
        [[ -n "${HOLD_JID:-}" ]] && FLAGS+=(-hold_jid "${HOLD_JID}")
        BODY="${HERE}/run_array.sh"
        ;;
    build)
        JOB_NAME="vagus_build"
        FLAGS=(
            -N "${JOB_NAME}"
            -pe ${PE_DIRECTIVE}
            -l "h_rt=${WALLTIME_BUILD},mem=${MEM_PER_TASK}"
            -o '$JOB_NAME.$JOB_ID.log'
        )
        BODY="${HERE}/run_build.sh"
        ;;
    eeg)
        JOB_NAME="eeg_${TARGET_TAG}"
        FLAGS=(
            -N "${JOB_NAME}"
            -pe ${PE_DIRECTIVE}
            -l "h_rt=${WALLTIME_EEG:-${WALLTIME_ARRAY}},mem=${MEM_EEG:-${MEM_PER_TASK}}"
            -o '$JOB_NAME.$JOB_ID.log'
        )
        BODY="${HERE}/run_eeg.sh"
        ;;
    reduce)
        JOB_NAME="reduce_${TARGET_TAG}"
        # Reduce is single-threaded; force the smaller PE if available.
        REDUCE_PE="${PE_DIRECTIVE%% *} 1"
        if [[ "${PE_DIRECTIVE}" == mpi* ]]; then
            # On Kathleen MPI nodes the smallest viable allocation is the full node.
            REDUCE_PE="${PE_DIRECTIVE}"
        fi
        FLAGS=(
            -N "${JOB_NAME}"
            -pe ${REDUCE_PE}
            -l "h_rt=${WALLTIME_REDUCE},mem=${MEM_PER_TASK}"
            -hold_jid "fwd_${TARGET_TAG}${HOLD_JID:+,${HOLD_JID}}"
            -o '$JOB_NAME.$JOB_ID.log'
        )
        BODY="${HERE}/run_reduce.sh"
        ;;
esac

inob__log "qsub ${COMMON_FLAGS[*]} ${FLAGS[*]} ${BODY}"
qsub "${COMMON_FLAGS[@]}" "${FLAGS[@]}" "${BODY}"
