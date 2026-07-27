# Shared cluster helpers. Source this from every cluster/*.sh script.
#
# Required env vars (set by the profile or by the user):
#   CLUSTER_PROFILE         e.g. myriad / kathleen — selects profiles/<name>.env
#
# User-overridable:
#   INOB_LOCAL_DIR      local repo root (defaults to current working tree)
#   INOB_REMOTE_HOST    SSH alias / hostname (defaults to REMOTE_HOST in profile)
#   INOB_REMOTE_BASE    remote project dir (defaults to REMOTE_BASE in profile)
#
# The legacy hardcoded ``rmgpohk`` / ``Desktop/fif`` paths are gone — set the
# above env vars in your shell profile or ~/.inob.env. The profile defaults
# to ``${HOME}/Scratch/inob`` on the cluster, which works for any user.

set -euo pipefail

inob__here() {
    # Directory containing this lib.sh (i.e. <repo>/cluster).
    cd -- "$(dirname -- "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd
}

inob__load_user_env() {
    if [[ -f "${HOME}/.inob.env" ]]; then
        # shellcheck disable=SC1090,SC1091
        source "${HOME}/.inob.env"
    fi
}

inob__load_profile() {
    local profile="${CLUSTER_PROFILE:-}"
    if [[ -z "${profile}" ]]; then
        echo "[lib.sh] CLUSTER_PROFILE not set. Choose one of: $(ls "$(inob__here)/profiles/" | sed 's/\.env$//')" >&2
        return 2
    fi
    local profile_file
    profile_file="$(inob__here)/profiles/${profile}.env"
    if [[ ! -f "${profile_file}" ]]; then
        echo "[lib.sh] profile not found: ${profile_file}" >&2
        return 2
    fi
    # shellcheck disable=SC1090
    source "${profile_file}"

    # Allow user env overrides.
    REMOTE_HOST="${INOB_REMOTE_HOST:-${REMOTE_HOST}}"
    REMOTE_BASE="${INOB_REMOTE_BASE:-${REMOTE_BASE}}"
    LOCAL_DIR="${INOB_LOCAL_DIR:-$(cd -- "$(inob__here)/.." && pwd)}"
    export PROFILE_NAME REMOTE_HOST REMOTE_BASE LOCAL_DIR
    export N_CHUNKS TASKS CHUNKS_PER_TASK CORES_PER_TASK MEM_PER_TASK
    export WALLTIME_ARRAY WALLTIME_BUILD WALLTIME_REDUCE BLAS_THREADS
    export PE_DIRECTIVE
}

inob__load_modules() {
    if ! command -v module >/dev/null 2>&1; then
        # SGE runs body scripts with ``-S /bin/bash`` (non-login), so /etc/profile
        # is not sourced and the ``module`` function is undefined even though
        # MODULESHOME/MODULEPATH are imported via ``qsub -V``. Initialise it.
        for _init in "${MODULESHOME:-}/init/bash" /etc/profile.d/modules.sh; do
            if [[ -n "${_init}" && -f "${_init}" ]]; then
                # shellcheck disable=SC1090,SC1091
                source "${_init}" && break
            fi
        done
    fi
    if ! command -v module >/dev/null 2>&1; then
        echo "[lib.sh] 'module' not available; skipping module loads (dev machine?)" >&2
        return 0
    fi
    module purge
    for m in "${MODULES[@]}"; do
        module load "${m}"
    done
}

inob__activate_venv() {
    local base="${1:-${HOME}/Scratch/inob}"
    # shellcheck disable=SC1090,SC1091
    source "${base}/venv/bin/activate"
}

inob__log() {
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[${ts}] [${PROFILE_NAME:-?}] $*"
}

inob__source_target() {
    # Map SOURCE_TARGET -> TISSUES (comma-sep tissue labels for the solver) and
    # TARGET_TAG (slug for per-target chunk dirs, output NPZ, and job names).
    # Lets one pipeline run vagus, spine, or both without editing configs.
    local t="${SOURCE_TARGET:-vagus}"
    # SOURCE_SPACING: per-target dipole spacing (mm). Empty ⇒ use the config's
    # forward.source_spacing_mm. Muscle is volume-filled, so a 5 mm grid yields
    # thousands of sources; default it to a tractable 15 mm (~380 sources).
    SOURCE_SPACING=""
    # MUSCLE_ANISOTROPY: when "1", the solver gives the muscle compartment a
    # fibre-aligned conductivity tensor (σ∥≠σ⊥). Only the muscle target sets it,
    # so vagus/spine leadfields stay isotropic and byte-identical.
    MUSCLE_ANISOTROPY=""
    case "${t}" in
        vagus)       TISSUES="vagus_left";             TARGET_TAG="vagus" ;;
        spine)       TISSUES="spinal_cord";            TARGET_TAG="spine" ;;
        spine_vagus) TISSUES="spinal_cord,vagus_left"; TARGET_TAG="spine_vagus" ;;
        muscle)      TISSUES="muscle";                 TARGET_TAG="muscle"
                     SOURCE_SPACING="${SOURCE_SPACING:-15}"
                     MUSCLE_ANISOTROPY="${MUSCLE_ANISOTROPY:-1}" ;;
        *)
            echo "[lib.sh] unknown SOURCE_TARGET '${t}'. Use: vagus | spine | spine_vagus | muscle" >&2
            return 2 ;;
    esac
    # Honour a caller-provided SOURCE_SPACING override for any target.
    SOURCE_SPACING="${SOURCE_SPACING_OVERRIDE:-${SOURCE_SPACING}}"
    MUSCLE_ANISOTROPY="${MUSCLE_ANISOTROPY_OVERRIDE:-${MUSCLE_ANISOTROPY}}"
    export SOURCE_TARGET="${t}" TISSUES TARGET_TAG SOURCE_SPACING MUSCLE_ANISOTROPY
}

inob__init() {
    inob__load_user_env
    inob__load_profile
}
