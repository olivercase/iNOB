# Shared cluster helpers. Source this from every cluster/*.sh script.
#
# Required env vars (set by the profile or by the user):
#   CLUSTER_PROFILE         e.g. myriad / kathleen — selects profiles/<name>.env
#
# User-overridable:
#   VAGUS_FM_LOCAL_DIR      local repo root (defaults to current working tree)
#   VAGUS_FM_REMOTE_HOST    SSH alias / hostname (defaults to REMOTE_HOST in profile)
#   VAGUS_FM_REMOTE_BASE    remote project dir (defaults to REMOTE_BASE in profile)
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
    REMOTE_HOST="${VAGUS_FM_REMOTE_HOST:-${REMOTE_HOST}}"
    REMOTE_BASE="${VAGUS_FM_REMOTE_BASE:-${REMOTE_BASE}}"
    LOCAL_DIR="${VAGUS_FM_LOCAL_DIR:-$(cd -- "$(inob__here)/.." && pwd)}"
    export PROFILE_NAME REMOTE_HOST REMOTE_BASE LOCAL_DIR
    export N_CHUNKS TASKS CHUNKS_PER_TASK CORES_PER_TASK MEM_PER_TASK
    export WALLTIME_ARRAY WALLTIME_BUILD WALLTIME_REDUCE BLAS_THREADS
    export PE_DIRECTIVE
}

inob__load_modules() {
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

inob__init() {
    inob__load_user_env
    inob__load_profile
}
