#!/usr/bin/env bash
# One-time DUNE 2.10 + duneuro + duneuro-py build, profile-driven.
# Run on a compute node (not login):
#   qrsh -pe smp 4 -l mem=8G,h_rt=2:00:00 -now no
#   cd <base>/code && CLUSTER_PROFILE=myriad bash cluster/build_duneuro.sh

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Load user env early so INOB_REMOTE_BASE is available for the fallback
# below (inob__init hasn't run yet, so lib.sh hasn't sourced it yet).
[[ -f "${HOME}/.inob.env" ]] && source "${HOME}/.inob.env"
# Under SGE the body may run from a spool copy; fall back to the staged dir.
[[ -f "${HERE}/lib.sh" ]] || HERE="${INOB_REMOTE_BASE:-${HOME}/Scratch/inob}/code/cluster"
# shellcheck disable=SC1091
source "${HERE}/lib.sh"
inob__init

# REMOTE_BASE is now set by inob__init (profile default, overridable via
# INOB_REMOTE_BASE in ~/.inob.env). Use it for all paths so this script
# works regardless of what directory the user staged the code to.
BASE="${REMOTE_BASE}"
SRC="${BASE}/duneuro-src"
mkdir -p "${SRC}" "${BASE}"

inob__log "loading modules for ${PROFILE_NAME}"
inob__load_modules
export CC=gcc CXX=g++
# Force UTF-8 in Python/pip — these nodes' C.UTF-8 locale falls back to ascii,
# which makes pip choke on any non-ascii byte in a requirements file.
export PYTHONUTF8=1

# ── python venv with manylinux2014-friendly pins ───────────────────────────
if [[ ! -d "${BASE}/venv" ]]; then
    python3 -m venv "${BASE}/venv"
fi
# shellcheck disable=SC1091
source "${BASE}/venv/bin/activate"
pip install --upgrade pip wheel
pip install --only-binary=:all: -r "${HERE}/../requirements-cluster.txt"

# Install the inob package itself so cluster jobs can ``python -m inob.*``.
pip install --no-deps -e "${HERE}/.."

# ── DUNE 2.10 modules ──────────────────────────────────────────────────────
# The clone loop below probes several DUNE namespaces and *expects* most of
# them to miss. gitlab.dune-project.org answers a miss with an auth challenge
# rather than a plain 404, so an interactive git — or one whose GIT_ASKPASS
# points at an editor helper, as it does inside a VS Code terminal — sits on a
# credential prompt forever instead of failing through to the next namespace.
# Force every git here to be non-interactive so a miss is an immediate miss.
# (The Dockerfile sets GIT_TERMINAL_PROMPT=0 for the same reason.)
export GIT_TERMINAL_PROMPT=0
export GIT_ASKPASS=""
export SSH_ASKPASS=""
export GIT_CONFIG_PARAMETERS="'credential.helper='"

cd "${SRC}"
DUNE_MODS=(dune-common dune-geometry dune-grid dune-istl dune-localfunctions \
           dune-typetree dune-functions dune-uggrid dune-alugrid \
           dune-pdelab dune-subgrid)
for m in "${DUNE_MODS[@]}"; do
    # A run killed mid-clone leaves a directory with no .git behind; treat that
    # as absent rather than as a usable checkout.
    if [[ -d "${m}" && ! -d "${m}/.git" ]]; then
        inob__log "discarding partial ${m}"
        rm -rf "${m}"
    fi
    if [[ ! -d "${m}" ]]; then
        # Try each known DUNE namespace in priority order; fail loudly if none work.
        cloned=""
        for ns in core staging extensions pdelab; do
            if git clone -b releases/2.10 "https://gitlab.dune-project.org/${ns}/${m}.git" 2>/dev/null; then
                cloned="${ns}"; break
            fi
            rm -rf "${m}"     # git leaves the directory behind on a failed clone
        done
        if [[ -z "${cloned}" ]]; then
            inob__log "ERROR: could not clone DUNE module ${m} from any namespace"
            exit 1
        fi
        inob__log "cloned ${m} from ${cloned}"
    fi
done

# duneuro
[[ -d duneuro    ]] || git clone https://gitlab.dune-project.org/duneuro/duneuro.git
[[ -d duneuro-py ]] || git clone https://gitlab.dune-project.org/duneuro/duneuro-py.git

# Pin duneuro + apply the Eigen-5 / DUNE-2.10 source patch. The patch itself
# lives in the separate olivercase/duneuro-build repo (the build recipe),
# not in this repo — fetch it fresh each run from a tagged release so there's
# one source of truth and the URL can't drift out from under us via a future
# push to that repo's main branch.
DUNEURO_COMMIT="8f344b4da9c128ddf3e47af5ec136d05a3aeb162"
DUNEURO_BUILD_TAG="v1.0.1"
DUNEURO_PATCH="${SRC}/duneuro-eigen5-dune210.patch"
inob__log "fetching duneuro patch from olivercase/duneuro-build@${DUNEURO_BUILD_TAG}"
curl -fsSL -o "${DUNEURO_PATCH}" \
    "https://raw.githubusercontent.com/olivercase/duneuro-build/${DUNEURO_BUILD_TAG}/scripts/patches/duneuro-eigen5-dune210.patch"
# Reset to the pinned commit and discard any prior patch/edits so re-runs are
# idempotent (a stale half-patched tree makes `git apply` fail both ways).
git -C "${SRC}/duneuro" reset --hard "${DUNEURO_COMMIT}" >/dev/null
git -C "${SRC}/duneuro" clean -fdq
inob__log "applying duneuro patch"
# git apply has no fuzz factor and is strict about blank-context lines, so fall
# back to GNU patch (which tolerates whitespace/offset drift) if it refuses.
if git -C "${SRC}/duneuro" apply --reverse --check "${DUNEURO_PATCH}" 2>/dev/null; then
    inob__log "duneuro patch already applied — skipping"
elif git -C "${SRC}/duneuro" apply --check "${DUNEURO_PATCH}" 2>/dev/null; then
    git -C "${SRC}/duneuro" apply "${DUNEURO_PATCH}"
else
    inob__log "git apply refused — falling back to GNU patch --fuzz=3"
    patch -d "${SRC}/duneuro" -p1 --fuzz=3 --ignore-whitespace < "${DUNEURO_PATCH}"
fi

# duneuro-py configures with the deprecated find_package(PythonLibs), which does
# not locate the module/venv Python on its own (dunecontrol aborts with
# "Could NOT find PythonLibs"). Compute explicit hints so both the old (PYTHON_*)
# and new (Python3_*) CMake finders resolve to our venv Python.
PYEXE="${BASE}/venv/bin/python"
PYINC="$("${PYEXE}" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
PYLIBDIR="$("${PYEXE}" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
PYLDLIB="$("${PYEXE}" -c 'import sysconfig; print(sysconfig.get_config_var("LDLIBRARY"))')"
PYLIB="${PYLIBDIR}/${PYLDLIB}"
if [[ ! -f "${PYLIB}" ]]; then
    # LDLIBRARY may be a static lib or live in a multiarch subdir; find the .so.
    PYLIB="$(find "${PYLIBDIR}" -maxdepth 2 -name 'libpython3.*.so*' 2>/dev/null | head -1)"
fi
inob__log "python for duneuro-py: exe=${PYEXE} inc=${PYINC} lib=${PYLIB}"
[[ -f "${PYLIB}" && -f "${PYINC}/Python.h" ]] || {
    inob__log "ERROR: could not locate libpython/Python.h (inc=${PYINC} lib=${PYLIB})"; exit 1; }

cat > "${SRC}/release.opts" <<OPTS
CMAKE_FLAGS="
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CXX_STANDARD=20
  -DCMAKE_CXX_FLAGS='-O3 -DNDEBUG -fPIC'
  -DCMAKE_C_FLAGS='-O3 -DNDEBUG -fPIC'
  -DBUILD_SHARED_LIBS=ON
  -DDUNE_ENABLE_PYTHONBINDINGS=ON
  -DDUNE_PYTHON_INSTALL_LOCATION=none
  -DPython3_EXECUTABLE=${PYEXE}
  -DPython3_INCLUDE_DIR=${PYINC}
  -DPython3_LIBRARY=${PYLIB}
  -DPYTHON_EXECUTABLE=${PYEXE}
  -DPYTHON_INCLUDE_DIR=${PYINC}
  -DPYTHON_LIBRARY=${PYLIB}
"
OPTS

inob__log "running dunecontrol all (this is the long step)"
"${SRC}/dune-common/bin/dunecontrol" --opts="${SRC}/release.opts" all

# Install the duneuro-py extension into the venv. dunecontrol already built
# duneuropy.so under duneuro-py/build-cmake as part of `all`; a standalone cmake
# here fails because it can't resolve the dune-common package config outside the
# dunecontrol build, so just copy the artefact it produced.
inob__log "installing duneuropy into venv"
PYSITE="$("${BASE}/venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
DUNEUROPY_SO="$(find "${SRC}/duneuro-py/build-cmake" -name 'duneuropy*.so' 2>/dev/null | head -1)"
[[ -n "${DUNEUROPY_SO}" ]] || { inob__log "ERROR: duneuropy.so not found under duneuro-py/build-cmake"; exit 1; }
cp "${DUNEUROPY_SO}" "${PYSITE}/"
inob__log "copied ${DUNEUROPY_SO} -> ${PYSITE}/"

inob__log "verifying import"
"${BASE}/venv/bin/python" -c "import duneuropy as dp; print('duneuro OK:', dir(dp)[:5])"
inob__log "build complete"
