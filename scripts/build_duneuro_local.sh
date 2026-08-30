#!/usr/bin/env bash
# One-time DUNE 2.10 + duneuro + duneuro-py build for a *local* dev machine
# (macOS/Homebrew or a Linux box without environment modules). The cluster
# equivalent, which drives the same sources through SGE + `module load`, is
# cluster/build_duneuro.sh; this is the version environment.yml points at.
#
#   bash scripts/build_duneuro_local.sh
#
# Overridable:
#   INOB_VENV           venv to install duneuropy into   (default: <repo>/.venv)
#   INOB_DUNEURO_BASE   where the sources are built      (default: ~/.local/share/inob-duneuro)
#
# Re-running is idempotent: clones, patch and cmake configure all no-op when
# they are already in the state they want to be.

set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${INOB_VENV:-${REPO}/.venv}"
BASE="${INOB_DUNEURO_BASE:-${HOME}/.local/share/inob-duneuro}"
SRC="${BASE}/duneuro-src"

log() { printf '[build_duneuro_local] %s\n' "$*" >&2; }

# The module clones below probe several DUNE namespaces and *expect* most of
# them to 404. A 404 on gitlab.dune-project.org is served as an auth challenge,
# so an interactive git (or an editor's askpass helper, which is what
# GIT_ASKPASS points at inside a VS Code terminal) will sit forever on a
# credential prompt instead of failing through to the next namespace. Force
# every git in this script to be non-interactive so a miss is an immediate miss.
export GIT_TERMINAL_PROMPT=0
export GIT_ASKPASS=""
export SSH_ASKPASS=""
export GIT_CONFIG_PARAMETERS="'credential.helper='"

# ── prerequisites ──────────────────────────────────────────────────────────
[[ -x "${VENV}/bin/python" ]] || {
    log "ERROR: no venv at ${VENV}. Create one first:"
    log "  python3.11 -m venv ${VENV} && ${VENV}/bin/pip install -r requirements-docker.txt"
    exit 1; }

PYEXE="${VENV}/bin/python"
PYVER="$("${PYEXE}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
log "venv python ${PYVER} at ${PYEXE}"

for tool in cmake git curl pkg-config; do
    command -v "${tool}" >/dev/null 2>&1 || { log "ERROR: ${tool} not on PATH"; exit 1; }
done

# ── C++ dependencies ───────────────────────────────────────────────────────
# Two supported sources, in priority order: an activated conda env built from
# environment.yml, or Homebrew. SuperLU is deliberately not among them:
# dune-istl 2.10's superlufunctions.hh is written against the SuperLU <=5 ABI,
# where the single-precision complex type is ::complex. SuperLU 6 renamed it to
# singlecomplex, so anything current (7.x on both conda-forge and Homebrew)
# makes duneuro fail to compile with "no type named 'complex' in the global
# namespace". It is only an optional direct solver — the verified cluster
# profiles load no SuperLU either — so the finder is switched off below and
# dune-istl falls back to UMFPACK from SuiteSparse.
CMAKE_PREFIXES=()
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/include/eigen3" ]]; then
    log "C++ dependencies from the active conda env: ${CONDA_PREFIX}"
    CMAKE_PREFIXES+=("${CONDA_PREFIX}")
elif command -v brew >/dev/null 2>&1; then
    log "C++ dependencies from Homebrew"
    MISSING=()
    for pkg in eigen gmp metis suite-sparse tbb; do
        if prefix="$(brew --prefix "${pkg}" 2>/dev/null)" && [[ -d "${prefix}" ]]; then
            CMAKE_PREFIXES+=("${prefix}")
        else
            MISSING+=("${pkg}")
        fi
    done
    [[ ${#MISSING[@]} -eq 0 ]] || { log "ERROR: brew install ${MISSING[*]}"; exit 1; }
else
    log "ERROR: no conda env active and no brew on PATH — see environment.yml"
    exit 1
fi
PREFIX_PATH="$(IFS=';'; echo "${CMAKE_PREFIXES[*]}")"

# Both halves of the Eigen lookup have to be told where it lives:
# find_package(Eigen3) via Eigen3_DIR, and the compiler itself via -I, because
# duneuro includes <Eigen/Dense> directly from headers that DUNE's target-level
# include paths never reach (both packagers nest them under include/eigen3).
# The Dockerfile passes the same pair for the same reason.
EIGEN_PREFIX=""
for cand in "${CONDA_PREFIX:-}" "$(brew --prefix eigen 2>/dev/null || true)" /usr/local /usr; do
    [[ -n "${cand}" && -f "${cand}/include/eigen3/Eigen/Dense" ]] && { EIGEN_PREFIX="${cand}"; break; }
done
EIGEN_PREFIX="${EIGEN_PREFIX:-/usr/local}"
EIGEN_INC="${EIGEN_PREFIX}/include/eigen3"
EIGEN_CMAKE_DIR="${EIGEN_PREFIX}/share/eigen3/cmake"
[[ -f "${EIGEN_INC}/Eigen/Dense" ]] || { log "ERROR: Eigen/Dense not under ${EIGEN_INC}"; exit 1; }
log "eigen: inc=${EIGEN_INC} cmake=${EIGEN_CMAKE_DIR}"

mkdir -p "${SRC}"

# ── python hints for duneuro-py's CMake ────────────────────────────────────
# duneuro-py configures with the deprecated find_package(PythonLibs), which
# does not locate a venv/framework Python on its own. Compute explicit hints
# for both the old (PYTHON_*) and new (Python3_*) finders. On macOS the
# framework layout makes sysconfig's LIBDIR/LDLIBRARY pair a path that does
# not exist, so probe the real library instead of trusting it.
PYINC="$("${PYEXE}" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
PYLIBDIR="$("${PYEXE}" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
PYPREFIX="$("${PYEXE}" -c 'import sys; print(sys.base_prefix)')"
PYLIB=""
for cand in \
    "${PYLIBDIR}/libpython${PYVER}.dylib" \
    "${PYLIBDIR}/libpython${PYVER}.so" \
    "${PYPREFIX}/Python" \
    "${PYLIBDIR}/$("${PYEXE}" -c 'import sysconfig; print(sysconfig.get_config_var("LDLIBRARY") or "")')"
do
    [[ -f "${cand}" ]] && { PYLIB="${cand}"; break; }
done
if [[ -z "${PYLIB}" ]]; then
    PYLIB="$(find "${PYLIBDIR}" -maxdepth 2 \( -name "libpython${PYVER}*.dylib" -o -name "libpython${PYVER}*.so*" \) 2>/dev/null | head -1)"
fi
[[ -f "${PYLIB}" && -f "${PYINC}/Python.h" ]] || {
    log "ERROR: could not locate libpython/Python.h (inc=${PYINC} lib=${PYLIB:-none})"; exit 1; }
log "python for duneuro-py: inc=${PYINC} lib=${PYLIB}"

# ── DUNE 2.10 modules ──────────────────────────────────────────────────────
cd "${SRC}"
DUNE_MODS=(dune-common dune-geometry dune-grid dune-istl dune-localfunctions \
           dune-typetree dune-functions dune-uggrid dune-alugrid \
           dune-pdelab dune-subgrid)
for m in "${DUNE_MODS[@]}"; do
    # A previous run killed mid-clone leaves a directory with no .git; treat
    # that as absent rather than as a usable checkout.
    [[ -d "${m}" && ! -d "${m}/.git" ]] && { log "discarding partial ${m}"; rm -rf "${m}"; }
    if [[ ! -d "${m}" ]]; then
        log "cloning ${m}"
        cloned=""
        for ns in core staging extensions pdelab; do
            if git clone -q -b releases/2.10 "https://gitlab.dune-project.org/${ns}/${m}.git" 2>/dev/null; then
                cloned="${ns}"; break
            fi
            rm -rf "${m}"     # git leaves the directory behind on a failed clone
        done
        [[ -n "${cloned}" ]] || { log "ERROR: could not clone DUNE module ${m} from any namespace"; exit 1; }
        log "  ${m} <- ${cloned}"
    fi
done

[[ -d duneuro    ]] || git clone -q https://gitlab.dune-project.org/duneuro/duneuro.git
[[ -d duneuro-py ]] || git clone -q https://gitlab.dune-project.org/duneuro/duneuro-py.git

# ── pin duneuro + apply the Eigen-5 / DUNE-2.10 patch ──────────────────────
# Same pins and same single source of truth as cluster/build_duneuro.sh: the
# patch lives in olivercase/duneuro-build and is fetched by tag, never from a
# branch that could drift underneath us.
DUNEURO_COMMIT="8f344b4da9c128ddf3e47af5ec136d05a3aeb162"
DUNEURO_BUILD_TAG="v1.0.1"
DUNEURO_PATCH="${SRC}/duneuro-eigen5-dune210.patch"
log "fetching duneuro patch from olivercase/duneuro-build@${DUNEURO_BUILD_TAG}"
curl -fsSL -o "${DUNEURO_PATCH}" \
    "https://raw.githubusercontent.com/olivercase/duneuro-build/${DUNEURO_BUILD_TAG}/scripts/patches/duneuro-eigen5-dune210.patch"
git -C "${SRC}/duneuro" fetch -q origin "${DUNEURO_COMMIT}" 2>/dev/null || true
git -C "${SRC}/duneuro" reset --hard -q "${DUNEURO_COMMIT}"
git -C "${SRC}/duneuro" clean -fdq
log "applying duneuro patch"
if git -C "${SRC}/duneuro" apply --reverse --check "${DUNEURO_PATCH}" 2>/dev/null; then
    log "duneuro patch already applied — skipping"
elif git -C "${SRC}/duneuro" apply --check "${DUNEURO_PATCH}" 2>/dev/null; then
    git -C "${SRC}/duneuro" apply "${DUNEURO_PATCH}"
else
    log "git apply refused — falling back to GNU patch --fuzz=3"
    patch -d "${SRC}/duneuro" -p1 --fuzz=3 --ignore-whitespace < "${DUNEURO_PATCH}"
fi

# ── configure + build ──────────────────────────────────────────────────────
cat > "${SRC}/release.opts" <<OPTS
CMAKE_FLAGS="
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CXX_STANDARD=20
  -DCMAKE_CXX_FLAGS='-O3 -DNDEBUG -fPIC -I${EIGEN_INC}'
  -DCMAKE_C_FLAGS='-O3 -DNDEBUG -fPIC'
  -DCMAKE_PREFIX_PATH='${PREFIX_PATH}'
  -DEigen3_DIR=${EIGEN_CMAKE_DIR}
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  -DCMAKE_DISABLE_FIND_PACKAGE_SuperLU=ON
  -DCMAKE_DISABLE_FIND_PACKAGE_MPI=ON
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

log "running dunecontrol all (this is the long step)"
"${SRC}/dune-common/bin/dunecontrol" --opts="${SRC}/release.opts" all

# ── install the extension into the venv ────────────────────────────────────
# dunecontrol already built duneuropy under duneuro-py/build-cmake; a
# standalone cmake here cannot resolve dune-common's package config outside
# that build, so copy the artefact it produced.
log "installing duneuropy into ${VENV}"
PYSITE="$("${PYEXE}" -c 'import site; print(site.getsitepackages()[0])')"
DUNEUROPY_SO="$(find "${SRC}/duneuro-py/build-cmake" \( -name 'duneuropy*.so' -o -name 'duneuropy*.dylib' \) 2>/dev/null | head -1)"
[[ -n "${DUNEUROPY_SO}" ]] || { log "ERROR: duneuropy not found under duneuro-py/build-cmake"; exit 1; }
cp "${DUNEUROPY_SO}" "${PYSITE}/duneuropy.so"
# duneuro-py also emits pure-python helpers next to the extension; the Docker
# image installs them alongside it, so keep the two environments identical.
cp "${SRC}"/duneuro-py/build-cmake/src/*.py "${PYSITE}/" 2>/dev/null || true
log "copied ${DUNEUROPY_SO} -> ${PYSITE}/duneuropy.so"

log "verifying import"
"${PYEXE}" -c "import duneuropy as dp; print('duneuro OK:', dir(dp)[:5])"
log "build complete"
