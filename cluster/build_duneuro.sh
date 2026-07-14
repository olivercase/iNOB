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
cd "${SRC}"
DUNE_MODS=(dune-common dune-geometry dune-grid dune-istl dune-localfunctions \
           dune-typetree dune-functions dune-uggrid dune-alugrid \
           dune-pdelab dune-subgrid)
for m in "${DUNE_MODS[@]}"; do
    if [[ ! -d "${m}" ]]; then
        # Try each known DUNE namespace in priority order; fail loudly if none work.
        if   git clone -b releases/2.10 "https://gitlab.dune-project.org/core/${m}.git"        2>/dev/null; then :;
        elif git clone -b releases/2.10 "https://gitlab.dune-project.org/staging/${m}.git"     2>/dev/null; then :;
        elif git clone -b releases/2.10 "https://gitlab.dune-project.org/extensions/${m}.git"  2>/dev/null; then :;
        elif git clone -b releases/2.10 "https://gitlab.dune-project.org/pdelab/${m}.git"      2>/dev/null; then :;
        else
            inob__log "ERROR: could not clone DUNE module ${m} from any namespace"
            exit 1
        fi
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
DUNEURO_BUILD_TAG="v1.0.0"
DUNEURO_PATCH="${SRC}/duneuro-eigen5-dune210.patch"
inob__log "fetching duneuro patch from olivercase/duneuro-build@${DUNEURO_BUILD_TAG}"
curl -fsSL -o "${DUNEURO_PATCH}" \
    "https://raw.githubusercontent.com/olivercase/duneuro-build/${DUNEURO_BUILD_TAG}/scripts/patches/duneuro-eigen5-dune210.patch"
# Reset to the pinned commit and discard any prior patch/edits so re-runs are
# idempotent (a stale half-patched tree makes `git apply` fail both ways).
git -C "${SRC}/duneuro" reset --hard "${DUNEURO_COMMIT}" >/dev/null
git -C "${SRC}/duneuro" clean -fdq
inob__log "applying duneuro patch"
git -C "${SRC}/duneuro" apply "${DUNEURO_PATCH}"

cat > "${SRC}/release.opts" <<'OPTS'
CMAKE_FLAGS="
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CXX_STANDARD=20
  -DCMAKE_CXX_FLAGS='-O3 -DNDEBUG -fPIC'
  -DCMAKE_C_FLAGS='-O3 -DNDEBUG -fPIC'
  -DBUILD_SHARED_LIBS=ON
  -DDUNE_ENABLE_PYTHONBINDINGS=ON
"
OPTS

inob__log "running dunecontrol all (this is the long step)"
"${SRC}/dune-common/bin/dunecontrol" --opts="${SRC}/release.opts" all

# Install duneuro-py extension into the venv.
cd "${SRC}/duneuro-py"
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=20 \
      -DPython3_EXECUTABLE="${BASE}/venv/bin/python" \
      ..
make -j"${CORES_PER_TASK:-4}"
PYSITE="$("${BASE}/venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
cp -r src/duneuropy* "${PYSITE}/" 2>/dev/null || \
    find . -name 'duneuropy*.so' -exec cp {} "${PYSITE}/" \;

inob__log "verifying import"
"${BASE}/venv/bin/python" -c "import duneuropy as dp; print('duneuro OK:', dir(dp)[:5])"
inob__log "build complete"
