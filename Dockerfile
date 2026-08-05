# Reproducible build of the iNOB pipeline INCLUDING DUNEuro.
#
# DUNEuro has no PyPI wheel; it is compiled from source (DUNE 2.10 +
# duneuro + duneuro-py) with the Eigen-5 / DUNE-2.10 patches maintained in
# the separate olivercase/duneuro-build repo (fetched below by tag, the same
# single source of truth cluster/build_duneuro.sh uses). This image bakes
# the whole toolchain so reviewers can regenerate every leadfield without a
# local build.
#
#   docker build -t inob .
#   docker run --rm -v "$PWD/outputs:/work/outputs" inob \
#       inob-forward --config configs/default.yaml
#
# Build is long (DUNE+duneuro compile from source). The DUNE layer is cached
# separately from the duneuro layer so iterating on the patch is cheap.

FROM python:3.11-bookworm AS build

ENV DEBIAN_FRONTEND=noninteractive
# DUNE/duneuro C++ build dependencies. Eigen is deliberately NOT taken from
# apt — see the pinned build below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl pkg-config ca-certificates \
        libgmp-dev libgmpxx4ldbl \
        libmetis-dev libsuperlu-dev libsuitesparse-dev \
        libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

# Eigen, pinned to the version the patch is written against.
#
# The duneuro patch rewrites `m.jacobiSvd(ComputeThinU | ComputeThinV)` into
# the template form `m.jacobiSvd<ComputeThinU | ComputeThinV>()`, which only
# exists from Eigen 5. Bookworm ships 3.4, where that line does not parse —
# the compiler reads `.jacobiSvd` as a member reference and the build died
# with "invalid use of non-static member function" 6 minutes in. Eigen is
# header-only, so pinning it is cheap, and 5.0.1 is the version the working
# local build (cluster/build_duneuro.sh, via Homebrew) compiles against.
#
# BLAS/LAPACK are switched off deliberately: duneuro uses only the headers,
# and those two targets are the one part of Eigen that must be compiled — an
# install without a build looked for libeigen_blas_static.a and failed.
#
# The version is confirmed by compiling against the headers rather than by
# grepping a file: Eigen 5 moved the version macros out of Macros.h, so the
# grep found nothing, exited 1, and took the whole build down with it. This
# check answers the question that actually matters — can a compiler find
# these headers, and what version do they report — and cannot go stale when
# the layout changes again.
ARG EIGEN_VERSION=5.0.1
RUN set -eux; \
    curl -fsSL -o /tmp/eigen.tar.gz \
        "https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz"; \
    tar -xzf /tmp/eigen.tar.gz -C /tmp; \
    cmake -S "/tmp/eigen-${EIGEN_VERSION}" -B /tmp/eigen-build \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_TESTING=OFF \
        -DEIGEN_BUILD_BLAS=OFF -DEIGEN_BUILD_LAPACK=OFF \
        -DEIGEN_BUILD_DOC=OFF; \
    cmake --install /tmp/eigen-build; \
    rm -rf /tmp/eigen.tar.gz "/tmp/eigen-${EIGEN_VERSION}" /tmp/eigen-build; \
    printf '%s\n' \
        '#include <Eigen/Core>' \
        '#include <cstdio>' \
        'int main() { printf("Eigen %d.%d.%d\\n", EIGEN_WORLD_VERSION,' \
        '  EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION); }' > /tmp/eigen_version.cpp; \
    g++ -I/usr/local/include/eigen3 /tmp/eigen_version.cpp -o /tmp/eigen_version; \
    /tmp/eigen_version; \
    rm -f /tmp/eigen_version.cpp /tmp/eigen_version

ENV DUNE_SRC=/opt/dune-src
WORKDIR ${DUNE_SRC}

# ── DUNE 2.10 core/staging/extensions modules ──────────────────────────────
# Each module lives in one of four namespaces and we do not track which, so
# every namespace is tried until one clones. GIT_TERMINAL_PROMPT=0 matters:
# without it a miss makes git ask for a username and fail with "could not read
# Username ... No such device or address", which reads like a credentials
# problem and buried the real (compile) error in the log. A module that never
# clones is a hard failure here rather than a confusing one much later.
ENV GIT_TERMINAL_PROMPT=0
RUN set -eux; \
    for m in dune-common dune-geometry dune-grid dune-istl dune-localfunctions \
             dune-typetree dune-functions dune-uggrid dune-alugrid \
             dune-pdelab dune-subgrid; do \
        for ns in core staging extensions pdelab; do \
            if git clone -q -b releases/2.10 \
                 "https://gitlab.dune-project.org/${ns}/${m}.git" 2>/dev/null; then break; fi; \
        done; \
        test -d "${m}" || { echo "could not clone ${m} from any namespace" >&2; exit 1; }; \
    done

# Pinned duneuro + the source patch (separate layer → fast patch iteration).
# The patch itself lives in olivercase/duneuro-build (the build recipe), not
# in this repo — fetched fresh from a tagged release, same as
# cluster/build_duneuro.sh, so there's one source of truth.
ARG DUNEURO_COMMIT=8f344b4da9c128ddf3e47af5ec136d05a3aeb162
ARG DUNEURO_BUILD_TAG=v1.0.1
RUN set -eux; \
    git clone -q https://gitlab.dune-project.org/duneuro/duneuro.git; \
    git clone -q https://gitlab.dune-project.org/duneuro/duneuro-py.git; \
    git -C duneuro checkout -q "${DUNEURO_COMMIT}"; \
    curl -fsSL -o /tmp/duneuro.patch \
        "https://raw.githubusercontent.com/olivercase/duneuro-build/${DUNEURO_BUILD_TAG}/scripts/patches/duneuro-eigen5-dune210.patch"; \
    git -C duneuro apply /tmp/duneuro.patch

# ── python venv + runtime deps (pinned in requirements-docker.txt) ─────────
ENV VENV=/opt/venv
RUN python -m venv ${VENV}
ENV PATH="${VENV}/bin:${PATH}"
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir --upgrade pip wheel setuptools \
    && pip install --no-cache-dir -r /tmp/requirements-docker.txt

# ── build DUNE + duneuro + duneuro-py ──────────────────────────────────────
# dunecontrol reads CMAKE_FLAGS from the environment (no opts file needed).
ENV EIGEN_INC=/usr/local/include/eigen3
ENV Eigen3_DIR=/usr/local/share/eigen3/cmake
# Eigen now lives under /usr/local, not /usr/include where apt put it, so
# both halves of the lookup have to be told: find_package(Eigen3) via
# Eigen3_DIR, and the compiler itself via -I, because duneuro includes
# <Eigen/Dense> directly in headers that DUNE's target-level include
# paths do not reach. This mirrors what build_duneuro_local.sh passes.
ENV CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CXX_FLAGS='-O3 -DNDEBUG -fPIC -I/usr/local/include/eigen3' -DCMAKE_C_FLAGS='-O3 -DNDEBUG -fPIC' -DBUILD_SHARED_LIBS=ON -DDUNE_ENABLE_PYTHONBINDINGS=ON -DCMAKE_DISABLE_FIND_PACKAGE_MPI=ON -DPython3_EXECUTABLE=/opt/venv/bin/python -DEigen3_DIR=/usr/local/share/eigen3/cmake"
RUN ${DUNE_SRC}/dune-common/bin/dunecontrol all

# Install the compiled duneuro-py extension into the venv's site-packages.
RUN set -eux; \
    PYSITE="$(python -c 'import site; print(site.getsitepackages()[0])')"; \
    LIB="$(find ${DUNE_SRC}/duneuro-py/build-cmake -name 'duneuropy*.so' -o -name 'duneuropy*.dylib' | head -1)"; \
    test -n "${LIB}"; \
    cp "${LIB}" "${PYSITE}/duneuropy.so"; \
    cp ${DUNE_SRC}/duneuro-py/build-cmake/src/*.py "${PYSITE}/" 2>/dev/null || true; \
    python -c "import duneuropy as dp; print('duneuro OK:', dir(dp)[:4])"

# ── install the inob package itself ────────────────────────────────────
WORKDIR /work
COPY . /work
RUN pip install --no-cache-dir --no-deps -e /work

# The whole point of this image is running the real-DUNEuro tests, and pytest
# was not in it — `docker run inob:duneuro pytest` failed with "executable file
# not found", which the duneuro-validation workflow would have hit the moment
# the build itself stopped failing. Installed late and on its own line so test
# tooling can change without invalidating the hour-long compile above.
RUN pip install --no-cache-dir "pytest>=8.0"

# Default to a shell on the venv; CMD is the command to run (e.g. a CLI).
ENV MPLBACKEND=Agg
ENTRYPOINT []
CMD ["bash"]
