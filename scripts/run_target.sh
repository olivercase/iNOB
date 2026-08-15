#!/usr/bin/env bash
# Solve one source target end to end and draw every figure it supports.
#
#   bash scripts/run_target.sh vagus
#   bash scripts/run_target.sh spine
#   bash scripts/run_target.sh muscle
#
# This is the local mirror of the cluster procedure in cluster/run_array.sh +
# run_reduce.sh + run_eeg.sh: the same --source-target machinery, the same
# per-target dipole spacing rule from cluster/lib.sh (config default 5 mm for
# vagus and spine; 15 mm for the volume-filled muscle target), and the same
# figures. Running it for each target in turn produces three sets that are
# comparable because nothing but the target changed.
#
# Needs duneuropy. On a machine without a DUNEuro build, run it in the image:
#
#   docker build -t inob:duneuro .
#   docker run --rm -v "$PWD/outputs:/work/outputs" -v "$PWD/data:/work/data" \
#       inob:duneuro bash scripts/run_target.sh vagus
#
# EEG is attempted for every target; muscle has no validated electrode patch of
# its own, so a failure there is expected and reported rather than fatal.
set -uo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-}"
case "${TARGET}" in
    vagus|spine|spine_vagus) SPACING="" ;;
    muscle|spine_muscle)     SPACING="15" ;;   # volume-filled: 5 mm gives thousands
    *) echo "usage: $0 {vagus|spine|spine_vagus|muscle|spine_muscle}" >&2; exit 2 ;;
esac

INOB="python -m inob.cli.main"
# macOS ships bash 3.2, where "${SET[@]}" on an EMPTY array trips `set -u`
# ("unbound variable") instead of expanding to nothing — which killed every
# target two seconds in. Expand through the ${x+y} guard, which is empty-safe
# on both 3.2 and 5.x.
SET=()
[[ -n "${SPACING}" ]] && SET=(--set "forward.source_spacing_mm=${SPACING}")
FAILED=()

run() { echo; echo "### $*"; if ! "$@"; then FAILED+=("$*"); echo "  ^^ FAILED"; fi; }

# SKIP_MEG / SKIP_EEG let a queue re-enter this script for a target whose
# expensive half is already solved, without re-running it.
SKIP_MEG="${SKIP_MEG:-0}"
SKIP_EEG="${SKIP_EEG:-0}"

echo "=== ${TARGET}: solving ==============================================="
# The OPM array is target-independent and shared by every leadfield, so it is
# NOT regenerated here: rebuilding it mid-queue would leave already-solved
# leadfields describing coil positions that no longer exist on disk.
if [[ ! -f outputs/sensors/sensor_array.mat ]]; then
    echo "outputs/sensors/sensor_array.mat missing — run 'inob sensors' first" >&2
    exit 1
fi
# The electrode patch IS target-dependent (it is sited over the target), so it
# is rebuilt per target into its own tagged file.
run $INOB electrodes  --source-target "${TARGET}"
[[ "${SKIP_MEG}" == "1" ]] || run $INOB forward --source-target "${TARGET}" ${SET[@]+"${SET[@]}"}
[[ "${SKIP_EEG}" == "1" ]] || run $INOB eeg     --source-target "${TARGET}" ${SET[@]+"${SET[@]}"}

echo
echo "=== ${TARGET}: figures =============================================== "
run $INOB topoplot    --source-target "${TARGET}"
run $INOB detect      --source-target "${TARGET}"
run $INOB cap-compare --source-target "${TARGET}"
run $INOB cross       --source-target "${TARGET}"
run $INOB physiology  --source-target "${TARGET}"
run python -m inob.cli.sensor_field --source-target "${TARGET}"
run python -m inob.cli.sensor_field --source-target "${TARGET}" --aggregate coherent
run python -m inob.cli.sensor_field --source-target "${TARGET}" --aggregate rms

# The cord's own Q ladder (1 / 5.11 / 10 / 20 nA·m), applied to whichever target
# this is. Run for every target it makes sense for, so the three are compared at
# the same source strengths rather than each at its own literature anchor.
run $INOB detect --source-target "${TARGET}" --q-nAm 1 5.11 10 20 --target detect \
    --out-detect "outputs/detect/${TARGET}_q1_to_20.png"

echo
$INOB status || true
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "=== ${TARGET}: every step succeeded ==="
else
    echo "=== ${TARGET}: ${#FAILED[@]} step(s) failed ==="
    printf '  %s\n' "${FAILED[@]}"
fi
