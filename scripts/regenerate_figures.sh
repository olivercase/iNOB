#!/usr/bin/env bash
# Regenerate every figure that can be produced from the forward models
# currently saved under outputs/forward/ and outputs/{fem,geometry}/.
#
# Models on disk drive what is possible:
#   duneuro_leadfield_spine.npz       spine MEG leadfield   -> spine figures
#   duneuro_eeg_leadfield_spine.npz   spine EEG (paddle)    -> cross-modality
#   duneuro_leadfield_muscle.npz      muscle MEG leadfield  -> muscle figures
# There is NO vagus leadfield on disk, so vagus figures are not regenerated.
#
# NOT regenerated here (need work this machine cannot do):
#   sensitivity        -> re-solves DUNEuro with perturbed conductivities (no duneuropy locally)
#   location           -> needs duneuro_eeg_leadfield_wholebody_spine.npz (not saved)
#   detect (muscle)    -> needs a muscle EEG leadfield (only MEG is saved for muscle)
#   physiology (muscle)-> no validated muscle dynamics model (PHYSIOLOGY-TODO)
#   snr                -> numbers/JSON only, produces no figure
#
# Sarvas (analytic-sphere) figures are produced separately into outputs/sarvas/
# by scripts/muscle_field_sarvas.py — kept apart from the FEM figures on purpose.
set -uo pipefail
cd "$(dirname "$0")/.."
INOB="python -m inob.cli.main"
FAILED=()

run() { echo "### $*"; if ! "$@"; then FAILED+=("$*"); echo "  ^^ FAILED"; fi; }

# --- Geometry / mesh (target-independent) --------------------------------
run $INOB visualise

# --- Spine (MEG + EEG) ----------------------------------------------------
run $INOB topoplot    --source-target spine
run $INOB detect      --source-target spine
run $INOB cap-compare --source-target spine
run $INOB cross       --source-target spine
run $INOB physiology  --source-target spine
run python -m inob.cli.sensor_field --source-target spine
run python -m inob.cli.sensor_field --source-target spine --aggregate coherent
run python -m inob.cli.sensor_field --source-target spine --aggregate rms
run python -m inob.cli.sensor_field --source-target spine --level c7 --aggregate coherent
run python -m inob.cli.sensor_field --source-target spine --level c7 --aggregate rms

# --- Muscle (MEG only) ----------------------------------------------------
run $INOB topoplot    --source-target muscle --target meg
run $INOB cap-compare --source-target muscle
run python -m inob.cli.sensor_field --source-target muscle
run python -m inob.cli.sensor_field --source-target muscle --aggregate coherent

# --- Sarvas (analytic sphere) -> outputs/sarvas/ --------------------------
run python scripts/muscle_field_sarvas.py

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All figure commands succeeded."
else
  echo "Failed commands (${#FAILED[@]}):"
  printf '  %s\n' "${FAILED[@]}"
fi
