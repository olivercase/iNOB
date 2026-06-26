#!/bin/bash -l
# Body script for the one-time DUNEuro build job (qsub-driven version of
# build_duneuro.sh). Submitted by cluster/submit.sh.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Under SGE the body runs from a spool copy, so BASH_SOURCE is not the staged
# cluster/ dir; fall back to the known staged location.
[[ -f "${HERE}/build_duneuro.sh" ]] || HERE="${INOB_REMOTE_BASE:-${HOME}/Scratch/inob}/code/cluster"
exec bash "${HERE}/build_duneuro.sh"
