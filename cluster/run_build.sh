#!/bin/bash -l
# Body script for the one-time DUNEuro build job (qsub-driven version of
# build_duneuro.sh). Submitted by cluster/submit.sh.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/build_duneuro.sh"
