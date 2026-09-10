#!/usr/bin/env bash
# Bundle the solved outputs for an archival deposit (Zenodo / OSF).
#
#   bash scripts/package_results.sh [outfile.tar.gz]
#
# Collects the leadfields, FEM, geometry, sensor arrays and figures from
# outputs/, records the iNOB version and git commit, and writes a SHA-256
# manifest next to the tarball so a reviewer can check what they downloaded.
# Chunk directories and logs are left out; they are regenerable scratch.
set -euo pipefail
cd "$(dirname "$0")/.."

out="${1:-inob-results-$(python -c 'import inob;print(inob.__version__)').tar.gz}"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

mkdir -p "$stage/inob-results"
for d in forward fem geometry sensors sarvas detect cap; do
    [ -d "outputs/$d" ] || continue
    rsync -a --exclude 'chunks*' --exclude '*.log' "outputs/$d" "$stage/inob-results/"
done
find outputs -maxdepth 1 -name '*.png' -exec cp {} "$stage/inob-results/" \;
cp configs/default.yaml "$stage/inob-results/config_used.yaml"
{
    echo "inob $(python -c 'import inob;print(inob.__version__)')"
    echo "commit $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "packaged $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$stage/inob-results/PROVENANCE.txt"

( cd "$stage/inob-results" && find . -type f ! -name SHA256SUMS -print0 \
    | sort -z | xargs -0 shasum -a 256 > SHA256SUMS )
tar -C "$stage" -czf "$out" inob-results
shasum -a 256 "$out" > "$out.sha256"
echo "wrote $out ($(du -h "$out" | cut -f1)); manifest inside, checksum in $out.sha256"
