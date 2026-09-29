#!/usr/bin/env bash
# Rebuild main.pdf every time a .tex or .bib file in this folder is saved,
# and keep it open in Preview, which reloads it when the file changes.
#   ./watch.sh          (Ctrl-C to stop)
set -u
cd "$(dirname "$0")"

build() {
    # One TeX pass reusing last run's .aux/.bbl (~8 s instead of ~50 s);
    # --only-cached skips the online bundle check. A new \cite or \label
    # settles on the following save. If a package is missing, fall back
    # to a full online build, which fetches it.
    if tectonic -X compile --only-cached --keep-intermediates --reruns 0 \
           --synctex --keep-logs main.tex >build.log 2>&1 \
       || tectonic -X compile --keep-intermediates --synctex --keep-logs \
           main.tex >build.log 2>&1; then
        echo "$(date +%H:%M:%S)  built main.pdf"
    else
        echo "$(date +%H:%M:%S)  BUILD FAILED:"
        grep -E '^error|^!' build.log | head -5
    fi
}

stamp() { stat -f '%m' *.tex *.bib figures/* 2>/dev/null | sort -n | tail -1; }

build
open -a Preview main.pdf
last=$(stamp)
while sleep 0.5; do
    now=$(stamp)
    if [ "$now" != "$last" ]; then
        last=$now
        build
    fi
done
