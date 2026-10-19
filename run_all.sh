#!/usr/bin/env bash
# Rebuild everything from raw data: expected points -> projections -> dashboard.
# Usage: ./run_all.sh [--refresh] [--check] [--force]
#   --refresh  delete ./data and download the latest nflverse files
#   --check    stop after the readiness check (exit 0 ready, 2 week incomplete, 3 nothing new)
#   --force    skip the readiness and "already published" checks
set -euo pipefail
cd "$(dirname "$0")"
REFRESH=0; CHECK=0; FORCE=""
for a in "$@"; do case $a in --refresh) REFRESH=1;; --check) CHECK=1;; --force) FORCE="--force";; esac; done
[ "$REFRESH" = 1 ] && rm -rf data
[ -d data ] || ./fetch_data.sh
mkdir -p output
set +e; python3 src/check_ready.py data output $FORCE; rc=$?; set -e
[ $rc -ne 0 ] && exit $rc
[ "$CHECK" = 1 ] && exit 0
python3 src/build_xfp.py data output
SOS_COL=ppr LAMS=20 python3 src/model.py data output
python3 src/build_html.py output data
python3 src/sanity.py output
cp output/dashboard.html docs/index.html
python3 src/check_ready.py data output --record
echo "Done. Open docs/index.html in a browser."
