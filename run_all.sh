#!/usr/bin/env bash
# Rebuild everything from raw data: expected points -> projections -> dashboard.
set -euo pipefail
[ -d data ] || ./fetch_data.sh
mkdir -p output
python3 src/build_xfp.py data output
SOS_COL=ppr LAMS=20 python3 src/model.py data output
python3 src/build_html.py output
cp output/dashboard.html docs/index.html
echo "Open docs/index.html in a browser."
