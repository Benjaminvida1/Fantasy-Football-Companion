#!/usr/bin/env bash
# Downloads the nflverse data this project uses into ./data (about 250 MB).
set -euo pipefail
B=https://github.com/nflverse/nflverse-data/releases/download
mkdir -p data && cd data
for f in stats_player/stats_player_week_{2023,2024,2025,2026}.csv \
         snap_counts/snap_counts_{2024,2025,2026}.csv \
         pbp/play_by_play_{2024,2025,2026}.csv \
         injuries/injuries_2026.csv players/players.csv schedules/games.csv; do
  echo "fetching $f"; curl -fsSL -o "$(basename "$f")" "$B/$f"
done
