"""Is the latest NFL week complete in the downloaded data, and is it new?

Usage: python3 check_ready.py <data_dir> <out_dir> [--force] [--record]
Exit codes: 0 = a new complete week is ready, 2 = latest week is not complete in all datasets,
            3 = nothing new since the last recorded update.
--record writes <out_dir>/last_update.json with the last complete week (run after a successful build).
"""
import sys, os, json
import pandas as pd

SEASON = 2026
data, out = sys.argv[1], sys.argv[2]
force, record = '--force' in sys.argv, '--record' in sys.argv

g = pd.read_csv(f'{data}/games.csv', usecols=['game_id', 'season', 'week', 'game_type', 'away_score'])
g = g[(g.season == SEASON) & (g.game_type == 'REG')]
done = sorted(w for w, x in g.groupby('week') if x.away_score.notna().all())
if not done:
    print('NOT READY: no fully played week in the schedule file'); sys.exit(2)
W = max(done)
ids = set(g[g.week == W].game_id)

st = pd.read_csv(f'{data}/stats_player_week_{SEASON}.csv', usecols=['week', 'season_type', 'team'])
teams_with_stats = st[(st.season_type == 'REG') & (st.week == W)].team.nunique()
snaps = set(pd.read_csv(f'{data}/snap_counts_{SEASON}.csv', usecols=['game_id', 'week']).query('week == @W').game_id)
pbp = set(pd.read_csv(f'{data}/play_by_play_{SEASON}.csv', usecols=['game_id', 'week'], low_memory=False).query('week == @W').game_id)
problems = []
if teams_with_stats < 2 * len(ids): problems.append(f'weekly stats cover {teams_with_stats} teams, expected {2 * len(ids)}')
if not ids <= snaps: problems.append(f'snap counts missing {len(ids - snaps)} games')
if not ids <= pbp: problems.append(f'play-by-play missing {len(ids - pbp)} games')
if problems and not force:
    print(f'NOT READY for week {W}: ' + '; '.join(problems)); sys.exit(2)

path = f'{out}/last_update.json'
last = json.load(open(path))['week'] if os.path.exists(path) else None
if record:
    json.dump({'week': W, 'games': len(ids)}, open(path, 'w')); print(f'recorded last complete week = {W}'); sys.exit(0)
if last == W and not force:
    print(f'NOTHING NEW: week {W} was already published'); sys.exit(3)
print(f'READY: week {W} is complete ({len(ids)} games); previous update: {last}')
