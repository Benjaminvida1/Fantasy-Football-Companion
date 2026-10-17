"""Step 1: per-play expected fantasy points (xFP, PPR) -> player-week table.

Usage: python3 -I build_xfp.py <data_dir> <out_dir>
Cross-fit: xFP for season s comes from a model trained on the *other* full seasons,
so backtests never see their own outcomes.
"""
import sys, os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DATA, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
SEASONS = [2024, 2025, 2026]
TRAIN_SEASONS = [2024, 2025]
SKILL = ['QB', 'RB', 'WR', 'TE']

# ---------- players / ids ----------
pl = pd.read_csv(f'{DATA}/players.csv', usecols=['gsis_id', 'pfr_id', 'position_group', 'display_name'])
pl = pl.dropna(subset=['gsis_id']).drop_duplicates('gsis_id')
pos_of = pl.set_index('gsis_id')['position_group'].to_dict()
name_of = pl.set_index('gsis_id')['display_name'].to_dict()
pfr2gsis = pl.dropna(subset=['pfr_id']).drop_duplicates('pfr_id').set_index('pfr_id')['gsis_id'].to_dict()

# ---------- play-by-play ----------
cols = ['game_id', 'week', 'season_type', 'posteam', 'defteam', 'yardline_100', 'down', 'ydstogo', 'goal_to_go',
        'air_yards', 'receiver_player_id', 'rusher_player_id', 'passer_player_id', 'pass_attempt', 'rush_attempt',
        'qb_scramble', 'complete_pass', 'pass_touchdown', 'rush_touchdown', 'interception', 'receiving_yards',
        'rushing_yards', 'passing_yards', 'pass_location', 'run_location', 'run_gap', 'shotgun',
        'two_point_attempt', 'play_deleted']
frames = []
for s in SEASONS:
    d = pd.read_csv(f'{DATA}/play_by_play_{s}.csv', usecols=cols, low_memory=False)
    d['season'] = s
    d = d[(d.season_type == 'REG') & (d.play_deleted == 0) & (d.two_point_attempt == 0)]
    frames.append(d)
pb = pd.concat(frames, ignore_index=True)
for c in ['air_yards', 'receiving_yards', 'rushing_yards', 'passing_yards', 'pass_touchdown', 'rush_touchdown',
          'interception', 'complete_pass', 'yardline_100', 'down', 'ydstogo', 'goal_to_go', 'shotgun']:
    pb[c] = pd.to_numeric(pb[c], errors='coerce')
for c in ['pass_location', 'run_location', 'run_gap']:
    pb[c] = pb[c].fillna('na').astype('category').cat.codes
POSCODE = {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3}

def fit_predict(X, y, season_col, feat, cat_idx):
    """Cross-fit predictions per season."""
    pred = pd.Series(index=X.index, dtype=float)
    for s in SEASONS:
        tr_seasons = [t for t in TRAIN_SEASONS if t != s]
        tr = season_col.isin(tr_seasons)
        te = season_col == s
        m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=60, l2_regularization=1.0,
                                          categorical_features=cat_idx, random_state=0)
        m.fit(X.loc[tr, feat], y[tr])
        pred[te] = m.predict(X.loc[te, feat])
    return pred

# ---- receiving targets ----
tg = pb[(pb.pass_attempt == 1) & pb.receiver_player_id.notna()].copy()
tg['pos'] = tg.receiver_player_id.map(pos_of).map(POSCODE).fillna(2)
tg['y'] = tg.complete_pass.fillna(0) * (1 + 0.1 * tg.receiving_yards.fillna(0)) + 6 * tg.pass_touchdown.fillna(0)
f_t = ['air_yards', 'yardline_100', 'down', 'ydstogo', 'goal_to_go', 'pass_location', 'pos', 'shotgun']
tg['xfp_rec'] = fit_predict(tg, tg.y, tg.season, f_t, [f_t.index('pass_location'), f_t.index('pos')])
print('target model: mean y %.3f mean xfp %.3f n=%d' % (tg.y.mean(), tg.xfp_rec.mean(), len(tg)))

# ---- rushes ----
ru = pb[(pb.rush_attempt == 1) & pb.rusher_player_id.notna()].copy()
ru['pos'] = ru.rusher_player_id.map(pos_of).map(POSCODE).fillna(1)
ru['y'] = 0.1 * ru.rushing_yards.fillna(0) + 6 * ru.rush_touchdown.fillna(0)
f_r = ['yardline_100', 'down', 'ydstogo', 'goal_to_go', 'run_location', 'run_gap', 'shotgun', 'qb_scramble', 'pos']
ru['qb_scramble'] = ru.qb_scramble.fillna(0)
ru['xfp_rush'] = fit_predict(ru, ru.y, ru.season, f_r,
                             [f_r.index('run_location'), f_r.index('run_gap'), f_r.index('pos')])
print('rush model: mean y %.3f mean xfp %.3f n=%d' % (ru.y.mean(), ru.xfp_rush.mean(), len(ru)))

# ---- QB passing (per attempt, passer perspective) ----
pa = pb[(pb.pass_attempt == 1) & pb.passer_player_id.notna()].copy()
pa['y'] = 0.04 * pa.passing_yards.fillna(0) + 4 * pa.pass_touchdown.fillna(0) - 2 * pa.interception.fillna(0)
f_p = ['air_yards', 'yardline_100', 'down', 'ydstogo', 'goal_to_go', 'pass_location', 'shotgun']
pa['xfp_pass'] = fit_predict(pa, pa.y, pa.season, f_p, [f_p.index('pass_location')])
print('pass model: mean y %.3f mean xfp %.3f n=%d' % (pa.y.mean(), pa.xfp_pass.mean(), len(pa)))

# ---------- aggregate to player-week ----------
K = ['season', 'week']
rec = tg.groupby(K + ['receiver_player_id', 'posteam']).agg(
    targets=('y', 'size'), air_yards=('air_yards', 'sum'), xfp_rec=('xfp_rec', 'sum'),
    rz_tgt=('yardline_100', lambda v: (v <= 20).sum()), ez_tgt=('yardline_100', lambda v: (v <= 10).sum())
).reset_index().rename(columns={'receiver_player_id': 'gsis_id', 'posteam': 'team'})
rus = ru.groupby(K + ['rusher_player_id', 'posteam']).agg(
    carries=('y', 'size'), xfp_rush=('xfp_rush', 'sum'),
    rz_car=('yardline_100', lambda v: (v <= 20).sum()), gl_car=('yardline_100', lambda v: (v <= 5).sum())
).reset_index().rename(columns={'rusher_player_id': 'gsis_id', 'posteam': 'team'})
qbp = pa.groupby(K + ['passer_player_id', 'posteam']).agg(
    pass_att=('y', 'size'), xfp_pass=('xfp_pass', 'sum')
).reset_index().rename(columns={'passer_player_id': 'gsis_id', 'posteam': 'team'})

pw = rec.merge(rus, on=K + ['gsis_id', 'team'], how='outer').merge(qbp, on=K + ['gsis_id', 'team'], how='outer')

# team totals for shares
tt = tg.groupby(K + ['posteam']).agg(t_tgt=('y', 'size'), t_air=('air_yards', 'sum'),
                                       t_rz_tgt=('yardline_100', lambda v: (v <= 20).sum())).reset_index()
ru_np = ru[ru.pos != 0]
tr = ru_np.groupby(K + ['posteam']).agg(t_car=('y', 'size'), t_rz_car=('yardline_100', lambda v: (v <= 20).sum()),
                                         t_gl_car=('yardline_100', lambda v: (v <= 5).sum())).reset_index()
team = tt.merge(tr, on=K + ['posteam'], how='outer').rename(columns={'posteam': 'team'})
pw = pw.merge(team, on=K + ['team'], how='left')

# ---------- snaps ----------
sn = []
for s in SEASONS:
    d = pd.read_csv(f'{DATA}/snap_counts_{s}.csv')
    d = d[(d.game_type == 'REG') & d.position.isin(['QB', 'RB', 'FB', 'WR', 'TE'])]
    sn.append(d)
sn = pd.concat(sn)
sn['gsis_id'] = sn.pfr_player_id.map(pfr2gsis)
sn = sn.dropna(subset=['gsis_id'])
sn = sn.groupby(K + ['gsis_id', 'team']).agg(off_snaps=('offense_snaps', 'sum'), snap_pct=('offense_pct', 'max')).reset_index()
pw = pw.merge(sn, on=K + ['gsis_id', 'team'], how='outer')

# ---------- actual PPR ----------
st = []
for s in SEASONS:
    d = pd.read_csv(f'{DATA}/stats_player_week_{s}.csv', usecols=['player_id', 'season', 'week', 'season_type', 'team',
                                                                  'position', 'fantasy_points_ppr', 'receptions'])
    st.append(d[d.season_type == 'REG'])
st = pd.concat(st).rename(columns={'player_id': 'gsis_id', 'fantasy_points_ppr': 'ppr', 'position': 'pos_stats'})
st = st.drop(columns='season_type')
pw = pw.merge(st, on=K + ['gsis_id', 'team'], how='left')

pw['pos'] = pw.gsis_id.map(pos_of)
pw['name'] = pw.gsis_id.map(name_of)
pw['pos'] = pw.pos.fillna(pw.pos_stats)
pw = pw[pw.pos.isin(SKILL)].copy()

for c in ['targets', 'air_yards', 'xfp_rec', 'rz_tgt', 'ez_tgt', 'carries', 'xfp_rush', 'rz_car', 'gl_car', 'pass_att',
          'xfp_pass', 'off_snaps', 'ppr', 'receptions']:
    pw[c] = pw[c].fillna(0)
pw['xfp'] = pw.xfp_rec + pw.xfp_rush + pw.xfp_pass
# a game counts as played if the player saw offensive snaps or any opportunity
pw = pw[(pw.off_snaps > 0) | (pw.targets > 0) | (pw.carries > 0) | (pw.pass_att > 0)].copy()

def share(a, b):
    return np.where(pw[b] > 0, pw[a] / pw[b], np.nan)
pw['tgt_share'] = share('targets', 't_tgt')
pw['air_share'] = share('air_yards', 't_air')
pw['rz_tgt_share'] = share('rz_tgt', 't_rz_tgt')
pw['car_share'] = np.where(pw.pos == 'QB', np.nan, share('carries', 't_car'))
pw['rz_car_share'] = np.where(pw.pos == 'QB', np.nan, share('rz_car', 't_rz_car'))
pw['gl_car_share'] = np.where(pw.pos == 'QB', np.nan, share('gl_car', 't_gl_car'))
pw['wopr'] = 1.5 * pw.tgt_share.fillna(0) + 0.7 * pw.air_share.fillna(0)

# opponent
g = pd.read_csv(f'{DATA}/games.csv', usecols=['season', 'week', 'game_type', 'away_team', 'home_team'])
g = g[g.game_type == 'REG']
opp = pd.concat([g.rename(columns={'away_team': 'team', 'home_team': 'opp'})[['season', 'week', 'team', 'opp']],
                 g.rename(columns={'home_team': 'team', 'away_team': 'opp'})[['season', 'week', 'team', 'opp']]])
pw = pw.merge(opp, on=['season', 'week', 'team'], how='left')
print('rows without opponent:', pw.opp.isna().sum())
print('team labels not in schedule:', sorted(set(pw.team) - set(opp.team)))

pw.to_csv(f'{OUT}/playerweek.csv', index=False)
print(pw.shape)
print(pw.groupby(['season', 'pos']).agg(n=('xfp', 'size'), ppr=('ppr', 'mean'), xfp=('xfp', 'mean')).round(2))
