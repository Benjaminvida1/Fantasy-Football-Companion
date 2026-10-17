"""Step 2: over/under z-scores, ROS projection (ridge, validated season-out), ridge-shrunk strength of schedule.

Usage: python3 -I model.py <data_dir> <out_dir>
"""
import sys, json
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge

DATA, OUT = sys.argv[1], sys.argv[2]
POS = ['QB', 'RB', 'WR', 'TE']
END_WEEK = 17
CUTS = [4, 5, 6, 7, 8]
import os
LAMS = [float(x) for x in os.environ.get('LAMS','3,10,30,100,300').split(',')]
SOS_COL_ENV = os.environ.get('SOS_COL','ppr')

pw = pd.read_csv(f'{OUT}/playerweek.csv')
pw = pw.sort_values(['season', 'week'])

# ---------------- prior-season PPG (from raw stats, consistent "had an opportunity" games) -------------
pr = []
for s in [2023, 2024, 2025]:
    d = pd.read_csv(f'{DATA}/stats_player_week_{s}.csv',
                    usecols=['player_id', 'season', 'season_type', 'position_group', 'fantasy_points_ppr',
                             'targets', 'carries', 'attempts'])
    d = d[(d.season_type == 'REG') & d.position_group.isin(POS)]
    d = d[(d.targets.fillna(0) > 0) | (d.carries.fillna(0) > 0) | (d.attempts.fillna(0) > 0)]
    a = d.groupby('player_id').fantasy_points_ppr.agg(['mean', 'size']).reset_index()
    a = a[a['size'] >= 6]
    a['season'] = s + 1
    pr.append(a.rename(columns={'player_id': 'gsis_id', 'mean': 'prior_ppg'})[['season', 'gsis_id', 'prior_ppg']])
prior = pd.concat(pr)

# ---------------- schedule ----------------
g = pd.read_csv(f'{DATA}/games.csv', usecols=['season', 'week', 'game_type', 'away_team', 'home_team', 'away_score'])
g = g[(g.game_type == 'REG') & g.season.isin([2024, 2025, 2026])]
g['played'] = g.away_score.notna()
sched = pd.concat([
    g.rename(columns={'away_team': 'team', 'home_team': 'opp'})[['season', 'week', 'team', 'opp', 'played']],
    g.rename(columns={'home_team': 'team', 'away_team': 'opp'})[['season', 'week', 'team', 'opp', 'played']]])

# ---------------- residual variance model: var(r | xfp) = v0 + v1*xfp, per position ----------------
pw['resid'] = pw.ppr - pw.xfp
hist = pw[pw.season < 2026]
VAR = {}
for p in POS:
    h = hist[hist.pos == p]
    A = np.c_[np.ones(len(h)), h.xfp.values]
    coef, *_ = np.linalg.lstsq(A, (h.resid.values - h.resid.mean()) ** 2, rcond=None)
    VAR[p] = (max(coef[0], 1.0), max(coef[1], 0.0))
print('residual variance model (v0, v1):', {k: tuple(round(x, 2) for x in v) for k, v in VAR.items()})

def game_var(df):
    v0 = df.pos.map(lambda p: VAR[p][0]); v1 = df.pos.map(lambda p: VAR[p][1])
    return v0 + v1 * df.xfp

# ---------------- DvP (two-way ridge: offense + defense effects) ----------------
SOS_COL = SOS_COL_ENV
def dvp_rel(season, upto_week, lam):
    d = pw[(pw.season == season) & (pw.week <= upto_week)]
    tgm = d.groupby(['week', 'team', 'opp', 'pos'])[SOS_COL].sum().reset_index()
    out = {}
    for p in POS:
        t = tgm[tgm.pos == p]
        X = pd.get_dummies(t[['team', 'opp']], prefix=['o', 'd']).astype(float)
        m = Ridge(alpha=lam).fit(X, t[SOS_COL].values)
        coef = pd.Series(m.coef_, index=X.columns)
        league = t[SOS_COL].mean()
        out[p] = {c[2:]: v / league for c, v in coef.items() if c.startswith('d_')}
    return out

# ---------------- feature rows ----------------
def feat_rows(season, c, rel, future_mode='week'):
    d = pw[(pw.season == season) & (pw.week <= c)]
    rows = []
    for gid, x in d.groupby('gsis_id'):
        x = x.sort_values('week')
        n = len(x)
        rows.append(dict(gsis_id=gid, season=season, c=c, pos=x.pos.iloc[-1], team=x.team.iloc[-1], n=n,
                         xfp_pg=x.xfp.mean(), ppr_pg=x.ppr.mean(), xfp_l3=x.xfp.tail(3).mean(),
                         snap=x.snap_pct.mean() if x.snap_pct.notna().any() else np.nan))
    r = pd.DataFrame(rows)
    r['resid_pg'] = r.ppr_pg - r.xfp_pg
    r = r.merge(prior, on=['season', 'gsis_id'], how='left')
    r['has_prior'] = r.prior_ppg.notna().astype(float)
    r['prior_f'] = r.prior_ppg.fillna(r.xfp_pg)
    r['snap'] = r.snap.fillna(r.snap.median())
    # future schedule + sos
    s = sched[(sched.season == season) & (sched.week <= END_WEEK)]
    fut = s[~s.played] if season == 2026 else s[s.week > c]
    fmap = fut.groupby('team').opp.apply(list).to_dict()
    r['sos'] = [np.mean([rel[p].get(o, 0.0) for o in fmap.get(t, [])]) if fmap.get(t) else 0.0
                for p, t in zip(r.pos, r.team)]
    return r

def targets(season, c):
    f = pw[(pw.season == season) & (pw.week > c) & (pw.week <= END_WEEK)]
    a = f.groupby('gsis_id').ppr.agg(['mean', 'size']).reset_index().rename(columns={'mean': 'y', 'size': 'n_f'})
    return a[a.n_f >= 3]

FEATS = ['xfp_pg', 'xfp_l3', 'resid_pg', 'snap', 'prior_f', 'has_prior']

def dataset(lam):
    out = []
    for s in [2024, 2025]:
        for c in CUTS:
            rel = dvp_rel(s, c, lam)
            r = feat_rows(s, c, rel).merge(targets(s, c), on='gsis_id')
            r = r[(r.n >= 3) & (r.xfp_pg >= 2.0)]
            out.append(r)
    return pd.concat(out, ignore_index=True)

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

results = {}
best = None
for lam in LAMS:
    D = dataset(lam)
    tot = {'naive': [], 'xfp': [], 'model': [], 'model_sos': []}
    ys = []
    betas = []
    cors = {}
    for test_s in [2024, 2025]:
        tr, te = D[D.season != test_s], D[D.season == test_s]
        te = te.copy(); te['pred'] = np.nan; tr = tr.copy(); tr['pred'] = np.nan
        for p in POS:
            a, b = tr[tr.pos == p], te[te.pos == p]
            m = Ridge(alpha=1.0).fit(a[FEATS], a.y)
            tr.loc[a.index, 'pred'] = m.predict(a[FEATS]); te.loc[b.index, 'pred'] = m.predict(b[FEATS])
        ps = tr.pred * tr.sos
        beta = float(((tr.y - tr.pred) * ps).sum() / (ps ** 2).sum())
        betas.append(beta)
        te['res'] = te.y - te.pred
        for pp in POS:
            q = te[te.pos == pp]
            cors.setdefault(pp, []).append(round(float(np.corrcoef(q.res, q.sos)[0, 1]), 3))
        te['pred_sos'] = te.pred * (1 + beta * te.sos)
        tot['naive'].append(te.ppr_pg.values); tot['xfp'].append(te.xfp_pg.values)
        tot['model'].append(te.pred.values); tot['model_sos'].append(te.pred_sos.values); ys.append(te.y.values)
    y = np.concatenate(ys)
    res = {k: rmse(np.concatenate(v), y) for k, v in tot.items()}
    res['beta_by_fold'] = [round(b, 2) for b in betas]
    res['corr_resid_sos'] = cors
    results[lam] = res
    print('lambda', lam, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})
    if best is None or res['model_sos'] < results[best]['model_sos']:
        best = lam
print('best lambda by OOS RMSE with SOS:', best)

# ---------------- final fit (both seasons) at best lambda ----------------
LAM = best
D = dataset(LAM)
final = {}
rows_report = {}
for p in POS:
    a = D[D.pos == p]
    m = Ridge(alpha=1.0).fit(a[FEATS], a.y)
    final[p] = m
    cv_res = []
    rows_report[p] = dict(zip(FEATS, [round(c, 3) for c in m.coef_]), intercept=round(m.intercept_, 3), n=int(len(a)))
print('final coefficients:', json.dumps(rows_report, indent=1))
D['pred'] = np.nan
for p in POS:
    a = D[D.pos == p]; D.loc[a.index, 'pred'] = final[p].predict(a[FEATS])
ps = D.pred * D.sos
BETA = float(((D.y - D.pred) * ps).sum() / (ps ** 2).sum())
print('beta (SOS pass-through) =', round(BETA, 3))
sd_by_pos = {p: float(rmse(D[D.pos == p].pred.values, D[D.pos == p].y.values)) for p in POS}
print('in-sample RMSE by pos (approx ROS ppg error):', {k: round(v, 2) for k, v in sd_by_pos.items()})
# out-of-sample sd by pos for uncertainty (leave-season-out)
oos = []
for test_s in [2024, 2025]:
    tr, te = D[D.season != test_s], D[D.season == test_s].copy()
    for p in POS:
        a, b = tr[tr.pos == p], te[te.pos == p]
        m = Ridge(alpha=1.0).fit(a[FEATS], a.y)
        te.loc[b.index, 'pred'] = m.predict(b[FEATS])
    oos.append(te)
oos = pd.concat(oos)
SD = {p: float(rmse(oos[oos.pos == p].pred.values, oos[oos.pos == p].y.values)) for p in POS}
print('OOS RMSE by pos:', {k: round(v, 2) for k, v in SD.items()})

# ---------------- 2026 current state ----------------
rel26 = dvp_rel(2026, 99, LAM)
cur = feat_rows(2026, 99, rel26)
cur['base'] = np.nan
# small-sample shrinkage: the model was trained on players with >=3 games, so for n<3 pull the in-season
# signals toward the player's prior (or a position-level default) with weight w = n/3.
pos_default = cur[cur.n >= 3].groupby('pos').xfp_pg.median().to_dict()
w = np.minimum(cur.n / 3.0, 1.0)
anchor = cur.prior_ppg.fillna(cur.pos.map(pos_default))
for col in ['xfp_pg', 'xfp_l3']:
    cur[col + '_raw'] = cur[col]
    cur[col] = w * cur[col] + (1 - w) * anchor
cur['resid_pg'] = w * cur['resid_pg']
cur['prior_f'] = cur.prior_ppg.fillna(cur.xfp_pg)
for p in POS:
    a = cur[cur.pos == p]
    cur.loc[a.index, 'base'] = np.clip(final[p].predict(a[FEATS]), 0, None)

# over/under z-score
d26 = pw[pw.season == 2026].copy()
d26['gv'] = game_var(d26)
zz = d26.groupby('gsis_id').apply(lambda x: pd.Series({'resid_sum': x.resid.sum(), 'var_sum': x.gv.sum()}), include_groups=False).reset_index()
zz['z'] = zz.resid_sum / np.sqrt(zz.var_sum)
cur = cur.merge(zz, on='gsis_id', how='left')

# usage summary
def wmean(x, col):
    v = x[col]
    return v.mean() if v.notna().any() else np.nan
u = d26.groupby('gsis_id').apply(lambda x: pd.Series({
    'tgt_pg': x.targets.mean(), 'tgt_share': wmean(x, 'tgt_share'), 'air_share': wmean(x, 'air_share'),
    'rz_tgt_share': wmean(x, 'rz_tgt_share'), 'car_pg': x.carries.mean(), 'car_share': wmean(x, 'car_share'),
    'rz_car_share': wmean(x, 'rz_car_share'), 'gl_car_share': wmean(x, 'gl_car_share'), 'wopr': x.wopr.mean(),
    'snap_pct': x.snap_pct.mean(), 'xfp_l3': x.xfp.tail(3).mean()}), include_groups=False).reset_index()
cur = cur.merge(u.drop(columns=['xfp_l3']), on='gsis_id', how='left')
names = pw.drop_duplicates('gsis_id', keep='last').set_index('gsis_id').name.to_dict()
cur['name'] = cur.gsis_id.map(names)

# injuries
inj = pd.read_csv(f'{DATA}/injuries_2026.csv')
inj = inj.sort_values('week').drop_duplicates('gsis_id', keep='last')
cur = cur.merge(inj[['gsis_id', 'report_status', 'report_primary_injury', 'practice_status', 'week']].rename(
    columns={'week': 'inj_week'}), on='gsis_id', how='left')

# players worth listing
keep = cur[(cur.xfp_pg >= 3.0) | (cur.base >= 5.0)].copy()
keep = keep[keep.n >= 1]
print('players kept:', keep.groupby('pos').size().to_dict())

# replacement levels via greedy league fill (12 teams: QB1 RB2 WR2 TE1 FLEX1)
pool = keep.sort_values('base', ascending=False)
taken = set()
def take(pos_list, k):
    c = 0
    for i, r in pool.iterrows():
        if i in taken or r.pos not in pos_list: continue
        taken.add(i); c += 1
        if c == k: break
take(['QB'], 12); take(['RB'], 24); take(['WR'], 24); take(['TE'], 12); take(['RB', 'WR', 'TE'], 12)
repl = {}
for p in POS:
    rest = pool[(pool.pos == p) & (~pool.index.isin(taken))]
    repl[p] = float(rest.base.iloc[0]) if len(rest) else 0.0
print('replacement ppg (best non-starter):', {k: round(v, 2) for k, v in repl.items()})

# upcoming schedule (unplayed regular-season games through week 18)
s26 = sched[(sched.season == 2026) & (~sched.played)]
schedule = {t: {int(w): o for w, o in zip(x.week, x.opp)} for t, x in s26.groupby('team')}

summary = dict(lam=LAM, beta=BETA, sd=SD, repl=repl, coefs=rows_report, rmse_table=results,
               var=VAR)
keep.to_csv(f'{OUT}/players_2026.csv', index=False)
json.dump(dict(summary=summary, schedule=schedule, dvp=rel26), open(f'{OUT}/model_meta.json', 'w'), default=float)
print(keep.sort_values('z').head(8)[['name', 'pos', 'team', 'n', 'ppr_pg', 'xfp_pg', 'z', 'base']].round(2).to_string())
print(keep.sort_values('z').tail(8)[['name', 'pos', 'team', 'n', 'ppr_pg', 'xfp_pg', 'z', 'base']].round(2).to_string())
