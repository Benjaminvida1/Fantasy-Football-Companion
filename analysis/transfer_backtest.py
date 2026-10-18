"""Does stripping boosted games and adding the pooled transfer boost improve out-of-sample ROS projections?

Folds: train on one of 2024/2025, test on the other (rates are estimated on the training season only).
Usage: python3 -B transfer_backtest.py <data_dir> <out_dir>
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
import opportunity as op

DATA, OUT = sys.argv[1], sys.argv[2]
POS = ['QB', 'RB', 'WR', 'TE']
CUTS = [4, 5, 6, 7, 8]
END_WEEK = 17

pw = pd.read_csv(f'{OUT}/playerweek.csv').sort_values(['season', 'week'])
g = pd.read_csv(f'{DATA}/games.csv', usecols=['season', 'week', 'game_type', 'away_team', 'home_team', 'away_score'])
games = op.played_games(g[g.season.isin([2024, 2025, 2026])])
cache = op.build_cache(pw, games)

# prior-season PPG (same construction as model.py)
pr = []
for s in [2023, 2024, 2025]:
    d = pd.read_csv(f'{DATA}/stats_player_week_{s}.csv', usecols=['player_id', 'season', 'season_type', 'position_group',
                                                                  'fantasy_points_ppr', 'targets', 'carries', 'attempts'])
    d = d[(d.season_type == 'REG') & d.position_group.isin(POS)]
    d = d[(d.targets.fillna(0) > 0) | (d.carries.fillna(0) > 0) | (d.attempts.fillna(0) > 0)]
    a = d.groupby('player_id').fantasy_points_ppr.agg(['mean', 'size']).reset_index()
    a = a[a['size'] >= 6]; a['season'] = s + 1
    pr.append(a.rename(columns={'player_id': 'gsis_id', 'mean': 'prior_ppg'})[['season', 'gsis_id', 'prior_ppg']])
prior = pd.concat(pr)

# boosted-week flags (a team game in which any RB/WR/TE star was out)
BOOST = {}
for (s, t), ts in cache.items():
    for w in ts.weeks:
        BOOST[(s, t, w)] = len(ts.absent_stars(w)) > 0


def build(season, c):
    d = pw[(pw.season == season) & (pw.week <= c)]
    rows = []
    for gid, x in d.groupby('gsis_id'):
        x = x.sort_values('week')
        pos = x.pos.iloc[-1]
        flags = np.array([BOOST.get((season, t, w), False) for t, w in zip(x.team, x.week)]) if pos in op.POS3 else np.zeros(len(x), bool)
        clean = x[~flags]
        xfp = x.xfp.mean()
        rows.append(dict(gsis_id=gid, season=season, c=c, pos=pos, team=x.team.iloc[-1], n=len(x), n_boost=int(flags.sum()),
                         xfp_pg=xfp, ppr_pg=x.ppr.mean(), xfp_l3=x.xfp.tail(3).mean(),
                         xfp_clean=clean.xfp.mean() if len(clean) >= 2 else xfp,
                         xfp_l3c=clean.xfp.tail(3).mean() if len(clean) >= 1 else x.xfp.tail(3).mean(),
                         snap=x.snap_pct.mean() if x.snap_pct.notna().any() else np.nan))
    r = pd.DataFrame(rows)
    r['resid_pg'] = r.ppr_pg - r.xfp_pg
    r = r.merge(prior, on=['season', 'gsis_id'], how='left')
    r['has_prior'] = r.prior_ppg.notna().astype(float)
    r['prior_f'] = r.prior_ppg.fillna(r.xfp_pg)
    r['snap'] = r.snap.fillna(r.snap.median())
    f = pw[(pw.season == season) & (pw.week > c) & (pw.week <= END_WEEK)]
    a = f.groupby('gsis_id').ppr.agg(['mean', 'size']).reset_index().rename(columns={'mean': 'y', 'size': 'n_f'})
    r = r.merge(a, on='gsis_id'); r = r[(r.n_f >= 3) & (r.n >= 3) & (r.xfp_pg >= 2.0)].copy()
    # boost components per recipient under each timing scenario
    cn, co = {}, {}
    for t, grp in r.groupby('team'):
        ts = cache.get((season, t))
        if ts is None:
            continue
        rec = [p for p in grp.gsis_id if ts.pos.get(p) in op.POS3]
        bclean = dict(zip(grp.gsis_id, grp.xfp_clean))
        fut = [w for w in ts.weeks if c < w <= END_WEEK]
        cn.update(op.boost_components(ts, c, 'naive', rec, bclean, fut))
        co.update(op.boost_components(ts, c, 'oracle', rec, bclean, fut))
    zero = {(pa, k): 0.0 for pa in op.POS3 for k in op.KINDS}
    r['comp_naive'] = [cn.get(p, zero) for p in r.gsis_id]
    r['comp_oracle'] = [co.get(p, zero) for p in r.gsis_id]
    return r


D = pd.concat([build(s, c) for s in [2024, 2025] for c in CUTS], ignore_index=True)
print('rows:', len(D), '| rows with any boosted history game:', int((D.n_boost > 0).sum()))

F_BASE = ['xfp_pg', 'xfp_l3', 'resid_pg', 'snap', 'prior_f', 'has_prior']
F_CLEAN = ['xfp_clean', 'xfp_l3c', 'resid_pg', 'snap', 'prior_f', 'has_prior']


def fit_pred(tr, te, feats):
    p = np.full(len(te), np.nan); coefs = {}
    for pos in POS:
        a, b = tr[tr.pos == pos], te[te.pos == pos]
        if len(b) == 0: continue
        m = Ridge(alpha=1.0).fit(a[feats], a.y); p[(te.pos == pos).values] = m.predict(b[feats]); coefs[pos] = m
    return p, coefs


def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


res = {k: [] for k in ['base', 'clean', 'clean+fix_naive', 'clean+fix_oracle', 'clean+fit_naive', 'clean+fit_oracle']}
ys, aff, rate_log = [], [], {}
for test_s in [2024, 2025]:
    rates, nev, E, C = op.estimate_rates(cache, [s for s in [2024, 2025] if s != test_s])
    rate_log[test_s] = {pa: (round(r['same'], 2), round(r['other'], 2), nev[pa]) for pa, r in rates.items()}
    tr, te = D[D.season != test_s].copy(), D[D.season == test_s].copy()
    for df in (tr, te):
        df['b_naive'] = [op.apply_rates(c_, rates) for c_ in df.comp_naive]
        df['b_oracle'] = [op.apply_rates(c_, rates) for c_ in df.comp_oracle]
    p_base, _ = fit_pred(tr, te, F_BASE)
    p_clean, _ = fit_pred(tr, te, F_CLEAN)
    res['base'].append(p_base); res['clean'].append(p_clean)
    res['clean+fix_naive'].append(p_clean + te.b_naive.values)
    res['clean+fix_oracle'].append(p_clean + te.b_oracle.values)
    for tag, col in [('naive', 'b_naive'), ('oracle', 'b_oracle')]:
        tr2, te2 = tr.copy(), te.copy()
        tr2['boost'] = tr2[col]; te2['boost'] = te2[col]
        pp, _ = fit_pred(tr2, te2, F_CLEAN + ['boost'])
        res[f'clean+fit_{tag}'].append(pp)
    ys.append(te.y.values)
    aff.append(((te.n_boost > 0) | (te.b_oracle.abs() > 0.25) | (te.b_naive.abs() > 0.25)).values)
y = np.concatenate(ys); A = np.concatenate(aff)
print('\nrates estimated on the training season (same, other, n events) by position of the absent star:')
print(json.dumps(rate_log))
print(f'\nOut-of-sample RMSE (PPG), n={len(y)}, affected subset n={int(A.sum())}')
print(f"{'variant':20s} {'all':>7s} {'fold24':>7s} {'fold25':>7s} {'affected':>9s}")
for k, v in res.items():
    pr_ = np.concatenate(v)
    f24 = rmse(v[0], ys[0]); f25 = rmse(v[1], ys[1])
    print(f'{k:20s} {rmse(pr_, y):7.3f} {f24:7.3f} {f25:7.3f} {rmse(pr_[A], y[A]):9.3f}')

# final pooled rates (all of 2024-25) and the fitted pass-through of the boost feature
rates_all, nev_all, E_all, C_all = op.estimate_rates(cache, [2024, 2025])
print('\npooled rates 2024-25 (same, other, n):', {pa: (round(r['same'], 2), round(r['other'], 2), nev_all[pa]) for pa, r in rates_all.items()})
D['b_naive'] = [op.apply_rates(c_, rates_all) for c_ in D.comp_naive]
D['b_oracle'] = [op.apply_rates(c_, rates_all) for c_ in D.comp_oracle]
for tag in ['naive', 'oracle']:
    D['boost'] = D[f'b_{tag}']
    out = {}
    for pos in op.POS3:
        a = D[D.pos == pos]; m = Ridge(alpha=1.0).fit(a[F_CLEAN + ['boost']], a.y)
        out[pos] = round(float(m.coef_[-1]), 2)
    print(f'fitted pass-through on boost ({tag} timing):', out)
json.dump({'rates': {pa: r for pa, r in rates_all.items()}, 'n_events': nev_all}, open(f'{OUT}/transfer_rates.json', 'w'))
D.drop(columns=['comp_naive', 'comp_oracle']).to_csv(f'{OUT}/transfer_backtest_rows.csv', index=False)
