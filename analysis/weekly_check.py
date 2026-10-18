"""One-week-ahead check of the boost badge: when exactly one RB/WR/TE star is out in week w (known in advance),
does baseline + pooled-rate boost predict teammates' week-w xFP better than the trailing baseline alone?
Rates come from the other season (out-of-sample). Usage: python3 -B weekly_check.py <data_dir> <out_dir>"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np, pandas as pd
import opportunity as op
DATA, OUT = sys.argv[1], sys.argv[2]
pw = pd.read_csv(f'{OUT}/playerweek.csv')
g = pd.read_csv(f'{DATA}/games.csv', usecols=['season','week','game_type','away_team','home_team','away_score'])
cache = op.build_cache(pw, op.played_games(g[g.season.isin([2024,2025,2026])]))
rows = []
for test_s in [2024, 2025]:
    rates, _, _, _ = op.estimate_rates(cache, [s for s in [2024, 2025] if s != test_s])
    for (s, t), ts in cache.items():
        if s != test_s: continue
        for w in ts.weeks:
            if w < 4 or w > 17: continue
            stars, qbs = ts.absent_stars(w), ts.absent_qb(w)
            if len(stars) != 1 or qbs: continue
            a = stars[0]; pa = ts.pos[a]; base = ts.base(w)
            rec = [p for p in ts.X.index if p != a and ts.pos[p] in op.POS3 and ts.pres(p, w) and p in base]
            S_same = sum(base[p] for p in rec if ts.pos[p] == pa); S_oth = sum(base[p] for p in rec if ts.pos[p] != pa)
            for p in rec:
                share = (base[p] / S_same if ts.pos[p] == pa and S_same > 0 else 0, base[p] / S_oth if ts.pos[p] != pa and S_oth > 0 else 0)
                boost = (rates[pa]['same'] * base[a] * share[0]) + (rates[pa]['other'] * base[a] * share[1])
                rows.append(dict(season=s, pos=ts.pos[p], base=base[p], boost=boost, actual=ts.X.at[p, w]))
R = pd.DataFrame(rows)
r = lambda e: float(np.sqrt(np.mean(e ** 2)))
print('teammate-weeks:', len(R), '| mean boost %.2f xFP, mean (actual - baseline) %.2f' % (R.boost.mean(), (R.actual - R.base).mean()))
for lab, m in [('all', R.pos.notna()), ('base>=5 xFP/g', R.base >= 5), ('RB', R.pos == 'RB'), ('WR', R.pos == 'WR'), ('TE', R.pos == 'TE')]:
    d = R[m]
    print(f"{lab:15s} n={len(d):4d}  RMSE baseline {r(d.actual-d.base):.3f}  baseline+boost {r(d.actual-d.base-d.boost):.3f}  corr(boost, actual-baseline) {np.corrcoef(d.boost, d.actual-d.base)[0,1]:.2f}")
for s in [2024, 2025]:
    d = R[R.season == s]; print(s, f'RMSE baseline {r(d.actual-d.base):.3f}  with boost {r(d.actual-d.base-d.boost):.3f}  n={len(d)}')
