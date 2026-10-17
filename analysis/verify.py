import pandas as pd, numpy as np
st = pd.read_csv('data/stats_player_week_2026.csv', usecols=['player_display_name','position','season_type','week','fantasy_points_ppr','team'])
st = st[st.season_type=='REG']
pl = pd.read_csv('output/players_2026.csv')
for nm in ['Jaxon Smith-Njigba','Jahmyr Gibbs','Brock Purdy','Travis Kelce']:
    raw = st[st.player_display_name==nm]
    r = pl[pl['name']==nm].iloc[0]
    print(f"{nm}: raw games={len(raw)} raw ppr/g={raw.fantasy_points_ppr.mean():.2f} | table n={r.n} ppr/g={r.ppr_pg:.2f} xfp/g={r.get('xfp_pg_raw', r.xfp_pg):.2f} z={r.z:.2f} base={r.base:.2f}")
# league-wide sanity: sum of player xfp vs actual by position in 2026
pw = pd.read_csv('output/playerweek.csv'); p26 = pw[pw.season==2026]
print(p26.groupby('pos')[['ppr','xfp']].sum().round(0).assign(ratio=lambda d:(d.xfp/d.ppr).round(3)))
print('z distribution (n>=3):', pl[pl.n>=3].z.describe()[['mean','std']].round(2).to_dict(), ' share |z|>=1.25:', round((pl[pl.n>=3].z.abs()>=1.25).mean(),3))
