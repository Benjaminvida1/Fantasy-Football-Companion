import sys, os, json, numpy as np, pandas as pd
OUT = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
cur = pd.read_csv(f'{OUT}/players_2026.csv')
meta = json.load(open(f'{OUT}/model_meta.json'))
S = meta['summary']
def nn(v, d=4):
    return None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), d)
players = []
for _, r in cur.iterrows():
    inj = r.report_status if (pd.notna(r.report_status) and r.inj_week >= 6) else None
    players.append(dict(id=r.gsis_id, name=r['name'], pos=r.pos, team=r.team, n=int(r.n),
        ppr=nn(r.ppr_pg, 2), xfp=nn(r.get('xfp_pg_raw', r.xfp_pg), 2), z=nn(r.z, 2) or 0.0, base=nn(r.base, 2),
        tgt=nn(r.tgt_share), air=nn(r.air_share), car=nn(r.car_share), snap=nn(r.snap_pct),
        rzt=nn(r.rz_tgt_share), glc=nn(r.gl_car_share),
        inj=inj, injd=(r.report_primary_injury if pd.notna(r.report_primary_injury) and inj else None)))
rm = S['rmse_table']; k = list(rm.keys())[0]
data = dict(players=players, sched=meta['schedule'], dvp={p: {t: round(v, 4) for t, v in d.items()} for p, d in meta['dvp'].items()},
            repl={p: round(v, 3) for p, v in S['repl'].items()}, sd={p: round(v, 3) for p, v in S['sd'].items()},
            summary=dict(rmse=dict(naive=rm[k]['naive'], xfp=rm[k]['xfp'], model=rm[k]['model']), coefs=S['coefs'], lam=S['lam']),
            nextWeek=6, asOf='Week 5 plus the Week 6 Thursday game (SEA at DEN)')
tpl = open(os.path.join(HERE, 'template.html')).read()
html = tpl.replace('__DATA__', json.dumps(data, separators=(',', ':'), default=float))
open(f'{OUT}/dashboard.html', 'w').write(html)
print(len(players), 'players;', len(html) // 1024, 'KB')
