"""Sanity checks on freshly built outputs. Exit 1 if any check fails.  Usage: python3 sanity.py <out_dir>"""
import sys, json
import numpy as np, pandas as pd

out = sys.argv[1]
cur = pd.read_csv(f'{out}/players_2026.csv')
pw = pd.read_csv(f'{out}/playerweek.csv'); p26 = pw[pw.season == 2026]
checks = []
def chk(name, ok, detail): checks.append((name, bool(ok), detail))

chk('enough players', len(cur) >= 250, f'{len(cur)} players')
z = cur[cur.n >= 3].z
chk('z-scores calibrated', 0.75 <= z.std() <= 1.25 and abs(z.mean()) <= 0.3, f'mean {z.mean():.2f}, sd {z.std():.2f}')
for pos, d in p26.groupby('pos'):
    r = d.xfp.sum() / d.ppr.sum()
    chk(f'xFP vs actual, {pos}', 0.92 <= r <= 1.08, f'ratio {r:.3f}')
chk('projections valid', cur.base.notna().all() and cur.base.between(0, 45).all(), f'range {cur.base.min():.1f} to {cur.base.max():.1f}')
meta = json.load(open(f'{out}/model_meta.json'))
chk('schedule present', len(meta['schedule']) >= 28, f'{len(meta["schedule"])} teams with upcoming games')
chk('latest week has data', p26.week.max() >= 4, f'latest week in data: {p26.week.max()}')
html = open(f'{out}/dashboard.html').read()
D = json.loads(html.split('const D = ')[1].split(';\nconst POS')[0])
chk('dashboard data parses', len(D['players']) == len(cur) and len(D['inj']) >= 1, f"{len(D['players'])} players, {len(D['inj'])} injury rows, as of {D['asOf']}")
bad = [c for c in checks if not c[1]]
for n, ok, d in checks: print(('PASS ' if ok else 'FAIL ') + n + ': ' + d)
sys.exit(1 if bad else 0)
