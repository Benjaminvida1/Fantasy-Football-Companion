"""Injects model output and the injury snapshot into the dashboard template.
Usage: python3 build_html.py <out_dir>   (reads players_2026.csv, model_meta.json; writes dashboard.html)"""
import sys, os, json, re, numpy as np, pandas as pd

OUT = sys.argv[1]
DATA = sys.argv[2] if len(sys.argv) > 2 else 'data'
HERE = os.path.dirname(os.path.abspath(__file__))
# Display cutoff for the injury list and badges (looser than the starter thresholds used in the transfer analysis,
# so partial-game players such as a banged-up RB1 still show up).
STAR_XFP = {'QB': 14.0, 'RB': 6.0, 'WR': 6.0, 'TE': 5.0}
SUFFIX = {'jr', 'sr', 'ii', 'iii', 'iv'}


def nn(v, d=4):
    return None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), d)


def norm(name):
    toks = re.sub(r"[.'’]", '', str(name).lower()).replace('-', ' ').split()
    return ''.join(t for t in toks if t not in SUFFIX)


cur = pd.read_csv(f'{OUT}/players_2026.csv')
meta = json.load(open(f'{OUT}/model_meta.json'))
S = meta['summary']

players = []
for _, r in cur.iterrows():
    players.append(dict(id=r.gsis_id, name=r['name'], pos=r.pos, team=r.team, n=int(r.n),
        ppr=nn(r.ppr_pg, 2), xfp=nn(r.get('xfp_pg_raw', r.xfp_pg), 2), z=nn(r.z, 2) or 0.0, base=nn(r.base, 2),
        tgt=nn(r.tgt_share), air=nn(r.air_share), car=nn(r.car_share), snap=nn(r.snap_pct),
        rzt=nn(r.rz_tgt_share), glc=nn(r.gl_car_share), inj=None, injd=None, tm=[]))

# ---- injury snapshot (authoritative for current statuses; players not listed are treated as healthy) ----
snap = json.load(open(os.path.join(HERE, '..', 'reference', 'nfl_injuries_latest.json')))
idx = {(norm(p['name']), p['team']): p for p in players}
unmatched = []
for e in snap['players']:
    p = idx.get((norm(e['name']), e['team']))
    if p is None:
        unmatched.append(f"{e['name']} ({e['pos']} {e['team']})"); continue
    p['inj'], p['injd'] = e['status'], e['injury']

is_star = lambda p: p['n'] >= 2 and p['xfp'] is not None and p['xfp'] >= STAR_XFP[p['pos']]
by_team = {}
for p in players:
    by_team.setdefault(p['team'], []).append(p)
inj_rows = []
for t, ps in by_team.items():
    out_stars = [q for q in ps if is_star(q) and q['inj'] in ('Out', 'Doubtful')]
    for p in ps:
        p['tm'] = [dict(id=q['id'], n=q['name'].split(' ', 1)[1] if ' ' in q['name'] else q['name'], pos=q['pos'], st=q['inj'])
                   for q in out_stars if q['id'] != p['id']]
    for q in ps:
        if is_star(q) and q['inj'] in ('Out', 'Doubtful', 'Questionable'):
            nxt = sorted([x for x in ps if x['id'] != q['id'] and x['pos'] in ('RB', 'WR', 'TE') and x['pos'] == q['pos'] and x['inj'] not in ('Out', 'Doubtful')],
                         key=lambda x: -(x['xfp'] or 0))[:2]
            if q['pos'] == 'QB':
                nxt = sorted([x for x in ps if x['id'] != q['id'] and x['pos'] == 'QB' and x['inj'] not in ('Out', 'Doubtful')], key=lambda x: -(x['xfp'] or 0))[:1]
            inj_rows.append(dict(id=q['id'], name=q['name'], pos=q['pos'], team=t, status=q['inj'], injury=q['injd'], xfp=q['xfp'],
                                 next=[dict(name=x['name'], pos=x['pos'], xfp=x['xfp']) for x in nxt]))
order = {'Out': 0, 'Doubtful': 1, 'Questionable': 2}
inj_rows.sort(key=lambda x: (order[x['status']], -x['xfp']))
print('injury entries matched:', len(snap['players']) - len(unmatched), 'of', len(snap['players']), '| unmatched:', unmatched)
print('stars on report:', len(inj_rows), '| players with a teammate-out badge:', sum(1 for p in players if p['tm']))

# ---- current week from the schedule: last fully played week W; projections start the week after W,
# or the one after that when the following week is already mostly played ----
gm = pd.read_csv(f'{DATA}/games.csv', usecols=['season', 'week', 'game_type', 'away_score'])
gm = gm[(gm.season == 2026) & (gm.game_type == 'REG')]
full = sorted(w for w, x in gm.groupby('week') if x.away_score.notna().all())
W = max(full)
nxt = gm[gm.week == W + 1]
k_played = int(nxt.away_score.notna().sum())
next_week = W + 2 if len(nxt) and k_played / len(nxt) >= 0.5 else W + 1
as_of = f'Week {W}' + (f' plus {k_played} Week {W + 1} game{"s" if k_played != 1 else ""}' if k_played else '')
print('as of:', as_of, '| projections start week', next_week)

rm = S['rmse_table']; k = list(rm.keys())[0]
data = dict(players=players, sched=meta['schedule'], dvp={p: {t: round(v, 4) for t, v in d.items()} for p, d in meta['dvp'].items()},
            repl={p: round(v, 3) for p, v in S['repl'].items()}, sd={p: round(v, 3) for p, v in S['sd'].items()},
            summary=dict(rmse=dict(naive=rm[k]['naive'], xfp=rm[k]['xfp'], model=rm[k]['model']), coefs=S['coefs'], lam=S['lam']),
            nextWeek=next_week, asOf=as_of,
            inj=inj_rows, injSource=snap['source'])
html = open(os.path.join(HERE, 'template.html')).read().replace('__DATA__', json.dumps(data, separators=(',', ':'), default=float))
open(f'{OUT}/dashboard.html', 'w').write(html)
print(len(players), 'players;', len(html) // 1024, 'KB')
