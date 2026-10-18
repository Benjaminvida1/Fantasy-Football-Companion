"""Opportunity-transfer library: absent-starter detection, pooled transfer rates, stripped baselines, boost components.

A "star" is an RB/WR/TE whose trailing-3-appearance xFP/g meets a position threshold. A star is "absent" in a team
game when he has no row or 10 or fewer offensive snaps. A team-week with an absent star is a "boosted" week for his
teammates. QB absences are not modeled (the backup QB, not the receivers, absorbs the lost production).
"""
import numpy as np, pandas as pd

POS3 = ['RB', 'WR', 'TE']
THR = {'RB': 9.0, 'WR': 9.0, 'TE': 7.0}
QB_THR = 14.0
NAIVE_K = 2          # naive timing: a star out at the cutoff stays out for the next K team games
KINDS = ['same', 'other']


class TeamSeason:
    """Per team-season cache of xFP/snaps by player and week, plus trailing baselines."""

    def __init__(self, d, weeks):
        self.weeks = sorted(weeks)
        self.X = d.pivot_table(index='gsis_id', columns='week', values='xfp', aggfunc='sum')
        self.SN = d.pivot_table(index='gsis_id', columns='week', values='off_snaps', aggfunc='sum')
        u = d.drop_duplicates('gsis_id', keep='last').set_index('gsis_id')
        self.pos = u.pos.to_dict()
        self.name = u['name'].to_dict()
        self._pres = {}
        self._base = {}

    def pres(self, p, w):
        k = (p, w)
        if k not in self._pres:
            ok = (w in self.X.columns) and pd.notna(self.X.at[p, w])
            if ok and w in self.SN.columns and pd.notna(self.SN.at[p, w]):
                ok = self.SN.at[p, w] > 10
            self._pres[k] = bool(ok)
        return self._pres[k]

    def base(self, w):
        """Trailing-3-appearance xFP/g of every player as of the start of week w (needs 2+ games, one in last 5 weeks)."""
        if w not in self._base:
            prev = [x for x in self.weeks if x < w]
            out = {}
            for p in self.X.index:
                app = [x for x in prev if self.pres(p, x)]
                if len(app) >= 2 and app[-1] >= w - 5:
                    out[p] = float(np.mean([self.X.at[p, x] for x in app[-3:]]))
            self._base[w] = out
        return self._base[w]

    def absent_stars(self, w):
        b = self.base(w)
        return [p for p, v in b.items() if self.pos[p] in POS3 and v >= THR[self.pos[p]] and not self.pres(p, w)]

    def absent_qb(self, w):
        b = self.base(w)
        return [p for p, v in b.items() if self.pos[p] == 'QB' and v >= QB_THR and not self.pres(p, w)]


def build_cache(pw, games):
    """games: played regular-season games with columns season, week, team."""
    cache = {}
    tw = games.groupby(['season', 'team']).week.apply(list).to_dict()
    for (s, t), d in pw.groupby(['season', 'team']):
        if (s, t) in tw:
            cache[(s, t)] = TeamSeason(d, tw[(s, t)])
    return cache


def played_games(g):
    g = g[(g.game_type == 'REG') & g.away_score.notna()]
    a = g.rename(columns={'away_team': 'team'})[['season', 'week', 'team']]
    b = g.rename(columns={'home_team': 'team'})[['season', 'week', 'team']]
    return pd.concat([a, b])


def events_and_controls(cache, seasons, first_week=4, last_week=17):
    ev, ct = [], []
    for (s, t), ts in cache.items():
        if s not in seasons:
            continue
        for w in ts.weeks:
            if w < first_week or w > last_week:
                continue
            stars, qbs = ts.absent_stars(w), ts.absent_qb(w)
            base = ts.base(w)

            def split(a):
                others = [p for p in ts.X.index if p != a and ts.pos[p] in POS3 and ts.pres(p, w)]
                gain = sum(ts.X.at[p, w] - base.get(p, 0.0) for p in others)
                same = sum(ts.X.at[p, w] - base.get(p, 0.0) for p in others if ts.pos[p] == ts.pos[a])
                return gain, same

            if len(stars) == 1 and not qbs:
                a = stars[0]; gain, same = split(a)
                ev.append(dict(season=s, team=t, week=w, name=ts.name[a], pos=ts.pos[a], base=base[a], k=1,
                               same=same / base[a], other=(gain - same) / base[a]))
            elif not stars and not qbs:
                pres_stars = [p for p, v in base.items() if ts.pos[p] in POS3 and v >= THR[ts.pos[p]] and ts.pres(p, w)]
                if pres_stars:
                    a = pres_stars[0]; gain, same = split(a)
                    ct.append(dict(season=s, pos=ts.pos[a], same=same / base[a], other=(gain - same) / base[a]))
    return pd.DataFrame(ev), pd.DataFrame(ct)


def estimate_rates(cache, seasons):
    """Pooled net transfer rates: share of the absent star's lost xFP that lands on same-position / other-position
    teammates, net of what teams with nobody out show against the same trailing baselines."""
    E, C = events_and_controls(cache, seasons)
    rates, n = {}, {}
    for pa in POS3:
        e, c = E[E.pos == pa], C[C.pos == pa]
        rates[pa] = {'same': max(0.0, e.same.mean() - c.same.mean()), 'other': max(0.0, e.other.mean() - c.other.mean())}
        n[pa] = len(e)
    return rates, n, E, C


def boost_components(ts, cutoff, scenario, recipients, bclean, fut_weeks):
    """Per-recipient components comp[(pos_a, kind)] = sum over future weeks of star_base * share / n_future_games.
    The projected boost in xFP/g is sum(rate[pos_a][kind] * comp).  scenario: 'naive' | 'oracle' | dict(star -> set of out weeks)."""
    out = {p: {(pa, k): 0.0 for pa in POS3 for k in KINDS} for p in recipients}
    if not fut_weeks:
        return out
    base = ts.base(cutoff + 1)
    stars = [p for p, v in base.items() if ts.pos[p] in POS3 and v >= THR[ts.pos[p]]]
    for j, w in enumerate(fut_weeks):
        if isinstance(scenario, dict):
            ab = [a for a in stars if w in scenario.get(a, set())]
        elif scenario == 'oracle':
            ab = [a for a in stars if not ts.pres(a, w)]
        else:  # naive
            ab = [a for a in stars if cutoff in ts.weeks and not ts.pres(a, cutoff) and j < NAIVE_K]
        for a in ab:
            pa = ts.pos[a]
            act = [p for p in recipients if p != a and p not in ab]
            s_same = sum(bclean[p] for p in act if ts.pos[p] == pa)
            s_oth = sum(bclean[p] for p in act if ts.pos[p] != pa)
            for p in act:
                if ts.pos[p] == pa and s_same > 0:
                    out[p][(pa, 'same')] += base[a] * bclean[p] / s_same / len(fut_weeks)
                elif ts.pos[p] != pa and s_oth > 0:
                    out[p][(pa, 'other')] += base[a] * bclean[p] / s_oth / len(fut_weeks)
    return out


def apply_rates(comp, rates):
    return sum(rates[pa][k] * v for (pa, k), v in comp.items())
