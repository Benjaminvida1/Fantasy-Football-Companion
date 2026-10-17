# Fantasy Football Companion

A quantitative model for a 12-team PPR fantasy football league (1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX). It uses opportunity data (targets, air yards, carries, snap share, red-zone usage) to estimate how many points a player *should* have scored, flags who is over- or under-performing, projects the rest of the season, and evaluates trades against your actual roster.

**Dashboard:** open [`docs/index.html`](docs/index.html) in a browser. It is a self-contained page with three tabs: Over / Under (sortable player table and scatter plot), Trade evaluator, and Method. The snapshot covers the 2026 season through Week 5 plus the Week 6 Thursday game.

## What it answers

- **Who is running hot or cold?** Actual PPR points per game against the points their usage predicts, with a z-score so one big game does not look like a trend.
- **Who should I buy low or sell high?** Players whose production is well above or below expectation over at least 3 games.
- **What is a player worth the rest of the way?** Projected points per game, total points through Week 17, and value over a waiver-level replacement.
- **Does this trade help me?** Enter your roster, then the players you would give and get. The tool re-optimizes your best weekly lineup with and without the trade and reports the point difference.

## How it works

1. **Expected points (xFP).** Gradient-boosted trees estimate the PPR value of every target, rush and pass attempt from the situation alone: air yards, yard line, down, distance, pass or run location, and position. Models are trained on 2024-25 play-by-play and cross-fitted by season, so no season is scored by a model that saw it. A player's xFP is the sum over his plays.
2. **Over/under-performance.** `gap = PPR - xFP`, and `z = sum(gap) / sqrt(sum(Var(gap | xFP)))`, with the variance fit per position. A signal needs at least 3 games and |z| >= 1.25.
3. **Rest-of-season projection.** One ridge regression per position on season xFP, last-3-weeks xFP, the gap, snap share and last year's PPG, trained on 2024-25 at several cutoff weeks and tested on the held-out season. Players with fewer than 3 games are shrunk toward last year's PPG.
4. **Strength of schedule.** A two-way ridge model (offense effect + defense effect) gives each defense a softness rating per position. Defense softness does persist through a season (correlation about 0.15-0.35), but in backtests it did **not** improve player projections once usage was known, so the dashboard shows it and defaults its weight to 0. A slider applies it anyway.
5. **Trade value.** Value over replacement across the remaining weeks. With a roster entered, each week's best lineup is recomputed with and without the trade, including bye weeks. Replacement level is the best non-starter in a 12-team league at each position.

### Backtest

Predicting rest-of-season PPG for players with at least 3 games, trained on one season and tested on the other:

| Method | RMSE (PPG) |
|---|---|
| Naive: current PPG | 4.12 |
| Current xFP per game | 3.97 |
| This model | 3.88 |

Errors are still large (about 3-5 PPG per player depending on position), so treat flags as prompts to check a player's usage, not as verdicts.

## Run it

```bash
pip install -r requirements.txt
./run_all.sh
```

`run_all.sh` downloads about 250 MB of nflverse data into `data/` (not committed), builds the expected-points table, fits the projection model, and regenerates `docs/index.html`. To refresh during the season, delete `data/` and run it again.

## Project layout

| Path | Purpose |
|---|---|
| `src/build_xfp.py` | Fits the per-play models and builds the player-week table (xFP, shares, snaps, actual PPR) |
| `src/model.py` | Over/under z-scores, backtested projection, schedule ratings, replacement levels |
| `src/build_html.py`, `src/template.html` | Injects model output into the dashboard page |
| `analysis/` | Schedule-persistence diagnostic and a spot-check against raw weekly stats |
| `output/` | Latest model output (`players_2026.csv`, `model_meta.json`) |
| `docs/index.html` | The dashboard |
| `fetch_data.sh` | Downloads the raw nflverse data |

## Limits

- Injuries beyond the latest report, role changes after a teammate's injury, trades and coaching changes are not modeled.
- Routes run and personnel groupings are not in the public data.
- The trade error band treats each player's projection error as independent, which understates risk when both sides of a trade involve the same team.

## Data

[nflverse](https://github.com/nflverse/nflverse-data): play-by-play, weekly player stats, snap counts, injuries, schedules and player IDs for 2023-2026.
