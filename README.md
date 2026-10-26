# Fantasy Football Companion

A quantitative model for a 12-team PPR fantasy football league (1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX). It uses opportunity data (targets, air yards, carries, snap share, red-zone usage) to estimate how many points a player *should* have scored, flags who is over- or under-performing, projects the rest of the season, and evaluates trades against your actual roster.

**Dashboard:** https://benjaminvida1.github.io/Fantasy-Football-Companion/ (GitHub Pages; refreshed weekly, the header shows the as-of week). You can also open [`docs/index.html`](docs/index.html) locally in a browser. It is a self-contained page with four tabs: Over / Under (sortable player table and scatter plot), Trade evaluator, Injuries, and Method. 

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

## Weekly refresh

`./run_all.sh --refresh` downloads the latest data, checks that the newest NFL week is complete in every dataset, rebuilds the model, runs sanity checks, and writes `output/dashboard.html` (page fragment for the Claude artifact) and `docs/index.html` (standalone page for GitHub Pages). A scheduled Claude task runs this every Tuesday at 7:00 AM ET (with a Wednesday retry) following [`WEEKLY_UPDATE.md`](WEEKLY_UPDATE.md): it also refreshes the injury report from NFL.com, republishes the dashboard link and pushes here. Tuesday morning is deliberate, since Monday Night Football finishes around midnight ET. The season constant (2026) is set in `src/check_ready.py`, `src/build_html.py` and `fetch_data.sh`.

## Teammate injuries: tested, not applied

When a starter is out, do his teammates' projections need adjusting? I measured it and backtested it. Short answer: teammates do absorb part of the lost usage, but adjusting the projection for it did not improve accuracy, so the model ignores it.

- **Transfer rates.** Across 207 games in 2024-25 with exactly one RB, WR or TE starter out (trailing-3-game xFP of at least 9, 9 and 7), his RB/WR/TE teammates together gained about 0.64 (RB out), 0.49 (WR out) and 0.61 (TE out) of his lost xFP, net of teams with nobody out. Standard errors are large.
- **Season-long backtest** (same folds and filters as above):

| Variant | RMSE (PPG) |
|---|---|
| Current model | **3.880** |
| Strip games played while a starter was out | 3.906 |
| Strip + boost, starter assumed out 2 more games | 3.914 |
| Strip + boost, exact return weeks known | 4.046 |
| Same, boost weight fitted to the data | 3.892 |

- **One week ahead** (1,557 teammate-weeks): the pooled-rate boost predicted +0.95 xFP against +0.54 observed, and improved predictions only for RB teammates (RMSE 5.25 to 5.20).
- **What shipped.** The Injuries tab lists injured starters from the NFL.com report with a "back in" note you can set per player (saved in your browser), and the player table shows a "Teammate out" badge. Both are context only and change no projection or trade value.
- **Injury data.** `reference/nfl_injuries_latest.json` is a snapshot of the NFL.com report (skill positions only), refreshed by the weekly Claude task. It has game-day status but no return timelines.

Reproduce with `python3 analysis/transfer_backtest.py data output` and `python3 analysis/weekly_check.py data output` (run `./run_all.sh` first so `output/playerweek.csv` exists).

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
| `src/build_html.py`, `src/template.html` | Injects model output and the injury snapshot into the dashboard page |
| `src/opportunity.py` | Absent-starter detection, pooled transfer rates, stripped baselines (used by the injury analysis) |
| `analysis/` | Schedule-persistence diagnostic, spot-check against raw stats, and the injury-transfer backtests |
| `reference/` | Latest NFL.com injury report snapshot (refreshed weekly) |
| `src/check_ready.py`, `src/sanity.py` | Week-complete check and output sanity checks used by `run_all.sh` |
| `WEEKLY_UPDATE.md` | Runbook for the weekly scheduled update |
| `output/` | Latest model output (`players_2026.csv`, `model_meta.json`) |
| `docs/index.html` | The dashboard as a standalone page (served by GitHub Pages; the root `index.html` redirects here) |
| `fetch_data.sh` | Downloads the raw nflverse data |

## Limits

- Teammate injuries are shown but not applied to projections (see above). Role changes, trades and coaching changes are not modeled, and QB absences are not analyzed at all.
- Routes run and personnel groupings are not in the public data.
- The trade error band treats each player's projection error as independent, which understates risk when both sides of a trade involve the same team.

## Data

[nflverse](https://github.com/nflverse/nflverse-data): play-by-play, weekly player stats, snap counts, injuries, schedules and player IDs for 2023-2026.
