# Weekly update runbook

Run every Tuesday 7:00 AM ET (and again Wednesday as a retry) by a scheduled Claude task. It refreshes the data, rebuilds
the model, refreshes the injury report, republishes the dashboard and pushes to this repo. Do not change the model design
during a weekly run; if something fails, stop and report.

Dashboard (claude.ai artifact, update in place): https://claude.ai/artifact/4PJgEUY5atgYxrJSBHucF6

## Steps

1. **Set up.** Attach and clone this repo (`Benjaminvida1/Fantasy-Football-Companion`, push access), then `pip install -r requirements.txt`.
2. **Check the data.** `./run_all.sh --refresh --check`
   - exit 0: a new complete week is ready, continue.
   - exit 3: that week was already published. Stop, and report "nothing new" (the Wednesday retry normally lands here).
   - exit 2: the latest week is not complete in nflverse yet. Stop and report what was missing; the next run retries.
3. **Refresh injuries.** Read https://www.nfl.com/injuries/ with the web fetch tool. Write `reference/nfl_injuries_latest.json`:
   `{"week": N, "fetched": "YYYY-MM-DD", "source": "NFL.com injury report, Week N, read on <date> (nfl.com/injuries). Offensive skill positions only; games already played omitted.", "players": [{"name","pos","team","injury","status"}]}`
   Keep only QB/RB/WR/TE with status Out, Doubtful or Questionable, and omit teams whose game that week is already final.
   Use nflverse team codes (`LA` for the Rams, `LV`, `JAX`, `NE`, `NO`, `NYG`, `NYJ`, `SF`, `TB`, `WAS`, ...).
   If the page cannot be read, build the file from `data/injuries_2026.csv` (latest week, same filters) and say so in `source`.
4. **Rebuild.** `./run_all.sh` (uses the downloaded data). It must end with all sanity checks passing. If a check fails, stop and report; do not publish.
   A name that does not match a player is printed as `unmatched`; players with little or no usage are expected there.
5. **Republish the dashboard.** Read the artifact URL above once, then publish `output/dashboard.html` to the same URL. Do not publish `docs/index.html` to the artifact: it is the standalone copy for GitHub Pages and already contains its own document wrapper.
6. **Push.** `git add -A && git commit -m "Weekly update: through Week N"` and push `main`.
7. **Report briefly:** the week now included, the five biggest sell-high and buy-low flags (players with 3+ games), the injury source used, and any skipped step or warning.

## Why Tuesday morning

Monday Night Football ends around midnight ET. Running Tuesday morning means that game's stats are in nflverse; the
readiness check refuses to run on an incomplete week.
