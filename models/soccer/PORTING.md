# PORTING.md — soccer engine from research/soccer_research.ipynb

Source: capstone "Calibrated Monte Carlo State-Space Simulator" (paper NOT in repo; this file
is the spec). Do NOT run the notebook. Extract from listed cells only.

## Cell map
- Cell 2: API-Football acquisition + cache (fixtures, events, team stats). Port the client +
  caching; add live-fixture polling (see Live data below).
- Cells 4, 6: cleaning, event integrity checks (drop shootout goals, VAR-overturned, score-vs-event
  consistency). Port as-is — this is the data-quality layer.
- Cell 9: Dixon-Coles per-competition fits (L2-regularized L-BFGS-B, low-score correction),
  cross-competition Elo, rolling form. PORT the DC fitter. SKIP the 171-feature GBM (see below).
- Cells 12, 14: minute-level state stream (t, sh, sa, rh, ra) + Poisson hazard GLM:
  log lambda_k(t|s) = b0 + time buckets + state indicators (leading, big lead, man up) + log DC strengths.
  This is the core asset. Port fit + persisted coefficients.
- Cell 16: per-fixture calibration — multiplicative (alpha_h, alpha_a) via KL grid search [0.60,1.55]^2.
  Port mechanism, CHANGE TARGET (below).
- Cell 19: forward simulator (2,000-4,000 paths from arbitrary state). Port; vectorize with numpy
  so a full revaluation runs <1s on the VPS.
- Cell 21: trigger validation logic. Port trigger detection only.
- Cells 23-31: synthetic backtest + figures. DO NOT PORT (research only).

## PRODUCTION CHANGE 1 — anchor to market, not the GBM
Paper finding: pre-match GBM loses to Bet365 closing (Brier 0.6266 vs 0.6063). Therefore the
production calibration target in eq (2) is the DE-OVERROUNDED MARKET 1X2 vector (Kalshi mid or
closing odds), not the GBM. GBM is not ported. Consequence: simulator carries zero pre-match
opinion; ALL claimed edge is in-match state response. This is the honest version of the paper's
conclusion ("its value, if any, has to come after kickoff").

## PRODUCTION CHANGE 2 — trigger watcher = phase-0 deployable (paper's stated future work)
Rule: pre-match favorite (market p > 0.55) concedes first goal before minute 35.
Conservative hit-rate basis: held-out 77.4% BTTS (NOT full-sample 88.1%), n=84 — small, so:
- Margin: require quoted price to imply <= 72% (5pt haircut) before any stake.
- Stake: parlay-tier sizing ($20 cap) regardless of single-leg status, until live sample > 100 triggers.
- LOG-ONLY MODE FIRST: on every trigger, record all available live quotes (Kalshi + any
  reference odds) with timestamps. This builds the matched live-price dataset the paper lacked.
  Betting enables only after logged quotes show executable prices clear the threshold.
- Map trigger bets to contracts that actually exist on Kalshi (verify at build: 3-way winner
  likely; BTTS/totals possibly absent). "Fav avoid loss" = buy NO on opponent-win contract.
  If BTTS untradable on Kalshi, the trigger sleeve logs anyway — data is the product.

## Live data
- API-Football live endpoints (existing key/pipeline from cell 2). Poll 30-60s during tracked
  fixtures; extract goals, red cards, minute. Budget: stays on an affordable tier; no SportRadar.
- Execution discipline (ports the paper's state-change ban): never take a quote captured before
  a detected state change; after goal/red card, wait for fresh quote, use limit orders only.

## World Cup adaptation (models/soccer/wc.py)
- National-team DC strengths are sparse. v1: derive lambda_DC inputs from market-implied outright
  and match odds rather than fitting on thin international data.
- Pooled hazard STATE coefficients (time buckets, score state, red cards) transfer from club data
  as an explicit assumption — document it. Edge threshold for WC: MIN_EDGE + 0.02 (stricter).
- Prior widening: simulator path count 4,000 minimum; treat probabilities within +/-3pts of
  breakeven as no-trade.

## Target files
- models/soccer/data.py        API-Football client, cache, integrity checks, live poller
- models/soccer/dixon_coles.py per-competition DC fitter (persisted params)
- models/soccer/hazard.py      Poisson GLM fit + coefficient persistence
- models/soccer/simulator.py   calibrate-to-market + forward simulate from state
- models/soccer/triggers.py    trigger detection + quote logging (phase-0)
- models/soccer/wc.py          World Cup market-implied strength layer
- models/soccer/score.py       score(client) -> list[Signal] for run_cycle

## Constraints
- Hazard/DC fitting offline (Colab/local); VPS loads frozen coefficients + runs simulator/poller.
- Trigger log schema: add table trigger_events(ts, fixture_id, trigger_type, state_json,
  quotes_json, acted) to db/schema.sql.
- All paper caveats hold: n=84 holdout, not pre-registered, no multiple-testing correction.
  The system's first job is to collect the prospective sample, not to bet it.
