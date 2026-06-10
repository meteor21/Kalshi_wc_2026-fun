# kalshi-quant

Autonomous Kalshi trading system. Soccer (World Cup) + tennis. Bankroll $1000.

## Token discipline (IMPORTANT)
- Be terse. No preamble, no recap of changes unless asked.
- Read only files needed for the task. Never read db/*.db or data/*.parquet.
- Prefer grep/targeted reads over whole-file reads.
- One concern per session; /compact when long. No speculative refactors, no verbose docstrings.

## Architecture
- execution/kalshi.py — API client (RSA-PSS auth, markets, orders, balance). Env: KALSHI_ENV=demo|prod.
- db/schema.sql, db/store.py — SQLite. Log every model output, price seen, order, fill.
- strategy/rules.py — hard constraints. Execution NEVER bypasses rules.
- models/* — probability models. Output rows: (market_ticker, p_model, ts).
- scripts/run_cycle.py — scan→decide→trade, cron entry point.

## Hard rules (never relax in code)
- Multi-leg/parlay-style stake <= $20, only if total payout >= 5x stake.
- Single bet stake <= $100; only if total return <= 3x stake (i.e. price >= ~0.33).
- Min edge: p_model - p_breakeven >= 0.04, where p_breakeven includes Kalshi fee 0.07*p*(1-p)/contract.
- Max open exposure 30% of bankroll. Daily loss >= 10% -> halt(), require manual reset.
- KALSHI_ENV=demo until calibration proves out. Never default prod.

## API facts (verified 2026-06)
- Prod https://api.elections.kalshi.com/trade-api/v2 | Demo https://demo-api.kalshi.co/trade-api/v2
- Sign RSA-PSS(SHA256, salt=32) over f"{ts_ms}{METHOD}{path}"; path includes /trade-api/v2, EXCLUDES query string. ts in MILLISECONDS.
- Headers: KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP, KALSHI-ACCESS-SIGNATURE (b64).
- ~10 req/s authed. Fees: ~$0.02/contract executed + trading fee ceil(0.07*p*(1-p)) per contract.
- Secrets via env: KALSHI_KEY_ID, KALSHI_PRIVATE_KEY_PATH. Never commit keys.

## Conventions
- Python 3.12. Deps: requests, cryptography (polars later for research). No frameworks.
- Prices as float in [0,1] internally (Kalshi cents/100). Money in cents (int) in db. UTC everywhere.
