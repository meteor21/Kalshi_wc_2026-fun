# kalshi-quant
Autonomous Kalshi sports trading. See CLAUDE.md for architecture and hard rules,
**DEPLOY.md** for the full deployment runbook.

Two models plug into `scripts/run_cycle.py` via `score(client) -> list[Signal]`:
- **Tennis** (`models/tennis/`): pre-match win-probability (Elo + features, isotonic-calibrated).
- **Soccer** (`models/soccer/`): calibrated Monte-Carlo state-space simulator + a phase-0
  LOG-ONLY trigger watcher (`scripts/soccer_watch.py`).

Each `score()` safely no-ops until its frozen artifact exists, so the cycle never crashes.

## Setup
1. Create demo account + API key: demo.kalshi.co → Settings → API. Save private key PEM.
2. `cp .env.example .env`, fill in key ID + PEM path (+ `API_FOOTBALL_KEY` for soccer).
3. `pip install -r requirements.txt`
4. `python -m scripts.healthcheck` — balance + model/API readiness before anything trades.
5. `python -m scripts.run_cycle` — full cycle (no-op until a model artifact is present).

## Deploy (Hetzner CX22 / any small VPS)
cron: `*/30 * * * * cd /opt/kalshi-quant && .venv/bin/python -m scripts.run_cycle >> cycle.log 2>&1`

Fit offline, then copy artifacts to the VPS. Full steps, systemd unit for the watcher, and
the go-live checklist are in **DEPLOY.md**.
