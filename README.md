# kalshi-quant
Autonomous Kalshi sports trading. See CLAUDE.md for architecture and hard rules.

## Setup
1. Create demo account + API key: demo.kalshi.co → Settings → API. Save private key PEM.
2. `cp .env.example .env`, fill in key ID and PEM path.
3. `pip install -r requirements.txt`
4. `python -m scripts.healthcheck` — must print balance before anything else.
5. `python -m scripts.run_cycle` — full cycle (no-op until models plugged in).

## Deploy (Hetzner CX22 / any small VPS)
cron: `*/30 * * * * cd /opt/kalshi-quant && .venv/bin/python -m scripts.run_cycle >> cycle.log 2>&1`
