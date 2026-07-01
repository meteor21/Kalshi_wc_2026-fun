# Deployment

Small VPS (e.g. Hetzner CX22). Fitting runs offline; the VPS loads frozen artifacts and
runs the cron cycle + the live trigger watcher.

## 1. Setup
```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in Kalshi key id + PEM path, API_FOOTBALL_KEY
python -m scripts.healthcheck # must print balance + readiness before anything trades
```

## 2. Secrets / safety posture (do not skip)
- `.env` and `*.pem` are gitignored — never commit them.
- `KALSHI_ENV=demo` until calibration proves out. `run_cycle` refuses to trade `prod`
  unless `ALLOW_PROD=1` is also set.
- Hard rules live in `strategy/rules.py` and are enforced in `run_cycle` every cycle:
  halt below the equity floor, halt at the daily-loss limit (computed from settled P&L),
  edge/stake/exposure caps. Trigger legs are sized at the parlay tier ($20 cap).

## 3. Fit the models (offline: Colab / local / any box with data access)
```
# Tennis (Sackmann + tennis-data.co.uk are public)
python -m models.tennis.train           # -> models/tennis/model.joblib
python -m models.tennis.backtest        # GO / NO-GO vs Bet365 closing; singles stay off on NO-GO

# Soccer (needs API_FOOTBALL_KEY)
python -m scripts.soccer_pull --seasons 2018-2025   # cached, resumable, parallel
python -m scripts.soccer_analyze                    # -> hazard_models.joblib, dc_params.json
```
Copy the produced `*.joblib` / `dc_params.json` next to their modules on the VPS.
Missing artifacts => that model's `score()` safely no-ops (no crash, no trades).

## 4. Run
**Trading cycle** (cron, every 30 min). No-ops for any model without an artifact:
```
*/30 * * * * cd /opt/kalshi-quant && .venv/bin/python -m scripts.run_cycle >> cycle.log 2>&1
```

**Phase-0 soccer trigger watcher** (LOG-ONLY — collects the prospective live-quote sample
the paper lacked; does NOT bet until `SOCCER_TRIGGER_BETTING=1`). Run as a long-lived
service, e.g. systemd:
```
[Service]
WorkingDirectory=/opt/kalshi-quant
Environment=API_FOOTBALL_KEY=...
ExecStart=/opt/kalshi-quant/.venv/bin/python -m scripts.soccer_watch --every 45
Restart=always
```

## 5. Go-live checklist
1. `healthcheck` green (balance + artifacts + API-Football quota).
2. Tennis backtest = GO; soccer trigger sample built and prices seen to clear the threshold.
3. Flip `KALSHI_ENV=prod` + `ALLOW_PROD=1` (and `SOCCER_TRIGGER_BETTING=1`) only then.
