"""Deployment readiness check. Run first after setting .env: verifies Kalshi auth,
model artifacts, and API-Football access before anything trades."""
import os
from pathlib import Path

from models.soccer import hazard
from models.soccer.dixon_coles import PARAMS_PATH as DC_PARAMS
from models.soccer.triggers import betting_enabled
from models.tennis.score import ARTIFACT as TENNIS_ARTIFACT


def _ok(b):
    return "OK" if b else "--"


def main():
    from execution.kalshi_client import KalshiClient
    print("=== Kalshi ===")
    c = KalshiClient()
    print(f"env={c.env} base={c.base}")
    bal = c.balance()
    print(f"balance: {bal}")
    for mk in c.markets(limit=3).get("markets", []):
        print(f"  {mk['ticker']}: {mk.get('title', '')[:60]}")

    print("\n=== models (frozen artifacts) ===")
    print(f"[{_ok(Path(TENNIS_ARTIFACT).exists())}] tennis  {TENNIS_ARTIFACT}")
    print(f"[{_ok(Path(hazard.MODELS_PATH).exists())}] soccer hazard  {hazard.MODELS_PATH}")
    print(f"[{_ok(Path(DC_PARAMS).exists())}] soccer dixon-coles  {DC_PARAMS}")
    print("  (missing artifacts => that model's score() safely no-ops)")

    print("\n=== API-Football ===")
    if not os.environ.get("API_FOOTBALL_KEY"):
        print("[--] API_FOOTBALL_KEY not set (soccer pull + trigger watcher disabled)")
    else:
        from models.soccer.data import APIFootball
        st = APIFootball().status()
        sub, req = st.get("subscription", {}), st.get("requests", {})
        print(f"[OK] plan={sub.get('plan')}  requests today={req.get('current')}/{req.get('limit_day')}")

    print(f"\nsoccer trigger betting: {'ENABLED' if betting_enabled() else 'log-only (phase-0)'}")


if __name__ == "__main__":
    main()
