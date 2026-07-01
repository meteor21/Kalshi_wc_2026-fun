"""One full cycle: refresh data -> score markets -> filter by rules -> place orders.
Cron this. Models plug in via score(client) -> list[Signal]. A model raising must never
kill the cycle, so each is wrapped; a failure is logged and skipped."""
import os
import traceback

import db
from strategy import rules

MODELS = [
    ("tennis", "models.tennis.score"),
    ("soccer", "models.soccer.score"),
]


def get_signals(client):
    """Run every plugged-in model. Returns list[(model_name, Signal)]. Fail-safe:
    a model that errors is logged and skipped rather than aborting the cycle."""
    out = []
    for name, module in MODELS:
        try:
            mod = __import__(module, fromlist=["score"])
            for sig in mod.score(client) or []:
                out.append((name, sig))
        except Exception as e:  # noqa: BLE001 - never let one model kill the cycle
            print(f"MODEL {name} failed: {e!r}")
            traceback.print_exc()
    return out


def current_state(client):
    bal = client.balance().get("balance", 0) / 100.0  # cents -> dollars
    pos = client.positions()
    exposure = sum(abs(p.get("market_exposure", 0))
                   for p in pos.get("market_positions", [])) / 100.0
    return bal, exposure


def _stake_for(sig, bankroll):
    """Kelly stake, capped at the parlay tier for trigger legs, single tier otherwise."""
    cap = rules.PARLAY_MAX_STAKE if sig.is_parlay_leg else rules.SINGLE_MAX_STAKE
    return min(rules.kelly_stake(sig, bankroll), cap)


def run():
    from execution.kalshi_client import KalshiClient
    client = KalshiClient()
    if client.env == "prod" and os.environ.get("ALLOW_PROD") != "1":
        raise SystemExit("refusing to trade prod without ALLOW_PROD=1 (calibrate on demo first)")

    db.init()
    bankroll, exposure = current_state(client)
    with db.conn() as c:
        db.record_equity(c, bankroll, exposure)
        daily_pnl = db.daily_realized_pnl(c)

        if bankroll <= rules.HALT_FLOOR:
            print(f"HALT: bankroll {bankroll:.2f} <= floor {rules.HALT_FLOOR}"); return
        if daily_pnl <= -rules.DAILY_LOSS_LIMIT:
            print(f"HALT: daily loss {daily_pnl:.2f} <= -{rules.DAILY_LOSS_LIMIT}"); return

        for model, sig in get_signals(client):
            sig.fee_per_contract = rules.kalshi_fee(sig.price)
            e = rules.edge(sig)
            pred_id = db.log_prediction(c, model, sig.ticker, sig.side, sig.model_prob, sig.price, e)
            stake = _stake_for(sig, bankroll)
            ok, reason = rules.check_single(sig, stake, bankroll, exposure, daily_pnl)
            if not ok or stake <= 0:
                print(f"SKIP {sig.ticker} {sig.side}: {reason if not ok else 'zero stake'}")
                continue
            price_cents = max(1, min(99, round(sig.price * 100)))
            count = int(stake / sig.price)
            if count < 1:
                print(f"SKIP {sig.ticker} {sig.side}: stake {stake:.2f} < 1 contract")
                continue
            resp = client.create_order(sig.ticker, sig.side, "buy", count, price_cents)
            oid = resp.get("order", {}).get("order_id")
            db.log_order(c, pred_id, sig.ticker, sig.side, "buy", count, price_cents, oid,
                         is_parlay=int(sig.is_parlay_leg))
            exposure += stake
            print(f"ORDER [{model}] {sig.ticker} {sig.side} x{count} @ {price_cents}c (edge {e:.3f})")


if __name__ == "__main__":
    run()
