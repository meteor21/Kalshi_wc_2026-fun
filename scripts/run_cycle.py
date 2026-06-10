"""One full cycle: refresh data -> score markets -> filter by rules -> place orders.
Cron this. Models plug in via score() functions returning Signal lists."""
import db
from execution.kalshi_client import KalshiClient
from strategy import rules
from strategy.rules import Signal


def get_signals(client):
    """Plug models in here. Each returns list[Signal]. Stub until models land."""
    signals = []
    # from models.soccer.score import score as wc_score; signals += wc_score(client)
    # from models.tennis.score import score as tn_score; signals += tn_score(client)
    return signals


def current_state(client):
    bal = client.balance().get("balance", 0) / 100.0  # cents -> dollars
    pos = client.positions()
    exposure = sum(abs(p.get("market_exposure", 0)) for p in pos.get("market_positions", [])) / 100.0
    return bal, exposure


def run():
    db.init()
    client = KalshiClient()
    bankroll, exposure = current_state(client)
    daily_pnl = 0.0  # TODO: compute from settlements table for today

    with db.conn() as c:
        for sig in get_signals(client):
            sig.fee_per_contract = rules.kalshi_fee(sig.price)
            e = rules.edge(sig)
            pred_id = db.log_prediction(c, "stub", sig.ticker, sig.side, sig.model_prob, sig.price, e)
            stake = min(rules.kelly_stake(sig, bankroll), rules.SINGLE_MAX_STAKE)
            ok, reason = rules.check_single(sig, stake, bankroll, exposure, daily_pnl)
            if not ok:
                print(f"SKIP {sig.ticker} {sig.side}: {reason}")
                continue
            count = int(stake / sig.price)
            price_cents = int(sig.price * 100)
            resp = client.create_order(sig.ticker, sig.side, "buy", count, price_cents)
            oid = resp.get("order", {}).get("order_id")
            db.log_order(c, pred_id, sig.ticker, sig.side, "buy", count, price_cents, oid)
            exposure += stake
            print(f"ORDER {sig.ticker} {sig.side} x{count} @ {price_cents}c (edge {e:.3f})")


if __name__ == "__main__":
    run()
