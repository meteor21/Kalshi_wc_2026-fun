"""Phase-0 trigger watcher (PORTING.md change 2; capstone Sec 9 — the cleanest
empirical result, no simulator or market price required).

Rule: a pre-match favorite (de-overrounded MARKET p > 0.55) concedes the first goal
before minute 35. Held-out BTTS hit rate 77.4% (n=84) — small, exploratory, not
pre-registered. So phase-0 is LOG-ONLY: on every fired trigger, record all available
live quotes with timestamps to build the matched live-price dataset the paper lacked.
Betting turns on only once logged quotes show executable prices clearing the threshold."""
import json
import os

from strategy.rules import Signal

TRIGGER_TYPE = "big_fav_conceded_first_35"
FAV_MARKET_THRESHOLD = 0.55      # "big favorite" by de-overrounded market prob
EARLY_MINUTE = 35
HELDOUT_BTTS = 0.774             # conservative held-out rate (NOT full-sample 88.1%)
MARGIN_HAIRCUT = 0.05            # 5pt haircut -> only bet if quote implies <= 0.724
MAX_IMPLIED = HELDOUT_BTTS - MARGIN_HAIRCUT
MIN_LIVE_SAMPLE = 100            # parlay-tier sizing until prospective sample exceeds this


def betting_enabled():
    """Phase-0 default is False (log only). Flip via env once the sample is built."""
    return os.environ.get("SOCCER_TRIGGER_BETTING") == "1"


def detect(state):
    """state: {fixture_id, minute, fav_side, fav_market_prob, first_goal_side,
    first_goal_minute, ...}. Returns a trigger dict or None."""
    if state.get("fav_market_prob", 0.0) <= FAV_MARKET_THRESHOLD:
        return None
    side, minute = state.get("first_goal_side"), state.get("first_goal_minute")
    if side is None or minute is None:
        return None
    if side == state.get("fav_side") or minute > EARLY_MINUTE:
        return None             # favorite did NOT concede first, or not early enough
    return {"trigger_type": TRIGGER_TYPE, "fixture_id": state["fixture_id"], "state": state}


def log(conn, trigger, quotes):
    """LOG-ONLY: persist trigger + all captured live quotes. Data is the product."""
    import db
    return db.log_trigger_event(conn, trigger["fixture_id"], trigger["trigger_type"],
                                json.dumps(trigger["state"]), json.dumps(quotes))


def to_signals(trigger, contracts):
    """Map a fired trigger to parlay-tier Signals IFF betting is enabled and a quote
    clears the conservative threshold. contracts: resolved Kalshi contracts as
    {ticker, side_for_bet, ask_cents}. 'Fav avoid loss' = buy NO on opponent-win;
    BTTS = buy YES if listed. Returns [] in phase-0 (log-only)."""
    if not betting_enabled():
        return []
    out = []
    for c in contracts:
        price = c["ask_cents"] / 100.0
        if price > MAX_IMPLIED:          # market already prices the edge away
            continue
        out.append(Signal(c["ticker"], c["side_for_bet"], HELDOUT_BTTS, price, is_parlay_leg=True))
    return out                            # NB: size at parlay tier ($20 cap), not single
