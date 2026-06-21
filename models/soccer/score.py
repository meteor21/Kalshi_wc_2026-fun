"""Live soccer scorer for run_cycle.get_signals(). Two sleeves:

  1. Trigger watcher (phase-0, LOG-ONLY): detect the favorite-concedes-early trigger and
     record live quotes. This is the immediately deployable, market-free piece.
  2. State-space simulator: anchor DC strengths to the de-overrounded market so the
     PRE-match simulation reproduces the market (alpha=1, zero pre-match opinion per
     PORTING.md change 1); all edge is the in-match hazard response. Revalue from the
     current state and compare to live Kalshi prices.

VPS-side: loads frozen hazard coefficients and runs the simulator/poller. Fitting is
offline. No-ops cleanly when coefficients or markets are absent."""
from pathlib import Path

from strategy import rules
from strategy.rules import Signal
from . import hazard, triggers, wc
from .simulator import revalue


def evaluate_fixture(fx, haz, conn=None):
    """Pure core: given a resolved live fixture, return Signals. Always logs a fired
    trigger (log-only); emits simulator Signals unless a state change just occurred.

    fx keys: fixture_id, is_world_cup, market_1x2 [h,d,a], state (start_min,hs,as,hr,ar),
    state_changed (bool), trigger_state (dict), contracts {outcome: {ticker, ask_cents}},
    trigger_contracts (list)."""
    sigs = []

    # --- sleeve 1: trigger watcher (market-free, log-only) ---
    trig = triggers.detect(fx.get("trigger_state", {}))
    if trig:
        if conn is not None:
            triggers.log(conn, trig, fx.get("quotes", {}))
        sigs += triggers.to_signals(trig, fx.get("trigger_contracts", []))

    # --- sleeve 2: state-space simulator ---
    # state-change ban: never act on a quote captured before a detected state change
    if fx.get("state_changed"):
        return sigs
    dc_lh, dc_la = wc.lambdas_from_market(fx["market_1x2"])   # anchor to market (alpha=1)
    n_sims = wc.WC_N_SIMS if fx.get("is_world_cup") else 4000
    sim = revalue(fx["state"], haz, dc_lh, dc_la, 1.0, 1.0, n_sims=n_sims)

    for outcome, c in fx.get("contracts", {}).items():
        if outcome not in sim:
            continue
        price = c["ask_cents"] / 100.0
        fee = rules.kalshi_fee(price)
        breakeven = price + fee
        model_p = sim[outcome]
        # WC is held to a stricter edge + no-trade band; club leaves rules.check_single to run_cycle
        if fx.get("is_world_cup") and not wc.tradable(model_p, breakeven):
            continue
        sigs.append(Signal(c["ticker"], "yes", model_p, price, fee))
    return sigs


def tracked_fixtures(client):
    """SEAM: resolve live soccer fixtures the system is tracking into the fx dicts
    evaluate_fixture() consumes — joins API-Football live state (models.soccer.data)
    with Kalshi soccer contracts + de-overrounded market 1X2, and flags state changes
    vs the previous cycle. Returns [] until wired to the live API + Kalshi schema."""
    return []


def score(client):
    """Return list[Signal] for tracked soccer fixtures. No-op until hazard coefficients
    are fit and live fixtures are wired."""
    if not Path(hazard.MODELS_PATH).exists():
        return []
    haz = hazard.load()
    import db
    out = []
    with db.conn() as conn:
        for fx in tracked_fixtures(client):
            out += evaluate_fixture(fx, haz, conn)
    return out
