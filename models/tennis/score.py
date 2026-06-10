"""Live pre-match tennis scorer. Runs on the VPS: loads the frozen artifact
(model + isotonic + FeatureState) and emits Signals for run_cycle.get_signals().
Training/backtest happen offline; here we only read state and predict."""
import os
from pathlib import Path

import joblib
import pandas as pd

from strategy.rules import Signal
from .features import surface_type

ARTIFACT = Path(__file__).parent / "model.joblib"
DEFAULT_SURFACE = "Hard_outdoor"


def _load(artifact=ARTIFACT):
    return joblib.load(artifact)


def _predict(art, player_a, player_b, surface):
    """P(player_a beats player_b) from the calibrated model + persisted FeatureState."""
    row = art["state"].prematch(player_a, player_b, surface)
    X = pd.DataFrame([row])[art["features"]]
    raw = art["clf"].predict_proba(X)[:, 1][0]
    return float(art["iso"].predict([raw])[0])


def parse_matchup(market, known_players):
    """SEAM: map a Kalshi tennis market -> (yes_player, opp_player, surface).
    Kalshi's tennis schema isn't pinned in this repo, so we match the two known
    player names that appear in the market text and treat the YES contract's player
    as player_A. Returns None if the market can't be resolved. Override per the
    real series schema once confirmed against the API."""
    text = " ".join(str(market.get(k, "")) for k in
                    ("title", "yes_sub_title", "subtitle", "yes_subtitle"))
    hits = [p for p in known_players if p and p in text]
    yes_player = market.get("yes_sub_title") or market.get("yes_subtitle")
    if yes_player not in known_players:
        yes_player = next((p for p in hits if p == yes_player), hits[0] if hits else None)
    opp = next((p for p in hits if p != yes_player), None)
    if not yes_player or not opp:
        return None
    surface = surface_type(market.get("tourney_name", ""), market.get("surface", "Hard")) \
        if market.get("surface") else DEFAULT_SURFACE
    return yes_player, opp, surface


def _tennis_markets(client, series):
    if series:
        return client.markets(series_ticker=series).get("markets", [])
    return client.markets().get("markets", [])


def score(client, artifact=ARTIFACT, series=None):
    """Return list[Signal] for open tennis markets. run_cycle sets fees and applies rules.
    No-op until a trained artifact exists (singles stay off until the backtest gate passes)."""
    if not Path(artifact).exists():
        return []
    art = _load(artifact)
    known = set(art["state"].last) | set(art["state"].elo.overall)
    series = series or os.environ.get("KALSHI_TENNIS_SERIES")

    signals = []
    for m in _tennis_markets(client, series):
        parsed = parse_matchup(m, known)
        if not parsed:
            continue
        yes_player, opp, surface = parsed
        p = _predict(art, yes_player, opp, surface)
        ticker = m["ticker"]
        ya, na = m.get("yes_ask"), m.get("no_ask")
        if ya:
            signals.append(Signal(ticker, "yes", p, ya / 100.0))
        if na:
            signals.append(Signal(ticker, "no", 1.0 - p, na / 100.0))
    return signals
