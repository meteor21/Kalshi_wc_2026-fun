"""Backtest the tennis model vs market CLOSING odds (tennis-data.co.uk, free CSVs:
Pinnacle PSW/PSL, fallback B365). GO/NO-GO gate: the model must beat the closing
line and show fee-adjusted edge pockets, else tennis singles stay off. Offline only."""
import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

from strategy import rules
from strategy.rules import Signal
from .features import FEATURES, build_training_frame, clean, load_sackmann

TENNIS_DATA = "http://www.tennis-data.co.uk/{y}/{y}.zip"
MIN_BETS = 50            # don't trust a gate decision on a handful of bets


def _key(name):
    """'Roger Federer' / 'Federer R.' -> 'federer r' for cross-source joins."""
    name = str(name).replace(".", "").strip()
    parts = name.split()
    if not parts:
        return ""
    if "," in name or (len(parts) == 2 and len(parts[1]) == 1):   # 'Federer R'
        return f"{parts[0]} {parts[1][0]}".lower()
    return f"{parts[-1]} {parts[0][0]}".lower()                   # 'Roger Federer'


def load_odds(years):
    """tennis-data.co.uk yearly workbooks -> winner/loser keys + closing decimal odds."""
    frames = []
    for y in years:
        r = requests.get(TENNIS_DATA.format(y=y), timeout=60)
        if r.status_code != 200:
            continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        name = next(n for n in z.namelist() if n.lower().endswith((".xlsx", ".xls", ".csv")))
        with z.open(name) as fh:
            df = pd.read_csv(fh) if name.lower().endswith(".csv") else pd.read_excel(fh)
        df["w_key"], df["l_key"] = df["Winner"].map(_key), df["Loser"].map(_key)
        psw = df.get("PSW", df.get("B365W"))
        psl = df.get("PSL", df.get("B365L"))
        df["odds_w"], df["odds_l"] = psw, psl
        frames.append(df[["w_key", "l_key", "odds_w", "odds_l"]].dropna())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _implied(odds_w, odds_l):
    """De-vigged two-way closing probability for the winner."""
    pw, pl = 1.0 / odds_w, 1.0 / odds_l
    return pw / (pw + pl)


def fit_predict(frame):
    """Train <=2022, isotonic-calibrate 2023, predict 2024-25. Returns the test frame
    with p_model = P(actual winner wins)."""
    frame = frame.dropna(subset=FEATURES + ["target"])
    tr, cal = frame[frame.year <= 2022], frame[frame.year == 2023]
    te = frame[frame.year >= 2024].copy()
    clf = GradientBoostingClassifier(random_state=42).fit(tr[FEATURES], tr["target"])
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        clf.predict_proba(cal[FEATURES])[:, 1], cal["target"].to_numpy())
    p_a = iso.predict(clf.predict_proba(te[FEATURES])[:, 1])
    te["p_model"] = np.where(te["flip"] == 1, p_a, 1.0 - p_a)   # P(winner)
    return te


def simulate(joined, bankroll=rules.BANKROLL_START):
    """Walk matches chronologically; bet the side with fee-adjusted edge per the rule
    engine; settle on the known result. Returns (pnl, n_bets, edges)."""
    pnl, n_bets, edges = 0.0, 0, []
    for r in joined.sort_values("tourney_date").itertuples(index=False):
        for price, p_model, won in ((1.0 / r.odds_w, r.p_model, True),
                                    (1.0 / r.odds_l, 1.0 - r.p_model, False)):
            if not (0 < price < 1):
                continue
            sig = Signal(r.winner_name, "yes", p_model, price, rules.kalshi_fee(price))
            e = rules.edge(sig)
            edges.append(e)
            stake = min(rules.kelly_stake(sig, bankroll), rules.SINGLE_MAX_STAKE)
            ok, _ = rules.check_single(sig, stake, bankroll, 0.0, 0.0)
            if not ok or stake <= 0:
                continue
            contracts = int(stake / price)
            cost = contracts * (price + sig.fee_per_contract)
            ret = contracts * 1.0 if won else 0.0
            bankroll += ret - cost
            pnl += ret - cost
            n_bets += 1
    return pnl, n_bets, np.array(edges)


def main(years=(2024, 2025)):
    matches = clean(load_sackmann())
    frame, _ = build_training_frame(matches, keep_meta=True)
    te = fit_predict(frame)
    te["w_key"], te["l_key"] = te["winner_name"].map(_key), te["loser_name"].map(_key)

    odds = load_odds(years)
    joined = te.merge(odds, on=["w_key", "l_key"], how="inner")
    print(f"test matches={len(te)}  matched to odds={len(joined)}")
    if joined.empty:
        print("NO-GO: no odds overlap"); return

    y = np.ones(len(joined))   # p_model is P(winner); the winner always won -> label 1
    market_imp = _implied(joined["odds_w"].to_numpy(), joined["odds_l"].to_numpy())
    model_ll = log_loss(y, joined["p_model"].to_numpy(), labels=[0, 1])
    market_ll = log_loss(y, market_imp, labels=[0, 1])
    print(f"log_loss  model={model_ll:.4f}  market(closing)={market_ll:.4f}")

    pnl, n_bets, edges = simulate(joined)
    pos = (edges >= rules.MIN_EDGE).sum()
    print(f"bets={n_bets}  P&L=${pnl:.2f}  edge>=MIN pockets={pos}/{len(edges)}  "
          f"edge[min/med/max]={edges.min():.3f}/{np.median(edges):.3f}/{edges.max():.3f}")

    go = n_bets >= MIN_BETS and pnl > 0 and model_ll < market_ll
    print(f"VERDICT: {'GO — singles allowed' if go else 'NO-GO — tennis singles stay off'}")
    return go


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2024,2025")
    a = ap.parse_args()
    main(tuple(int(y) for y in a.years.split(",")))
