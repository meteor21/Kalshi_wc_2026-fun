"""State-dependent Poisson scoring-hazard model (cells 12/14, capstone eq 1):
  log lambda_k(t|s) = b0 + time_buckets(t) + state(s) + b_h*log_dc_h + b_a*log_dc_a
Two GLMs (home-goal, away-goal) fit on minute-level rows. Fit offline; the simulator
loads frozen coefficients. 45-60' is the reference time bucket."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

MODELS_PATH = Path(__file__).parent / "hazard_models.joblib"

TIME_BUCKETS = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90)]
# final feature order (min_45_60 is the dropped reference bucket); index order matters
HAZARD_FEATS = [
    "min_0_15", "min_15_30", "min_30_45", "min_60_75", "min_75_90",
    "h_leading", "a_leading", "big_lead", "h_man_up", "a_man_up",
    "log_dc_lh", "log_dc_la",
]


def build_state_stream(events, fixtures, dc_lambdas):
    """Minute-level (t, score, reds) rows + next-minute goal labels for every fixture.
    events: fixture_id, minute, type ('Goal'|'Card'), detail, team_id.
    fixtures: fixture_id, home_team_id, away_team_id, competition.
    dc_lambdas: dict fixture_id -> (lambda_home, lambda_away)."""
    fl = fixtures.set_index("fixture_id")[["home_team_id", "away_team_id"]].to_dict("index")
    by_fid = events.groupby("fixture_id")
    rows = []
    for fid, fx in fl.items():
        if fid not in by_fid.groups or fid not in dc_lambdas:
            continue
        ev = by_fid.get_group(fid).sort_values("minute")
        ev = ev.assign(side=np.where(ev.team_id == fx["home_team_id"], "H",
                                     np.where(ev.team_id == fx["away_team_id"], "A", None)))
        # goals scored in minute m -> label on state at minute m-1 (predicts (t,t+1])
        goals = ev[ev.type == "Goal"]
        gmin = goals.groupby(["minute", "side"]).size()
        lh, la = dc_lambdas[fid]
        hs = as_ = hr = ar = 0
        evb = ev.groupby("minute")
        for t in range(0, 90):
            ghn = int(gmin.get((t + 1, "H"), 0))
            gan = int(gmin.get((t + 1, "A"), 0))
            rows.append((fid, t, hs - as_, hr - ar, lh, la, ghn, gan))
            if t in evb.groups:
                for _, e in evb.get_group(t).iterrows():
                    if e.side not in ("H", "A"):
                        continue
                    if e.type == "Goal":
                        hs += e.side == "H"; as_ += e.side == "A"
                    elif e.type == "Card" and "Red" in str(e.detail):
                        hr += e.side == "H"; ar += e.side == "A"
    cols = ["fixture_id", "minute", "score_diff", "red_diff", "dc_lh", "dc_la",
            "h_goal_next", "a_goal_next"]
    return add_features(pd.DataFrame(rows, columns=cols))


def add_features(state):
    """Attach the HAZARD_FEATS columns to a minute-level state frame."""
    s = state.copy()
    for lo, hi in TIME_BUCKETS:
        if (lo, hi) == (45, 60):
            continue
        s[f"min_{lo}_{hi}"] = ((s.minute >= lo) & (s.minute < hi)).astype(int)
    s["h_leading"] = (s.score_diff > 0).astype(int)
    s["a_leading"] = (s.score_diff < 0).astype(int)
    s["big_lead"] = (s.score_diff.abs() >= 2).astype(int)
    s["h_man_up"] = (s.red_diff < 0).astype(int)   # away has more reds -> home up a man
    s["a_man_up"] = (s.red_diff > 0).astype(int)
    s["log_dc_lh"] = np.log(s.dc_lh.clip(lower=0.05))
    s["log_dc_la"] = np.log(s.dc_la.clip(lower=0.05))
    return s


def fit(state, alpha=1e-4, max_iter=500):
    """Fit home/away goal hazards. Returns a frozen-coefficients dict."""
    X = state[HAZARD_FEATS].to_numpy()
    hh = PoissonRegressor(alpha=alpha, max_iter=max_iter).fit(X, state.h_goal_next.to_numpy())
    ha = PoissonRegressor(alpha=alpha, max_iter=max_iter).fit(X, state.a_goal_next.to_numpy())
    return {"hazard_h": hh, "hazard_a": ha, "features": HAZARD_FEATS}


def coef_arrays(haz):
    """(b0_h, beta_h, b0_a, beta_a, F_IDX) — the simulator's hot-path inputs."""
    hh, ha = haz["hazard_h"], haz["hazard_a"]
    f_idx = {f: i for i, f in enumerate(haz["features"])}
    return hh.intercept_, hh.coef_, ha.intercept_, ha.coef_, f_idx


def save(haz, path=MODELS_PATH):
    joblib.dump(haz, path)


def load(path=MODELS_PATH):
    return joblib.load(path)
