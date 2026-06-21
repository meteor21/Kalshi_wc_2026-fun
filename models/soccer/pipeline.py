"""International soccer pipeline: assemble API-Football pulls into the minute-level
modeling frame, chronologically block-split (train/calib/valid/test — no shuffling, so
no temporal leakage), fit Dixon-Coles + the hazard GLM on the train block, and report
held-out metrics + the empirical trigger rates. Offline (Colab/local/this box)."""
import numpy as np
import pandas as pd

from . import hazard
from .dixon_coles import DixonColes

# Senior men's national-team competitions for a World Cup model (API-Football league ids).
# Verify against /leagues at pull time; ids occasionally shift. Club comps deliberately excluded.
INTL_LEAGUES = {
    1: "World Cup", 4: "Euro Championship", 5: "UEFA Nations League",
    6: "Africa Cup of Nations", 7: "Asian Cup", 9: "Copa America",
    10: "Friendlies", 22: "CONCACAF Gold Cup",
    29: "WC Qualification - Africa", 30: "WC Qualification - Asia",
    31: "WC Qualification - CONCACAF", 32: "WC Qualification - Europe",
    33: "WC Qualification - Oceania", 34: "WC Qualification - South America",
}
BLOCKS = ("train", "calib", "valid", "test")


# --- assembly from API-Football JSON ------------------------------------------

def assemble_fixtures(raw):
    """raw: iterable of API-Football /fixtures rows. Keep finished matches with a score."""
    rows = []
    for f in raw:
        fx, lg, tm, gl = f["fixture"], f["league"], f["teams"], f["goals"]
        if (fx.get("status", {}) or {}).get("short") not in ("FT", "AET", "PEN"):
            continue
        if gl.get("home") is None or gl.get("away") is None:
            continue
        rows.append(dict(
            fixture_id=fx["id"], date=fx["date"], competition=lg["id"], season=lg["season"],
            home_team_id=tm["home"]["id"], away_team_id=tm["away"]["id"],
            home_team=tm["home"]["name"], away_team=tm["away"]["name"],
            home_score=int(gl["home"]), away_score=int(gl["away"]),
        ))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


def assemble_events(events_by_fid, fixtures):
    """Flatten /fixtures/events into (fixture_id, minute, type, detail, team_id), keeping
    only goals/cards for fixtures that pass the score-vs-event integrity check."""
    fl = fixtures.set_index("fixture_id")[["home_team_id", "away_team_id",
                                           "home_score", "away_score"]].to_dict("index")
    rows, bad = [], 0
    for fid, evs in events_by_fid.items():
        fx = fl.get(fid)
        if not fx:
            continue
        keep, gh, ga = [], 0, 0
        for e in evs:
            etype, detail = e.get("type"), str(e.get("detail", ""))
            elapsed = (e.get("time", {}) or {}).get("elapsed", 0) or 0
            tid = (e.get("team", {}) or {}).get("id")
            if tid not in (fx["home_team_id"], fx["away_team_id"]):
                continue
            if etype == "Goal":
                if elapsed > 120 or "Penalty Shootout" in detail:      # shootout
                    continue
                gh += tid == fx["home_team_id"]; ga += tid == fx["away_team_id"]
            if etype == "Var" and ("cancel" in detail.lower() or "disallow" in detail.lower()):
                continue                                               # VAR-overturned
            if etype in ("Goal", "Card"):
                keep.append((fid, min(int(elapsed), 90), etype, detail, tid))
        if gh == fx["home_score"] and ga == fx["away_score"]:          # integrity gate
            rows.extend(keep)
        else:
            bad += 1
    if bad:
        print(f"  integrity: dropped {bad} fixtures (event goals != scoreline)")
    return pd.DataFrame(rows, columns=["fixture_id", "minute", "type", "detail", "team_id"])


# --- chronological block split -------------------------------------------------

def block_split(fixtures, fracs=(0.60, 0.15, 0.15, 0.10)):
    """Tag each fixture with a contiguous date-ordered block. No shuffling."""
    df = fixtures.sort_values("date").reset_index(drop=True)
    n = len(df)
    bounds = np.cumsum([int(round(f * n)) for f in fracs])
    block = np.empty(n, dtype=object)
    start = 0
    for name, end in zip(BLOCKS, bounds):
        block[start:end] = name
        start = end
    block[start:] = BLOCKS[-1]
    df["block"] = block
    return df


# --- fitting + evaluation ------------------------------------------------------

def fit_dc(fixtures, train_mask):
    """Pooled Dixon-Coles over all national teams from the train block; returns the
    fitted model + per-fixture (lambda_home, lambda_away) for EVERY fixture."""
    tr = fixtures[train_mask]
    dc = DixonColes().fit(tr.home_team_id.tolist(), tr.away_team_id.tolist(),
                          tr.home_score.tolist(), tr.away_score.tolist())
    lams = {r.fixture_id: dc.lambdas(r.home_team_id, r.away_team_id)
            for r in fixtures.itertuples()}
    return dc, lams


def _poisson_ll(model, X, y):
    lam = np.clip(model.predict(X), 1e-6, 1.0)
    return float((y * np.log(lam) - lam).mean())


def wilson(x, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def trigger_table(fixtures, events, dc, lams, holdout_mask):
    """Empirical big-favorite-concedes-first-<=35' rates (capstone Sec 9). Favorite is
    DC-implied here (historical); production uses the market. Reports full + held-out."""
    by_fid = events.groupby("fixture_id")
    recs = []
    for r in fixtures.itertuples():
        lh, la = lams[r.fixture_id]
        p = dc.outcome_probs(lh, la)
        fav = "H" if p["home"] >= p["away"] else "A"
        fav_p = max(p["home"], p["away"])
        fg_min, fg_side = 999, None
        if r.fixture_id in by_fid.groups:
            g = by_fid.get_group(r.fixture_id)
            g = g[g.type == "Goal"].sort_values("minute")
            if len(g):
                fg_min = int(g.iloc[0].minute)
                fg_side = "H" if g.iloc[0].team_id == r.home_team_id else "A"
        total = r.home_score + r.away_score
        recs.append(dict(
            fixture_id=r.fixture_id, block=r.block, big_fav=fav_p > 0.55,
            fav_conceded_early=(fg_side is not None and fg_side != fav and fg_min <= 35),
            btts=(r.home_score >= 1 and r.away_score >= 1), over25=total >= 3,
            fav_avoid_loss=(fav == ("H" if r.home_score > r.away_score else
                                    "A" if r.away_score > r.home_score else fav))
                           or r.home_score == r.away_score,
        ))
    S = pd.DataFrame(recs)
    trig = S.big_fav & S.fav_conceded_early
    out = []
    for label, mask in [("full sample", pd.Series(True, index=S.index)), ("held-out", holdout_mask.values)]:
        sub = S[trig & mask]
        n = len(sub)
        row = {"sample": label, "n": n}
        for bet in ("btts", "over25", "fav_avoid_loss"):
            hits = int(sub[bet].sum())
            rate = hits / n if n else 0.0
            lo, hi = wilson(hits, n)
            row[bet] = f"{rate*100:.1f}% [{lo*100:.0f}-{hi*100:.0f}]  be={rate*100:.0f}c"
        out.append(row)
    return pd.DataFrame(out)


def run(fixtures, events):
    """Full analysis: block-split, fit DC + hazard on train, report held-out Poisson
    log-likelihood and the trigger table. Returns a dict of artifacts + a printed report."""
    fixtures = block_split(fixtures)
    fmap = dict(zip(fixtures.fixture_id, fixtures.block))
    train_mask = fixtures.block == "train"
    print(f"fixtures={len(fixtures)}  " + "  ".join(
        f"{b}={int((fixtures.block==b).sum())}" for b in BLOCKS))

    dc, lams = fit_dc(fixtures, train_mask)
    print(f"Dixon-Coles: teams={len(dc.teams)}  home_adv={dc.home_adv:.3f}  rho={dc.rho:.3f}")

    state = hazard.build_state_stream(events, fixtures, lams)
    state["block"] = state.fixture_id.map(fmap)
    tr = state[state.block == "train"]
    haz = hazard.fit(tr)

    X = {b: state[state.block == b][hazard.HAZARD_FEATS].to_numpy() for b in BLOCKS}
    print("\nhazard held-out Poisson logL/row (higher=better; baseline = mean-rate):")
    for b in BLOCKS:
        sb = state[state.block == b]
        if not len(sb):
            continue
        for side, m in (("H", haz["hazard_h"]), ("A", haz["hazard_a"])):
            y = sb[f"{'h' if side=='H' else 'a'}_goal_next"].to_numpy()
            ll = _poisson_ll(m, X[b], y)
            base = float((y * np.log(max(y.mean(), 1e-6)) - y.mean()).mean())
            print(f"  {b:5} {side}  logL={ll:+.5f}  base={base:+.5f}  lift={ll-base:+.5f}")

    print("\nhazard coefficients (home / away):")
    _, bh, _, ba, F = hazard.coef_arrays(haz)
    for f in hazard.HAZARD_FEATS:
        print(f"  {f:12} {bh[F[f]]:+.3f} / {ba[F[f]]:+.3f}")

    holdout = fixtures.block.isin(("valid", "test"))
    print("\nTrigger — big favorite concedes first <=35' (DC-implied favorite):")
    tt = trigger_table(fixtures, events, dc, lams, holdout)
    print(tt.to_string(index=False))
    return {"fixtures": fixtures, "dc": dc, "lams": lams, "hazard": haz,
            "state": state, "triggers": tt}
