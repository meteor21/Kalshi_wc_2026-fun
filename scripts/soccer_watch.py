"""Phase-0 soccer trigger watcher (LOG-ONLY). Polls API-Football live fixtures, detects
the big-favorite-concedes-first-<=35' trigger (favorite taken from pre-match Match Winner
odds), and logs the state + a live-quote snapshot to the trigger_events table. Betting
stays OFF: the system's first job is to build the prospective sample, not to bet it.

Run: API_FOOTBALL_KEY=... python -m scripts.soccer_watch --leagues 1,5,32 --every 45
"""
import argparse
import time

import db
from models.soccer import data, triggers
from models.soccer.pipeline import INTL_LEAGUES
from models.soccer.simulator import de_overround


def favorite_from_dec(dec):
    """(fav_side 'H'/'A', fav_market_prob) from [home,draw,away] decimal odds, or (None,None)."""
    if not dec:
        return None, None
    p = de_overround([1.0 / dec[0], 1.0 / dec[1], 1.0 / dec[2]])
    return ("H" if p[0] >= p[2] else "A"), float(max(p[0], p[2]))


def live_state(fx, events, fav_side, fav_p):
    """Build the trigger-detection state dict from a live fixture + its events."""
    home_id, away_id = fx["teams"]["home"]["id"], fx["teams"]["away"]["id"]
    fg_min, fg_side = data.first_goal(events, home_id, away_id)
    goals = fx.get("goals", {}) or {}
    return {
        "fixture_id": fx["fixture"]["id"],
        "minute": (fx["fixture"].get("status", {}) or {}).get("elapsed"),
        "h_score": goals.get("home"), "a_score": goals.get("away"),
        "fav_side": fav_side, "fav_market_prob": fav_p,
        "first_goal_side": fg_side, "first_goal_minute": fg_min,
    }


def process(api, fx, logged, conn):
    """Detect + log the trigger for one live fixture (once). Returns True if it fired."""
    fid = fx["fixture"]["id"]
    fav_side, fav_p = favorite_from_dec(data.match_winner_odds(api.odds(fid)))
    if fav_side is None:
        return False
    events = api._get("/fixtures/events", {"fixture": fid})   # live, uncached
    trig = triggers.detect(live_state(fx, events, fav_side, fav_p))
    if not trig or fid in logged:
        return False
    quotes = {"api_football_odds": api.odds(fid), "goals": fx.get("goals"),
              "elapsed": (fx["fixture"].get("status", {}) or {}).get("elapsed")}
    triggers.log(conn, trig, quotes)
    logged.add(fid)
    st = trig["state"]
    print(f"TRIGGER logged: fixture {fid} fav={fav_side} p={fav_p:.2f} "
          f"conceded first @{st['first_goal_minute']}' (score {st['h_score']}-{st['a_score']})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", default="", help="comma ids; default = INTL_LEAGUES")
    ap.add_argument("--every", type=int, default=45, help="poll interval seconds")
    ap.add_argument("--rounds", type=int, default=0, help="0 = until no live fixtures remain")
    a = ap.parse_args()

    api = data.APIFootball()
    if not api.key:
        raise SystemExit("set API_FOOTBALL_KEY (e.g. in .env)")
    leagues = {int(x) for x in a.leagues.split(",")} if a.leagues else set(INTL_LEAGUES)
    db.init()
    print(f"watching leagues={sorted(leagues)}  betting_enabled={triggers.betting_enabled()}")

    logged, r = set(), 0
    while True:
        live = [fx for fx in api.live() if (fx.get("league", {}) or {}).get("id") in leagues]
        print(f"[round {r+1}] live tracked fixtures: {len(live)}")
        with db.conn() as conn:
            for fx in live:
                try:
                    process(api, fx, logged, conn)
                except Exception as e:  # noqa: BLE001 - one bad fixture must not stop the watcher
                    print(f"  fixture {fx.get('fixture', {}).get('id')} error: {e!r}")
        r += 1
        if (a.rounds and r >= a.rounds) or not live:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
