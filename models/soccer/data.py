"""API-Football acquisition + on-disk cache + event integrity (cell 2 / capstone Sec 3).
Static endpoints (fixtures, events, statistics) are cached to disk for offline fitting;
live endpoints are polled during tracked fixtures and never cached. Key via env
API_FOOTBALL_KEY. Stays on an affordable tier — no SportRadar."""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

BASE = "https://v3.football.api-sports.io"
CACHE = Path(os.environ.get("SOCCER_CACHE", "data/api_football"))


class _RateLimiter:
    """Spread calls to <= rpm requests/minute across threads (token-bucket-lite)."""

    def __init__(self, rpm):
        self.interval = 60.0 / max(rpm, 1)
        self.lock = threading.Lock()
        self.next_t = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.next_t:
                time.sleep(self.next_t - now)
            self.next_t = max(now, self.next_t) + self.interval


class APIFootball:
    def __init__(self, key=None, cache=CACHE):
        self.key = key or os.environ.get("API_FOOTBALL_KEY")
        self.cache = Path(cache)
        self.s = requests.Session()
        self.s.headers["x-apisports-key"] = self.key or ""

    def _one(self, path, params, limiter, tries):
        """Single GET with optional rate-limit + 429/5xx backoff. Returns parsed json."""
        for i in range(tries):
            if limiter:
                limiter.wait()
            r = self.s.get(BASE + path, params=params, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** i)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"giving up after {tries} tries: {path} {params}")

    def _paged(self, path, params, limiter=None, tries=1):
        """Fetch ALL pages (API-Football paginates) and concatenate their 'response' lists."""
        params = dict(params or {})
        out, page = [], 1
        while True:
            params["page"] = page
            j = self._one(path, params, limiter, tries)
            out += j.get("response", []) or []
            total = (j.get("paging") or {}).get("total") or 1
            if page >= total:
                return out
            page += 1

    def _get(self, path, params=None, cache_key=None, limiter=None, tries=1):
        """Paged GET. If cache_key given, serve/store the full 'response' list on disk."""
        if cache_key:
            fp = self.cache / f"{cache_key}.json"
            if fp.exists():
                return json.loads(fp.read_text())
        resp = self._paged(path, params, limiter, tries)
        if cache_key:
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(json.dumps(resp))
        return resp

    # --- static (cached) ---
    def fixtures(self, league, season):
        return self._get("/fixtures", {"league": league, "season": season},
                         cache_key=f"fixtures_{league}_{season}")

    def events(self, fixture_id):
        return self._get("/fixtures/events", {"fixture": fixture_id},
                         cache_key=f"events_{fixture_id}")

    def statistics(self, fixture_id):
        return self._get("/fixtures/statistics", {"fixture": fixture_id},
                         cache_key=f"stats_{fixture_id}")

    def odds(self, fixture_id):
        return self._get("/odds", {"fixture": fixture_id}, cache_key=f"odds_{fixture_id}")

    # --- account / quota ---
    def status(self):
        """Plan + rate-limit info. Probe this first to size the pull against quota."""
        return self._one("/status", {}, None, 1).get("response", {})

    # --- parallel bulk fetch (rate-limited, cached, 429/5xx backoff, paginated) ---
    def bulk_events(self, fixture_ids, workers=8, rpm=240):
        """Fetch events for many fixtures in parallel (cached). Yields (fixture_id, events)."""
        lim = _RateLimiter(rpm)
        ids = list(fixture_ids)

        def one(fid):
            return fid, self._get("/fixtures/events", {"fixture": fid},
                                  cache_key=f"events_{fid}", limiter=lim, tries=6)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            yield from ex.map(one, ids)

    # --- live (never cached) ---
    def live(self, league=None):
        params = {"live": "all"}
        if league:
            params["league"] = league
        return self._get("/fixtures", params)

    def poll(self, fixture_ids, every=45, rounds=None):
        """Yield (fixture, events) for tracked fixtures every `every` seconds.
        Bounded by `rounds` (None = until all tracked fixtures finish)."""
        tracked, r = set(fixture_ids), 0
        while tracked and (rounds is None or r < rounds):
            for fx in self.live():
                fid = fx.get("fixture", {}).get("id")
                if fid not in tracked:
                    continue
                yield fx, self._get("/fixtures/events", {"fixture": fid})  # live: uncached
                if fx.get("fixture", {}).get("status", {}).get("short") in ("FT", "AET", "PEN"):
                    tracked.discard(fid)
            r += 1
            if tracked:
                time.sleep(every)


# --- integrity layer (cell 4/6) -------------------------------------------------

def clean_events(events, home_id, away_id, final_h, final_a):
    """Drop penalty-shootout goals and VAR-overturned goals; keep only home/away events.
    Then check the surviving goal count against the recorded scoreline. Returns
    (clean_events, ok) where ok=False means the fixture fails the integrity check."""
    out = []
    for e in events:
        etype = e.get("type")
        detail = str(e.get("detail", ""))
        elapsed = (e.get("time", {}) or {}).get("elapsed", 0) or 0
        tid = (e.get("team", {}) or {}).get("id")
        if tid not in (home_id, away_id):
            continue
        if etype == "Goal" and (elapsed > 120 or "Penalty Shootout" in detail):
            continue                                   # shootout goal
        if etype == "Var" and ("cancelled" in detail.lower() or "disallow" in detail.lower()):
            continue                                   # VAR-overturned
        out.append(e)
    gh = sum(1 for e in out if e.get("type") == "Goal" and (e.get("team", {}) or {}).get("id") == home_id)
    ga = sum(1 for e in out if e.get("type") == "Goal" and (e.get("team", {}) or {}).get("id") == away_id)
    return out, (gh == final_h and ga == final_a)


def scoring_team(event, home_id, away_id):
    """Team credited with a goal on the scoreboard, flipping own goals to the opponent
    (API-Football tags an own goal with the CONCEDING team)."""
    tid = (event.get("team", {}) or {}).get("id")
    if "own goal" in str(event.get("detail", "")).lower():
        return away_id if tid == home_id else home_id
    return tid


def first_goal(events, home_id, away_id):
    """(minute, side) of the first goal, or (None, None). Side is 'H'/'A', own-goal aware."""
    goals = sorted((e for e in events if e.get("type") == "Goal"),
                   key=lambda e: (e.get("time", {}) or {}).get("elapsed", 999) or 999)
    for g in goals:
        if (g.get("team", {}) or {}).get("id") in (home_id, away_id):
            return (g["time"]["elapsed"], "H" if scoring_team(g, home_id, away_id) == home_id else "A")
    return (None, None)


def match_winner_odds(odds_response):
    """[home, draw, away] decimal odds from the 'Match Winner' market, averaged across
    bookmakers, or None. Feeds the pre-match favorite for the trigger watcher."""
    hs, ds, aws = [], [], []
    for row in odds_response or []:
        for bk in row.get("bookmakers", []) or []:
            for bet in bk.get("bets", []) or []:
                if bet.get("name") not in ("Match Winner", "1X2"):
                    continue
                m = {v.get("value"): v.get("odd") for v in bet.get("values", []) or []}
                try:
                    hs.append(float(m["Home"])); ds.append(float(m["Draw"])); aws.append(float(m["Away"]))
                except (KeyError, TypeError, ValueError):
                    continue
    if not hs:
        return None
    return [sum(hs) / len(hs), sum(ds) / len(ds), sum(aws) / len(aws)]
