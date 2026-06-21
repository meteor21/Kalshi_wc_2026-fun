"""Pull international fixtures + events from API-Football into cached parquet frames.
Probes quota first, fetches events in parallel (rate-limited, resumable via on-disk
cache). Run: API_FOOTBALL_KEY=... python -m scripts.soccer_pull --seasons 2018-2025

Re-running is cheap: anything already cached on disk is not re-requested."""
import argparse
from pathlib import Path

import pandas as pd

from models.soccer import data, pipeline

OUT = Path("data")


def seasons_arg(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=seasons_arg, default=seasons_arg("2018-2025"))
    ap.add_argument("--leagues", default="", help="comma ids; default = INTL_LEAGUES")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rpm", type=int, default=240, help="requests/min cap (free tier: 10)")
    ap.add_argument("--max-fixtures", type=int, default=0, help="0 = all (bound a run vs quota)")
    a = ap.parse_args()

    api = data.APIFootball()
    if not api.key:
        raise SystemExit("set API_FOOTBALL_KEY (e.g. in .env)")
    st = api.status()
    sub, req = st.get("subscription", {}), st.get("requests", {})
    print(f"plan={sub.get('plan')} active={sub.get('active')}  "
          f"requests today={req.get('current')}/{req.get('limit_day')}")

    leagues = ([int(x) for x in a.leagues.split(",")] if a.leagues else list(pipeline.INTL_LEAGUES))
    print(f"leagues={len(leagues)}  seasons={a.seasons[0]}-{a.seasons[-1]}")

    raw = []
    for lg in leagues:
        for yr in a.seasons:
            try:
                raw += api.fixtures(lg, yr)
            except Exception as e:
                print(f"  fixtures {lg}/{yr}: {e}")
    fixtures = pipeline.assemble_fixtures(raw)
    print(f"finished fixtures: {len(fixtures)}")

    fids = fixtures.fixture_id.tolist()
    uncached = [f for f in fids if not (api.cache / f"events_{f}.json").exists()]
    print(f"events to fetch: {len(uncached)} (cached: {len(fids)-len(uncached)})")
    if a.max_fixtures:
        fids = fids[: a.max_fixtures]

    ev_by_fid = {}
    for i, (fid, evs) in enumerate(api.bulk_events(fids, workers=a.workers, rpm=a.rpm), 1):
        ev_by_fid[fid] = evs
        if i % 200 == 0:
            print(f"  events {i}/{len(fids)}")
    events = pipeline.assemble_events(ev_by_fid, fixtures)

    OUT.mkdir(exist_ok=True)
    fixtures.to_parquet(OUT / "soccer_intl_fixtures.parquet", index=False)
    events.to_parquet(OUT / "soccer_intl_events.parquet", index=False)
    print(f"saved -> {OUT}/soccer_intl_fixtures.parquet ({len(fixtures)} fixtures), "
          f"soccer_intl_events.parquet ({len(events)} events)")


if __name__ == "__main__":
    main()
