"""Run the international soccer analysis on the pulled parquet frames: block split,
Dixon-Coles + hazard fit, held-out metrics, trigger rates. Saves frozen coefficients.
Run: python -m scripts.soccer_analyze"""
from pathlib import Path

import pandas as pd

from models.soccer import dixon_coles, hazard, pipeline

DATA = Path("data")


def main():
    fixtures = pd.read_parquet(DATA / "soccer_intl_fixtures.parquet")
    events = pd.read_parquet(DATA / "soccer_intl_events.parquet")
    print(f"loaded {len(fixtures)} fixtures, {len(events)} events\n")

    art = pipeline.run(fixtures, events)

    hazard.save(art["hazard"])
    dixon_coles.save_all({"intl_pooled": art["dc"]})
    print(f"\nsaved frozen coefficients -> {hazard.MODELS_PATH.name}, "
          f"{dixon_coles.PARAMS_PATH.name}")


if __name__ == "__main__":
    main()
