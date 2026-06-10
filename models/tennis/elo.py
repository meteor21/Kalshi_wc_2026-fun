"""Incremental ATP Elo: overall + per-surface. Ported as-is from research cells 5/11.
Reads pre-match ratings before applying the result (leakage-safe). Pure-python +
JSON state so the VPS scorer can update without pandas/sklearn."""
import json
from pathlib import Path

BASE_ELO = 1500.0
STATE_PATH = Path(__file__).parent / "elo_state.json"


def k_factor(n):
    """Match-count K: 250 / (n+5)^0.4 (cell 5)."""
    return 250.0 / ((n + 5) ** 0.4)


def expected(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


class EloModel:
    """Overall + surface Elo. update() returns the pre-match ratings it used,
    then advances state. Grand Slam (level 'G') gets a 1.1 K multiplier."""

    def __init__(self):
        self.overall = {}        # player -> rating
        self.n = {}              # player -> matches played
        self.surface = {}        # "player||surface" -> rating
        self.sn = {}             # "player||surface" -> matches on surface

    def rating(self, p):
        return self.overall.get(p, BASE_ELO)

    def surface_rating(self, p, s):
        return self.surface.get(f"{p}||{s}", BASE_ELO)

    def _step(self, ra, rb, ka, kb, klvl):
        ea = expected(ra, rb)
        return ra + klvl * ka * (1 - ea), rb + klvl * kb * (0 - (1 - ea))

    def update(self, winner, loser, surface, grand_slam=False):
        """Apply one finished match. Returns pre-match ratings (for feature rows)."""
        klvl = 1.1 if grand_slam else 1.0
        ra, rb = self.rating(winner), self.rating(loser)
        sra, srb = self.surface_rating(winner, surface), self.surface_rating(loser, surface)

        ra2, rb2 = self._step(ra, rb, k_factor(self.n.get(winner, 0)),
                              k_factor(self.n.get(loser, 0)), klvl)
        self.overall[winner], self.overall[loser] = ra2, rb2
        self.n[winner] = self.n.get(winner, 0) + 1
        self.n[loser] = self.n.get(loser, 0) + 1

        wk, lk = f"{winner}||{surface}", f"{loser}||{surface}"
        sra2, srb2 = self._step(sra, srb, k_factor(self.sn.get(wk, 0)),
                                k_factor(self.sn.get(lk, 0)), klvl)
        self.surface[wk], self.surface[lk] = sra2, srb2
        self.sn[wk] = self.sn.get(wk, 0) + 1
        self.sn[lk] = self.sn.get(lk, 0) + 1

        return {"w_elo": ra, "l_elo": rb, "w_surface_elo": sra, "l_surface_elo": srb}

    def save(self, path=STATE_PATH):
        Path(path).write_text(json.dumps(
            {"overall": self.overall, "n": self.n, "surface": self.surface, "sn": self.sn}))

    @classmethod
    def load(cls, path=STATE_PATH):
        m = cls()
        p = Path(path)
        if p.exists():
            d = json.loads(p.read_text())
            m.overall, m.n = d.get("overall", {}), d.get("n", {})
            m.surface, m.sn = d.get("surface", {}), d.get("sn", {})
        return m
