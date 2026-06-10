"""Pre-match tennis features. Every value uses only data available BEFORE the match.
Ported from research cells 0,3,8-14,26-31,37 (PORTING.md), with the mandatory
in-match leakage fix on bp_ratio. pandas lives here; execution/ never imports this."""
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import requests
from io import StringIO

from .elo import EloModel

SACKMANN = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{}.csv"

# cell 8: tournaments treated as indoor (lowercased substrings)
INDOOR_TOURNEYS = [
    "rotterdam", "marseille", "metz", "antwerp", "basel", "vienna", "paris masters",
    "sofia", "stockholm", "cologne", "dallas", "st. petersburg", "atp finals", "nitto",
    "tour finals", "masters cup", "next gen finals", "nextgen finals", "milan", "astana",
]
SURFACES = ["Clay", "Grass", "Hard_indoor", "Hard_outdoor"]

# PORTING.md v1 feature list (NOT cell 33). bp_ratio_diff is the FIXED trailing version.
FEATURES = [
    "elo_diff", "surface_elo_diff", "rank_diff", "age_diff", "ht_diff",
    "h2h_total", "dominant_h2h", "lefty_win_pct", "player_A_win_pct", "bp_ratio_diff",
] + [f"surface_{s}" for s in SURFACES]

BP_WINDOW = 15          # trailing matches for bp_ratio (leakage fix)
DOMINANT_MIN = 8        # cell 27: dominant H2H needs >= 8 meetings
DOMINANT_PCT = 0.75


def surface_type(tourney_name, surface):
    """Hard split into indoor/outdoor by tourney name (cells 9-10); Clay/Grass kept."""
    if surface == "Hard":
        indoor = any(t in str(tourney_name).lower() for t in INDOOR_TOURNEYS)
        return "Hard_indoor" if indoor else "Hard_outdoor"
    return surface


def load_sackmann(years=range(2000, 2026)):
    """Cell 0: pull Jeff Sackmann ATP CSVs. NOT for the VPS (training/backtest only)."""
    frames = []
    for y in years:
        r = requests.get(SACKMANN.format(y), timeout=30)
        if r.status_code == 200:
            df = pd.read_csv(StringIO(r.text))
            df["year"] = y
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(matches):
    """Cell 3: drop seed/entry cols and rows missing core serve stats."""
    df = matches.drop(columns=["winner_entry", "loser_entry", "winner_seed", "loser_seed"],
                      errors="ignore")
    serve = ["l_ace", "l_svpt", "l_1stIn", "l_1stWon", "l_df", "w_1stIn", "w_ace", "w_df",
             "w_svpt", "w_2ndWon", "w_1stWon", "w_SvGms", "l_2ndWon", "l_SvGms", "l_bpSaved"]
    df = df.dropna(subset=[c for c in serve if c in df.columns])
    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d")
    return df.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)


def _bp_ratio(saved, faced):
    return float(saved) / (float(faced) + 1.0)   # +1 avoids divide-by-zero (cell 14)


# module-level factories so FeatureState stays picklable (joblib persists it for scoring)
def _new_bp():
    return deque(maxlen=BP_WINDOW)


def _zero_pair():
    return [0, 0]


class FeatureState:
    """All leakage-safe running state. Drives batch training and live scoring from
    the same code path: read features from current state, then observe() the result."""

    def __init__(self):
        self.elo = EloModel()
        self.bp = defaultdict(_new_bp)        # player -> recent bp ratios
        self.h2h = defaultdict(_zero_pair)    # (lo,hi) -> [wins_lo, wins_hi]
        self.lefty = defaultdict(_zero_pair)  # player -> [wins, total] vs lefties
        self.last = {}                        # player -> rank/age/ht/hand

    # --- reads (pre-match) ---
    def trailing_bp(self, p):
        h = self.bp[p]
        return sum(h) / len(h) if h else 0.5

    def h2h_feats(self, a, b):
        lo, hi = sorted([a, b])
        wlo, whi = self.h2h[(lo, hi)]
        total = wlo + whi
        wins_a = wlo if a == lo else whi
        pct = wins_a / total if total else 0.5
        dominant = int(total >= DOMINANT_MIN and pct >= DOMINANT_PCT)
        return pct, total, dominant

    def lefty_pct(self, p):
        w, t = self.lefty[p]
        return w / t if t else 0.5   # cell 37 fillna(0.5)

    def prematch(self, a, b, surface):
        """Feature row for an upcoming match (no state change). rank/age/ht from
        last-seen values; live callers may override via set_player()."""
        pa, pb = self.last.get(a, {}), self.last.get(b, {})
        pct, total, dom = self.h2h_feats(a, b)
        row = {
            "elo_diff": self.elo.rating(a) - self.elo.rating(b),
            "surface_elo_diff": self.elo.surface_rating(a, surface) - self.elo.surface_rating(b, surface),
            "rank_diff": _diff(pa.get("rank"), pb.get("rank")),
            "age_diff": _diff(pa.get("age"), pb.get("age")),
            "ht_diff": _diff(pa.get("ht"), pb.get("ht")),
            "h2h_total": total,
            "dominant_h2h": dom,
            "lefty_win_pct": self.lefty_pct(a),
            "player_A_win_pct": pct,
            "bp_ratio_diff": self.trailing_bp(a) - self.trailing_bp(b),
        }
        for s in SURFACES:
            row[f"surface_{s}"] = int(s == surface)
        return row

    # --- write (after the match is known) ---
    def observe(self, winner, loser, surface, grand_slam, w_bp, l_bp, w_hand, l_hand,
                w_rank=None, l_rank=None, w_age=None, l_age=None, w_ht=None, l_ht=None):
        self.elo.update(winner, loser, surface, grand_slam)
        self.bp[winner].append(w_bp)
        self.bp[loser].append(l_bp)
        lo, hi = sorted([winner, loser])
        self.h2h[(lo, hi)][0 if winner == lo else 1] += 1
        # symmetric lefty record: each player's results vs left-handed opponents
        if l_hand == "L":
            self.lefty[winner][0] += 1
            self.lefty[winner][1] += 1
        if w_hand == "L":
            self.lefty[loser][1] += 1
        for p, rank, age, ht, hand in ((winner, w_rank, w_age, w_ht, w_hand),
                                       (loser, l_rank, l_age, l_ht, l_hand)):
            self.last[p] = {"rank": rank, "age": age, "ht": ht, "hand": hand}


def _diff(x, y):
    if x is None or y is None or pd.isna(x) or pd.isna(y):
        return 0.0
    return float(x) - float(y)


def build_training_frame(matches, seed=42, state=None, keep_meta=False):
    """Walk matches in time order; for each, read pre-match features then observe().
    Random A/B flip (cell 14/26) removes winner-position leakage. Returns (frame, state)
    so the final state can be persisted for live scoring. keep_meta adds winner/loser
    names, date and flip so the backtest can join market odds and recover P(winner).
    Expects clean()'d input (datetime tourney_date); re-sorts defensively."""
    df = matches.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    st = state or FeatureState()
    rng = np.random.default_rng(seed)
    rows = []
    for m in df.itertuples(index=False):
        w, l = m.winner_name, m.loser_name
        s = surface_type(m.tourney_name, m.surface)
        gs = (getattr(m, "tourney_level", "") == "G")
        flip = bool(rng.random() < 0.5)
        a, b = (w, l) if flip else (l, w)

        pct, total, dom = st.h2h_feats(a, b)
        a_bp, b_bp = st.trailing_bp(a), st.trailing_bp(b)
        ra, rb = st.elo.rating(a), st.elo.rating(b)
        sra, srb = st.elo.surface_rating(a, s), st.elo.surface_rating(b, s)
        a_rank, b_rank = (m.winner_rank, m.loser_rank) if flip else (m.loser_rank, m.winner_rank)
        a_age, b_age = (m.winner_age, m.loser_age) if flip else (m.loser_age, m.winner_age)
        a_ht, b_ht = (m.winner_ht, m.loser_ht) if flip else (m.loser_ht, m.winner_ht)

        row = {
            "elo_diff": ra - rb, "surface_elo_diff": sra - srb,
            "rank_diff": _diff(a_rank, b_rank), "age_diff": _diff(a_age, b_age),
            "ht_diff": _diff(a_ht, b_ht), "h2h_total": total, "dominant_h2h": dom,
            "lefty_win_pct": st.lefty_pct(a), "player_A_win_pct": pct,
            "bp_ratio_diff": a_bp - b_bp, "target": int(flip), "year": int(m.year),
        }
        for surf in SURFACES:
            row[f"surface_{surf}"] = int(surf == s)
        if keep_meta:
            row.update(winner_name=w, loser_name=l, tourney_date=m.tourney_date, flip=int(flip))
        rows.append(row)

        st.observe(w, l, s, gs,
                   _bp_ratio(m.w_bpSaved, m.w_bpFaced), _bp_ratio(m.l_bpSaved, m.l_bpFaced),
                   m.winner_hand, m.loser_hand, m.winner_rank, m.loser_rank,
                   m.winner_age, m.loser_age, m.winner_ht, m.loser_ht)
    return pd.DataFrame(rows), st
