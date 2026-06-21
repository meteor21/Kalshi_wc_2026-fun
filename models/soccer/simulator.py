"""Forward Monte Carlo state-space simulator (cell 19, capstone Sec 5).
From any in-match state, draw N continuations to full time under the fitted
state-dependent Poisson hazards and read outcome probabilities off the path shares.
Fully vectorized so a revaluation runs <1s on the VPS.

PRODUCTION CHANGE (PORTING.md): per-fixture calibration anchors to the de-overrounded
MARKET 1X2 vector, not the dropped GBM. The simulator therefore carries zero pre-match
opinion of its own; all claimed edge is the in-match state response."""
import numpy as np

from .hazard import coef_arrays

CAL_GRID = np.round(np.arange(0.60, 1.551, 0.05), 3)   # alpha grid [0.60, 1.55]


def de_overround(p):
    """Proportional normalization: raw implied probabilities -> sum to 1 (capstone Sec 7)."""
    p = np.asarray(p, float)
    return p / p.sum()


def implied_from_decimal(odds):
    """De-overrounded 1X2 from decimal odds [home, draw, away]."""
    return de_overround(1.0 / np.asarray(odds, float))


def simulate(state, haz, dc_lh, dc_la, alpha_h=1.0, alpha_a=1.0, n_sims=4000, seed=42):
    """state: (start_minute, h_score, a_score, h_red, a_red). Returns final score arrays."""
    start, hs0, as0, hr0, ar0 = state
    b0_h, beta_h, b0_a, beta_a, F = coef_arrays(haz)
    rng = np.random.default_rng(seed)
    log_lh, log_la = np.log(max(dc_lh, 0.05)), np.log(max(dc_la, 0.05))

    hs = np.full(n_sims, hs0, np.int32)
    as_ = np.full(n_sims, as0, np.int32)
    # reds are fixed at the observed state (sim does not generate new cards, per notebook)
    h_man_up, a_man_up = float(ar0 > hr0), float(hr0 > ar0)

    def time_vec(minute):
        v = np.zeros(len(beta_h), np.float64)
        if minute < 15:   v[F["min_0_15"]] = 1
        elif minute < 30: v[F["min_15_30"]] = 1
        elif minute < 45: v[F["min_30_45"]] = 1
        elif minute < 60: pass                       # 45-60 reference bucket
        elif minute < 75: v[F["min_60_75"]] = 1
        else:             v[F["min_75_90"]] = 1
        v[F["h_man_up"]], v[F["a_man_up"]] = h_man_up, a_man_up
        v[F["log_dc_lh"]], v[F["log_dc_la"]] = log_lh, log_la
        return v

    for minute in range(start, 90):
        tv = time_vec(minute)
        base_h, base_a = b0_h + beta_h @ tv, b0_a + beta_a @ tv
        sd = hs - as_
        h_lead, a_lead = (sd > 0).astype(float), (sd < 0).astype(float)
        big = (np.abs(sd) >= 2).astype(float)
        dh = beta_h[F["h_leading"]] * h_lead + beta_h[F["a_leading"]] * a_lead + beta_h[F["big_lead"]] * big
        da = beta_a[F["h_leading"]] * h_lead + beta_a[F["a_leading"]] * a_lead + beta_a[F["big_lead"]] * big
        hs += rng.poisson(np.exp(base_h + dh) * alpha_h).astype(np.int32)
        as_ += rng.poisson(np.exp(base_a + da) * alpha_a).astype(np.int32)
    return hs, as_


def markets(hs, as_):
    """Outcome probabilities from final-score path arrays."""
    total = hs + as_
    return {
        "home": float((hs > as_).mean()), "draw": float((hs == as_).mean()),
        "away": float((hs < as_).mean()),
        "over15": float((total >= 2).mean()), "over25": float((total >= 3).mean()),
        "over35": float((total >= 4).mean()),
        "btts": float(((hs >= 1) & (as_ >= 1)).mean()),
    }


def _kl(p, q):
    p, q = np.asarray(p, float), np.clip(np.asarray(q, float), 1e-6, None)
    return float((p * np.log(p / q)).sum())


def calibrate_to_market(market_1x2, haz, dc_lh, dc_la, n_sims=2000, grid=CAL_GRID, seed=42):
    """Grid-search multiplicative (alpha_h, alpha_a) so the simulator's PRE-match 1X2
    matches the de-overrounded market vector (KL). Returns (alpha_h, alpha_a, kl)."""
    target = de_overround(market_1x2)
    best = (1.0, 1.0, np.inf)
    for ah in grid:
        for aa in grid:
            hs, as_ = simulate((0, 0, 0, 0, 0), haz, dc_lh, dc_la, ah, aa, n_sims, seed)
            m = markets(hs, as_)
            kl = _kl(target, [m["home"], m["draw"], m["away"]])
            if kl < best[2]:
                best = (float(ah), float(aa), kl)
    return best


def revalue(state, haz, dc_lh, dc_la, alpha_h, alpha_a, n_sims=4000, seed=42):
    """Single in-match revaluation from `state` with cached calibration alphas."""
    return markets(*simulate(state, haz, dc_lh, dc_la, alpha_h, alpha_a, n_sims, seed))
