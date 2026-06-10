"""All betting constraints. The execution layer must reject anything failing check()."""
from dataclasses import dataclass

BANKROLL_START = 1000.0
HALT_FLOOR = 500.0          # absolute equity floor: stop all trading
DAILY_LOSS_LIMIT = 100.0    # halt for the day
MAX_OPEN_EXPOSURE = 0.30    # fraction of current bankroll
MIN_EDGE = 0.04             # model prob - breakeven prob, after fees
KELLY_FRACTION = 0.25       # quarter-Kelly

SINGLE_MAX_STAKE = 100.0
SINGLE_MAX_RATIO = 3.0      # total payout / stake (decimal odds cap)
PARLAY_MAX_STAKE = 20.0
PARLAY_MIN_PAYOUT = 5.0     # total payout / stake


@dataclass
class Signal:
    ticker: str
    side: str            # yes|no
    model_prob: float    # our probability that side wins
    price: float         # cost per contract in dollars (0-1)
    fee_per_contract: float = 0.0
    is_parlay_leg: bool = False


def kalshi_fee(price):
    """Trading fee per contract in dollars: 0.07 * p * (1-p), rounded up to next cent."""
    import math
    return math.ceil(7.0 * price * (1 - price)) / 100.0


def breakeven_prob(price, fee=0.0):
    return (price + fee) / 1.0


def edge(sig: Signal):
    return sig.model_prob - breakeven_prob(sig.price, sig.fee_per_contract)


def kelly_stake(sig: Signal, bankroll):
    """Fractional Kelly for binary contract paying $1."""
    p, c = sig.model_prob, sig.price + sig.fee_per_contract
    if c <= 0 or c >= 1:
        return 0.0
    b = (1 - c) / c                      # net odds
    f = (p * b - (1 - p)) / b            # full Kelly fraction
    return max(0.0, f * KELLY_FRACTION * bankroll)


def check_single(sig: Signal, stake, bankroll, open_exposure, daily_pnl):
    """Returns (ok, reason)."""
    if bankroll <= HALT_FLOOR:
        return False, "halt floor breached"
    if daily_pnl <= -DAILY_LOSS_LIMIT:
        return False, "daily loss limit"
    if edge(sig) < MIN_EDGE:
        return False, f"edge {edge(sig):.3f} < {MIN_EDGE}"
    if stake > SINGLE_MAX_STAKE:
        return False, "stake > single max"
    ratio = 1.0 / sig.price if sig.price > 0 else 999
    if ratio > SINGLE_MAX_RATIO:
        return False, f"payout ratio {ratio:.2f} > {SINGLE_MAX_RATIO}"
    if open_exposure + stake > MAX_OPEN_EXPOSURE * bankroll:
        return False, "exposure cap"
    return True, "ok"


def check_parlay(legs, stake, payout_mult, bankroll, open_exposure, daily_pnl):
    if bankroll <= HALT_FLOOR:
        return False, "halt floor breached"
    if daily_pnl <= -DAILY_LOSS_LIMIT:
        return False, "daily loss limit"
    if stake > PARLAY_MAX_STAKE:
        return False, "stake > parlay max"
    if payout_mult < PARLAY_MIN_PAYOUT:
        return False, f"payout {payout_mult:.1f}x < {PARLAY_MIN_PAYOUT}x"
    combo_prob = 1.0
    for leg in legs:
        combo_prob *= leg.model_prob
    if combo_prob * payout_mult < 1.0 + MIN_EDGE:
        return False, f"combo EV {combo_prob * payout_mult:.2f} below threshold"
    if open_exposure + stake > MAX_OPEN_EXPOSURE * bankroll:
        return False, "exposure cap"
    return True, "ok"
