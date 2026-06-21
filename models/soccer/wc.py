"""World Cup adaptation (PORTING.md). National-team Dixon-Coles strengths are too
sparse to fit, so v1 DERIVES the simulator's lambda_DC inputs from de-overrounded
market 1X2 odds instead. The pooled hazard STATE coefficients are transferred from
club data as an explicit modeling assumption. WC trading is held to a stricter bar."""
import numpy as np
from scipy.optimize import minimize

from strategy import rules
from .dixon_coles import DixonColes
from .simulator import de_overround

WC_MIN_EDGE = rules.MIN_EDGE + 0.02     # stricter than club play (sparse-data caution)
WC_N_SIMS = 4000                        # prior widening: more simulation paths
NO_TRADE_BAND = 0.03                    # within +/-3pts of breakeven -> no trade

_DC = DixonColes(rho=0.0)               # closed-form Poisson outcome probs (rho = 0)


def lambdas_from_market(market_1x2, max_goals=10):
    """Solve (lambda_home, lambda_away) whose closed-form 1X2 matches the de-overrounded
    market vector. Stands in for a Dixon-Coles fit on thin international data."""
    target = de_overround(market_1x2)

    def loss(z):
        lh, la = np.exp(z)
        p = _DC.outcome_probs(lh, la, max_goals)
        d = np.array([p["home"], p["draw"], p["away"]]) - target
        return float(d @ d)

    res = minimize(loss, np.log([1.4, 1.1]), method="L-BFGS-B",
                   bounds=[(np.log(0.2), np.log(3.5))] * 2)
    lh, la = np.exp(res.x)
    return float(lh), float(la)


def tradable(model_p, breakeven_p):
    """WC gate: clear the stricter edge AND sit outside the no-trade band."""
    edge = model_p - breakeven_p
    return edge >= WC_MIN_EDGE and abs(edge) > NO_TRADE_BAND
