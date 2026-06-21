"""Dixon-Coles per-competition scoring-strength fitter (cell 9 / capstone Sec 4).
L2-regularized bounded L-BFGS-B with the low-score correction. Fit offline; the
fitted lambda_home/lambda_away feed the hazard model and the simulator's DC prior.

Parameterization (sum-to-zero att/def kept in check by L2 + bounds):
  log lambda_home = c + home_adv + att[home] - def[away]
  log lambda_away = c             + att[away] - def[home]
att = attacking strength (higher scores more); def = defensive solidity (higher
concedes less). rho is the Dixon-Coles low-score dependence correction."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

PARAMS_PATH = Path(__file__).parent / "dc_params.json"


def tau(x, y, lh, la, rho):
    """Dixon-Coles low-score correction for the (x,y) cell."""
    out = np.ones_like(lh, dtype=float)
    out = np.where((x == 0) & (y == 0), 1 - lh * la * rho, out)
    out = np.where((x == 0) & (y == 1), 1 + lh * rho, out)
    out = np.where((x == 1) & (y == 0), 1 + la * rho, out)
    out = np.where((x == 1) & (y == 1), 1 - rho, out)
    return out


class DixonColes:
    def __init__(self, teams=None, att=None, deff=None, home_adv=0.25, rho=-0.05, c=0.0):
        self.teams = list(teams) if teams is not None else []
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.att = np.zeros(len(self.teams)) if att is None else np.asarray(att, float)
        self.deff = np.zeros(len(self.teams)) if deff is None else np.asarray(deff, float)
        self.home_adv, self.rho, self.c = home_adv, rho, c

    # --- rates ---
    def _lams(self, hi, ai, att, deff, home_adv, c):
        lh = np.exp(c + home_adv + att[hi] - deff[ai])
        la = np.exp(c + att[ai] - deff[hi])
        return lh, la

    def lambdas(self, home, away):
        """(lambda_home, lambda_away) for a matchup; falls back to baseline if unseen."""
        hi, ai = self.idx.get(home), self.idx.get(away)
        if hi is None or ai is None:
            base = np.exp(self.c)
            return float(base * np.exp(self.home_adv)), float(base)
        lh, la = self._lams(np.array([hi]), np.array([ai]), self.att, self.deff,
                            self.home_adv, self.c)
        return float(lh[0]), float(la[0])

    def fit(self, home, away, hg, ag, l2=0.01, max_goals=10):
        """Fit on arrays of team names + goals (one competition). Returns self."""
        teams = sorted(set(home) | set(away))
        self.teams, self.idx = teams, {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = np.array([self.idx[t] for t in home])
        ai = np.array([self.idx[t] for t in away])
        hg, ag = np.asarray(hg, int), np.asarray(ag, int)

        def unpack(p):
            return p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1], p[2 * n + 2]  # att, def, home, rho, c

        def nll(p):
            att, deff, home_adv, rho, c = unpack(p)
            lh, la = self._lams(hi, ai, att, deff, home_adv, c)
            ll = (poisson.logpmf(hg, lh) + poisson.logpmf(ag, la)
                  + np.log(np.clip(tau(hg, ag, lh, la, rho), 1e-9, None)))
            pen = l2 * (att @ att + deff @ deff)        # L2 shrinkage (identifiability)
            return -ll.sum() + pen

        x0 = np.concatenate([np.zeros(2 * n), [0.25, -0.05, np.log(max(hg.mean(), 0.3))]])
        bounds = [(-3, 3)] * (2 * n) + [(-0.5, 1.0), (-0.2, 0.2), (-1.0, 1.5)]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
        self.att, self.deff, self.home_adv, self.rho, self.c = unpack(res.x)
        return self

    def outcome_probs(self, lh, la, max_goals=10):
        """Closed-form 1X2 / Over / BTTS from (lambda_home, lambda_away) with DC tau.
        Used as the simulator's DC-state prior and for descriptive checks."""
        x = np.arange(max_goals + 1)
        ph, pa = poisson.pmf(x, lh), poisson.pmf(x, la)
        M = np.outer(ph, pa)
        for (i, j) in ((0, 0), (0, 1), (1, 0), (1, 1)):
            M[i, j] *= float(tau(np.array(i), np.array(j), np.array([lh]), np.array([la]), self.rho)[0])
        M /= M.sum()
        gh, ga = np.indices(M.shape)
        total = gh + ga
        return {
            "home": float(M[gh > ga].sum()), "draw": float(M[gh == ga].sum()),
            "away": float(M[gh < ga].sum()),
            "over15": float(M[total >= 2].sum()), "over25": float(M[total >= 3].sum()),
            "btts": float(M[(gh >= 1) & (ga >= 1)].sum()),
        }

    # --- persistence (per-competition dict of params) ---
    def to_dict(self):
        return {"teams": self.teams, "att": self.att.tolist(), "deff": self.deff.tolist(),
                "home_adv": self.home_adv, "rho": self.rho, "c": self.c}

    @classmethod
    def from_dict(cls, d):
        return cls(d["teams"], d["att"], d["deff"], d["home_adv"], d["rho"], d["c"])


def save_all(models_by_comp, path=PARAMS_PATH):
    """models_by_comp: {competition: DixonColes}."""
    Path(path).write_text(json.dumps({k: m.to_dict() for k, m in models_by_comp.items()}))


def load_all(path=PARAMS_PATH):
    d = json.loads(Path(path).read_text()) if Path(path).exists() else {}
    return {k: DixonColes.from_dict(v) for k, v in d.items()}
