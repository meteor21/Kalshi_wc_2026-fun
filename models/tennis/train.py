"""Train + calibrate the pre-match tennis model. Run in Colab/locally, NEVER the VPS.
Time split: train <=2022, calibrate (isotonic) 2023, test 2024-25. Writes a joblib
artifact (model + isotonic + FeatureState) that score.py loads frozen on the VPS."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from .features import FEATURES, build_training_frame, clean, load_sackmann

ARTIFACT = Path(__file__).parent / "model.joblib"


def time_split(frame):
    return (frame[frame.year <= 2022], frame[frame.year == 2023], frame[frame.year >= 2024])


def _report(tag, y, p):
    print(f"  {tag:18} log_loss={log_loss(y, p):.4f}  brier={brier_score_loss(y, p):.4f}  n={len(y)}")


def reliability(y, p, bins=10):
    """Reliability curve points (mean predicted vs observed) without matplotlib."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            out.append((float(p[mask].mean()), float(y[mask].mean()), int(mask.sum())))
    return out


def main(out=ARTIFACT, plot=False):
    matches = clean(load_sackmann())
    frame, state = build_training_frame(matches)
    frame = frame.dropna(subset=FEATURES + ["target"])
    tr, cal, te = time_split(frame)
    print(f"frame={len(frame)}  train={len(tr)} (<=2022)  cal={len(cal)} (2023)  test={len(te)} (2024-25)")

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(tr[FEATURES], tr["target"])

    # isotonic calibration on the 2023 fold (PORTING.md: isotonic, not Platt)
    raw_cal = clf.predict_proba(cal[FEATURES])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, cal["target"].to_numpy())

    yte = te["target"].to_numpy()
    raw_te = clf.predict_proba(te[FEATURES])[:, 1]
    cal_te = iso.predict(raw_te)
    print("test metrics:")
    _report("uncalibrated", yte, raw_te)
    _report("isotonic", yte, cal_te)

    print("reliability (pred, obs, n):")
    for pred, obs, n in reliability(yte, cal_te):
        print(f"  {pred:.2f} -> {obs:.2f}  (n={n})")

    joblib.dump({"clf": clf, "iso": iso, "features": FEATURES, "state": state}, out)
    print(f"saved artifact -> {out}")

    if plot:
        import matplotlib.pyplot as plt
        pts = reliability(yte, cal_te)
        plt.plot([p for p, _, _ in pts], [o for _, o, _ in pts], marker="o")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("predicted"); plt.ylabel("observed"); plt.title("Reliability (test)")
        plt.savefig(Path(out).parent / "reliability.png", dpi=120)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ARTIFACT))
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    main(Path(a.out), a.plot)
