"""
Cross-Asset Contagion Project — Regime Label Fix
Ryan: Data & Econometrics Lead

Fix: Pull raw VIX levels directly from Yahoo Finance
     (returns_matrix.csv has log returns, not VIX levels)

VIX thresholds:
  VIX < 20  → calm   (label=0)
  VIX 20-30 → stress (label=1)
  VIX > 30  → crisis (label=2)

Run from repo root:
    python scripts/fix_regime_labels.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import warnings

warnings.filterwarnings("ignore")

PROCESSED_PATH       = "data/processed"
VIX_STRESS_THRESHOLD = 20.0
VIX_CRISIS_THRESHOLD = 30.0
START_DATE           = "2015-01-01"
END_DATE             = "2024-12-31"


def get_vix_levels():
    """Pull raw VIX closing levels — NOT returns."""
    print("  Pulling raw VIX levels from Yahoo Finance...")
    raw = yf.download("^VIX", start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)
    vix = raw["Close"].squeeze()
    vix = vix[vix.index.dayofweek < 5]  # weekdays only
    vix = vix.ffill().dropna()
    print(f"  VIX range: {vix.min():.1f} to {vix.max():.1f}")
    print(f"  VIX mean:  {vix.mean():.1f}")
    return vix


def compute_regime_labels(vix):
    print("\n  Applying VIX thresholds...")
    labels = pd.Series(0, index=vix.index, name="regime", dtype=int)
    labels[vix >= VIX_STRESS_THRESHOLD] = 1
    labels[vix >= VIX_CRISIS_THRESHOLD] = 2

    total = len(labels)
    names = {0: "calm", 1: "stress", 2: "crisis"}

    print(f"\n  Thresholds: calm<{VIX_STRESS_THRESHOLD}, "
          f"stress {VIX_STRESS_THRESHOLD}-{VIX_CRISIS_THRESHOLD}, "
          f"crisis>{VIX_CRISIS_THRESHOLD}\n")
    print("  Regime distribution:")
    for label in [0, 1, 2]:
        count = (labels == label).sum()
        pct   = 100 * count / total
        mean_vix = vix[labels == label].mean()
        print(f"    {names[label]:8s} (label={label}): "
              f"{count} days ({pct:.1f}%) — avg VIX={mean_vix:.1f}")

    # Sanity check known crisis periods
    print("\n  Crisis period checks:")
    covid = labels["2020-02-01":"2020-05-01"]
    covid_crisis = (covid == 2).sum()
    print(f"    COVID crash (Feb-May 2020): {covid_crisis} crisis days "
          f"{'[OK]' if covid_crisis > 20 else '[CHECK]'}")

    gfc_available = "2008-09-01" in labels.index
    if gfc_available:
        gfc = labels["2008-09-01":"2009-03-01"]
        gfc_crisis = (gfc == 2).sum()
        print(f"    GFC (Sep 2008-Mar 2009):    {gfc_crisis} crisis days "
              f"{'[OK]' if gfc_crisis > 20 else '[CHECK]'}")

    return labels


def save_labels(labels):
    out_path = os.path.join(PROCESSED_PATH, "regime_labels.csv")
    labels.to_csv(out_path, header=True)
    print(f"\n  [SAVED] {out_path}")


def main():
    print("\n" + "=" * 55)
    print("  REGIME LABEL FIX — VIX THRESHOLD METHOD")
    print("  Ryan: Data & Econometrics Lead")
    print("=" * 55 + "\n")

    os.makedirs(PROCESSED_PATH, exist_ok=True)
    vix    = get_vix_levels()
    labels = compute_regime_labels(vix)
    save_labels(labels)

    print("\n" + "=" * 55)
    print("  DONE. Now run:")
    print("    git add .")
    print('    git commit -m "Ryan: fix regime labels - VIX threshold method"')
    print("    git push")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
