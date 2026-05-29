"""
Cross-Asset Contagion Project — Regime Label Fix (v2)
Ryan: Data & Econometrics Lead

Fix: Output BOTH daily and weekly regime labels so Srujan's
     build_graphs.py swap_in_regime_labels() finds exact date matches.

VIX thresholds:
  VIX < 20  → calm   (label=0) ~60-70%
  VIX 20-30 → stress (label=1) ~20-25%
  VIX > 30  → crisis (label=2) ~5-10%

Run from repo root:
    python scripts/fix_regime_labels.py

Outputs:
    data/processed/regime_labels.csv         (daily — original)
    data/processed/regime_labels_weekly.csv  (weekly — for build_graphs.py)
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
    print("  Pulling raw VIX levels from Yahoo Finance...")
    raw = yf.download("^VIX", start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)
    vix = raw["Close"].squeeze()
    vix = vix[vix.index.dayofweek < 5]
    vix = vix.ffill().dropna()
    # Remove timezone info to match returns_matrix index
    vix.index = vix.index.tz_localize(None)
    print(f"  VIX range: {vix.min():.1f} to {vix.max():.1f}, mean: {vix.mean():.1f}")
    return vix


def compute_regime_labels(vix):
    labels = pd.Series(0, index=vix.index, name="regime", dtype=int)
    labels[vix >= VIX_STRESS_THRESHOLD] = 1
    labels[vix >= VIX_CRISIS_THRESHOLD] = 2
    return labels


def print_distribution(labels, vix, title):
    total = len(labels)
    names = {0: "calm", 1: "stress", 2: "crisis"}
    print(f"\n  {title}:")
    for label in [0, 1, 2]:
        count    = (labels == label).sum()
        pct      = 100 * count / total
        mean_vix = vix.reindex(labels[labels == label].index).mean()
        print(f"    {names[label]:8s} (label={label}): "
              f"{count:4d} ({pct:.1f}%) — avg VIX={mean_vix:.1f}")


def main():
    print("\n" + "=" * 55)
    print("  REGIME LABEL FIX v2 — DAILY + WEEKLY OUTPUT")
    print("  Ryan: Data & Econometrics Lead")
    print("=" * 55 + "\n")

    os.makedirs(PROCESSED_PATH, exist_ok=True)

    vix    = get_vix_levels()
    labels = compute_regime_labels(vix)

    # ── Daily labels (original) ───────────────────────────────────────────
    daily_path = os.path.join(PROCESSED_PATH, "regime_labels.csv")
    labels.to_csv(daily_path, header=True)
    print_distribution(labels, vix, "Daily regime distribution")
    print(f"\n  [SAVED] {daily_path}")

    # ── Weekly labels — resample to weekly frequency ──────────────────────
    # Use last trading day of each week (same as build_graphs.py snapshot logic)
    # mode = most common regime in that week
    weekly_labels = labels.resample("W-FRI").agg(
        lambda x: x.mode()[0] if len(x) > 0 else 0
    ).astype(int)
    weekly_labels.name = "regime"

    # Also produce forward-filled version for any missing weeks
    weekly_labels = weekly_labels.ffill().dropna().astype(int)

    weekly_path = os.path.join(PROCESSED_PATH, "regime_labels_weekly.csv")
    weekly_labels.to_csv(weekly_path, header=True)
    print_distribution(weekly_labels, vix.resample("W-FRI").last().ffill(), 
                      "Weekly regime distribution")
    print(f"\n  [SAVED] {weekly_path}")

    # ── Sanity check ──────────────────────────────────────────────────────
    print("\n  Crisis period checks:")
    covid = labels["2020-02-01":"2020-05-01"]
    print(f"    COVID (Feb-May 2020): {(covid==2).sum()} crisis days "
          f"{'[OK]' if (covid==2).sum() > 20 else '[CHECK]'}")

    print("\n" + "=" * 55)
    print("  DONE. Now run:")
    print("    git add .")
    print('    git commit -m "Ryan: regime labels v2 - daily + weekly output"')
    print("    git push")
    print("=" * 55 + "\n")
    print("  Tell Srujan to update build_graphs.py to read")
    print("  regime_labels_weekly.csv instead of regime_labels.csv")


if __name__ == "__main__":
    main()
