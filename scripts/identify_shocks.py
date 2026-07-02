"""
scripts/identify_shocks.py

Programmatic shock-event identification (Phase 2, Kailash's task per the
project plan: "Define shock events formally: use threshold return < -2 sigma
on each asset to programmatically identify shock dates").

This file previously existed but was empty (0 lines) — the hand-picked
events used in scripts/evaluate.py (COVID, FTX, SVB, 2022 rate shock) were
never actually validated against a systematic threshold rule. This script
does that validation and produces the full candidate list so Kailash can
build the curated event catalog on top of something reproducible, not just
memory of well-known dates.

Method:
    Rolling z-score per asset: z_t = (r_t - mean_252) / std_252
    using a trailing 252-trading-day (~1yr) window, NOT full-sample stats.
    Full-sample mean/std would leak future volatility information into
    early-period shock detection (e.g. COVID's -2 sigma bar would be set
    partly using 2021-2024 data). Rolling window avoids that look-ahead bias.
    An asset-day is flagged as a shock when z_t < -2.0.

Outputs:
    results/shock_events_identified.csv
        Every asset-day breaching the -2 sigma threshold, with z-score.
    results/shock_events_market_wide.csv
        Days where 2+ assets breach simultaneously (candidate contagion
        events — cross-asset, not idiosyncratic single-name moves).
    Console report cross-checking the four hand-picked events in
    evaluate.py's shock_events list against this systematic rule.

Run from repo root (after scripts/data_ingestion.py has produced
data/processed/returns_matrix.csv):
    python scripts/identify_shocks.py
"""

import os
import pandas as pd
import numpy as np

RETURNS_PATH = "data/processed/returns_matrix.csv"
RESULTS_DIR  = "results"

ROLLING_WINDOW = 252     # ~1 trading year
Z_THRESHOLD    = -2.0
MIN_HISTORY    = 60      # need at least this many days before computing z

# The four events currently hardcoded in scripts/evaluate.py — cross-checked
# below against the systematic rule so Kailash knows which are backed by the
# -2 sigma definition and which are curated for other reasons (e.g. the
# 2022 rate shock is a slower-moving regime event, not a single-day -2 sigma
# print, so it's expected NOT to show up here).
HANDPICKED_EVENTS = [
    {"name": "COVID Crash",     "date": "2020-03-12", "asset": "SPY"},
    {"name": "FTX Collapse",    "date": "2022-11-09", "asset": "BTC"},
    {"name": "SVB Crisis",      "date": "2023-03-10", "asset": "SPY"},
    {"name": "2022 Rate Shock", "date": "2022-06-13", "asset": "TLT"},
]


def load_returns(path=RETURNS_PATH):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index().ffill().dropna()
    return df


def compute_rolling_zscores(returns_df, asset_cols, window=ROLLING_WINDOW):
    """
    Trailing rolling z-score per asset. min_periods=MIN_HISTORY so early
    rows (before a full window is available) still get a z-score computed
    off whatever history exists, rather than being silently NaN for the
    first ~year of data.
    """
    r = returns_df[asset_cols]
    roll_mean = r.rolling(window=window, min_periods=MIN_HISTORY).mean()
    roll_std  = r.rolling(window=window, min_periods=MIN_HISTORY).std()
    roll_std  = roll_std.replace(0, np.nan)
    z = (r - roll_mean) / roll_std
    return z


def identify_shocks(returns_df, asset_cols, threshold=Z_THRESHOLD):
    z = compute_rolling_zscores(returns_df, asset_cols)

    records = []
    for date in z.index:
        row = z.loc[date]
        for asset in asset_cols:
            zval = row[asset]
            if pd.notna(zval) and zval < threshold:
                records.append({
                    "date":  date,
                    "asset": asset,
                    "return": float(returns_df.loc[date, asset]),
                    "z_score": float(zval),
                })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(["date", "z_score"]).reset_index(drop=True)
    return df, z


def market_wide_events(shock_df, min_assets=2):
    """
    Collapse asset-level shocks into market-wide candidate events: days
    where at least `min_assets` distinct assets breach -2 sigma. These are
    the days most likely to represent genuine cross-asset contagion rather
    than an idiosyncratic single-name move, and are the natural pool to draw
    the paper's curated event catalog from.
    """
    if shock_df.empty:
        return pd.DataFrame()

    grouped = shock_df.groupby("date").agg(
        n_assets=("asset", "nunique"),
        assets=("asset", lambda s: ",".join(sorted(s))),
        worst_z=("z_score", "min"),
    ).reset_index()

    wide = grouped[grouped["n_assets"] >= min_assets].sort_values(
        "worst_z"
    ).reset_index(drop=True)
    return wide


def crosscheck_handpicked(shock_df, returns_df, asset_cols):
    """
    For each hand-picked event in evaluate.py, report whether the source
    asset actually breached -2 sigma on (or within 3 days of) that date,
    per the systematic rule. This is a sanity check on the paper's curated
    event catalog, not a replacement for it — some genuinely important
    events (slow-moving regime shifts like the 2022 rate hike cycle) will
    not show up as single-day -2 sigma prints, and that's fine as long as
    the paper is explicit about which events are threshold-derived vs
    narrative-selected.
    """
    print("\nCross-check: hand-picked events in scripts/evaluate.py")
    print("-" * 70)
    for ev in HANDPICKED_EVENTS:
        target_date = pd.Timestamp(ev["date"])
        asset = ev["asset"]
        window = shock_df[
            (shock_df["asset"] == asset) &
            (shock_df["date"] >= target_date - pd.Timedelta(days=3)) &
            (shock_df["date"] <= target_date + pd.Timedelta(days=3))
        ]
        if not window.empty:
            best = window.loc[window["z_score"].idxmin()]
            print(f"  [CONFIRMED] {ev['name']:16s} {asset:5s} "
                  f"z={best['z_score']:.2f} on {best['date'].date()} "
                  f"(target {target_date.date()})")
        else:
            print(f"  [NOT A -2sigma PRINT] {ev['name']:16s} {asset:5s} "
                  f"near {target_date.date()} — likely a regime/narrative "
                  f"event rather than a single-day statistical outlier; "
                  f"document this distinction in the Methodology section.")
    print("-" * 70)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(RETURNS_PATH):
        print(f"returns_matrix.csv not found at {RETURNS_PATH}.")
        print("Run scripts/data_ingestion.py first.")
        return

    returns_df = load_returns()
    asset_cols = [c for c in returns_df.columns if c != "VIX"]

    print("=" * 60)
    print("  Shock Event Identification (rolling -2 sigma)")
    print("=" * 60)
    print(f"  Assets  : {asset_cols}")
    print(f"  Window  : {ROLLING_WINDOW} trading days (trailing)")
    print(f"  Threshold: z < {Z_THRESHOLD}")

    shock_df, z_scores = identify_shocks(returns_df, asset_cols)
    print(f"\n  Total asset-day shock flags: {len(shock_df)}")

    out_path = os.path.join(RESULTS_DIR, "shock_events_identified.csv")
    shock_df.to_csv(out_path, index=False)
    print(f"  [SAVED] {out_path}")

    wide_df = market_wide_events(shock_df, min_assets=2)
    wide_path = os.path.join(RESULTS_DIR, "shock_events_market_wide.csv")
    wide_df.to_csv(wide_path, index=False)
    print(f"  [SAVED] {wide_path}")
    print(f"  Market-wide candidate events (>=2 assets same day): {len(wide_df)}")

    if not wide_df.empty:
        print("\n  Top 15 market-wide shock days (by worst single-asset z-score):")
        print(wide_df.head(15).to_string(index=False))

    crosscheck_handpicked(shock_df, returns_df, asset_cols)

    print("\nDone. Use results/shock_events_market_wide.csv as the candidate")
    print("pool for the curated event catalog in the paper's Data section.")


if __name__ == "__main__":
    main()
