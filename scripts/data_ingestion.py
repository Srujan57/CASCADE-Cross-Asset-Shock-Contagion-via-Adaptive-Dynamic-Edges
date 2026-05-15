"""
Cross-Asset Contagion Project — Phase 1 Data Ingestion
Ryan: Data & Econometrics Lead

Matches config.yaml exactly:
  - Tickers from assets section
  - Date range from dates section
  - Output paths from paths section (data/raw/, data/processed/)

Run from repo root:
    python scripts/data_ingestion.py

Outputs:
    data/raw/prices_raw.csv
    data/processed/returns_matrix.csv
    data/processed/data_quality_report.txt
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG — mirrors config.yaml exactly
# ─────────────────────────────────────────────

START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"

TICKERS = {
    "SPY": "SPY",
    "EEM": "EEM",
    "LQD": "LQD",
    "HYG": "HYG",
    "TLT": "TLT",
    "GLD": "GLD",
    "USO": "USO",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
}

RAW_PATH       = os.path.join("data", "raw")
PROCESSED_PATH = os.path.join("data", "processed")


def create_folders():
    os.makedirs(RAW_PATH, exist_ok=True)
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    print("  [OK] Folders ready: data/raw/ and data/processed/\n")


def pull_prices() -> pd.DataFrame:
    print("=" * 55)
    print("STEP 1: Pulling price data from Yahoo Finance...")
    print("=" * 55)

    frames = {}
    for name, ticker in TICKERS.items():
        try:
            raw = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
            if raw.empty:
                print(f"  [WARNING] No data for {name} ({ticker})")
                continue
            close = raw["Close"].squeeze()
            close.name = name
            frames[name] = close
            print(f"  [OK] {name:5s} — {len(close)} days")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    prices = pd.DataFrame(frames)
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"
    print(f"\n  Raw shape: {prices.shape}\n")
    return prices


def align_and_clean(prices: pd.DataFrame) -> pd.DataFrame:
    print("=" * 55)
    print("STEP 2: Aligning calendars & cleaning data...")
    print("=" * 55)

    prices = prices[prices.index.dayofweek < 5]
    missing_before = prices.isnull().sum()
    prices = prices.ffill(limit=3)
    threshold = int(prices.shape[1] * 0.7)
    prices = prices.dropna(thresh=threshold)
    missing_after = prices.isnull().sum()

    print("  Missing values (before → after forward-fill):")
    for col in prices.columns:
        print(f"    {col:5s}: {missing_before[col]:4d} → {missing_after[col]:4d}")
    print(f"\n  Aligned shape: {prices.shape}\n")
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    print("=" * 55)
    print("STEP 3: Computing log returns...")
    print("=" * 55)

    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.dropna(how="all")

    if log_returns.empty:
        print("  [WARNING] Empty returns — check data pull.")
        return log_returns

    print("  Return statistics:\n")
    stats = log_returns.describe().T[["mean", "std", "min", "max"]]
    stats.columns = ["Mean", "Std Dev", "Min", "Max"]
    print(stats.round(4).to_string())
    print(f"\n  Returns shape: {log_returns.shape}\n")
    return log_returns


def save_outputs(prices: pd.DataFrame, returns: pd.DataFrame):
    print("=" * 55)
    print("STEP 4: Saving outputs...")
    print("=" * 55)

    prices_path = os.path.join(RAW_PATH, "prices_raw.csv")
    prices.to_csv(prices_path)
    print(f"  [SAVED] {prices_path}")

    returns_path = os.path.join(PROCESSED_PATH, "returns_matrix.csv")
    returns.to_csv(returns_path)
    print(f"  [SAVED] {returns_path}")

    report_path = os.path.join(PROCESSED_PATH, "data_quality_report.txt")
    with open(report_path, "w") as f:
        f.write("CROSS-ASSET CONTAGION — DATA QUALITY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}\n")
        f.write(f"Total trading days: {len(returns)}\n\n")
        f.write("Crypto/Equity Alignment Rule:\n")
        f.write("  Weekend crypto prices dropped. Friday close forward-filled to Monday.\n\n")
        f.write("Asset coverage:\n")
        for col in returns.columns:
            n_valid = returns[col].notna().sum()
            pct = 100 * n_valid / len(returns)
            f.write(f"  {col:5s}: {n_valid} valid days ({pct:.1f}% coverage)\n")
        f.write("\nReturn statistics:\n")
        f.write(returns.describe().round(6).to_string())
    print(f"  [SAVED] {report_path}\n")


def main():
    print("\n" + "=" * 55)
    print("  CROSS-ASSET CONTAGION — PHASE 1 DATA INGESTION")
    print("  Ryan: Data & Econometrics Lead")
    print("=" * 55 + "\n")

    create_folders()
    prices  = pull_prices()
    prices  = align_and_clean(prices)
    returns = compute_log_returns(prices)
    save_outputs(prices, returns)

    print("=" * 55)
    print("  DONE. Hand these files to Srujan:")
    print("    → data/processed/returns_matrix.csv")
    print("    → data/processed/data_quality_report.txt")
    print("=" * 55 + "\n")

    return returns


if __name__ == "__main__":
    returns = main()
