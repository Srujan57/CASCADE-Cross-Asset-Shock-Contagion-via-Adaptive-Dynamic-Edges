"""
scripts/validate_data.py

Checks all data files before they enter the GNN:
  - No NaN values
  - Index alignment across all files
  - Value ranges look correct
  - DCC correlations are valid matrices
  - Regime labels are 0, 1, or 2 only

Run from repo root:
    python scripts/validate_data.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings

warnings.filterwarnings("ignore")

PROCESSED_PATH = "data/processed"

RETURNS_PATH    = os.path.join(PROCESSED_PATH, "returns_matrix.csv")
DCC_PATH        = os.path.join(PROCESSED_PATH, "dcc_correlations.pkl")
GRANGER_PATH    = os.path.join(PROCESSED_PATH, "granger_pvalues.csv")
REGIME_PATH     = os.path.join(PROCESSED_PATH, "regime_labels.csv")
REGIME_W_PATH   = os.path.join(PROCESSED_PATH, "regime_labels_weekly.csv")

ASSET_COLS = ["SPY", "EEM", "LQD", "HYG", "TLT", "GLD", "USO", "BTC", "ETH", "DXY", "VIX"]

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

errors   = []
warnings_list = []


def check(condition, label, detail=""):
    if condition:
        print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} {label}" + (f" — {detail}" if detail else ""))
        errors.append(label)


def warn(condition, label, detail=""):
    if not condition:
        print(f"  {WARN} {label}" + (f" — {detail}" if detail else ""))
        warnings_list.append(label)
    else:
        print(f"  {PASS} {label}")


# ─────────────────────────────────────────────
# CHECK 1: Returns matrix
# ─────────────────────────────────────────────

def validate_returns():
    print("=" * 55)
    print("CHECK 1: returns_matrix.csv")
    print("=" * 55)

    if not os.path.exists(RETURNS_PATH):
        print(f"  {FAIL} File not found: {RETURNS_PATH}")
        errors.append("returns_matrix.csv missing")
        return None

    df = pd.read_csv(RETURNS_PATH, index_col=0, parse_dates=True)

    check(df.shape[1] == 11, f"Has 11 asset columns (got {df.shape[1]})")
    check(df.shape[0] > 2000, f"Has >2000 trading days (got {df.shape[0]})")
    check(df.isnull().sum().sum() == 0, "No NaN values in returns")
    check(str(df.index[0].year) == "2017" or str(df.index[0].year) == "2015",
          f"Date range starts correctly (got {df.index[0].date()})")

    for col in ASSET_COLS:
        check(col in df.columns, f"Column '{col}' present")

    # Value range checks
    max_return = df[ASSET_COLS[:-1]].abs().max().max()  # exclude VIX
    warn(max_return < 0.5, f"Max daily return reasonable (got {max_return:.4f})",
         "values above 0.5 may indicate data error")

    crypto_std = df[["BTC", "ETH"]].std().mean()
    equity_std = df[["SPY", "EEM"]].std().mean()
    warn(crypto_std > equity_std,
         f"Crypto more volatile than equities (crypto std={crypto_std:.4f}, equity std={equity_std:.4f})")

    print(f"\n  Shape: {df.shape}, Range: {df.index[0].date()} to {df.index[-1].date()}\n")
    return df


# ─────────────────────────────────────────────
# CHECK 2: DCC correlations
# ─────────────────────────────────────────────

def validate_dcc(returns_df):
    print("=" * 55)
    print("CHECK 2: dcc_correlations.pkl")
    print("=" * 55)

    if not os.path.exists(DCC_PATH):
        print(f"  {FAIL} File not found: {DCC_PATH}")
        errors.append("dcc_correlations.pkl missing")
        return None

    with open(DCC_PATH, "rb") as f:
        dcc = pickle.load(f)

    check(isinstance(dcc, dict), "Is a dictionary")
    check(len(dcc) > 1000, f"Has >1000 daily matrices (got {len(dcc)})")

    # Check matrix properties
    sample_dates = list(dcc.keys())[:10]
    all_valid = True
    for d in sample_dates:
        mat = dcc[d]
        if not (mat.shape == (10, 10)):
            all_valid = False
        if not np.allclose(np.diag(mat), 1.0, atol=0.01):
            all_valid = False
        if not (mat.min() >= -1.0 and mat.max() <= 1.0):
            all_valid = False

    check(all_valid, "Matrices are 10x10, diagonal=1, values in [-1,1]")

    # Check date alignment with returns
    if returns_df is not None:
        dcc_dates  = set(dcc.keys())
        ret_dates  = set(returns_df.index)
        overlap    = len(dcc_dates & ret_dates)
        check(overlap > 1000,
              f"DCC dates align with returns ({overlap} overlapping dates)")

    # Check correlations move over time (not static)
    dates = list(dcc.keys())
    first = dcc[dates[0]]
    last  = dcc[dates[-1]]
    diff  = np.abs(first - last).mean()
    warn(diff > 0.01,
         f"Correlations change over time (mean change={diff:.4f})",
         "static correlations would defeat the purpose of DCC")

    print(f"\n  {len(dcc)} daily matrices, shape (10x10)\n")
    return dcc


# ─────────────────────────────────────────────
# CHECK 3: Granger p-values
# ─────────────────────────────────────────────

def validate_granger():
    print("=" * 55)
    print("CHECK 3: granger_pvalues.csv")
    print("=" * 55)

    if not os.path.exists(GRANGER_PATH):
        print(f"  {FAIL} File not found: {GRANGER_PATH}")
        errors.append("granger_pvalues.csv missing")
        return None

    df = pd.read_csv(GRANGER_PATH, index_col=0)
    assets = ["SPY", "EEM", "LQD", "HYG", "TLT", "GLD", "USO", "BTC", "ETH", "DXY"]

    check(df.shape == (10, 10), f"Shape is 10x10 (got {df.shape})")
    check(df.isnull().sum().sum() == 0, "No NaN values")
    check((df.values >= 0).all() and (df.values <= 1).all(),
          "All p-values in [0, 1]")

    sig_edges = (df < 0.05).sum().sum() - 10  # subtract diagonal
    check(sig_edges > 10,
          f"Has significant edges (p<0.05): {sig_edges} out of 90")

    warn(sig_edges < 80,
         f"Not too many significant edges ({sig_edges})",
         "too many may indicate multicollinearity")

    print(f"\n  {sig_edges} significant directional edges\n")
    return df


# ─────────────────────────────────────────────
# CHECK 4: Regime labels
# ─────────────────────────────────────────────

def validate_regime(returns_df):
    print("=" * 55)
    print("CHECK 4: regime_labels.csv + regime_labels_weekly.csv")
    print("=" * 55)

    for path, label in [(REGIME_PATH, "daily"), (REGIME_W_PATH, "weekly")]:
        if not os.path.exists(path):
            print(f"  {FAIL} File not found: {path}")
            errors.append(f"regime_labels ({label}) missing")
            continue

        df = pd.read_csv(path, index_col=0, parse_dates=True)
        col = df.columns[0]

        check(df.isnull().sum().sum() == 0, f"{label}: No NaN values")
        check(set(df[col].unique()).issubset({0, 1, 2}),
              f"{label}: Only labels 0, 1, 2 present")

        counts = df[col].value_counts().sort_index()
        total  = len(df)
        calm_pct   = 100 * counts.get(0, 0) / total
        stress_pct = 100 * counts.get(1, 0) / total
        crisis_pct = 100 * counts.get(2, 0) / total

        warn(calm_pct > 50,
             f"{label}: Calm regime >50% ({calm_pct:.1f}%)")
        warn(crisis_pct < 30,
             f"{label}: Crisis regime <30% ({crisis_pct:.1f}%)")
        warn(crisis_pct > 2,
             f"{label}: Crisis regime >2% ({crisis_pct:.1f}%)",
             "too few crisis periods may hurt model")

        print(f"    Distribution: calm={calm_pct:.1f}%, "
              f"stress={stress_pct:.1f}%, crisis={crisis_pct:.1f}%")

    print()


# ─────────────────────────────────────────────
# CHECK 5: Cross-file alignment
# ─────────────────────────────────────────────

def validate_alignment(returns_df):
    print("=" * 55)
    print("CHECK 5: Cross-file date alignment")
    print("=" * 55)

    if not os.path.exists(REGIME_PATH) or returns_df is None:
        print(f"  {WARN} Skipping — missing files")
        return

    regime = pd.read_csv(REGIME_PATH, index_col=0, parse_dates=True)
    ret_start = returns_df.index[0]
    ret_end   = returns_df.index[-1]
    reg_start = regime.index[0]
    reg_end   = regime.index[-1]

    check(abs((ret_start - reg_start).days) < 10,
          f"Returns and regime start dates align "
          f"(returns={ret_start.date()}, regime={reg_start.date()})")
    check(abs((ret_end - reg_end).days) < 10,
          f"Returns and regime end dates align "
          f"(returns={ret_end.date()}, regime={reg_end.date()})")

    overlap = len(set(returns_df.index) & set(regime.index))
    check(overlap > 1000,
          f"Returns and regime have >1000 overlapping dates (got {overlap})")
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  CROSS-ASSET CONTAGION — DATA VALIDATION")
    print("=" * 55 + "\n")

    returns = validate_returns()
    dcc     = validate_dcc(returns)
    granger = validate_granger()
    validate_regime(returns)
    validate_alignment(returns)

    print("=" * 55)
    if len(errors) == 0:
        print(f"  ALL CHECKS PASSED — {len(warnings_list)} warnings")
        print("  Data is clean and ready for the GNN.")
    else:
        print(f"  {len(errors)} ERRORS, {len(warnings_list)} warnings")
        print("  Fix errors before running build_graphs.py:")
        for e in errors:
            print(f"    - {e}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
