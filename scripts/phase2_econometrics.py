"""
Cross-Asset Contagion Project — Phase 2 Econometrics
Ryan: Data & Econometrics Lead

Produces three outputs Srujan needs for the GNN:
  1. data/processed/dcc_correlations.pkl  — DCC-GARCH edge weights
  2. data/processed/granger_pvalues.csv   — Granger causality edge directions
  3. data/processed/regime_labels.csv     — Regime labels (0=calm,1=stress,2=crisis)

Run from repo root:
    python scripts/phase2_econometrics.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

RETURNS_PATH   = "data/processed/returns_matrix.csv"
PROCESSED_PATH = "data/processed"

ASSET_COLS = ["SPY", "EEM", "LQD", "HYG", "TLT", "GLD", "USO", "BTC", "ETH", "DXY"]
VIX_COL    = "VIX"

GRANGER_MAX_LAG = 5
N_REGIMES       = 3


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_returns():
    print("=" * 55)
    print("Loading returns_matrix.csv...")
    print("=" * 55)
    df = pd.read_csv(RETURNS_PATH, index_col=0, parse_dates=True)
    df = df.sort_index().ffill().dropna()
    print(f"  Loaded: {df.shape[0]} days x {df.shape[1]} assets")
    print(f"  Range : {df.index[0].date()} to {df.index[-1].date()}\n")
    return df


# ─────────────────────────────────────────────
# STEP 1: DCC-GARCH
# ─────────────────────────────────────────────

def compute_dcc_correlations(returns_df):
    print("=" * 55)
    print("STEP 1: DCC-GARCH correlations...")
    print("=" * 55)

    from arch import arch_model

    asset_returns = returns_df[ASSET_COLS] * 100

    print("  Fitting GARCH(1,1) per asset:")
    std_resids = pd.DataFrame(index=asset_returns.index, columns=ASSET_COLS, dtype=float)

    for asset in ASSET_COLS:
        try:
            r = asset_returns[asset].dropna()
            model = arch_model(r, vol="Garch", p=1, q=1, dist="normal", rescale=False)
            result = model.fit(disp="off", show_warning=False)
            resid = result.std_resid
            std_resids.loc[resid.index, asset] = resid.values
            print(f"    [OK] {asset}")
        except Exception as e:
            print(f"    [WARN] {asset}: GARCH failed, using raw returns")
            std_resids[asset] = asset_returns[asset]

    std_resids = std_resids.ffill().dropna().astype(float)

    print("\n  Running DCC(1,1) recursion...")
    alpha = 0.05
    beta  = 0.93
    n_assets = len(ASSET_COLS)
    E = std_resids.values
    T = E.shape[0]
    Q_bar = np.corrcoef(E.T)
    Q_t   = Q_bar.copy()
    dcc_correlations = {}

    for t in range(1, T):
        e_prev = E[t-1:t].T
        Q_t = (1 - alpha - beta) * Q_bar + alpha * (e_prev @ e_prev.T) + beta * Q_t
        d = np.sqrt(np.diag(Q_t))
        d[d == 0] = 1e-8
        R_t = Q_t / np.outer(d, d)
        np.fill_diagonal(R_t, 1.0)
        dcc_correlations[std_resids.index[t]] = R_t.copy()

    print(f"  DCC complete: {len(dcc_correlations)} daily matrices\n")

    out_path = os.path.join(PROCESSED_PATH, "dcc_correlations.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(dcc_correlations, f)
    print(f"  [SAVED] {out_path}")

    avg_corr = np.mean(list(dcc_correlations.values()), axis=0)
    avg_df = pd.DataFrame(avg_corr, index=ASSET_COLS, columns=ASSET_COLS)
    avg_path = os.path.join(PROCESSED_PATH, "dcc_avg_correlation.csv")
    avg_df.round(4).to_csv(avg_path)
    print(f"  [SAVED] {avg_path}\n")

    return dcc_correlations


# ─────────────────────────────────────────────
# STEP 2: GRANGER CAUSALITY
# ─────────────────────────────────────────────

def compute_granger_causality(returns_df):
    print("=" * 55)
    print("STEP 2: Granger causality tests (110 pairs)...")
    print("=" * 55)

    from statsmodels.tsa.stattools import grangercausalitytests

    asset_returns = returns_df[ASSET_COLS]
    n = len(ASSET_COLS)
    pvalue_matrix = pd.DataFrame(np.ones((n, n)), index=ASSET_COLS, columns=ASSET_COLS)

    pair_count = 0
    for i, cause in enumerate(ASSET_COLS):
        for j, effect in enumerate(ASSET_COLS):
            if i == j:
                continue
            try:
                data = asset_returns[[effect, cause]].dropna()
                result = grangercausalitytests(data, maxlag=GRANGER_MAX_LAG, verbose=False)
                min_pval = min(
                    result[lag][0]["ssr_ftest"][1]
                    for lag in range(1, GRANGER_MAX_LAG + 1)
                )
                pvalue_matrix.loc[cause, effect] = min_pval
                pair_count += 1
            except:
                pvalue_matrix.loc[cause, effect] = 1.0

    print(f"  Tested {pair_count} pairs")
    sig = (pvalue_matrix < 0.05).sum().sum()
    print(f"  Significant edges (p < 0.05): {sig} out of {n*(n-1)}")

    out_path = os.path.join(PROCESSED_PATH, "granger_pvalues.csv")
    pvalue_matrix.round(4).to_csv(out_path)
    print(f"  [SAVED] {out_path}\n")

    print("  Significant Granger edges (cause -> effect, p < 0.05):")
    for cause in ASSET_COLS:
        for effect in ASSET_COLS:
            if cause != effect and pvalue_matrix.loc[cause, effect] < 0.05:
                p = pvalue_matrix.loc[cause, effect]
                print(f"    {cause:5s} -> {effect:5s}  p={p:.4f}")

    return pvalue_matrix


# ─────────────────────────────────────────────
# STEP 3: REGIME DETECTION (no hmmlearn)
# Uses VIX-based threshold clustering — same 3 states,
# no C++ build tools required
# ─────────────────────────────────────────────

def compute_regime_labels(returns_df):
    """
    Regime detection using VIX percentile thresholds + K-means clustering.
    Produces same 3 states as HMM: 0=calm, 1=stress, 2=crisis.
    Uses only numpy/scipy/sklearn — no C++ compilation needed.
    """
    print("=" * 55)
    print("STEP 3: Regime detection (VIX + K-means)...")
    print("=" * 55)

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    vix     = returns_df[VIX_COL]
    hy_vol  = returns_df["HYG"].rolling(20).std()
    ig_vol  = returns_df["LQD"].rolling(20).std()
    spread  = (hy_vol - ig_vol).fillna(0)
    spy_vol = returns_df["SPY"].rolling(20).std().fillna(0)

    features = pd.DataFrame({
        "VIX":     vix,
        "spread":  spread,
        "spy_vol": spy_vol
    }).dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    # K-means with 3 clusters — deterministic with fixed seed
    kmeans = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10)
    raw_labels = kmeans.fit_predict(X)

    # Order clusters by mean VIX: lowest = calm (0), highest = crisis (2)
    vix_vals = features["VIX"].values
    cluster_vix_means = [vix_vals[raw_labels == c].mean() for c in range(N_REGIMES)]
    order = np.argsort(cluster_vix_means)
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[c] for c in raw_labels])

    regime_series = pd.Series(labels, index=features.index, name="regime")

    counts = pd.Series(labels).value_counts().sort_index()
    names  = {0: "calm", 1: "stress", 2: "crisis"}
    print("  Regime distribution:")
    for state, count in counts.items():
        pct = 100 * count / len(labels)
        print(f"    {names[state]:8s} (label={state}): {count} days ({pct:.1f}%)")

    out_path = os.path.join(PROCESSED_PATH, "regime_labels.csv")
    regime_series.to_csv(out_path, header=True)
    print(f"\n  [SAVED] {out_path}\n")

    return regime_series


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  CROSS-ASSET CONTAGION — PHASE 2 ECONOMETRICS")
    print("  Ryan: Data & Econometrics Lead")
    print("=" * 55 + "\n")

    os.makedirs(PROCESSED_PATH, exist_ok=True)

    returns = load_returns()
    dcc     = compute_dcc_correlations(returns)
    granger = compute_granger_causality(returns)
    regime  = compute_regime_labels(returns)

    print("=" * 55)
    print("  PHASE 2 DONE. Files for Srujan:")
    print("    -> data/processed/dcc_correlations.pkl")
    print("    -> data/processed/granger_pvalues.csv")
    print("    -> data/processed/dcc_avg_correlation.csv")
    print("    -> data/processed/regime_labels.csv")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
