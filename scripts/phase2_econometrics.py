"""
scripts/phase2_econometrics.py

Produces three inputs the GNN pipeline needs:
  1. data/processed/dcc_correlations.pkl  — DCC-GARCH edge weights
  2. data/processed/granger_pvalues.csv   — Granger causality edge directions
  3. data/processed/regime_labels_kmeans_ALTERNATIVE.csv — an alternative
     K-means regime detector (NOT used by the main pipeline — see
     compute_regime_labels()'s header comment below)

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
# K-means clustering (3 states) on VIX level + credit spread + equity vol —
# see compute_regime_labels()'s docstring below for the actual method.
#
# NOT USED BY THE MAIN PIPELINE: this writes to the same path
# (data/processed/regime_labels.csv) as scripts/fix_regime_labels.py's
# simpler VIX-threshold rule, which is what build_graphs.py / evaluate.py
# actually consume for the reported results. Running this script would
# silently overwrite that file with a different regime definition — do not
# run it as part of the standard pipeline. Kept here as an alternative
# regime-detection method, not wired into results/.
# ─────────────────────────────────────────────

def compute_regime_labels(returns_df):
    """
    Regime detection using VIX LEVELS + K-means clustering.

    Critical fix: returns_df["VIX"] contains VIX LOG RETURNS, not levels.
    K-means on returns is meaningless for regime detection because a +10%
    VIX return looks identical whether VIX is at 12 (calm) or 40 (crisis).
    We download VIX closing prices directly so the cluster centroids
    correspond to actual volatility regimes.
    """
    print("=" * 55)
    print("STEP 3: Regime detection (VIX levels + K-means)...")
    print("=" * 55)

    import yfinance as yf
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Download VIX price LEVELS (not returns)
    print("  Downloading VIX levels from Yahoo Finance...")
    start_str = str(returns_df.index[0].date())
    end_str   = str((returns_df.index[-1] + pd.Timedelta(days=10)).date())
    vix_raw   = yf.download("^VIX", start=start_str, end=end_str,
                             progress=False)["Close"].squeeze()
    vix_raw.index = pd.to_datetime(vix_raw.index)
    if hasattr(vix_raw.index, "tz") and vix_raw.index.tz is not None:
        vix_raw.index = vix_raw.index.tz_localize(None)

    # Align to returns index
    vix_levels = vix_raw.reindex(returns_df.index).ffill().bfill()
    print(f"  VIX level range : {vix_levels.min():.1f} to {vix_levels.max():.1f}")

    # Credit stress and equity vol features (still use rolling returns vol)
    hy_vol  = returns_df["HYG"].rolling(20).std()
    ig_vol  = returns_df["LQD"].rolling(20).std()
    spread  = (hy_vol - ig_vol).fillna(0)
    spy_vol = returns_df["SPY"].rolling(20).std().fillna(0)

    features = pd.DataFrame({
        "VIX_level": vix_levels,   # level, not return
        "spread":    spread,
        "spy_vol":   spy_vol,
    }).dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    kmeans = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10)
    raw_labels = kmeans.fit_predict(X)

    # Order clusters by mean VIX LEVEL: lowest = calm (0), highest = crisis (2)
    vix_vals = features["VIX_level"].values
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
        bar = "█" * int(pct / 2)
        print(f"    {names[state]:8s} (label={state}): {count:>5} days ({pct:.1f}%) {bar}")

    # Written to a distinct filename — NOT regime_labels.csv — so this never
    # collides with scripts/fix_regime_labels.py's VIX-threshold output, which
    # is the one build_graphs.py / evaluate.py actually read. See this
    # function's header comment above for why both exist.
    out_path = os.path.join(PROCESSED_PATH, "regime_labels_kmeans_ALTERNATIVE.csv")
    regime_series.to_csv(out_path, header=True)
    print(f"\n  [SAVED] {out_path}")
    print("  (alternative regime detector — NOT used by the main pipeline;")
    print("   the pipeline reads data/processed/regime_labels.csv, produced by")
    print("   scripts/fix_regime_labels.py instead)\n")

    return regime_series


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  CROSS-ASSET CONTAGION — ECONOMETRICS")
    print("=" * 55 + "\n")

    os.makedirs(PROCESSED_PATH, exist_ok=True)

    returns = load_returns()
    dcc     = compute_dcc_correlations(returns)
    granger = compute_granger_causality(returns)
    regime  = compute_regime_labels(returns)

    print("=" * 55)
    print("  DONE. Outputs:")
    print("    -> data/processed/dcc_correlations.pkl")
    print("    -> data/processed/granger_pvalues.csv")
    print("    -> data/processed/dcc_avg_correlation.csv")
    print("    -> data/processed/regime_labels_kmeans_ALTERNATIVE.csv")
    print("       (NOT used by the pipeline — run scripts/fix_regime_labels.py")
    print("        separately to produce the regime_labels.csv the model reads)")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
