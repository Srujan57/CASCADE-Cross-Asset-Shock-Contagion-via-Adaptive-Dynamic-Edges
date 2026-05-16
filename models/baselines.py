"""
models/baselines.py

Three baseline models for cross-asset contagion prediction.
Each baseline isolates one component to justify the full EvolveGCN-H architecture.

Baselines:
    1. RollingCorrelationBaseline  — time-varying but no learning
    2. VARBaseline                 — linear learning, no graph structure
    3. StaticGCN                   — graph structure but no temporal evolution

Expected result ordering (worst → best):
    RollingCorr < VAR < StaticGCN < EvolveGCN-H

Run: python models/baselines.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from statsmodels.tsa.vector_ar.var_model import VAR
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Data utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_returns(path):
    """
    Load returns_matrix.csv from Ryan's preprocessing pipeline.

    index_col=0      : first column (dates) becomes the row index
    parse_dates=True : parse the index as datetime objects so you can
                       slice with df["2020-01-01":"2020-12-31"]
    sort_index()     : guarantee chronological order
    dropna           : drop rows where ALL assets are missing
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df = df.ffill()        # forward-fill intra-series gaps
    df = df.dropna()       # drop rows where ANY column is still NaN
                           # (early ETH rows before ~Nov 2017 have no data
                           #  and can't be forward-filled — drop them cleanly)
    print(f"Data loaded: {len(df)} rows, {df.shape[1]} assets")
    print(f"Date range : {df.index[0].date()} → {df.index[-1].date()}")
    return df


def train_val_test_split(df, train_end="2020-12-31", val_end="2022-12-31"):
    """
    Chronological split — NEVER shuffle time series data.
    Random shuffling leaks future data into training (data leakage).

    Args:
        df        : full returns DataFrame with datetime index
        train_end : last date of training set
        val_end   : last date of validation set
        (test is implicitly val_end → end of df)

    Returns:
        train, val, test DataFrames
    """
    train = df[df.index <= train_end]
    val   = df[(df.index > train_end) & (df.index <= val_end)]
    test  = df[df.index > val_end]

    print(f"Train: {train.index[0].date()} → {train.index[-1].date()} "
          f"({len(train)} days)")
    print(f"Val  : {val.index[0].date()} → {val.index[-1].date()} "
          f"({len(val)} days)")
    print(f"Test : {test.index[0].date()} → {test.index[-1].date()} "
          f"({len(test)} days)")

    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1: Rolling Correlation
# ─────────────────────────────────────────────────────────────────────────────

class RollingCorrelationBaseline:
    """
    Predicts each asset's return as the correlation-weighted average of
    all assets' returns on the previous day.

    Time-varying (uses a rolling window) but no learning — pure statistics.
    Represents the ceiling of what you can do with correlations alone.

    Args:
        window : rolling window in days for correlation computation (60 from config)
    """

    def __init__(self, window=60):
        self.window      = window
        self.corr_matrix = None

    def fit(self, train_df):
        """
        Compute the static correlation matrix on training data.
        .corr() computes pairwise Pearson correlation for all column pairs.
        Shape: (num_assets, num_assets).
        """
        self.corr_matrix = train_df.corr()
        self.asset_names = train_df.columns.tolist()
        return self

    def _get_rolling_corr(self, df, t):
        """
        Compute correlation over the window ending at row t.
        Falls back to training correlation if window is too small.
        """
        window_data = df.iloc[max(0, t - self.window) : t]
        if len(window_data) < 5:
            return self.corr_matrix.values
        return window_data.corr().fillna(0).values

    def predict(self, df, horizon=1):
        """
        Rolling one-step-ahead prediction.

        For each timestep t, predict return at t+horizon for all assets.
        Prediction = correlation-weighted average of current returns.

        Args:
            df      : returns DataFrame (val or test set)
            horizon : 1, 5, or 10 days ahead

        Returns:
            predictions : np.array shape (n_valid_timesteps, num_assets)
            actuals     : np.array shape (n_valid_timesteps, num_assets)
        """
        returns  = df.values
        n, m     = returns.shape
        predictions = []
        actuals     = []

        for t in range(n - horizon):
            r_t = returns[t]

            C     = self._get_rolling_corr(df, t)
            C_pos = np.clip(C, 0, None)

            row_sums = C_pos.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            weights  = C_pos / row_sums

            pred   = weights @ r_t
            actual = returns[t + horizon]

            predictions.append(pred)
            actuals.append(actual)

        return np.array(predictions), np.array(actuals)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2: VAR (Vector Autoregression)
# ─────────────────────────────────────────────────────────────────────────────

class VARBaseline:
    """
    Vector Autoregression implemented manually using scipy.linalg.lstsq.

    Bypasses statsmodels entirely — numpy's lstsq has a known BLAS/LAPACK
    instability on Windows (DLASCLS error) that scipy's implementation avoids
    by using a different underlying driver ('gelsd' instead of 'gesdd').

    VAR is just OLS run separately for each asset:
        For each asset i:
            y_i = X @ beta_i
        where X is the matrix of all assets' lagged returns stacked as columns.

    Args:
        maxlags : number of lags (5 = one trading week)
    """

    def __init__(self, maxlags=5):
        self.maxlags     = maxlags
        self.coefs       = None   # shape: (num_assets, num_assets * maxlags)
        self.asset_names = None
        self.drop_cols   = []
        self.means       = None
        self.stds        = None

    def _build_lag_matrix(self, data, lags):
        """
        Build the design matrix X from lagged values.

        For data of shape (T, m) and lags=2:
            Row t of X = [data[t-1], data[t-2]]  (all assets, each lag)
            Row t of y = data[t]

        Returns:
            X : (T - lags, m * lags)
            y : (T - lags, m)
        """
        T, m = data.shape
        X_rows, y_rows = [], []

        for t in range(lags, T):
            # Concatenate lags: [t-1, t-2, ..., t-lags], each row is m values
            lag_row = np.concatenate([data[t - l] for l in range(1, lags + 1)])
            X_rows.append(lag_row)
            y_rows.append(data[t])

        X = np.array(X_rows)
        y = np.array(y_rows)

        # Safety filter: remove any row with NaN or inf
        # (shouldn't occur after load_returns cleans the data, but
        #  guards against edge cases like 0/0 returns on illiquid days)
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
        if not valid.all():
            print(f"  [VAR] Dropped {(~valid).sum()} rows with NaN/inf from lag matrix")
        return X[valid], y[valid]

    def fit(self, train_df):
        """
        Fit VAR via OLS using scipy.linalg.lstsq with gelsd driver.

        gelsd uses divide-and-conquer SVD — more numerically stable than
        numpy's gesdd on Windows BLAS builds.
        """
        from scipy.linalg import lstsq

        self.drop_cols = ["VIX"] if "VIX" in train_df.columns else []
        train_clean    = train_df.drop(columns=self.drop_cols)

        self.asset_names = train_clean.columns.tolist()

        # Standardize for numerical stability
        self.means = train_clean.mean()
        self.stds  = train_clean.std().replace(0, 1)
        data       = ((train_clean - self.means) / self.stds).values.astype(float)

        # Build lag matrix
        X, y = self._build_lag_matrix(data, self.maxlags)

        # Fit one OLS equation per asset using scipy lstsq
        # coefs shape: (num_assets * maxlags, num_assets)
        self.coefs, _, _, _ = lstsq(X, y, lapack_driver='gelsd')

        print(f"VAR fitted | lags={self.maxlags} | "
              f"assets={len(self.asset_names)}")
        return self

    def get_granger_summary(self):
        """
        Print the lag-1 coefficient matrix.
        coefs[:m] are the lag-1 coefficients — shape (m, m).
        Rows = target asset, Cols = source asset.
        """
        if self.coefs is None:
            raise RuntimeError("Call fit() first.")

        m = len(self.asset_names)
        # First m rows of coefs.T are lag-1 coefficients
        coef_lag1 = self.coefs[:m, :].T    # (m, m)
        df = pd.DataFrame(
            coef_lag1,
            index=self.asset_names,
            columns=self.asset_names
        )
        print("\nVAR Lag-1 Coefficient Matrix (rows=target, cols=source):")
        print(df.round(4))
        return df

    def predict(self, df, horizon=1):
        """
        Rolling forecast using the fitted coefficient matrix.
        """
        df_clean = df.drop(columns=self.drop_cols)
        df_std   = (df_clean - self.means) / self.stds
        data     = df_std.values.astype(float)
        n        = len(data)
        lag      = self.maxlags

        predictions = []
        actuals     = []

        for t in range(lag, n - horizon):
            # Build lag vector for this timestep
            lag_vec = np.concatenate(
                [data[t - l] for l in range(1, lag + 1)]
            )  # (m * lags,)

            # One-step prediction: lag_vec @ coefs
            # For multi-step: iteratively apply the model
            pred_std = lag_vec @ self.coefs   # (m,)

            if horizon > 1:
                # Iteratively forecast: feed prediction back as new lag
                current_data = data[t - lag + 1 : t].tolist()
                current_data.append(pred_std)

                for _ in range(horizon - 1):
                    lag_vec  = np.concatenate(current_data[-lag:])
                    pred_std = lag_vec @ self.coefs
                    current_data.append(pred_std)

            actual = data[t + horizon]

            predictions.append(pred_std)
            actuals.append(actual)

        # Un-standardize back to return scale
        std_vals    = self.stds.values
        mean_vals   = self.means.values
        preds_arr   = np.array(predictions) * std_vals + mean_vals
        actuals_arr = np.array(actuals)     * std_vals + mean_vals

        return preds_arr, actuals_arr


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 3: Static GCN
# ─────────────────────────────────────────────────────────────────────────────

class StaticGCN(nn.Module):
    """
    Standard GCN with fixed weight matrices — no temporal evolution.

    Isolates the contribution of EvolveGCN-H's GRU mechanism.
    Same architecture depth, width, and interface as EvolveGCNH.

    If EvolveGCN-H >> StaticGCN:
        → the temporal evolution of W is genuinely capturing regime changes
    If they're similar:
        → the dynamic W isn't helping — reconsider the GRU design

    Args:
        node_features : input features per node (4)
        hidden_dim    : internal representation size (64)
        num_layers    : GCN depth (2)
        dropout       : regularization (0.3)
    """

    def __init__(self, node_features, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()

        self.convs   = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for i in range(num_layers):
            in_dim = node_features if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_dim, hidden_dim))

        # Same prediction heads as EvolveGCNH — fair comparison
        self.predictors = nn.ModuleDict({
            "t1":  self._make_predictor(hidden_dim),
            "t5":  self._make_predictor(hidden_dim),
            "t10": self._make_predictor(hidden_dim),
        })

    def _make_predictor(self, hidden_dim):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, snapshots):
        """
        Identical interface to EvolveGCNH.forward().
        Key difference: W never changes between timesteps.
        The graph structure (edge weights) still changes each week,
        but the transformation parameters don't adapt to regimes.
        """
        preds = {"t1": [], "t5": [], "t10": []}

        for x, edge_index, edge_weight in snapshots:
            h = x
            for conv in self.convs:
                h = conv(h, edge_index, edge_weight)
                h = torch.relu(h)
                h = self.dropout(h)

            for horizon, predictor in self.predictors.items():
                preds[horizon].append(predictor(h))

        return {k: torch.stack(v, dim=0) for k, v in preds.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation utilities
# ─────────────────────────────────────────────────────────────────────────────

def mean_squared_error(predictions, actuals):
    """Average squared error. Penalizes large errors heavily."""
    return np.mean((predictions - actuals) ** 2)


def mean_absolute_error(predictions, actuals):
    """Average absolute error. Same units as returns — more interpretable."""
    return np.mean(np.abs(predictions - actuals))


def directional_accuracy(predictions, actuals):
    """
    Fraction of predictions with correct sign (up/down direction).
    50% = coin flip. Anything above ~53% is meaningful for daily returns.
    """
    correct = np.sign(predictions) == np.sign(actuals)
    return float(np.mean(correct))


def evaluate(predictions, actuals, label="Model"):
    """
    Compute and print all three metrics. Returns dict for paper tables.

    Args:
        predictions : np.array shape (n, num_assets)
        actuals     : np.array shape (n, num_assets)
        label       : string identifier for printing
    """
    mse  = mean_squared_error(predictions, actuals)
    mae  = mean_absolute_error(predictions, actuals)
    dacc = directional_accuracy(predictions, actuals)

    print(f"\n{'─' * 45}")
    print(f"  {label}")
    print(f"{'─' * 45}")
    print(f"  MSE                : {mse:.8f}")
    print(f"  MAE                : {mae:.8f}")
    print(f"  Directional Acc    : {dacc:.4f}  ({dacc*100:.1f}%)")

    return {"model": label, "mse": mse, "mae": mae,
            "directional_accuracy": dacc}


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check — run: python models/baselines.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import os

    REAL_DATA_PATH = "data/processed/returns_matrix.csv"
    USE_REAL_DATA  = os.path.exists(REAL_DATA_PATH)

    # ── Load data ─────────────────────────────────────────────────────────
    if USE_REAL_DATA:
        print("Loading real returns data...")
        df = load_returns(REAL_DATA_PATH)
    else:
        print("Real data not found — generating synthetic data for testing...")

        np.random.seed(42)
        dates  = pd.date_range("2015-01-01", "2024-12-31", freq="B")
        assets = ["BTC-USD", "ETH-USD", "SPY", "EEM", "LQD",
                  "HYG", "TLT", "GLD", "USO", "DX-Y.NYB"]

        market_factor = np.random.randn(len(dates)) * 0.01
        betas         = np.array([1.5, 1.8, 1.0, 1.2, 0.3,
                                   0.6, -0.2, 0.1, 0.8, -0.4])
        noise         = np.random.randn(len(dates), len(assets)) * 0.02
        returns_data  = market_factor[:, None] * betas[None, :] + noise

        df = pd.DataFrame(returns_data, index=dates, columns=assets)
        print(f"Synthetic data: {df.shape[0]} days × {df.shape[1]} assets")

    # ── Split ─────────────────────────────────────────────────────────────
    print("\nSplitting data...")
    train, val, test = train_val_test_split(df)

    results_table = []

    # ── Baseline 1: Rolling Correlation ───────────────────────────────────
    print("\n" + "=" * 50)
    print("BASELINE 1: Rolling Correlation")
    print("=" * 50)

    roll_model = RollingCorrelationBaseline(window=60)
    roll_model.fit(train)

    for horizon, label in [(1, "t+1"), (5, "t+5"), (10, "t+10")]:
        preds, actuals = roll_model.predict(val, horizon=horizon)
        metrics = evaluate(preds, actuals,
                           label=f"Rolling Correlation — {label}")
        results_table.append(metrics)

    # ── Baseline 2: VAR ───────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("BASELINE 2: VAR")
    print("=" * 50)

    var_model = VARBaseline(maxlags=5)
    var_model.fit(train)
    var_model.get_granger_summary()

    for horizon, label in [(1, "t+1"), (5, "t+5"), (10, "t+10")]:
        preds, actuals = var_model.predict(val, horizon=horizon)
        metrics = evaluate(preds, actuals, label=f"VAR — {label}")
        results_table.append(metrics)

    # ── Baseline 3: Static GCN ────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("BASELINE 3: Static GCN (forward pass shape check)")
    print("=" * 50)

    # Build fake graph snapshots — same structure as build_graphs.py will produce
    # VIX is included as a node feature here (column 10), not excluded like in VAR
    num_nodes = df.shape[1]
    num_edges = 25
    snapshots = []
    for _ in range(20):
        x           = torch.randn(num_nodes, 4)
        src         = torch.randint(0, num_nodes, (num_edges,))
        dst         = torch.randint(0, num_nodes, (num_edges,))
        edge_index  = torch.stack([src, dst], dim=0)
        edge_weight = torch.rand(num_edges)
        snapshots.append((x, edge_index, edge_weight))

    static_gcn = StaticGCN(node_features=4, hidden_dim=64)
    static_gcn.eval()

    with torch.no_grad():
        preds = static_gcn(snapshots)

    for horizon, pred in preds.items():
        status = "PASS" if pred.shape == (20, num_nodes, 1) else "FAIL"
        print(f"  StaticGCN {horizon}: {str(list(pred.shape)):20s} — {status}")

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("SUMMARY — Validation Set Results")
    print("=" * 50)
    summary = pd.DataFrame(results_table)
    print(summary.to_string(index=False))
    print("\nAll baselines operational.")