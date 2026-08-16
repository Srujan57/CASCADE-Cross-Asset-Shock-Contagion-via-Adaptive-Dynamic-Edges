"""
scripts/build_graphs.py

Converts returns_matrix.csv into a sequence of weekly graph snapshots
ready to feed directly into EvolveGCNH.forward().

Each snapshot is a tuple: (x, edge_index, edge_weight, date)
    x           : node features (num_assets, 4)
                  [return_t, 30d_vol, VIX_t, regime_label]
    edge_index  : connectivity (2, num_edges)
    edge_weight : correlation strengths (num_edges,)
    date        : snapshot date (for alignment and debugging)

Current edge weights:  rolling Pearson correlation (60-day window)
Swap-in when ready:    DCC-GARCH  → call swap_in_dcc_edges()
Current regime labels: placeholder 0 (calm) until data/processed/regime_labels.csv exists
Swap-in when ready:    VIX-threshold regime labels (scripts/fix_regime_labels.py)
                        → call swap_in_regime_labels()
                        NOTE: not an HMM, despite earlier comments in this file saying
                        so — the actual method is a simple VIX-level threshold rule
                        (calm/stress/crisis via VIX < 20 / 20-30 / > 30). A separate,
                        unrelated K-means-based regime detector also exists in
                        scripts/phase2_econometrics.py but is NOT used by this pipeline
                        (see that script's own note) — do not confuse the two.

Run: python scripts/build_graphs.py
"""

import numpy as np
import pandas as pd
import torch
import pickle
import yaml
import os


# ─────────────────────────────────────────────────────────────────────────────
# Config & data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path="config.yaml"):
    """
    Load config.yaml into a Python dict.
    safe_load prevents arbitrary code execution from malicious YAML.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def load_returns(path):
    """
    Load and clean returns_matrix.csv.
    ffill fills intra-series gaps.
    dropna removes early ETH rows (no ETH data before Nov 2017).
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df = df.ffill()
    df = df.dropna()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_rolling_vol(returns_df, window=30):
    """
    30-day rolling standard deviation per asset.

    .rolling(window).std() computes std of the previous `window` rows
    for each cell. First (window-1) rows become NaN — ffill/bfill fills them.

    Returns DataFrame same shape as returns_df.
    """
    vol = returns_df.rolling(window=window).std()
    vol = vol.ffill().bfill()
    return vol


def get_weekly_dates(returns_df, update_freq="W"):
    """
    Get one snapshot date per week — snapped to actual trading days.

    pd.date_range with freq="W" gives every Sunday. We find the last
    actual trading day on or before each Sunday. This guarantees every
    snapshot date exists in the returns index.
    """
    date_min     = returns_df.index.min()
    date_max     = returns_df.index.max()
    weekly       = pd.date_range(start=date_min, end=date_max, freq=update_freq)
    trading_days = returns_df.index

    snapshot_dates = []
    for d in weekly:
        valid = trading_days[trading_days <= d]
        if len(valid) > 0:
            snapshot_dates.append(valid[-1])

    return sorted(set(snapshot_dates))


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_edge_index_and_weights(corr_matrix, threshold=0.3):
    """
    Convert a correlation matrix into edge_index and edge_weight tensors.

    Only keeps edges where |correlation| > threshold.
    Each undirected edge is listed twice (i→j and j→i) because
    PyG's message passing needs both directions for symmetric aggregation.

    Args:
        corr_matrix : np.array (num_assets, num_assets)
        threshold   : minimum |r| to include an edge

    Returns:
        edge_index  : torch.LongTensor  shape (2, num_edges)
        edge_weight : torch.FloatTensor shape (num_edges,)
    """
    n = corr_matrix.shape[0]
    src_list, dst_list, weight_list = [], [], []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue   # skip self-loops — GCNLayer adds them internally
            if abs(corr_matrix[i, j]) > threshold:
                src_list.append(i)
                dst_list.append(j)
                # Absolute correlation as weight — sign is in the returns themselves
                weight_list.append(abs(float(corr_matrix[i, j])))

    # Safety: if no edges pass the threshold, add minimal connectivity
    # (very rare — would only happen in extreme low-volatility periods)
    if len(src_list) == 0:
        for i in range(n):
            src_list.append(i)
            dst_list.append(i)
            weight_list.append(1.0)
        print("  Warning: no edges exceeded threshold — using self-loops as fallback")

    edge_index  = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float)

    return edge_index, edge_weight


def build_node_features(returns_row, vol_row, vix_value, regime=0.0):
    """
    Build node feature matrix for one snapshot.

    Feature vector per node: [return_t, 30d_vol, VIX_t, regime_label]

    Args:
        returns_row : pd.Series — returns for all assets on snapshot date
        vol_row     : pd.Series — rolling vol for all assets on snapshot date
        vix_value   : float — VIX level on this date (same for all nodes)
        regime      : float — VIX-threshold regime label, 0=calm/1=stress/2=crisis
                      (0=placeholder until data/processed/regime_labels.csv exists;
                      NOT an HMM output, see module docstring)

    Returns:
        x : torch.FloatTensor (num_assets, 4)
    """
    features = []
    for asset in returns_row.index:
        r    = float(returns_row[asset])
        v    = float(vol_row[asset])
        feat = [r, v, vix_value, regime]
        features.append(feat)

    return torch.tensor(features, dtype=torch.float)


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────

def build_snapshots(returns_df, config, regime_labels=None, verbose=True):
    """
    Build the full sequence of weekly graph snapshots.

    For each weekly date:
        1. Compute rolling correlation over past `window` days → edges
        2. Build node features for that date
        3. Pack into (x, edge_index, edge_weight, date) tuple

    Args:
        returns_df    : cleaned returns DataFrame (dates × assets)
        config        : loaded config.yaml dict
        regime_labels : optional pd.Series — VIX-threshold regime labels indexed by
                        date (scripts/fix_regime_labels.py), NOT an HMM output.
                        If None, regime column is set to 0.0 (placeholder)
        verbose       : print progress

    Returns:
        snapshots   : list of (x, edge_index, edge_weight, date)
        asset_names : list of asset names in node index order
    """
    window    = config["graph"]["rolling_window_days"]
    threshold = config["graph"]["corr_threshold"]
    vol_win   = config["features"]["vol_window_days"]
    vix_col   = "VIX"

    # All asset columns excluding VIX (VIX is a feature, not a node)
    asset_cols = [c for c in returns_df.columns if c != vix_col]

    # Precompute rolling vol for all dates at once — much faster than per-snapshot
    vol_df = compute_rolling_vol(returns_df[asset_cols], window=vol_win)

    snapshot_dates = get_weekly_dates(
        returns_df, update_freq=config["graph"]["update_frequency"]
    )

    if verbose:
        print(f"Building snapshots...")
        print(f"  Total candidate dates : {len(snapshot_dates)}")
        print(f"  Correlation window    : {window} days")
        print(f"  Edge threshold        : |r| > {threshold}")
        print(f"  Vol window            : {vol_win} days")
        print(f"  Assets (nodes)        : {len(asset_cols)}")
        print(f"  Regime labels         : {'VIX-threshold' if regime_labels is not None else 'placeholder (0)'}")
        print()

    snapshots = []
    skipped   = 0

    for i, date in enumerate(snapshot_dates):

        # Need at least `window` days of history before this date
        history = returns_df.index[returns_df.index <= date]
        if len(history) < window:
            skipped += 1
            continue

        # ── Rolling correlation window ─────────────────────────────────
        # .tail(window) gives the last `window` rows up to and including `date`
        window_data = returns_df[asset_cols].loc[
            returns_df.index <= date
        ].tail(window)

        # Pairwise Pearson correlation — shape (num_assets, num_assets)
        corr_matrix = window_data.corr().values
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # ── Edges ──────────────────────────────────────────────────────
        edge_index, edge_weight = build_edge_index_and_weights(
            corr_matrix, threshold=threshold
        )

        # ── Node features ──────────────────────────────────────────────
        returns_row = returns_df[asset_cols].loc[date]
        vol_row     = vol_df.loc[date]
        vix_value   = float(returns_df[vix_col].loc[date]) \
                      if vix_col in returns_df.columns else 0.0

        regime = 0.0
        if regime_labels is not None and date in regime_labels.index:
            regime = float(regime_labels.loc[date])

        x = build_node_features(returns_row, vol_row, vix_value, regime)

        snapshots.append((x, edge_index, edge_weight, date))

        if verbose and (i + 1) % 50 == 0:
            print(f"  [{i+1:>4}/{len(snapshot_dates)}] {date.date()} | "
                  f"nodes={x.shape[0]:>2} | "
                  f"edges={edge_weight.shape[0]:>4}")

    if verbose:
        print(f"\nComplete: {len(snapshots)} snapshots built, "
              f"{skipped} skipped (insufficient history)")
        if len(snapshots) > 0:
            edge_counts = [s[2].shape[0] for s in snapshots]
            print(f"Edge count — min: {min(edge_counts):>4} | "
                  f"mean: {np.mean(edge_counts):.1f} | "
                  f"max: {max(edge_counts):>4}")

    return snapshots, asset_names if 'asset_names' in dir() else asset_cols


# ─────────────────────────────────────────────────────────────────────────────
# Train / val / test split for snapshots
# ─────────────────────────────────────────────────────────────────────────────

def split_snapshots(snapshots, config):
    """
    Split snapshot list chronologically into train / val / test.

    Uses the same date boundaries as config.yaml so the split is
    consistent with baselines.py and the GNN training loop.

    Returns three lists of snapshots.
    """
    train_end = pd.Timestamp(config["dates"]["train_end"])
    val_end   = pd.Timestamp(config["dates"]["val_end"])

    train = [(x, ei, ew, d) for x, ei, ew, d in snapshots if d <= train_end]
    val   = [(x, ei, ew, d) for x, ei, ew, d in snapshots
             if d > train_end and d <= val_end]
    test  = [(x, ei, ew, d) for x, ei, ew, d in snapshots if d > val_end]

    print(f"Split: train={len(train)} | val={len(val)} | test={len(test)} snapshots")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Swap-in functions — upgrade placeholders with real outputs
# ─────────────────────────────────────────────────────────────────────────────

def swap_in_regime_labels(snapshots, regime_labels_path):
    """
    Replace placeholder regime labels (0.0) with VIX-threshold regime labels
    (scripts/fix_regime_labels.py). NOT an HMM output — see module docstring.

    Call this after data/processed/regime_labels.csv exists.
    The rest of the pipeline (model, training) needs no changes.

    Args:
        snapshots          : existing snapshot list
        regime_labels_path : path to regime_labels.csv

    Returns:
        updated snapshot list with real regime labels in x[:, 3]
    """
    regime_df = pd.read_csv(regime_labels_path, index_col=0, parse_dates=True)
    regime_series = regime_df.iloc[:, 0].astype(float)

    updated = []
    swapped = 0
    for x, edge_index, edge_weight, date in snapshots:
        if date in regime_series.index:
            x_new = x.clone()
            x_new[:, 3] = float(regime_series.loc[date])
            updated.append((x_new, edge_index, edge_weight, date))
            swapped += 1
        else:
            updated.append((x, edge_index, edge_weight, date))

    print(f"Regime labels: swapped {swapped}/{len(snapshots)} snapshots")
    return updated


def swap_in_dcc_edges(snapshots, dcc_path, asset_names, threshold=0.3):
    """
    Replace rolling correlation edge weights with DCC-GARCH correlations.

    Ryan's DCC dict is keyed by DAILY dates. Snapshots are WEEKLY.
    We find the nearest DCC date within 7 days of each snapshot date
    rather than requiring exact match (which would silently skip everything).
    """
    with open(dcc_path, "rb") as f:
        dcc_data = pickle.load(f)

    dcc_dates = sorted(dcc_data.keys())
    dcc_dates_ts = pd.DatetimeIndex(dcc_dates)

    updated = []
    swapped = 0
    no_match = 0

    for x, edge_index, edge_weight, date in snapshots:
        # Find nearest DCC date within 7 calendar days
        diffs = abs(dcc_dates_ts - date)
        nearest_idx = diffs.argmin()
        nearest_diff = diffs[nearest_idx].days

        if nearest_diff <= 7:
            dcc_matrix = np.nan_to_num(
                dcc_data[dcc_dates[nearest_idx]], nan=0.0
            )
            ei, ew = build_edge_index_and_weights(
                dcc_matrix, threshold=threshold
            )
            updated.append((x, ei, ew, date))
            swapped += 1
        else:
            # No DCC data within 7 days — keep rolling correlation
            updated.append((x, edge_index, edge_weight, date))
            no_match += 1

    print(f"DCC edges: swapped {swapped}/{len(snapshots)} snapshots "
          f"({no_match} kept rolling corr — no nearby DCC date)")
    return updated


# ─────────────────────────────────────────────────────────────────────────────
# Save & load
# ─────────────────────────────────────────────────────────────────────────────

def save_snapshots(snapshots, asset_names, path):
    """
    Save snapshots to disk as a pickle file.

    Pickle serializes any Python object to binary.
    Always use 'wb' (write binary) / 'rb' (read binary).

    Saves a dict with both the snapshot list and the asset name ordering
    so you always know which node index corresponds to which asset.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"snapshots": snapshots, "asset_names": asset_names}
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    size_mb = os.path.getsize(path) / 1e6
    print(f"Saved {len(snapshots)} snapshots → {path} ({size_mb:.1f} MB)")


def load_snapshots(path):
    """
    Load snapshots from disk.
    Returns (snapshots, asset_names).
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    snapshots   = payload["snapshots"]
    asset_names = payload["asset_names"]
    print(f"Loaded {len(snapshots)} snapshots | assets: {asset_names}")
    return snapshots, asset_names


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Paths ─────────────────────────────────────────────────────────────
    CONFIG_PATH   = "config.yaml"
    RETURNS_PATH  = "data/processed/returns_matrix.csv"
    OUTPUT_PATH   = "data/processed/graph_snapshots.pkl"

    # Optional — swap in when Ryan delivers these
    REGIME_PATH   = "data/processed/regime_labels.csv"
    DCC_PATH      = "data/processed/dcc_correlations.pkl"

    # ── Load ──────────────────────────────────────────────────────────────
    config  = load_config(CONFIG_PATH)
    returns = load_returns(RETURNS_PATH)

    print(f"Returns loaded: {returns.shape[0]} days × {returns.shape[1]} assets")
    print(f"Date range    : {returns.index[0].date()} → {returns.index[-1].date()}")
    print()

    # ── Regime labels (optional) ──────────────────────────────────────────
    regime_labels = None
    if os.path.exists(REGIME_PATH) and os.path.getsize(REGIME_PATH) > 0:
        regime_df     = pd.read_csv(REGIME_PATH, index_col=0, parse_dates=True)
        regime_labels = regime_df.iloc[:, 0].astype(float)
        print(f"Regime labels loaded: {len(regime_labels)} days")
    else:
        print("Regime labels not ready — using placeholder (0 = calm)")
    print()

    # ── Build snapshots ───────────────────────────────────────────────────
    snapshots, asset_names = build_snapshots(
        returns_df    = returns,
        config        = config,
        regime_labels = regime_labels,
        verbose       = True
    )

    print(f"\nAsset → node index mapping:")
    for i, a in enumerate(asset_names):
        print(f"  {i:>2}: {a}")

    # ── Optionally swap in DCC edges ──────────────────────────────────────
    if os.path.exists(DCC_PATH) and os.path.getsize(DCC_PATH) > 0:
        print("\nDCC correlations found — swapping in edge weights...")
        snapshots = swap_in_dcc_edges(snapshots, DCC_PATH, asset_names,
                                      threshold=config["graph"]["corr_threshold"])
    else:
        print("\nDCC correlations not ready — keeping rolling correlation edges")

    # ── Split ─────────────────────────────────────────────────────────────
    print()
    train, val, test = split_snapshots(snapshots, config)

    # ── Validate one snapshot ─────────────────────────────────────────────
    print("\nValidating snapshot structure...")
    x, edge_index, edge_weight, date = snapshots[len(train) // 2]
    print(f"  Sample snapshot date : {date.date()}")
    print(f"  x shape              : {list(x.shape)}  (should be [10, 4])")
    print(f"  edge_index shape     : {list(edge_index.shape)}")
    print(f"  edge_weight shape    : {list(edge_weight.shape)}")
    print(f"  x dtype              : {x.dtype}")
    print(f"  edge_index dtype     : {edge_index.dtype}")
    print(f"  NaN in x             : {torch.isnan(x).any().item()}")
    print(f"  NaN in edge_weight   : {torch.isnan(edge_weight).any().item()}")

    # Check shapes are consistent with EvolveGCNH expectations
    n_assets = len(asset_names)
    assert x.shape == (n_assets, 4), \
        f"x shape mismatch: expected ({n_assets}, 4), got {x.shape}"
    assert edge_index.shape[0] == 2, \
        f"edge_index should have 2 rows, got {edge_index.shape[0]}"
    assert edge_weight.shape[0] == edge_index.shape[1], \
        "edge_weight length must match number of edges"
    assert not torch.isnan(x).any(), "NaN found in node features"
    assert not torch.isnan(edge_weight).any(), "NaN found in edge weights"
    print("\nAll validation checks PASSED")

    # ── Save ──────────────────────────────────────────────────────────────
    save_snapshots(snapshots, asset_names, OUTPUT_PATH)

    print(f"\nDone. Next step: train.py loads graph_snapshots.pkl and runs EvolveGCNH.")
