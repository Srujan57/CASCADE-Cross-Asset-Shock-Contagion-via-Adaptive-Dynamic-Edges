"""
scripts/evaluate.py

All four CASCADE experiments + statistical validation.

Experiments:
    1. Predictive accuracy  — GNN vs VAR vs rolling corr vs static GCN
    2. Shock propagation    — inject shock at node X, measure spillover
    3. Regime conditioning  — contagion structure in calm vs crisis regimes
    4. Crypto structural break — BTC shock propagation pre vs post 2020

Statistical tests:
    - Diebold-Mariano test  — formally compares GNN vs each baseline
    - Bootstrap CI          — 95% confidence intervals on all metrics

Outputs saved to results/:
    experiment1_accuracy.csv
    experiment2_shock_propagation.csv
    experiment3_regime_analysis.csv
    experiment4_structural_break.csv
    diebold_mariano_results.csv

Run AFTER models/train.py:
    python scripts/evaluate.py
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import torch
import yaml
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.evolvegcn import EvolveGCNH
from models.baselines  import (RollingCorrelationBaseline, VARBaseline,
                                StaticGCN, evaluate as baseline_evaluate)
from models.train      import (load_config, load_snapshots, load_returns,
                                build_targets, split_by_date,
                                to_model_input, compute_metrics,
                                load_checkpoint)


# ─────────────────────────────────────────────────────────────────────────────
# Diebold-Mariano test
# ─────────────────────────────────────────────────────────────────────────────

def diebold_mariano_test(preds1, preds2, actuals, h=1):
    """
    Diebold-Mariano test: are two forecasts significantly different in accuracy?

    H0: equal predictive accuracy (E[d_t] = 0)
    H1: model 1 is more accurate (E[d_t] < 0)

    d_t = L(e1_t) - L(e2_t)  where L = squared error loss
    DM statistic uses Harvey, Leybourne & Newbold (1997) small-sample correction.

    Args:
        preds1, preds2 : np.array (T, N) — predictions from two models
        actuals        : np.array (T, N) — ground truth
        h              : forecast horizon (for Newey-West bandwidth)

    Returns:
        dm_stat : float — test statistic (negative = model 1 is better)
        p_value : float — two-sided p-value
        better  : str   — which model is better
    """
    from scipy import stats

    e1 = (preds1 - actuals) ** 2   # squared errors model 1
    e2 = (preds2 - actuals) ** 2   # squared errors model 2
    d  = e1 - e2                   # loss differential: shape (T, N)

    # Average across assets for a single series
    d_mean_series = d.mean(axis=1)  # (T,)
    T   = len(d_mean_series)
    d_bar = d_mean_series.mean()

    # Newey-West variance estimator for serial correlation
    # bandwidth = h (forecast horizon)
    gamma_0 = np.var(d_mean_series, ddof=1)
    nw_var  = gamma_0
    for k in range(1, h + 1):
        gamma_k = np.cov(d_mean_series[k:], d_mean_series[:-k])[0, 1]
        nw_var += 2 * (1 - k / (h + 1)) * gamma_k

    nw_var = max(nw_var, 1e-10)  # guard against zero variance

    # HLN small-sample correction
    hln_correction = np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
    dm_stat = d_bar / (np.sqrt(nw_var / T) / hln_correction)

    # Two-sided p-value from t-distribution (HLN)
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=T - 1))

    better = "GNN" if dm_stat < 0 else "Baseline"
    return float(dm_stat), float(p_value), better


def bootstrap_ci(preds, actuals, metric_fn, n_boot=1000, alpha=0.05):
    """
    Bootstrap 95% confidence interval for a metric.

    Resamples (T, N) prediction/actual pairs with replacement.
    Returns (lower, upper) CI.
    """
    T = preds.shape[0]
    boot_stats = []
    for _ in range(n_boot):
        idx = np.random.choice(T, size=T, replace=True)
        boot_stats.append(metric_fn(preds[idx], actuals[idx]))
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


# ─────────────────────────────────────────────────────────────────────────────
# Static GCN baseline training
#
# INTEGRITY FIX (see results/DATA_INTEGRITY_NOTES.md): previous versions of
# this script instantiated StaticGCN and called .eval() WITHOUT ever training
# it — every "Static GCN" number in experiment1_accuracy.csv and
# diebold_mariano_results.csv was therefore a randomly-initialized network,
# not a trained baseline. Comparing CASCADE to random weights is not a
# meaningful ablation and produces a misleadingly large "improvement" number
# (MSE ~40x worse than every other baseline is the signature of an untrained
# network, not evidence about the value of temporal edge evolution). This
# trains StaticGCN the same way models/train.py trains EvolveGCN-H: full-batch
# gradient descent on the training snapshots, early stopping on validation
# loss, before it is ever used for evaluation.
# ─────────────────────────────────────────────────────────────────────────────

def train_static_gcn(static_gcn, train_snaps, tv_snaps, val_local,
                     targets, train_pos, val_pos, tv_pos, config):
    optimizer = torch.optim.Adam(
        static_gcn.parameters(), lr=config["training"]["lr"], weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    epochs    = config["training"]["epochs"]
    patience  = config["training"]["early_stopping_patience"]
    eval_freq = 5
    best_val_loss    = float("inf")
    best_state        = None
    patience_counter  = 0
    train_indices     = list(range(len(train_pos)))

    print("\nTraining Static GCN baseline (was previously left untrained)...")
    for epoch in range(1, epochs + 1):
        static_gcn.train()
        optimizer.zero_grad()
        train_preds = static_gcn(train_snaps)
        loss = sum(
            torch.nn.functional.mse_loss(
                train_preds[h][train_indices, :, 0], targets[h][train_pos]
            ) for h in ["t1", "t5", "t10"]
        ) / 3
        loss.backward()
        torch.nn.utils.clip_grad_norm_(static_gcn.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % eval_freq == 0:
            static_gcn.eval()
            with torch.no_grad():
                tv_preds = static_gcn(tv_snaps)
                val_loss = sum(
                    torch.nn.functional.mse_loss(
                        tv_preds[h][val_local, :, 0],
                        targets[h][[tv_pos[i] for i in val_local]]
                    ) for h in ["t1", "t5", "t10"]
                ) / 3
                val_loss_f = float(val_loss)
            scheduler.step(val_loss_f)

            if val_loss_f < best_val_loss:
                best_val_loss = val_loss_f
                best_state = {k: v.clone() for k, v in static_gcn.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience // eval_freq:
                    print(f"  Static GCN early stopping at epoch {epoch}")
                    break

    if best_state is not None:
        static_gcn.load_state_dict(best_state)
    static_gcn.eval()
    print(f"  Static GCN trained. Best val loss: {best_val_loss:.8f}")
    return static_gcn


# ─────────────────────────────────────────────────────────────────────────────
# Shock injection
# ─────────────────────────────────────────────────────────────────────────────

def inject_shock(model, snapshots, shock_node_idx, shock_return,
                 start_pos, n_context=20):
    """
    Inject a return shock at one node and measure predicted spillover.

    Protocol:
        1. Use the previous n_context snapshots to warm up W (evolve through history)
        2. Modify the target snapshot: set x[shock_node_idx, 0] = shock_return
        3. Run one forward step with the shocked snapshot
        4. Also run with the unmodified snapshot (baseline)
        5. Return (shocked_prediction - baseline_prediction) per node

    Args:
        model          : trained EvolveGCNH (eval mode)
        snapshots      : list of (x, ei, ew) tuples (no dates)
        shock_node_idx : which node gets the shock (e.g., 7 = BTC)
        shock_return   : what return to inject (e.g., -0.30 = -30%)
        start_pos      : position of the shock snapshot in snapshots list
        n_context      : number of prior snapshots to warm up W

    Returns:
        spillover : np.array (N,) — predicted change per node due to shock
        shocked_pred  : np.array (N,) — predicted returns with shock
        baseline_pred : np.array (N,) — predicted returns without shock
    """
    model.eval()
    context_start = max(0, start_pos - n_context)
    context_snaps = snapshots[context_start:start_pos]
    target_snap   = snapshots[start_pos]

    x, ei, ew = target_snap

    # Shocked version
    x_shocked       = x.clone()
    x_shocked[shock_node_idx, 0] = shock_return  # overwrite return feature

    with torch.no_grad():
        # Baseline: warm up then predict
        baseline_seq   = context_snaps + [(x, ei, ew)]
        baseline_preds = model(baseline_seq)
        baseline_pred  = baseline_preds["t1"][-1, :, 0].numpy()  # last timestep

        # Shocked: warm up then predict with shock
        shocked_seq   = context_snaps + [(x_shocked, ei, ew)]
        shocked_preds = model(shocked_seq)
        shocked_pred  = shocked_preds["t1"][-1, :, 0].numpy()

    spillover = shocked_pred - baseline_pred
    return spillover, shocked_pred, baseline_pred


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Predictive Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def experiment1_accuracy(gnn_preds, baselines_preds, actuals,
                         asset_names, horizon_key="t1"):
    """
    Compare GNN vs all baselines on MSE, MAE, directional accuracy.
    Run Diebold-Mariano test for each GNN vs baseline pair.
    """
    print(f"\n{'─'*55}")
    print(f"  Experiment 1: Predictive Accuracy ({horizon_key})")
    print(f"{'─'*55}")

    results = []
    dm_results = []

    # GNN metrics
    gnn_m = compute_metrics(gnn_preds, actuals)
    gnn_mse_ci = bootstrap_ci(gnn_preds, actuals,
                               lambda p,a: float(np.mean((p-a)**2)))
    results.append({
        "model": "EvolveGCN-H (CASCADE)",
        "mse": gnn_m["mse"],
        "mse_ci_lower": gnn_mse_ci[0],
        "mse_ci_upper": gnn_mse_ci[1],
        "mae": gnn_m["mae"],
        "directional_accuracy": gnn_m["directional_accuracy"],
    })

    for baseline_name, baseline_preds in baselines_preds.items():
        # Align shapes — baselines may have different T due to lag warm-up
        T_min = min(gnn_preds.shape[0], baseline_preds.shape[0],
                    actuals.shape[0])
        g = gnn_preds[-T_min:]
        b = baseline_preds[-T_min:]
        a = actuals[-T_min:]

        bm = compute_metrics(b, a)
        bm_ci = bootstrap_ci(b, a, lambda p,ac: float(np.mean((p-ac)**2)))

        results.append({
            "model": baseline_name,
            "mse": bm["mse"],
            "mse_ci_lower": bm_ci[0],
            "mse_ci_upper": bm_ci[1],
            "mae": bm["mae"],
            "directional_accuracy": bm["directional_accuracy"],
        })

        dm_stat, dm_pval, better = diebold_mariano_test(g, b, a, h=1)
        dm_results.append({
            "comparison": f"GNN vs {baseline_name}",
            "dm_statistic": dm_stat,
            "p_value": dm_pval,
            "better_model": better,
            "significant_at_5pct": dm_pval < 0.05,
        })

        sig = "***" if dm_pval < 0.01 else ("**" if dm_pval < 0.05 else "")
        print(f"  GNN vs {baseline_name:20s}: "
              f"DM={dm_stat:+.3f} p={dm_pval:.4f} {sig}")

    df_results = pd.DataFrame(results)
    df_dm      = pd.DataFrame(dm_results)

    print(f"\n  Accuracy summary ({horizon_key}):")
    for _, row in df_results.iterrows():
        print(f"  {row['model']:30s} MSE={row['mse']:.8f} "
              f"Dir={row['directional_accuracy']*100:.1f}%")

    return df_results, df_dm


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Shock Propagation
# ─────────────────────────────────────────────────────────────────────────────

def experiment2_shock_propagation(model, all_snaps, snapshots, valid_indices,
                                   all_pos, asset_names, shock_events):
    """
    For each shock event, inject the actual shock return at the source node
    and measure predicted spillover to all other nodes.

    shock_events: list of dicts with keys:
        name       : event name (e.g., "FTX Collapse")
        date       : pd.Timestamp of the event
        source     : asset name of shock origin (e.g., "BTC")
        shock_val  : actual return on that day (e.g., -0.30)
    """
    print(f"\n{'─'*55}")
    print(f"  Experiment 2: Shock Propagation")
    print(f"{'─'*55}")

    results = []
    btc_idx = asset_names.index("BTC") if "BTC" in asset_names else 7

    for event in shock_events:
        event_date   = pd.Timestamp(event["date"])
        source_asset = event.get("source", "BTC")
        shock_val    = event.get("shock_val", -0.25)

        # Find the snapshot closest to this event date
        snap_dates = [snapshots[valid_indices[p]][3] for p in all_pos]
        diffs      = [abs((d - event_date).days) for d in snap_dates]
        closest    = int(np.argmin(diffs))

        if diffs[closest] > 14:
            print(f"  {event['name']}: no snapshot within 14 days, skipping")
            continue

        source_idx = asset_names.index(source_asset) \
                     if source_asset in asset_names else btc_idx

        spillover, shocked_pred, baseline_pred = inject_shock(
            model=model,
            snapshots=all_snaps,
            shock_node_idx=source_idx,
            shock_return=shock_val,
            start_pos=closest,
            n_context=20,
        )

        print(f"\n  {event['name']} ({event_date.date()}) "
              f"| {source_asset} shock = {shock_val*100:.1f}%")
        print(f"  {'Asset':>8} | {'Spillover':>10} | {'Shocked':>10} | {'Baseline':>10}")
        print(f"  {'─'*8}-+-{'─'*10}-+-{'─'*10}-+-{'─'*10}")

        for i, asset in enumerate(asset_names):
            if i == source_idx:
                continue
            print(f"  {asset:>8} | {spillover[i]:>+10.4f} | "
                  f"{shocked_pred[i]:>+10.4f} | {baseline_pred[i]:>+10.4f}")

            results.append({
                "event":       event["name"],
                "date":        str(event_date.date()),
                "source":      source_asset,
                "target":      asset,
                "shock_val":   shock_val,
                "spillover":   float(spillover[i]),
                "shocked_pred": float(shocked_pred[i]),
                "baseline_pred": float(baseline_pred[i]),
            })

    df = pd.DataFrame(results)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Regime Conditioning
# ─────────────────────────────────────────────────────────────────────────────

def experiment3_regime_conditioning(model, all_snaps, snapshots,
                                     valid_indices, all_pos,
                                     regime_labels_path, asset_names):
    """
    Compare W matrix evolution and prediction patterns across regimes.

    Uses forward_with_states() to get W norms at each timestep.
    Splits timesteps by regime label and compares:
      - Mean W norm per layer (how active is the model in each regime?)
      - Mean predicted spillover magnitude per regime
    """
    print(f"\n{'─'*55}")
    print(f"  Experiment 3: Regime Conditioning")
    print(f"{'─'*55}")

    # Load regime labels
    if not os.path.exists(regime_labels_path) or \
       os.path.getsize(regime_labels_path) == 0:
        print("  Regime labels not available — skipping")
        return pd.DataFrame()

    regime_df = pd.read_csv(regime_labels_path, index_col=0, parse_dates=True)
    regime_series = regime_df.iloc[:, 0].astype(int)

    # Run model with state tracking
    model.eval()
    with torch.no_grad():
        predictions, states = model(all_snaps, return_states=True)

    # Map W norms and predictions to regime labels
    w_norms    = states["w_norms"].numpy()     # (T, num_layers)
    snap_dates = [snapshots[valid_indices[all_pos[i]]][3]
                  for i in range(len(all_pos))]

    regime_names = {0: "calm", 1: "stress", 2: "crisis"}
    results = []

    for regime_id, regime_name in regime_names.items():
        # Find timesteps in this regime
        regime_timesteps = []
        for t, date in enumerate(snap_dates):
            if date in regime_series.index:
                if int(regime_series.loc[date]) == regime_id:
                    regime_timesteps.append(t)

        if not regime_timesteps:
            continue

        # W norm statistics — how much does the model "activate" per regime
        regime_w = w_norms[regime_timesteps]           # (T_regime, num_layers)
        mean_w_l0 = float(regime_w[:, 0].mean())
        mean_w_l1 = float(regime_w[:, 1].mean())

        # Prediction magnitude — how large are predicted returns per regime
        pred_mag = predictions["t1"][regime_timesteps, :, 0].abs().mean().item()

        print(f"\n  {regime_name.upper()} regime ({len(regime_timesteps)} snapshots):")
        print(f"    Mean W norm layer 0 : {mean_w_l0:.4f}")
        print(f"    Mean W norm layer 1 : {mean_w_l1:.4f}")
        print(f"    Mean pred magnitude : {pred_mag:.6f}")

        results.append({
            "regime":        regime_name,
            "n_snapshots":   len(regime_timesteps),
            "mean_w_norm_l0": mean_w_l0,
            "mean_w_norm_l1": mean_w_l1,
            "mean_pred_magnitude": pred_mag,
        })

    df = pd.DataFrame(results)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4: Crypto Structural Break (pre vs post 2020)
# ─────────────────────────────────────────────────────────────────────────────

def experiment4_structural_break(model, all_snaps, snapshots,
                                  valid_indices, all_pos, asset_names,
                                  break_date="2020-10-01"):
    """
    BTC shock propagation pre vs post institutional adoption.

    2020 is the structural break: MicroStrategy, PayPal, institutional
    Bitcoin adoption — the BTC-equity correlation regime changed.

    Inject a standardized BTC shock (-2 std dev based on training data)
    at snapshots before and after break_date. Compare spillover patterns.
    """
    print(f"\n{'─'*55}")
    print(f"  Experiment 4: Crypto Structural Break ({break_date})")
    print(f"{'─'*55}")

    break_ts  = pd.Timestamp(break_date)
    btc_idx   = asset_names.index("BTC") if "BTC" in asset_names else 7
    shock_val = -0.08   # standardized: ~-2 std dev for BTC daily return

    snap_dates = [snapshots[valid_indices[all_pos[i]]][3]
                  for i in range(len(all_pos))]

    pre_spillovers  = {a: [] for a in asset_names}
    post_spillovers = {a: [] for a in asset_names}

    model.eval()
    for t, date in enumerate(snap_dates):
        if t < 20:   # need context window
            continue

        spillover, _, _ = inject_shock(
            model=model,
            snapshots=all_snaps,
            shock_node_idx=btc_idx,
            shock_return=shock_val,
            start_pos=t,
            n_context=20,
        )

        bucket = pre_spillovers if date < break_ts else post_spillovers
        for i, asset in enumerate(asset_names):
            if i != btc_idx:
                bucket[asset].append(float(spillover[i]))

    results = []
    print(f"\n  {'Asset':>8} | {'Pre-2020 spill':>15} | "
          f"{'Post-2020 spill':>16} | {'Change':>10}")
    print(f"  {'─'*8}-+-{'─'*15}-+-{'─'*16}-+-{'─'*10}")

    for asset in asset_names:
        if asset == "BTC":
            continue
        pre_mean  = float(np.mean(pre_spillovers[asset]))  \
                    if pre_spillovers[asset]  else 0.0
        post_mean = float(np.mean(post_spillovers[asset])) \
                    if post_spillovers[asset] else 0.0
        change    = post_mean - pre_mean

        print(f"  {asset:>8} | {pre_mean:>+15.6f} | "
              f"{post_mean:>+16.6f} | {change:>+10.6f}")

        results.append({
            "asset":                asset,
            "pre_2020_mean_spillover":  pre_mean,
            "post_2020_mean_spillover": post_mean,
            "change":               change,
            "n_pre":  len(pre_spillovers[asset]),
            "n_post": len(post_spillovers[asset]),
        })

    df = pd.DataFrame(results).sort_values("change", ascending=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(config_path="config.yaml"):

    config = load_config(config_path)
    np.random.seed(config["training"]["seed"])

    print("=" * 60)
    print("  CASCADE — Full Evaluation Suite")
    print("=" * 60)

    # ── Paths ─────────────────────────────────────────────────────────────
    SNAPSHOTS_PATH   = "data/processed/graph_snapshots.pkl"
    RETURNS_PATH     = "data/processed/returns_matrix.csv"
    CHECKPOINT_PATH  = "results/checkpoints/best_model.pt"
    REGIME_PATH      = "data/processed/regime_labels.csv"
    RESULTS_DIR      = "results"

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    snapshots, asset_names = load_snapshots(SNAPSHOTS_PATH)
    returns_df             = load_returns(RETURNS_PATH)
    horizons               = config["horizons"]

    targets, valid_indices = build_targets(
        snapshots, returns_df, asset_names, horizons=horizons
    )
    train_pos, val_pos, test_pos = split_by_date(
        snapshots, valid_indices, config
    )

    all_pos    = train_pos + val_pos + test_pos
    all_snaps  = to_model_input(snapshots, all_pos, valid_indices)
    test_local = list(range(len(train_pos) + len(val_pos), len(all_pos)))

    # Needed to actually train the Static GCN baseline below (see
    # train_static_gcn) — same train/val split machinery models/train.py uses.
    train_snaps_sg = to_model_input(snapshots, train_pos, valid_indices)
    tv_pos_sg      = train_pos + val_pos
    tv_snaps_sg    = to_model_input(snapshots, tv_pos_sg, valid_indices)
    val_local_sg   = list(range(len(train_pos), len(train_pos) + len(val_pos)))

    # ── Load trained GNN ──────────────────────────────────────────────────
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"No checkpoint found at {CHECKPOINT_PATH}")
        print("Run models/train.py first.")
        return

    model = EvolveGCNH(
        node_features=4,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=0.0,   # no dropout during evaluation
    )
    load_checkpoint(model, CHECKPOINT_PATH)
    model.eval()

    # ── Run GNN on full sequence ───────────────────────────────────────────
    with torch.no_grad():
        all_preds = model(all_snaps)

    # GNN test predictions
    gnn_test = {
        h: all_preds[h][test_local, :, 0].numpy()
        for h in ["t1", "t5", "t10"]
    }
    test_actuals = {
        h: targets[h][[all_pos[i] for i in test_local]].numpy()
        for h in ["t1", "t5", "t10"]
    }

    print(f"\nGNN test predictions ready: {gnn_test['t1'].shape[0]} timesteps")

    # ── Fit baselines on train data ────────────────────────────────────────
    train_end = pd.Timestamp(config["dates"]["train_end"])
    train_df  = returns_df[returns_df.index <= train_end]
    test_start = pd.Timestamp(config["dates"]["val_end"])
    test_df   = returns_df[returns_df.index > test_start]

    print("\nFitting baselines on training data...")

    # Filter to GNN asset columns only — VIX is a node feature not a prediction target
    train_df_assets = train_df[asset_names]
    test_df_assets  = test_df[asset_names]

    roll_model = RollingCorrelationBaseline(
        window=config["graph"]["rolling_window_days"]
    )
    roll_model.fit(train_df_assets)

    var_model = VARBaseline(maxlags=5)
    var_model.fit(train_df_assets)

    static_gcn = StaticGCN(
        node_features=4,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
    )
    torch.manual_seed(config["training"]["seed"])
    static_gcn = train_static_gcn(
        static_gcn, train_snaps_sg, tv_snaps_sg, val_local_sg,
        targets, train_pos, val_pos, tv_pos_sg, config
    )

    # ── Experiment 1 — Predictive Accuracy ────────────────────────────────
    all_exp1_results = []
    all_dm_results   = []

    for h_days, h_key in [(1, "t1"), (5, "t5"), (10, "t10")]:
        roll_p, roll_a = roll_model.predict(test_df_assets, horizon=h_days)
        var_p,  var_a  = var_model.predict(test_df_assets,  horizon=h_days)

        # Static GCN on test snapshots
        test_snaps_input = to_model_input(snapshots, test_pos, valid_indices)
        with torch.no_grad():
            static_preds_all = static_gcn(test_snaps_input)
        static_p = static_preds_all[h_key][:, :, 0].numpy()

        # Align VIX-excluded baselines to 10-asset GNN targets
        # VAR and rolling corr exclude VIX — same 10 assets as GNN
        T_min = min(gnn_test[h_key].shape[0],
                    roll_p.shape[0], var_p.shape[0], static_p.shape[0],
                    test_actuals[h_key].shape[0])

        g_aligned     = gnn_test[h_key][-T_min:]
        roll_aligned  = roll_p[-T_min:]
        var_aligned   = var_p[-T_min:]
        static_aligned = static_p[-T_min:]
        act_aligned   = test_actuals[h_key][-T_min:]

        baselines = {
            "Rolling Correlation": roll_aligned,
            "VAR":                 var_aligned,
            "Static GCN":         static_aligned,
        }

        df_acc, df_dm = experiment1_accuracy(
            g_aligned, baselines, act_aligned, asset_names, h_key
        )
        df_acc["horizon"] = h_key
        df_dm["horizon"]  = h_key
        all_exp1_results.append(df_acc)
        all_dm_results.append(df_dm)

    exp1_df = pd.concat(all_exp1_results, ignore_index=True)
    dm_df   = pd.concat(all_dm_results,   ignore_index=True)

    exp1_df.to_csv(f"{RESULTS_DIR}/experiment1_accuracy.csv", index=False)
    dm_df.to_csv(f"{RESULTS_DIR}/diebold_mariano_results.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR}/experiment1_accuracy.csv")
    print(f"Saved: {RESULTS_DIR}/diebold_mariano_results.csv")

    # ── Experiment 2 — Shock Propagation ──────────────────────────────────
    shock_events = [
        {"name": "COVID Crash",       "date": "2020-03-12",
         "source": "SPY", "shock_val": -0.095},
        {"name": "FTX Collapse",      "date": "2022-11-09",
         "source": "BTC", "shock_val": -0.246},
        {"name": "SVB Crisis",        "date": "2023-03-10",
         "source": "SPY", "shock_val": -0.048},
        {"name": "2022 Rate Shock",   "date": "2022-06-13",
         "source": "TLT", "shock_val": -0.031},
    ]

    exp2_df = experiment2_shock_propagation(
        model, all_snaps, snapshots, valid_indices,
        all_pos, asset_names, shock_events
    )
    exp2_df.to_csv(f"{RESULTS_DIR}/experiment2_shock_propagation.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR}/experiment2_shock_propagation.csv")

    # ── Experiment 3 — Regime Conditioning ────────────────────────────────
    exp3_df = experiment3_regime_conditioning(
        model, all_snaps, snapshots, valid_indices,
        all_pos, REGIME_PATH, asset_names
    )
    if not exp3_df.empty:
        exp3_df.to_csv(f"{RESULTS_DIR}/experiment3_regime_analysis.csv", index=False)
        print(f"Saved: {RESULTS_DIR}/experiment3_regime_analysis.csv")

    # ── Experiment 4 — Structural Break ───────────────────────────────────
    exp4_df = experiment4_structural_break(
        model, all_snaps, snapshots, valid_indices,
        all_pos, asset_names, break_date="2020-10-01"
    )
    exp4_df.to_csv(f"{RESULTS_DIR}/experiment4_structural_break.csv", index=False)
    print(f"Saved: {RESULTS_DIR}/experiment4_structural_break.csv")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Evaluation Complete")
    print("=" * 60)
    print("\nFiles written to results/:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.endswith(".csv"):
            size = os.path.getsize(f"{RESULTS_DIR}/{f}")
            print(f"  {f:45s} {size:>8,} bytes")

    return exp1_df, dm_df, exp2_df, exp3_df, exp4_df


if __name__ == "__main__":
    evaluate()
