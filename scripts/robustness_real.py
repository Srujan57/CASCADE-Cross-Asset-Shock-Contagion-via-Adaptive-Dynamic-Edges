"""
scripts/robustness_real.py

Replaces the FABRICATED robustness checks previously produced by
scripts/phase4_results.py (now deprecated — see that file's docstring).

phase4_results.py's compute_robustness_checks() did not vary any actual
pipeline parameter or retrain anything. It took the baseline MSE and
multiplied it by hand-picked formulas:

    noise_factor = 1.0 + (thresh - 0.3) * 0.15        # correlation threshold
    noise_factor = 1.0 + abs(window - 60) * 0.001      # rolling window
    mse_adj = row["mse"] * 1.08                        # 8-asset universe

Its output was therefore made-up numbers dressed up as a sensitivity
analysis — a real problem if presented as evidence of robustness.
Likewise, its compute_bootstrap_ci() re-derived "bootstrap" CIs by
sampling a normal distribution parameterized by the *already-reported* CI
bounds, rather than resampling actual predictions — it's circular, not a
second independent bootstrap. The real bootstrap (resampling actual
predictions) already happens correctly in scripts/evaluate.py's
bootstrap_ci(), so results/experiment1_accuracy.csv's CI columns are
trustworthy on their own.

This script does the real thing for three robustness checks:
    (a) Correlation threshold: 0.2, 0.3 (baseline), 0.4
    (b) Rolling window: 30, 60 (baseline), 90 days
    (c) Reduced 8-asset universe: drop GLD, USO

For each variant it rebuilds graph snapshots from the real returns data,
trains a fresh EvolveGCN-H from scratch, evaluates on the same held-out
test period (2023-2024) as the main model, and reports real MSE/MAE.

This is genuinely expensive: up to 7 full training runs. Use --quick for a
fast smoke test (reduced epochs) to confirm the code path works before
committing to the full run overnight.

Run from repo root (needs data/processed/returns_matrix.csv and, ideally,
data/processed/regime_labels.csv / dcc_correlations.pkl already built):
    python scripts/robustness_real.py            # full rigor, slow
    python scripts/robustness_real.py --quick     # fast smoke test only
"""

import os
import sys
import copy
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.evolvegcn import EvolveGCNH
from models.train import (load_config, build_targets, split_by_date,
                           to_model_input, compute_loss, compute_metrics)
from scripts.build_graphs import (load_returns, build_snapshots,
                                   swap_in_regime_labels, swap_in_dcc_edges)

RESULTS_DIR    = "results"
RETURNS_PATH   = "data/processed/returns_matrix.csv"
REGIME_PATH    = "data/processed/regime_labels.csv"
DCC_PATH       = "data/processed/dcc_correlations.pkl"
DROPPED_ASSETS = ["GLD", "USO"]   # matches phase4_results.py's documented choice


def run_variant(returns_df, config, asset_subset=None, quick=False, seed=42):
    """
    Build snapshots under a modified config (and optional asset subset),
    train EvolveGCN-H from scratch, evaluate on the test split.

    Returns dict of {horizon: {"mse":..., "mae":..., "n_test":...}}
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    rdf = returns_df.copy()
    if asset_subset is not None:
        keep_cols = asset_subset + (["VIX"] if "VIX" in rdf.columns else [])
        rdf = rdf[keep_cols]

    regime_labels = None
    if os.path.exists(REGIME_PATH) and os.path.getsize(REGIME_PATH) > 0:
        regime_df = pd.read_csv(REGIME_PATH, index_col=0, parse_dates=True)
        regime_labels = regime_df.iloc[:, 0].astype(float)

    snapshots, asset_names = build_snapshots(
        returns_df=rdf, config=config, regime_labels=regime_labels, verbose=False
    )

    if os.path.exists(DCC_PATH) and os.path.getsize(DCC_PATH) > 0 and asset_subset is None:
        # DCC correlations were fit on the full 10-asset universe; only
        # valid to swap in when the asset universe matches. For the
        # reduced-universe robustness check, fall back to rolling
        # correlation edges (rebuilt on the subset above) instead.
        snapshots = swap_in_dcc_edges(snapshots, DCC_PATH, asset_names,
                                       threshold=config["graph"]["corr_threshold"])

    horizons = config["horizons"]
    targets, valid_indices = build_targets(snapshots, rdf, asset_names, horizons=horizons)
    train_pos, val_pos, test_pos = split_by_date(snapshots, valid_indices, config)

    train_snaps = to_model_input(snapshots, train_pos, valid_indices)
    tv_pos = train_pos + val_pos
    tv_snaps = to_model_input(snapshots, tv_pos, valid_indices)
    val_local = list(range(len(train_pos), len(train_pos) + len(val_pos)))
    all_pos = train_pos + val_pos + test_pos
    all_snaps = to_model_input(snapshots, all_pos, valid_indices)
    test_local = list(range(len(train_pos) + len(val_pos), len(all_pos)))

    model = EvolveGCNH(
        node_features=4,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"],
                                  weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                             factor=0.5, patience=10)

    epochs   = 15 if quick else config["training"]["epochs"]
    patience = 5  if quick else config["training"]["early_stopping_patience"]
    eval_freq = 5

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_preds = model(train_snaps)
        train_indices = list(range(len(train_pos)))
        loss = compute_loss(train_preds, train_indices, targets, train_pos)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % eval_freq == 0:
            model.eval()
            with torch.no_grad():
                tv_preds = model(tv_snaps)
                val_loss = compute_loss(tv_preds, val_local, targets,
                                         [tv_pos[i] for i in val_local])
                val_loss_f = float(val_loss)
            scheduler.step(val_loss_f)

            if val_loss_f < best_val_loss:
                best_val_loss = val_loss_f
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience // eval_freq:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        all_preds = model(all_snaps)

    results = {}
    for h_key in ["t1", "t5", "t10"]:
        p = all_preds[h_key][test_local, :, 0].numpy()
        t = targets[h_key][[all_pos[i] for i in test_local]].numpy()
        m = compute_metrics(p, t)
        results[h_key] = {"mse": m["mse"], "mae": m["mae"],
                          "directional_accuracy": m["directional_accuracy"],
                          "n_test": p.shape[0]}
    return results


def build_job_list(returns_df):
    """
    Enumerate all 8 robustness-check variants as independent job descriptors
    (kind, label, run_kwargs, is_baseline). Each job is a full, real retrain —
    nothing here changes what gets trained, only how the 8 jobs get divided
    up across processes. Kept as a plain list (not generators) so --shard can
    index into it deterministically.
    """
    jobs = []

    for thresh in [0.2, 0.3, 0.4]:
        jobs.append({
            "check": "corr_threshold", "parameter": f"threshold={thresh}",
            "baseline": thresh == 0.3,
            "kwargs": {"config_overrides": {"graph": {"corr_threshold": thresh}}},
        })

    for window in [30, 60, 90]:
        jobs.append({
            "check": "rolling_window", "parameter": f"window={window}d",
            "baseline": window == 60,
            "kwargs": {"config_overrides": {"graph": {"rolling_window_days": window}}},
        })

    full_assets = [c for c in returns_df.columns if c != "VIX"]
    reduced_assets = [a for a in full_assets if a not in DROPPED_ASSETS]
    for label, subset in [("11-asset (full)", None),
                          (f"{len(reduced_assets)}-asset (drop {','.join(DROPPED_ASSETS)})", reduced_assets)]:
        jobs.append({
            "check": "asset_universe", "parameter": label,
            "baseline": subset is None,
            "kwargs": {"asset_subset": subset},
        })

    return jobs


def run_job(job, returns_df, base_config, quick):
    cfg = copy.deepcopy(base_config)
    overrides = job["kwargs"].get("config_overrides", {})
    for section, kv in overrides.items():
        cfg[section].update(kv)
    asset_subset = job["kwargs"].get("asset_subset")
    return run_variant(returns_df, cfg, asset_subset=asset_subset, quick=quick)


def parse_shard(shard_str):
    """'2/4' -> (2, 4). '1/1' (default) means no sharding — run everything."""
    try:
        k, n = shard_str.split("/")
        k, n = int(k), int(n)
        assert 1 <= k <= n
    except Exception:
        raise SystemExit(f"--shard must look like 'K/N' with 1<=K<=N, got {shard_str!r}")
    return k, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke test (15 epochs) instead of full training runs.")
    parser.add_argument("--shard", default="1/1",
                        help="Run only this shard of the 8 independent jobs, e.g. "
                             "'1/3' '2/3' '3/3' run in three separate terminals to use "
                             "more CPU cores at once. Every shard trains real models on "
                             "the real data — this only changes wall-clock time, not "
                             "what gets computed. Default '1/1' runs everything serially "
                             "in one process, same as before.")
    parser.add_argument("--threads", type=int, default=None,
                        help="torch.set_num_threads() for this process. Set this when "
                             "running multiple shards in parallel so they don't all "
                             "fight over every CPU core (e.g. 4 physical cores, 2 "
                             "shards -> --threads 2 each).")
    args = parser.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    if not os.path.exists(RETURNS_PATH):
        print(f"{RETURNS_PATH} not found. Run scripts/data_ingestion.py first.")
        return

    base_config = load_config("config.yaml")
    returns_df = load_returns(RETURNS_PATH)

    shard_k, shard_n = parse_shard(args.shard)
    all_jobs = build_job_list(returns_df)
    my_jobs = [j for i, j in enumerate(all_jobs) if i % shard_n == (shard_k - 1)]

    print("=" * 60)
    print("  REAL Robustness Checks (replaces phase4_results.py's")
    print("  fabricated formulas)")
    print("=" * 60)
    if args.quick:
        print("  --quick mode: 15 epochs per run, for pipeline smoke-testing")
        print("  ONLY. Re-run without --quick before citing these numbers")
        print("  anywhere.")
    if shard_n > 1:
        print(f"  Shard {shard_k}/{shard_n}: running {len(my_jobs)} of "
              f"{len(all_jobs)} total jobs in this process.")
        print("  Run the other shard(s) in separate terminals, then merge —")
        print("  see the merge command printed at the end of each shard.")
    print()

    rows = []
    for job in my_jobs:
        print(f"[{job['check']}] {job['parameter']} — training...")
        res = run_job(job, returns_df, base_config, quick=args.quick)
        for h, m in res.items():
            rows.append({"robustness_check": job["check"],
                        "parameter": job["parameter"], "horizon": h,
                        "mse": m["mse"], "mae": m["mae"],
                        "directional_accuracy": m["directional_accuracy"],
                        "baseline": job["baseline"]})
        print(f"    t1 MSE={res['t1']['mse']:.8f}")

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = "_quick" if args.quick else ""
    shard_suffix = f"_shard{shard_k}of{shard_n}" if shard_n > 1 else ""
    out_path = os.path.join(RESULTS_DIR, f"robustness_checks_real{suffix}{shard_suffix}.csv")
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print(f"  [SAVED] {out_path}")
    if shard_n > 1:
        print(f"  Once all {shard_n} shards finish, merge them:")
        print(f"    python -c \"import pandas as pd, glob; "
              f"pd.concat([pd.read_csv(f) for f in "
              f"glob.glob('results/robustness_checks_real{suffix}_shard*of{shard_n}.csv')])"
              f".to_csv('results/robustness_checks_real{suffix}.csv', index=False)\"")
    if args.quick:
        print("  This is a smoke test, not a citable result. Re-run without")
        print("  --quick and use robustness_checks_real.csv (no suffix) as")
        print("  the real robustness table.")
    else:
        print("  This is the real robustness table — supersedes the old")
        print("  fabricated-formula version this script replaced.")
    print("=" * 60)


if __name__ == "__main__":
    main()
