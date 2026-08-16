"""
scripts/run_ablations.py

Runs two ablation studies to justify CASCADE's full architecture:

    Ablation A — Static edges:
        EvolveGCN-H trained on rolling correlation edges only (no DCC-GARCH).
        Quantifies the contribution of dynamic DCC edge weights.

    Ablation B — Reduced capacity (hidden_dim=32):
        EvolveGCN-H trained with half the parameters.
        Confirms results aren't driven purely by model size.

Both are compared against the full CASCADE model on the test set.
Results saved to results/ablation_results.csv.

Run: python scripts/run_ablations.py
"""

import os
import sys
import copy
import pickle
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.evolvegcn import EvolveGCNH
from models.train     import (load_config, load_snapshots, load_returns,
                               build_targets, split_by_date,
                               to_model_input, compute_loss,
                               compute_metrics, save_checkpoint,
                               load_checkpoint)
from scripts.build_graphs import (load_returns as bg_load_returns,
                                   build_snapshots, save_snapshots,
                                   load_config  as bg_load_config)


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
CONFIG_PATH          = "config.yaml"
RETURNS_PATH         = "data/processed/returns_matrix.csv"
SNAPSHOTS_DCC_PATH   = "data/processed/graph_snapshots.pkl"        # full model
SNAPSHOTS_ROLL_PATH  = "data/processed/graph_snapshots_rolling.pkl" # ablation A
CKPT_FULL_PATH       = "results/checkpoints/best_model.pt"
CKPT_ABLATION_A_PATH = "results/checkpoints/ablation_a_static_edges.pt"
CKPT_ABLATION_B_PATH = "results/checkpoints/ablation_b_hidden32.pt"
ABLATION_RESULTS_PATH= "results/ablation_results.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Build rolling-only snapshots (Ablation A)
# ─────────────────────────────────────────────────────────────────────────────

def build_rolling_snapshots(config):
    """
    Build graph snapshots using rolling Pearson correlation edges only.
    Intentionally skips the DCC-GARCH swap.
    Saved separately so the DCC snapshots are not overwritten.
    """
    if os.path.exists(SNAPSHOTS_ROLL_PATH):
        print(f"  Rolling snapshots already exist — loading from cache")
        return load_snapshots(SNAPSHOTS_ROLL_PATH)

    print("  Building rolling-only snapshots (no DCC swap)...")
    returns_df = bg_load_returns(RETURNS_PATH)

    snapshots, asset_names = build_snapshots(
        returns_df    = returns_df,
        config        = config,
        regime_labels = None,    # placeholder — regime not part of this ablation
        verbose       = False,
    )
    save_snapshots(snapshots, asset_names, SNAPSHOTS_ROLL_PATH)
    return snapshots, asset_names


# ─────────────────────────────────────────────────────────────────────────────
# Training loop (shared by both ablations)
# ─────────────────────────────────────────────────────────────────────────────

def train_ablation(snapshots, asset_names, config, model_config,
                   checkpoint_path, label):
    """
    Train EvolveGCN-H with given snapshots and model config.
    Mirrors models/train.py but with configurable hyperparameters.
    """
    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])

    returns_df = load_returns(RETURNS_PATH)
    horizons   = config["horizons"]

    targets, valid_indices = build_targets(
        snapshots, returns_df, asset_names, horizons=horizons
    )
    train_pos, val_pos, test_pos = split_by_date(
        snapshots, valid_indices, config
    )

    train_snaps  = to_model_input(snapshots, train_pos, valid_indices)
    tv_pos       = train_pos + val_pos
    tv_snaps     = to_model_input(snapshots, tv_pos, valid_indices)
    all_pos      = train_pos + val_pos + test_pos
    all_snaps    = to_model_input(snapshots, all_pos, valid_indices)
    val_local    = list(range(len(train_pos),
                              len(train_pos) + len(val_pos)))
    test_local   = list(range(len(train_pos) + len(val_pos), len(all_pos)))

    model = EvolveGCNH(
        node_features = 4,
        hidden_dim    = model_config["hidden_dim"],
        num_layers    = model_config["num_layers"],
        dropout       = model_config["dropout"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    epochs           = config["training"]["epochs"]
    patience         = config["training"]["early_stopping_patience"]
    eval_freq        = 5
    best_val_loss    = float("inf")
    patience_counter = 0

    print(f"  Training {label} for up to {epochs} epochs...")
    print(f"  {'Epoch':>6} | {'Train':>10} | {'Val':>10} | {'Dir%':>7}")
    print(f"  {'─'*45}")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        train_preds = model(train_snaps)
        train_indices = list(range(len(train_pos)))
        train_loss = compute_loss(train_preds, train_indices,
                                  targets, train_pos)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % eval_freq == 0:
            model.eval()
            with torch.no_grad():
                tv_preds = model(tv_snaps)
                val_loss = compute_loss(
                    tv_preds, val_local,
                    targets, [tv_pos[i] for i in val_local]
                )
                val_loss_f = float(val_loss)

                vp   = tv_preds["t1"][val_local, :, 0].numpy()
                vt   = targets["t1"][[tv_pos[i] for i in val_local]].numpy()
                dacc = float(np.mean(np.sign(vp) == np.sign(vt))) * 100

            scheduler.step(val_loss_f)

            if epoch % 20 == 0 or val_loss_f < best_val_loss:
                print(f"  {epoch:>6} | {train_loss.detach().item():>10.6f} | "
                      f"{val_loss_f:>10.6f} | {dacc:>6.1f}%")

            if val_loss_f < best_val_loss:
                best_val_loss = val_loss_f
                patience_counter = 0
                save_ablation_checkpoint(model, epoch,
                                          val_loss_f, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience // eval_freq:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    # Load best and evaluate on test
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded checkpoint epoch={ckpt['epoch']} val_loss={ckpt['val_loss']:.8f}")
    model.eval()
    with torch.no_grad():
        all_preds = model(all_snaps)

    test_metrics = {}
    for h_key in ["t1", "t5", "t10"]:
        p = all_preds[h_key][test_local, :, 0].numpy()
        t = targets[h_key][[all_pos[i] for i in test_local]].numpy()
        test_metrics[h_key] = compute_metrics(p, t)

    return test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Load full model metrics from existing test predictions
# ─────────────────────────────────────────────────────────────────────────────

def load_full_model_metrics():
    """
    Load the already-computed test metrics for the full CASCADE model.
    Avoids retraining the full model just to get comparison numbers.
    """
    with open("results/test_predictions.pkl", "rb") as f:
        data = pickle.load(f)

    preds   = data["preds"]
    targets = data["targets"]
    metrics = {}
    for h_key in ["t1", "t5", "t10"]:
        metrics[h_key] = compute_metrics(preds[h_key], targets[h_key])
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def save_ablation_checkpoint(model, epoch, val_loss, path):
    """
    Lightweight checkpoint — model weights only, no optimizer state.
    Ablations don't need to resume training so we skip the optimizer
    state dict which cuts file size roughly in half.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":            epoch,
        "model_state_dict": model.state_dict(),
        "val_loss":         val_loss,
    }, path)


def build_summary_table(full_metrics, ablation_a_metrics, ablation_b_metrics):
    """
    Build and print a comparison table of the three configurations.
    Computes % MSE improvement of full model over each ablation.
    """
    rows = []
    horizon_map = {"t1": "t+1", "t5": "t+5", "t10": "t+10"}

    models = [
        ("CASCADE (full)",             full_metrics),
        ("Ablation A — static edges",  ablation_a_metrics),
        ("Ablation B — hidden_dim=32", ablation_b_metrics),
    ]

    for h_key, h_label in horizon_map.items():
        for model_name, metrics in models:
            m = metrics[h_key]
            rows.append({
                "horizon":    h_label,
                "model":      model_name,
                "mse":        round(m["mse"], 8),
                "mae":        round(m["mae"], 8),
                "dir_acc":    round(m["directional_accuracy"], 4),
            })

    df = pd.DataFrame(rows)

    # Add % improvement over each ablation
    for h_key, h_label in horizon_map.items():
        full_mse = full_metrics[h_key]["mse"]
        abl_a_mse = ablation_a_metrics[h_key]["mse"]
        abl_b_mse = ablation_b_metrics[h_key]["mse"]

        pct_vs_a = (abl_a_mse - full_mse) / abl_a_mse * 100
        pct_vs_b = (abl_b_mse - full_mse) / abl_b_mse * 100

        mask = (df["horizon"] == h_label) & (df["model"] == "CASCADE (full)")
        df.loc[mask, "pct_improvement_vs_ablation_a"] = round(pct_vs_a, 2)
        df.loc[mask, "pct_improvement_vs_ablation_b"] = round(pct_vs_b, 2)

    return df


def print_summary(df):
    print("\n" + "=" * 75)
    print("  ABLATION STUDY — Summary")
    print("=" * 75)

    for h_label in ["t+1", "t+5", "t+10"]:
        print(f"\n  Horizon {h_label}:")
        print(f"  {'Model':35s} {'MSE':>12} {'MAE':>12} {'Dir%':>8}")
        print(f"  {'─'*35}-+-{'─'*12}-+-{'─'*12}-+-{'─'*8}")

        sub = df[df["horizon"] == h_label]
        for _, row in sub.iterrows():
            marker = " ←" if "full" in row["model"] else ""
            print(f"  {row['model']:35s} {row['mse']:>12.8f} "
                  f"{row['mae']:>12.8f} {row['dir_acc']*100:>7.1f}%{marker}")

        # Print improvement lines
        full_row = sub[sub["model"] == "CASCADE (full)"].iloc[0]
        if pd.notna(full_row.get("pct_improvement_vs_ablation_a", np.nan)):
            pct_a = full_row["pct_improvement_vs_ablation_a"]
            pct_b = full_row["pct_improvement_vs_ablation_b"]
            print(f"\n  DCC edges contribution     : {pct_a:+.1f}% MSE reduction "
                  f"(full vs ablation A)")
            print(f"  Extra capacity contribution: {pct_b:+.1f}% MSE reduction "
                  f"(full vs ablation B)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_ablations():
    config = load_config(CONFIG_PATH)

    print("=" * 60)
    print("  CASCADE — Ablation Study")
    print("=" * 60)

    # ── Load full model metrics (already trained) ──────────────────────────
    print("\n[Full Model] Loading existing test predictions...")
    full_metrics = load_full_model_metrics()
    for h in ["t1", "t5", "t10"]:
        m = full_metrics[h]
        print(f"  {h}: MSE={m['mse']:.8f} | Dir={m['directional_accuracy']*100:.1f}%")

    # ── Ablation A: Static (rolling correlation) edges ────────────────────
    print("\n" + "─" * 60)
    print("[Ablation A] Static edges — rolling correlation only (no DCC)")
    print("─" * 60)

    roll_snaps, roll_assets = build_rolling_snapshots(config)

    ablation_a_metrics = train_ablation(
        snapshots      = roll_snaps,
        asset_names    = roll_assets,
        config         = config,
        model_config   = config["model"],   # same hidden_dim=64
        checkpoint_path= CKPT_ABLATION_A_PATH,
        label          = "Ablation A (static edges)",
    )

    print(f"\n  Ablation A test results:")
    for h in ["t1", "t5", "t10"]:
        m = ablation_a_metrics[h]
        print(f"  {h}: MSE={m['mse']:.8f} | Dir={m['directional_accuracy']*100:.1f}%")

    # ── Ablation B: Reduced capacity (hidden_dim=32) ──────────────────────
    print("\n" + "─" * 60)
    print("[Ablation B] Reduced capacity — hidden_dim=32 (vs 64 in full model)")
    print("─" * 60)

    dcc_snaps, dcc_assets = load_snapshots(SNAPSHOTS_DCC_PATH)

    small_model_config = copy.deepcopy(config["model"])
    small_model_config["hidden_dim"] = 32

    ablation_b_metrics = train_ablation(
        snapshots      = dcc_snaps,
        asset_names    = dcc_assets,
        config         = config,
        model_config   = small_model_config,
        checkpoint_path= CKPT_ABLATION_B_PATH,
        label          = "Ablation B (hidden_dim=32)",
    )

    print(f"\n  Ablation B test results:")
    for h in ["t1", "t5", "t10"]:
        m = ablation_b_metrics[h]
        print(f"  {h}: MSE={m['mse']:.8f} | Dir={m['directional_accuracy']*100:.1f}%")

    # ── Summary table ──────────────────────────────────────────────────────
    summary_df = build_summary_table(
        full_metrics, ablation_a_metrics, ablation_b_metrics
    )
    print_summary(summary_df)

    summary_df.to_csv(ABLATION_RESULTS_PATH, index=False)
    print(f"\nSaved: {ABLATION_RESULTS_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    run_ablations()
