"""
models/train.py

Full training loop for EvolveGCN-H.

Design:
  - Passes full temporal sequence in order — never shuffled
  - Training forward: train snapshots only
  - Validation forward: train+val snapshots, index val portion
    (ensures W is properly evolved through training period before val evaluation)
  - Early stopping on val loss with gradient clipping
  - Saves best checkpoint to results/checkpoints/

Run: python models/train.py
"""

import os
import sys
import json
import pickle
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.evolvegcn import EvolveGCNH

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Config & data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_snapshots(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["snapshots"], payload["asset_names"]


def load_returns(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index().ffill().dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Target construction
# ─────────────────────────────────────────────────────────────────────────────

def build_targets(snapshots, returns_df, asset_names, horizons=[1, 5, 10]):
    """
    For each snapshot at date d, look up actual returns at d + h TRADING DAYS.

    Snapshots are weekly. Horizons are in trading days.
    Snapshot at 2020-01-03 (Friday) -> t+1 = 2020-01-06 (Monday).
    We use positional indexing on returns_df to step forward exactly h days.

    Returns:
        targets       : dict {"t1","t5","t10"} each tensor shape (T_valid, N)
        valid_indices : list of indices into snapshots with all targets available
    """
    trading_days = returns_df.index
    date_to_pos  = {d: i for i, d in enumerate(trading_days)}

    raw_targets   = {h: [] for h in horizons}
    valid_indices = []

    for snap_i, (x, ei, ew, date) in enumerate(snapshots):
        if date not in date_to_pos:
            continue

        pos       = date_to_pos[date]
        all_valid = True
        h_targets = {}

        for h in horizons:
            target_pos = pos + h
            if target_pos >= len(trading_days):
                all_valid = False
                break
            target_returns = returns_df[asset_names].iloc[target_pos].values.astype(float)
            if not np.isfinite(target_returns).all():
                all_valid = False
                break
            h_targets[h] = target_returns

        if all_valid:
            for h in horizons:
                raw_targets[h].append(h_targets[h])
            valid_indices.append(snap_i)

    targets = {
        f"t{h}": torch.tensor(np.array(raw_targets[h]), dtype=torch.float)
        for h in horizons
    }  # each (T_valid, N)

    print(f"Targets built: {len(valid_indices)} valid snapshots "
          f"(of {len(snapshots)} total)")
    return targets, valid_indices


def split_by_date(snapshots, valid_indices, config):
    """
    Split valid snapshot positions chronologically into train/val/test.
    Returns three lists of positions INTO the valid_indices array.
    """
    train_end = pd.Timestamp(config["dates"]["train_end"])
    val_end   = pd.Timestamp(config["dates"]["val_end"])

    train_pos, val_pos, test_pos = [], [], []
    for local_i, global_i in enumerate(valid_indices):
        date = snapshots[global_i][3]
        if date <= train_end:
            train_pos.append(local_i)
        elif date <= val_end:
            val_pos.append(local_i)
        else:
            test_pos.append(local_i)

    print(f"Split: train={len(train_pos)} | val={len(val_pos)} | "
          f"test={len(test_pos)} valid snapshots")
    return train_pos, val_pos, test_pos


# ─────────────────────────────────────────────────────────────────────────────
# Model input helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_model_input(snapshots, positions, valid_indices):
    """Extract (x, edge_index, edge_weight) for a list of positions."""
    return [
        (snapshots[valid_indices[i]][0],
         snapshots[valid_indices[i]][1],
         snapshots[valid_indices[i]][2])
        for i in positions
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Loss & metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(predictions, pred_indices, targets, target_indices):
    """
    MSE loss averaged over t+1, t+5, t+10 horizons.

    predictions    : dict {"t1","t5","t10"} each (T_seq, N, 1)
    pred_indices   : which rows of predictions to use (0-based into T_seq)
    targets        : dict {"t1","t5","t10"} each (T_valid_total, N)
    target_indices : which rows of targets to use (positions in full targets tensor)
    """
    total = torch.tensor(0.0)
    for h_key in ["t1", "t5", "t10"]:
        pred   = predictions[h_key][pred_indices, :, 0]   # (T_split, N)
        target = targets[h_key][target_indices]            # (T_split, N)
        total  = total + nn.functional.mse_loss(pred, target)
    return total / 3


def compute_metrics(preds_np, targets_np):
    """Compute MSE, MAE, directional accuracy from numpy arrays."""
    mse  = float(np.mean((preds_np - targets_np) ** 2))
    mae  = float(np.mean(np.abs(preds_np - targets_np)))
    dacc = float(np.mean(np.sign(preds_np) == np.sign(targets_np)))
    return {"mse": mse, "mae": mae, "directional_accuracy": dacc}


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, val_loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss":             val_loss,
    }, path)


def load_checkpoint(model, path, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"Checkpoint loaded: epoch={ckpt['epoch']} "
          f"val_loss={ckpt['val_loss']:.8f}")
    return ckpt["epoch"], ckpt["val_loss"]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train(config_path="config.yaml"):

    config = load_config(config_path)
    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])

    print("=" * 60)
    print("  CASCADE — EvolveGCN-H Training")
    print("=" * 60)

    # ── Paths ─────────────────────────────────────────────────────────────
    SNAPSHOTS_PATH  = "data/processed/graph_snapshots.pkl"
    RETURNS_PATH    = "data/processed/returns_matrix.csv"
    CHECKPOINT_PATH = "results/checkpoints/best_model.pt"
    HISTORY_PATH    = "results/training_history.json"

    # ── Load ──────────────────────────────────────────────────────────────
    snapshots, asset_names = load_snapshots(SNAPSHOTS_PATH)
    returns_df             = load_returns(RETURNS_PATH)
    horizons               = config["horizons"]

    print(f"\nSnapshots  : {len(snapshots)}")
    print(f"Assets     : {asset_names}")
    print(f"Returns    : {returns_df.shape[0]} days\n")

    # ── Targets & splits ──────────────────────────────────────────────────
    targets, valid_indices = build_targets(
        snapshots, returns_df, asset_names, horizons=horizons
    )
    train_pos, val_pos, test_pos = split_by_date(
        snapshots, valid_indices, config
    )

    # Build model input sequences
    # Training: train snapshots only (strict causal)
    train_snaps = to_model_input(snapshots, train_pos, valid_indices)

    # Val eval: train+val (W must evolve through training period first)
    tv_pos    = train_pos + val_pos
    tv_snaps  = to_model_input(snapshots, tv_pos, valid_indices)
    val_local = list(range(len(train_pos), len(train_pos) + len(val_pos)))

    # Test eval: train+val+test
    all_pos   = train_pos + val_pos + test_pos
    all_snaps = to_model_input(snapshots, all_pos, valid_indices)
    test_local = list(range(len(train_pos) + len(val_pos),
                            len(all_pos)))

    # ── Model ─────────────────────────────────────────────────────────────
    model = EvolveGCNH(
        node_features=4,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: EvolveGCN-H | params={n_params:,}\n")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    if WANDB_AVAILABLE:
        wandb.init(project="CASCADE", config=config, name="evolvegcn-h")

    # ── Training loop ──────────────────────────────────────────────────────
    epochs           = config["training"]["epochs"]
    patience         = config["training"]["early_stopping_patience"]
    eval_freq        = 5
    best_val_loss    = float("inf")
    patience_counter = 0
    history          = {"train_loss": [], "val_loss": [], "lr": []}

    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | "
          f"{'t+1 Dir%':>9} | {'LR':>9}")
    print("─" * 60)

    for epoch in range(1, epochs + 1):

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()

        train_preds = model(train_snaps)
        train_indices = list(range(len(train_pos)))
        train_loss  = compute_loss(train_preds, train_indices, targets, train_pos)
        train_loss.backward()

        # Clip gradients — important for GRU stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        history["train_loss"].append(train_loss.detach().item())
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # ── Validate ──────────────────────────────────────────────────────
        if epoch % eval_freq == 0:
            model.eval()
            with torch.no_grad():
                tv_preds  = model(tv_snaps)
                val_loss  = compute_loss(
                    tv_preds, val_local,
                    targets, [tv_pos[i] for i in val_local]
                )
                val_loss_f = float(val_loss)

                # Directional accuracy on t+1 for reporting
                vp = tv_preds["t1"][val_local, :, 0].numpy()
                vt = targets["t1"][[tv_pos[i] for i in val_local]].numpy()
                dacc = float(np.mean(np.sign(vp) == np.sign(vt))) * 100

            scheduler.step(val_loss_f)
            history["val_loss"].append(val_loss_f)
            lr_now = optimizer.param_groups[0]["lr"]

            print(f"{epoch:>6} | {train_loss.detach().item():>12.8f} | "
                  f"{val_loss_f:>12.8f} | {dacc:>8.1f}% | {lr_now:>9.6f}")

            if WANDB_AVAILABLE:
                wandb.log({"epoch": epoch, "train_loss": train_loss.detach().item(),
                           "val_loss": val_loss_f, "val_t1_dacc": dacc / 100})

            # Early stopping
            if val_loss_f < best_val_loss:
                best_val_loss    = val_loss_f
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch,
                                val_loss_f, CHECKPOINT_PATH)
                print(f"         ✓ Best val loss — checkpoint saved")
            else:
                patience_counter += 1
                if patience_counter >= patience // eval_freq:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

    # ── Final evaluation ───────────────────────────────────────────────────
    print(f"\nLoading best checkpoint for final evaluation...")
    load_checkpoint(model, CHECKPOINT_PATH)
    model.eval()

    with torch.no_grad():
        all_preds = model(all_snaps)

    print("\n" + "=" * 60)
    print("  Final Results on Best Checkpoint")
    print("=" * 60)

    for split_name, local_idx, global_pos in [
        ("Validation", val_local,  tv_pos),
        ("Test",       test_local, all_pos),
    ]:
        print(f"\n{split_name} Set:")
        for h_key in ["t1", "t5", "t10"]:
            p = all_preds[h_key][local_idx, :, 0].numpy()
            t = targets[h_key][[all_pos[i] for i in local_idx]].numpy()
            m = compute_metrics(p, t)
            print(f"  {h_key}: MSE={m['mse']:.8f} | "
                  f"MAE={m['mae']:.8f} | "
                  f"Dir={m['directional_accuracy']*100:.1f}%")

    # ── Save outputs for evaluate.py ───────────────────────────────────────
    os.makedirs("results", exist_ok=True)

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    # Test predictions
    with open("results/test_predictions.pkl", "wb") as f:
        pickle.dump({
            "preds":       {h: all_preds[h][test_local,:,0].numpy()
                           for h in ["t1","t5","t10"]},
            "targets":     {h: targets[h][[all_pos[i] for i in test_local]].numpy()
                           for h in ["t1","t5","t10"]},
            "dates":       [snapshots[valid_indices[all_pos[i]]][3]
                           for i in test_local],
            "asset_names": asset_names,
        }, f)

    # Val predictions
    with open("results/val_predictions.pkl", "wb") as f:
        pickle.dump({
            "preds":       {h: all_preds[h][val_local,:,0].numpy()
                           for h in ["t1","t5","t10"]},
            "targets":     {h: targets[h][[all_pos[i] for i in val_local]].numpy()
                           for h in ["t1","t5","t10"]},
            "dates":       [snapshots[valid_indices[all_pos[i]]][3]
                           for i in val_local],
            "asset_names": asset_names,
        }, f)

    # Full snapshot index for evaluate.py
    with open("results/snapshot_index.pkl", "wb") as f:
        pickle.dump({
            "valid_indices": valid_indices,
            "train_pos":     train_pos,
            "val_pos":       val_pos,
            "test_pos":      test_pos,
            "all_pos":       all_pos,
            "val_local":     val_local,
            "test_local":    test_local,
            "asset_names":   asset_names,
        }, f)

    print(f"\nSaved: results/training_history.json")
    print(f"Saved: results/test_predictions.pkl")
    print(f"Saved: results/val_predictions.pkl")
    print(f"Saved: results/snapshot_index.pkl")

    if WANDB_AVAILABLE:
        wandb.finish()

    print("\nDone. Next: python scripts/evaluate.py")
    return model, history


if __name__ == "__main__":
    train()
