"""
scripts/edge_importance.py

Fills a real gap in the project plan: Phase 3 requires "attention weights
showing which edges activate during shocks" as an interpretability output,
and Phase 4 requires "attention weight evolution over time" as a paper
figure. EvolveGCN-H (the architecture actually built in models/evolvegcn.py)
has no attention mechanism — that trade-off belongs to TGAT, the plan's
alternative architecture. Searching models/ and scripts/ confirms "attention"
never appears in the codebase, so this output was never produced.

This script does NOT retrofit fake attention onto EvolveGCN-H. Instead it
computes gradient-based edge saliency, a standard and defensible substitute:
for a chosen shock event, backpropagate the model's t+1 prediction at the
shock's target node(s) with respect to every edge_weight in that snapshot's
graph, and rank edges by |gradient|. This answers the same question the
plan wants ("which edges activate during a shock") without pretending the
model has attention it doesn't have. Document this substitution explicitly
in the Methodology section — reviewers who know EvolveGCN-H will ask.

Method:
    1. Load the trained checkpoint and the full snapshot sequence.
    2. For each known shock event (reuses the shock_events list from
       scripts/evaluate.py), find the nearest graph snapshot.
    3. Clone that snapshot's edge_weight tensor with requires_grad=True.
    4. Warm up W through n_context prior snapshots (no grad needed there —
       only the target snapshot's edges are being attributed).
    5. Forward pass through the target snapshot, take the t+1 prediction
       at the shock's target nodes (or shocked node's spillover targets),
       sum, and backpropagate.
    6. |edge_weight.grad| is the saliency score per edge — how much moving
       that correlation up/down would move the prediction.

Outputs:
    results/edge_importance_<event_slug>.csv   — per-edge saliency ranking
    results/figures/fig7_edge_importance_<event_slug>.png/.pdf — bar chart
        of the top-N most influential edges for that event

Run AFTER models/train.py (needs results/checkpoints/best_model.pt):
    python scripts/edge_importance.py
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.evolvegcn import EvolveGCNH
from models.train import (load_config, load_snapshots, load_returns,
                           build_targets, split_by_date, to_model_input,
                           load_checkpoint)

RESULTS_DIR = "results"
FIG_DIR     = "results/figures"

# Same four events as scripts/evaluate.py's experiment2_shock_propagation —
# keep these in sync if the event list there changes.
SHOCK_EVENTS = [
    {"name": "COVID Crash",       "date": "2020-03-12",
     "source": "SPY", "shock_val": -0.095},
    {"name": "FTX Collapse",      "date": "2022-11-09",
     "source": "BTC", "shock_val": -0.246},
    {"name": "SVB Crisis",        "date": "2023-03-10",
     "source": "SPY", "shock_val": -0.048},
    {"name": "2022 Rate Shock",   "date": "2022-06-13",
     "source": "TLT", "shock_val": -0.031},
]

N_CONTEXT = 20
TOP_N     = 15


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def find_closest_snapshot(snap_dates, event_date, max_days=14):
    diffs = [abs((d - event_date).days) for d in snap_dates]
    idx = int(np.argmin(diffs))
    if diffs[idx] > max_days:
        return None
    return idx


def edge_saliency_for_event(model, all_snaps, asset_names, target_idx,
                             shock_node_idx, shock_return, n_context=N_CONTEXT):
    """
    Returns a DataFrame with columns [source, target, edge_weight, saliency]
    for every edge in the target snapshot's graph, ranked by |gradient of
    the shocked node's t+1 spillover prediction w.r.t. that edge's weight|.
    """
    model.eval()

    context_start = max(0, target_idx - n_context)
    context_snaps = all_snaps[context_start:target_idx]
    x, ei, ew = all_snaps[target_idx]

    x_shocked = x.clone()
    x_shocked[shock_node_idx, 0] = shock_return

    ew_grad = ew.clone().detach().requires_grad_(True)

    # Warm up W with no-grad context, then a single grad-tracked forward
    # pass through the target snapshot only. We re-run context with grad
    # disabled to keep the graph small — only the target edge_weight needs
    # gradient tracking for this saliency computation.
    with torch.no_grad():
        W_current = [w.clone() for w in model.W]
        h = None
        for cx, cei, cew in context_snaps:
            hh = cx
            for i in range(model.num_layers):
                W_flat = W_current[i].view(1, -1)
                W_evolved = model.grus[i](W_flat, W_flat)
                W_current[i] = W_evolved.view(model.w_shapes[i])
                hh = model.convs[i](hh, cei, cew, W_current[i])
                hh = torch.relu(hh)

    # Grad-tracked forward on the shocked target snapshot, continuing from
    # the warmed-up (detached) W state.
    h = x_shocked
    for i in range(model.num_layers):
        W_flat = W_current[i].detach().view(1, -1)
        W_evolved = model.grus[i](W_flat, W_flat)
        W_step = W_evolved.view(model.w_shapes[i])
        h = model.convs[i](h, ei, ew_grad, W_step)
        h = torch.relu(h)

    pred_t1 = model.predictors["t1"](h)[:, 0]   # (N,)

    # Attribute the sum of predicted spillover across all non-source nodes —
    # this is "what drove the model's predicted contagion," not just one
    # target node, matching the plan's framing of shock -> all-node spillover.
    mask = torch.ones_like(pred_t1)
    mask[shock_node_idx] = 0.0
    objective = (pred_t1 * mask).sum()

    model.zero_grad(set_to_none=True)
    if ew_grad.grad is not None:
        ew_grad.grad.zero_()
    objective.backward()

    grads = ew_grad.grad.detach().abs().numpy()
    src, dst = ei.numpy()

    rows = []
    for e in range(len(grads)):
        rows.append({
            "source":      asset_names[src[e]],
            "target":      asset_names[dst[e]],
            "edge_weight": float(ew[e].item()),
            "saliency":    float(grads[e]),
        })

    df = pd.DataFrame(rows).sort_values("saliency", ascending=False)
    return df.reset_index(drop=True)


def plot_top_edges(df, event_name, out_path, top_n=TOP_N):
    top = df.head(top_n).iloc[::-1]
    labels = [f"{r.source}->{r.target}" for r in top.itertuples()]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(labels, top["saliency"], color="#C0392B", alpha=0.85)
    ax.set_xlabel("Edge saliency  |d(prediction)/d(edge weight)|")
    ax.set_title(f"Most influential edges — {event_name}\n"
                 f"(gradient-based saliency, EvolveGCN-H has no attention "
                 f"mechanism)", fontsize=10)
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(f"{out_path}.png", dpi=200)
    fig.savefig(f"{out_path}.pdf")
    plt.close(fig)


def main():
    config = load_config("config.yaml")

    SNAPSHOTS_PATH  = "data/processed/graph_snapshots.pkl"
    RETURNS_PATH    = "data/processed/returns_matrix.csv"
    CHECKPOINT_PATH = "results/checkpoints/best_model.pt"

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"No checkpoint found at {CHECKPOINT_PATH}. Run models/train.py first.")
        return
    if not os.path.exists(SNAPSHOTS_PATH):
        print(f"No snapshots found at {SNAPSHOTS_PATH}. Run scripts/build_graphs.py first.")
        return

    snapshots, asset_names = load_snapshots(SNAPSHOTS_PATH)
    returns_df = load_returns(RETURNS_PATH)
    horizons = config["horizons"]

    targets, valid_indices = build_targets(snapshots, returns_df, asset_names, horizons=horizons)
    train_pos, val_pos, test_pos = split_by_date(snapshots, valid_indices, config)
    all_pos = train_pos + val_pos + test_pos
    all_snaps = to_model_input(snapshots, all_pos, valid_indices)
    snap_dates = [snapshots[valid_indices[p]][3] for p in all_pos]

    model = EvolveGCNH(
        node_features=4,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=0.0,
    )
    load_checkpoint(model, CHECKPOINT_PATH)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    summary_rows = []

    for event in SHOCK_EVENTS:
        event_date = pd.Timestamp(event["date"])
        idx = find_closest_snapshot(snap_dates, event_date)
        if idx is None or idx < N_CONTEXT:
            print(f"  {event['name']}: no usable snapshot with enough context, skipping")
            continue

        source_idx = asset_names.index(event["source"]) if event["source"] in asset_names else 0

        df = edge_saliency_for_event(
            model, all_snaps, asset_names,
            target_idx=idx,
            shock_node_idx=source_idx,
            shock_return=event["shock_val"],
        )

        slug = slugify(event["name"])
        csv_path = os.path.join(RESULTS_DIR, f"edge_importance_{slug}.csv")
        df.to_csv(csv_path, index=False)

        fig_path = os.path.join(FIG_DIR, f"fig7_edge_importance_{slug}")
        plot_top_edges(df, event["name"], fig_path)

        top_edge = df.iloc[0]
        print(f"  {event['name']:16s} -> top edge: "
              f"{top_edge['source']}->{top_edge['target']} "
              f"(saliency={top_edge['saliency']:.6f}) | saved {csv_path}")

        summary_rows.append({
            "event": event["name"],
            "date": str(event_date.date()),
            "snapshot_date": str(snap_dates[idx].date()),
            "top_edge_source": top_edge["source"],
            "top_edge_target": top_edge["target"],
            "top_edge_saliency": top_edge["saliency"],
            "n_edges": len(df),
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(RESULTS_DIR, "edge_importance_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[SAVED] {summary_path}")

    print("\nDone. These are gradient saliency maps, not attention weights —")
    print("state that explicitly in the paper's Methodology and Discussion")
    print("sections when interpreting fig7_edge_importance_*.")


if __name__ == "__main__":
    main()
