"""
scripts/propagation_tables.py

Fills another Phase 4 gap: "Produce the contagion propagation tables: for
each shock at node X, show predicted vs. actual changes at all other nodes
across t+1, t+5, t+10." scripts/evaluate.py's inject_shock() only ever reads
the "t1" prediction head — experiment2_shock_propagation.csv has no t+5 or
t+10 columns, and never compares against realized (actual) returns at all,
only "shocked_pred" vs "baseline_pred" (both model outputs). This script
adds the two missing pieces: multi-horizon predictions, and the actual
realized market move to compare them against.

Note on what "actual" means here: the model predicts return at t+1/t+5/t+10
GIVEN a hypothetical shock injected into node features. There's no ground
truth for "what would have happened under a counterfactual shock" — actuals
can only be compared against baseline_pred (unshocked forecast) at the *real*
event date, since the four events used here were actual historical shocks the
market already experienced. So "actual" in this table means: the realized
return that occurred, and "predicted" means the model's shocked-scenario
forecast issued from the pre-shock snapshot. Large predicted-vs-actual gaps
are informative (model over/under-estimates real contagion) but should not
be read as a backtested trading signal — document this framing in Results.

Outputs:
    results/contagion_propagation_table.csv
    results/propagation_latex_table.txt   (Table 3, appendix-ready)

Run AFTER models/train.py:
    python scripts/propagation_tables.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.evolvegcn import EvolveGCNH
from models.train import (load_config, load_snapshots, load_returns,
                           build_targets, split_by_date, to_model_input,
                           load_checkpoint)

RESULTS_DIR = "results"

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

HORIZON_DAYS = {"t1": 1, "t5": 5, "t10": 10}
N_CONTEXT = 20


def inject_shock_multi_horizon(model, snapshots, shock_node_idx, shock_return,
                               start_pos, n_context=N_CONTEXT):
    """
    Same protocol as scripts/evaluate.py's inject_shock(), but reads all
    three prediction heads instead of only t1.
    """
    model.eval()
    context_start = max(0, start_pos - n_context)
    context_snaps = snapshots[context_start:start_pos]
    x, ei, ew = snapshots[start_pos]

    x_shocked = x.clone()
    x_shocked[shock_node_idx, 0] = shock_return

    with torch.no_grad():
        baseline_preds = model(context_snaps + [(x, ei, ew)])
        shocked_preds  = model(context_snaps + [(x_shocked, ei, ew)])

    result = {}
    for h_key in ["t1", "t5", "t10"]:
        baseline = baseline_preds[h_key][-1, :, 0].numpy()
        shocked  = shocked_preds[h_key][-1, :, 0].numpy()
        result[h_key] = {"baseline": baseline, "shocked": shocked,
                         "spillover": shocked - baseline}
    return result


def actual_forward_returns(returns_df, asset_names, event_date, horizons_days):
    """
    Realized return from event_date to event_date + h trading days, per
    asset. Uses positional indexing on the trading calendar, matching
    models/train.py's build_targets() convention.
    """
    trading_days = returns_df.index
    if event_date not in trading_days:
        # snap to nearest trading day on or after the event date
        later = trading_days[trading_days >= event_date]
        if len(later) == 0:
            return None
        event_date = later[0]

    pos = trading_days.get_loc(event_date)
    out = {}
    for h_key, h in horizons_days.items():
        target_pos = pos + h
        if target_pos >= len(trading_days):
            out[h_key] = None
            continue
        out[h_key] = returns_df[asset_names].iloc[target_pos]
    return out


def build_latex_table(df):
    lines = []
    lines.append("% TABLE 3: Contagion Propagation — Predicted vs Actual")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Predicted spillover vs.\\ realized return by shock event and horizon}")
    lines.append("\\label{tab:propagation}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{llccccc}")
    lines.append("\\toprule")
    lines.append("Event & Target & Horizon & Predicted Spillover & Realized Return & Baseline Fcst & Gap \\\\")
    lines.append("\\midrule")

    for event in df["event"].unique():
        edf = df[df["event"] == event]
        first = True
        for _, row in edf.iterrows():
            if pd.isna(row["actual_return"]):
                continue
            ev_col = event if first else ""
            first = False
            lines.append(
                f"{ev_col} & {row['target']} & {row['horizon']} & "
                f"{row['spillover']:.5f} & {row['actual_return']:.5f} & "
                f"{row['baseline_pred']:.5f} & {row['gap']:.5f} \\\\"
            )
        lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


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
    model.eval()

    rows = []

    for event in SHOCK_EVENTS:
        event_date = pd.Timestamp(event["date"])
        diffs = [abs((d - event_date).days) for d in snap_dates]
        idx = int(np.argmin(diffs))
        if diffs[idx] > 14 or idx < N_CONTEXT:
            print(f"  {event['name']}: no usable snapshot, skipping")
            continue

        source_idx = asset_names.index(event["source"]) if event["source"] in asset_names else 0

        preds = inject_shock_multi_horizon(
            model, all_snaps, source_idx, event["shock_val"], idx
        )
        actuals = actual_forward_returns(returns_df, asset_names, event_date, HORIZON_DAYS)

        print(f"\n  {event['name']} ({event_date.date()}), source={event['source']}")
        for h_key in ["t1", "t5", "t10"]:
            actual_row = actuals[h_key] if actuals else None
            for i, asset in enumerate(asset_names):
                if i == source_idx:
                    continue
                spillover = float(preds[h_key]["spillover"][i])
                baseline  = float(preds[h_key]["baseline"][i])
                actual    = float(actual_row[asset]) if actual_row is not None else np.nan
                gap       = spillover - actual if not np.isnan(actual) else np.nan

                rows.append({
                    "event": event["name"],
                    "date": str(event_date.date()),
                    "source": event["source"],
                    "target": asset,
                    "horizon": h_key,
                    "spillover": spillover,
                    "baseline_pred": baseline,
                    "actual_return": actual,
                    "gap": gap,
                })

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "contagion_propagation_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] {csv_path}")

    latex = build_latex_table(df)
    latex_path = os.path.join(RESULTS_DIR, "propagation_latex_table.txt")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"[SAVED] {latex_path}")
    print("\nLaTeX-formatted version of results/contagion_propagation_table.csv.")


if __name__ == "__main__":
    main()
