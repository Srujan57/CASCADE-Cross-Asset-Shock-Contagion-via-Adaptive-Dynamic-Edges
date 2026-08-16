"""
scripts/event_catalog.py

Builds a manually-curatable event catalog for six key historical shock
events (FTX, COVID, SVB, the 2022 rate-hike cycle, the 2018 crypto winter,
and the 2020 institutional Bitcoin adoption inflection): one structured
record per event, ready for a human to attach the financial narrative.

This script does NOT write the financial narratives — that requires
domain judgment about transmission channels (dollar funding stress,
collateral channels, risk-off flows) that shouldn't be fabricated by a
script. What it does is assemble every *quantitative* artifact the model
pipeline already produces for each event — model spillover predictions,
realized returns, and the most salient edges — into one structured record
per event, with an empty "transmission_narrative" field left for a human
reviewer to fill in with the actual financial reasoning. This turns
"write six paragraphs from scratch" into "check these numbers against
what you already know happened, then explain the mechanism."

Reads (only if already produced by other scripts — degrades gracefully
if a given script hasn't been run yet):
    results/experiment2_shock_propagation.csv   (scripts/evaluate.py)
    results/experiment4_structural_break.csv    (scripts/evaluate.py)
    results/contagion_propagation_table.csv     (scripts/propagation_tables.py)
    results/edge_importance_summary.csv         (scripts/edge_importance.py)
    results/shock_events_market_wide.csv        (scripts/identify_shocks.py)

Output:
    results/event_catalog.json — one record per event, narrative field
    blank, everything else pre-filled from real pipeline outputs.

Run LAST, after evaluate.py, propagation_tables.py, edge_importance.py,
and identify_shocks.py have all been run at least once:
    python scripts/event_catalog.py
"""

import os
import json
import pandas as pd

RESULTS_DIR = "results"

# Six key events used to anchor the shock-propagation and structural-break
# analyses. Two (2018 crypto winter, 2020 institutional adoption) are
# regime/period events rather than single-day shocks, so they don't get a
# spillover injection — they're structural framing for Experiment 4 instead.
EVENTS = [
    {"name": "COVID Crash", "date": "2020-03-12", "type": "single_day_shock",
     "source_asset": "SPY"},
    {"name": "2022 Fed Rate Hike Cycle Onset", "date": "2022-01-01",
     "type": "regime_onset", "source_asset": "TLT"},
    {"name": "FTX Collapse", "date": "2022-11-09", "type": "single_day_shock",
     "source_asset": "BTC"},
    {"name": "SVB / Banking Stress", "date": "2023-03-10", "type": "single_day_shock",
     "source_asset": "SPY"},
    {"name": "2018 Crypto Winter", "date_range": ["2018-01-01", "2018-12-31"],
     "type": "regime_period", "source_asset": "BTC"},
    {"name": "2020 Institutional Bitcoin Adoption Inflection", "date": "2020-10-01",
     "type": "structural_break", "source_asset": "BTC"},
]


def safe_read_csv(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return pd.read_csv(path)
    return None


def attach_shock_propagation(record, exp2_df):
    if exp2_df is None:
        return
    rows = exp2_df[exp2_df["event"] == record["name"]]
    if rows.empty:
        return
    record["model_predicted_spillover_t1"] = [
        {"target": r["target"], "spillover": r["spillover"]}
        for _, r in rows.sort_values("spillover").iterrows()
    ]


def attach_propagation_table(record, prop_df):
    if prop_df is None:
        return
    rows = prop_df[prop_df["event"] == record["name"]]
    if rows.empty:
        return
    record["predicted_vs_actual"] = [
        {"target": r["target"], "horizon": r["horizon"],
         "predicted_spillover": r["spillover"],
         "actual_return": r["actual_return"], "gap": r["gap"]}
        for _, r in rows.iterrows()
    ]


def attach_edge_importance(record, edge_summary_df):
    if edge_summary_df is None:
        return
    rows = edge_summary_df[edge_summary_df["event"] == record["name"]]
    if rows.empty:
        return
    r = rows.iloc[0]
    record["most_salient_edge"] = f"{r['top_edge_source']}->{r['top_edge_target']}"
    record["most_salient_edge_saliency"] = float(r["top_edge_saliency"])


def attach_structural_break(record, exp4_df):
    if exp4_df is None or record["name"] != "2020 Institutional Bitcoin Adoption Inflection":
        return
    record["pre_post_2020_spillover_change"] = [
        {"asset": r["asset"], "pre_2020": r["pre_2020_mean_spillover"],
         "post_2020": r["post_2020_mean_spillover"], "change": r["change"]}
        for _, r in exp4_df.iterrows()
    ]


def main():
    exp2_df = safe_read_csv(os.path.join(RESULTS_DIR, "experiment2_shock_propagation.csv"))
    exp4_df = safe_read_csv(os.path.join(RESULTS_DIR, "experiment4_structural_break.csv"))
    prop_df = safe_read_csv(os.path.join(RESULTS_DIR, "contagion_propagation_table.csv"))
    edge_df = safe_read_csv(os.path.join(RESULTS_DIR, "edge_importance_summary.csv"))
    wide_df = safe_read_csv(os.path.join(RESULTS_DIR, "shock_events_market_wide.csv"))

    missing = [name for name, d in [
        ("experiment2_shock_propagation.csv", exp2_df),
        ("experiment4_structural_break.csv", exp4_df),
        ("contagion_propagation_table.csv", prop_df),
        ("edge_importance_summary.csv", edge_df),
        ("shock_events_market_wide.csv", wide_df),
    ] if d is None]
    if missing:
        print("Note: these upstream files aren't available yet, so the")
        print("catalog will have gaps for them. Run the corresponding")
        print("scripts first for a fully populated catalog:")
        for m in missing:
            print(f"  - {m}")
        print()

    catalog = []
    for ev in EVENTS:
        record = dict(ev)
        record["transmission_narrative"] = ""  # left for a human reviewer to fill in
        record["financial_interpretation_notes"] = ""  # e.g. dollar funding
                                                          # stress, collateral
                                                          # channels, risk-off flows

        attach_shock_propagation(record, exp2_df)
        attach_propagation_table(record, prop_df)
        attach_edge_importance(record, edge_df)
        attach_structural_break(record, exp4_df)

        catalog.append(record)

    out_path = os.path.join(RESULTS_DIR, "event_catalog.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, default=str)

    print(f"[SAVED] {out_path}")
    print(f"{len(catalog)} events scaffolded. Fill in")
    print("'transmission_narrative' and 'financial_interpretation_notes'")
    print("for each event by hand — everything else in this file is")
    print("pre-filled from real pipeline outputs.")


if __name__ == "__main__":
    main()
