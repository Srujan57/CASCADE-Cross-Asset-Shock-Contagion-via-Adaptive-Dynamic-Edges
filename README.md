# CASCADE — Cross-Asset Shock Contagion via Adaptive Dynamic Edges

A temporal graph neural network (EvolveGCN-H) that models return forecasting and
shock propagation across a 10-asset universe — equities, bonds, commodities, and
crypto — from 2015-2024, with VIX and DXY as macro features.

## What's here

- `models/` — EvolveGCN-H architecture (`evolvegcn.py`), baselines (VAR, rolling
  correlation, static GCN — `baselines.py`), and the training loop (`train.py`).
- `scripts/` — the full pipeline: data ingestion, graph construction, econometrics
  (DCC-GARCH, Granger causality, regime labeling), evaluation, ablations, robustness
  checks, and figure generation. Run in the order described in `DEPLOY.md`.
- `results/` — CSVs, figures, and checkpoints from the most recent pipeline run.
- `streamlit_app.py` — a read-only results dashboard. See `DEPLOY.md` to run it
  locally or deploy it to Streamlit Community Cloud.

## Honest headline finding

On the held-out 2023-2024 test period, CASCADE's directional accuracy sits at
48.2-49.0% across all three forecast horizons (t+1, t+5, t+10) — statistically
indistinguishable from a coin flip — and a Diebold-Mariano test finds it is not
significantly more accurate than a linear VAR baseline at any horizon. CASCADE does
significantly beat naive rolling-correlation at t+5/t+10, and beats a properly
trained Static GCN baseline at t+5 only. This is a defensible,
market-efficiency-consistent result, not a "beats everything" story — see the
dashboard's Overview and Predictive Accuracy tabs for the full picture.

## Data integrity

`results/DATA_INTEGRITY_NOTES.md` documents a data-integrity audit performed on
this project — what was found, what was fixed, and what's disclosed as a known
limitation rather than hidden. It's an internal record of that process. The
dashboard's **Methodology & Limitations** tab covers the current, forward-looking
version of the same caveats — how the model was trained and evaluated, what each
tab does and doesn't claim, and what to know before drawing conclusions from any
single number in it.

## Running the pipeline

See `DEPLOY.md` for environment setup and how to run the dashboard. The model
training/evaluation pipeline itself requires `data/` (gitignored — pull raw data via
`scripts/data_ingestion.py`) and is not needed to run the dashboard, which only reads
already-generated files in `results/`.
