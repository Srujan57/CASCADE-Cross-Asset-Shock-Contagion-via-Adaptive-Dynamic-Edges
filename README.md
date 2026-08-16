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

## Before you cite or present anything from this repo

Read `results/DATA_INTEGRITY_NOTES.md` first. This project went through a
pre-presentation data-integrity audit that found and fixed real issues — a
fabricated bootstrap/robustness-check script and an untrained baseline model, among
smaller things — and that file is the single source of truth for what's trustworthy,
what was fixed, and what's disclosed as a known limitation rather than hidden. The
dashboard (`streamlit_app.py`) surfaces the same information live, including checks
that re-verify certain files haven't gone stale every time it loads.

## Running the pipeline

See `DEPLOY.md` for environment setup and how to run the dashboard. The model
training/evaluation pipeline itself requires `data/` (gitignored — pull raw data via
`scripts/data_ingestion.py`) and is not needed to run the dashboard, which only reads
already-generated files in `results/`.
