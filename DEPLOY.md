# Deploying the CASCADE dashboard to Streamlit

`streamlit_app.py` is a **read-only results dashboard** — it never imports
`torch`, never touches `data/`, and never re-runs the model. It only reads
CSV/JSON/PNG files already sitting in `results/`. That keeps the deploy
dependency-light: repo-root `requirements.txt` now contains exactly
`streamlit`, `pandas`, `plotly` — nothing else.

> **Correction, if you deployed from an earlier version of this guide:**
> this doc previously said to set a custom requirements filename in
> Streamlit Cloud's "Advanced settings." That option doesn't exist. Cloud
> auto-detects a dependency file by exact name only — checking
> `uv.lock` → `Pipfile` → `environment.yml` → `requirements.txt` →
> `pyproject.toml`, first match wins — and this repo had both
> `requirements.txt` and `environment.yml` at root, so Cloud silently used
> `environment.yml` (the full research conda env, which never included
> `plotly`), producing `ModuleNotFoundError: No module named 'plotly'` in
> production. Fixed: `environment.yml` is renamed to
> `environment.research.yml` so it no longer collides, and repo-root
> `requirements.txt` is now the dashboard-only file described below. If
> your deployed app is still showing that error, push these two renames
> and Cloud will pick up the fix on its next rebuild (or trigger a manual
> reboot from "Manage app").

## Background — `results/DATA_INTEGRITY_NOTES.md`

This project went through a two-pass data-integrity audit that found and fixed
real issues (a fabricated bootstrap/robustness-check script, an untrained
baseline model, misleading code comments, stale derived files, and an
undisclosed methodological caveat). Everything is now fixed and reconfirmed
against a real re-run. That file is an internal record of the process; the
dashboard's "Methodology & Limitations" tab covers the current state of the
same caveats in plain language, without the audit history.

## 1. Push to GitHub

Streamlit Community Cloud deploys from a GitHub repo. Commit these (all
small — figures are the biggest at ~100-300KB each):

```
streamlit_app.py
requirements.txt
.streamlit/config.toml
results/*.csv
results/*.json
results/*.md
results/figures/*.png
```

**Critical: `requirements.txt` must be the only dependency file Streamlit
Cloud can find at repo root.** If you reintroduce a root-level
`environment.yml`, `Pipfile`, or `uv.lock`, it will silently take priority
over `requirements.txt` again (see the correction note above) — keep the
full research environment in `environment.research.yml` /
`requirements-core.txt` instead, which Cloud's auto-discovery won't match.

You do **not** need `data/`, `results/checkpoints/`, or `.pkl` files — the
dashboard doesn't read them, and `.gitignore` already excludes them along
with `results/_to_delete/` (pure cleanup scratch — safe to delete locally
too) and the usual `.venv/`, `wandb/`, `__pycache__/`, `.env`.

The old fabricated result files and the script that generated them have
been deleted from the repo entirely (see `results/DATA_INTEGRITY_NOTES.md`)
— nothing to decide there, there's simply nothing left to commit or exclude.

## 2. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. "New app" → pick this repo/branch.
3. Main file path: `streamlit_app.py`
4. Advanced settings only has Python version and secrets — nothing to
   change there for this app. Dependencies are picked up automatically from
   repo-root `requirements.txt` (see the correction note above for why this
   matters).
5. Deploy. First build takes well under a minute — `streamlit`, `pandas`,
   and `plotly` are the only packages installed, no `torch`/`torch-geometric`.

## 3. Local run / sanity check before sharing the link

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Smoke-tested with `streamlit.testing.v1.AppTest` after every content change
in this audit — zero exceptions across all 9 tabs and every radio/dropdown
combination. Re-run that check yourself after any further edits to
`results/`, since the app degrades gracefully (shows an info box) for any
missing file rather than crashing, which can silently hide a typo in a
filename:

```bash
python3 -c "
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('streamlit_app.py')
at.run(timeout=60)
assert not at.exception, at.exception
print('OK — no exceptions')
"
```

## 4. If you regenerate results/ later

`ablation_results.csv` and `results/figures/*` need to stay in sync with
`experiment1_accuracy.csv` / `training_history.json` — the dashboard checks
this live (Ablations, Figures, and Data Integrity tabs) and will show a
warning banner if they drift out of sync again, so you don't need to
remember to check by hand. If you see that warning: re-run
`python scripts/run_ablations.py` and/or `python scripts/generate_figures.py`
before redeploying.

Do **not** run `scripts/phase4_results.py` — it's deprecated and will refuse
to run, but it exists as a reminder of what not to reintroduce (see
`results/DATA_INTEGRITY_NOTES.md`, Finding 6). If you need robustness checks
or bootstrap CIs regenerated, use `scripts/robustness_real.py` and
`scripts/evaluate.py::bootstrap_ci()` instead — both do real computation, not
closed-form shortcuts.
