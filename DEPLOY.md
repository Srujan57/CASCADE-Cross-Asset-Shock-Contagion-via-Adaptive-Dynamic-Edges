# Deploying the CASCADE dashboard to Streamlit

`streamlit_app.py` is a **read-only results dashboard** — it never imports
`torch`, never touches `data/`, and never re-runs the model. It only reads
CSV/JSON/PNG files already sitting in `results/`. That keeps the deploy
dependency-light (see `requirements_streamlit.txt`: streamlit, pandas, plotly —
nothing else).

## Before you deploy — read `results/DATA_INTEGRITY_NOTES.md`

This project went through a two-pass data-integrity audit that found and fixed
real issues (a fabricated bootstrap/robustness-check script, an untrained
baseline model, misleading code comments, stale derived files, and an
undisclosed methodological caveat). Everything is now fixed, reconfirmed
against a real re-run, and disclosed where disclosure was the right fix rather
than a code change. Read that file so you can answer questions about it live —
the dashboard's "Data Integrity & Limitations" tab shows the same information
and re-verifies two of those fixes (ablation/figure freshness) on every page
load, but you should know the story yourself before presenting it.

## 1. Push to GitHub

Streamlit Community Cloud deploys from a GitHub repo. Commit these (all
small — figures are the biggest at ~100-300KB each):

```
streamlit_app.py
requirements_streamlit.txt
.streamlit/config.toml
results/*.csv
results/*.json
results/*.md
results/figures/*.png
```

You do **not** need `data/`, `results/checkpoints/`, or `.pkl` files — the
dashboard doesn't read them, and `.gitignore` already excludes them along
with `results/_to_delete/` (pure cleanup scratch — safe to delete locally
too) and the usual `.venv/`, `wandb/`, `__pycache__/`, `.env`.

Two folders are a judgment call, not excluded by `.gitignore`:

- `results/archive_fabricated_DO_NOT_USE/` — the old fabricated CSVs, kept
  as a paper trail. The dashboard never reads it. Commit it if you want the
  history to be part of the public repo, or leave it out (delete locally
  first) if you'd rather it not ship at all — either is honest, since the
  live app doesn't depend on it either way.
- `scripts/archive_deprecated_DO_NOT_USE/phase4_results.py` — same
  situation, for the deprecated fabrication script. It's guarded against
  actually running, so shipping it is safe, but it's also fine to leave
  out of the deploy repo.

## 2. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. "New app" → pick this repo/branch.
3. Main file path: `streamlit_app.py`
4. Under "Advanced settings" → set the requirements file to
   `requirements_streamlit.txt` (Streamlit Cloud looks for `requirements.txt`
   by default — if it insists on that name, either rename
   `requirements_streamlit.txt` to `requirements.txt` for the deploy, or add
   a root `requirements.txt` that just does `-r requirements_streamlit.txt`;
   don't point it at the full research `requirements.txt`/`environment.yml`,
   which pulls in `torch`/`torch-geometric` and will slow the build down for
   no benefit to this app).
5. Deploy. First build takes a minute or two.

## 3. Local run / sanity check before sharing the link

```bash
pip install -r requirements_streamlit.txt
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
