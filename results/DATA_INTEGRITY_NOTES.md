# Data Integrity Audit — CASCADE

Audit performed before an external presentation (Google employee review), started
2026-08-16. This file is the single source of truth for which results files are
trustworthy, which are not, and what changed in the code as a result. Two passes so
far: an initial audit + real re-run, and a full second pass covering the rest of the
codebase (this update).

## Status as of the second pass

| Issue | Status |
|---|---|
| Fabricated bootstrap CI / robustness checks (`ryan_*`) | **Resolved** — replaced with real retrains, old files archived |
| Fabrication script (`phase4_results.py`) could still regenerate them | **Resolved** — script refuses to run, moved to `scripts/archive_deprecated_DO_NOT_USE/` |
| Untrained Static GCN baseline | **Resolved** — confirmed fixed by inspecting the re-run output |
| `ablation_results.csv` stale vs. current checkpoint | **Resolved** — regenerated; dashboard checks freshness live on every load |
| `results/figures/*` stale vs. current results | **Resolved** — regenerated; dashboard checks freshness live on every load |
| Regime labels mislabeled "HMM" in comments/config | **Resolved** — corrected to describe the actual VIX-threshold method |
| `regime_labels.csv` filename collision between two different labeling methods | **Resolved** — the unused K-means variant now writes to a different filename |
| Experiments 2-4 run on all snapshots, not held-out only | **Disclosed**, not a bug — see below and each tab's caption |
| Parameter count vs. dataset size (~101M params / 152 train snapshots) | Disclosed, not a bug — see below |

## Second-pass findings (this update)

### 6. The fabrication script itself was still live — RESOLVED

`scripts/phase4_results.py` was the source of the three fabricated files (see
Finding 1 below), but the *script* was untouched by the first pass — anyone running
`python scripts/phase4_results.py` out of habit would have silently regenerated
`results/ryan_bootstrap_ci.csv`, `ryan_robustness_checks.csv`, and
`ryan_latex_tables.txt` right back into `results/`, undoing the cleanup. Worse,
`generate_latex_tables()` hardcoded a narrative conclusion as a literal string —
`"DM test: CASCADE significantly beats Static GCN at all horizons (p<0.0001)"` —
instead of computing it from `dm_df`. That claim is now confirmed **false**: the
real, fixed comparison is only significant at t+5 (see Finding 3). Fix: the script's
docstring now documents exactly what was wrong with it, `main()` exits immediately
with an explanation if run, and the file was moved to
`scripts/archive_deprecated_DO_NOT_USE/phase4_results.py`.

### 7. Regime labels mislabeled as "HMM" throughout — RESOLVED

`scripts/build_graphs.py` (module docstring, two function docstrings, one print
statement) and `config.yaml` all described the regime-label feature as "HMM output"
or "HMM states." The actual method, in `scripts/fix_regime_labels.py`, is a simple
VIX-level threshold rule (calm < 20, stress 20-30, crisis > 30) — not a hidden Markov
model. The labels themselves were always real and exogenous (this was flagged as fine
in the first pass), so this was a documentation accuracy issue, not a results bug —
but it's exactly the kind of inconsistency a technical reviewer reading the code
would reasonably catch, so all references were corrected to describe the real method.

### 8. Two different regime-labeling scripts wrote to the same filename — RESOLVED

`scripts/phase2_econometrics.py::compute_regime_labels()` implements a *different*,
legitimate regime detector — K-means clustering (3 states) on VIX level + HY/IG
credit spread + SPY realized vol — but its header comment called this "VIX-based
threshold clustering," contradicting its own docstring, which correctly describes
K-means. More importantly, it wrote its output to
`data/processed/regime_labels.csv` — the *exact same path* `fix_regime_labels.py`
writes to, and the path `build_graphs.py`/`evaluate.py` actually read. Running
`phase2_econometrics.py` (needed for its legitimate DCC-GARCH and Granger-causality
outputs) would have silently overwritten the regime labels the model actually uses
with a different definition, with no error and no warning. Confirmed the currently
active `regime_labels.csv` is the VIX-threshold version (the calm/stress/crisis
split in `experiment3_regime_analysis.csv` — 217/105/26 snapshots, 62%/30%/7% — matches
`fix_regime_labels.py`'s documented expected range, not what K-means would produce).
Fixed by renaming `phase2_econometrics.py`'s output to
`regime_labels_kmeans_ALTERNATIVE.csv` and correcting its header comment.

### 9. Experiments 2-4 run on all snapshots, not held-out test data only — DISCLOSED

Read `scripts/evaluate.py`'s `evaluate()` end to end: Experiment 1 (Predictive
Accuracy) and its Diebold-Mariano tests correctly use `test_local`/held-out
predictions only. But Experiments 2 (Shock Propagation), 3 (Regime Analysis), and 4
(Structural Break) are all called with `all_snaps` = train + val + test snapshots
combined. This is defensible for what they're actually measuring — how the *trained*
model represents different regimes/shocks across its full input, an interpretability
question, not a generalization claim — but it hadn't been disclosed anywhere, and a
reviewer skimming the dashboard could reasonably assume all four experiments carry
the same held-out rigor as Experiment 1. Now called out explicitly in each of those
three tabs' captions and in the dashboard.

### 10. Ablations and figures re-generated — RESOLVED

Both were flagged "Open" in the first pass. Since then: `ablation_results.csv` was
regenerated via `python scripts/run_ablations.py` and now matches
`experiment1_accuracy.csv`'s CASCADE MSE to within 8-decimal rounding (confirmed:
diffs of ~2e-9, from `ablation_results.csv` rounding to `round(mse, 8)`).
`results/figures/*` were regenerated via `python scripts/generate_figures.py` and now
postdate `training_history.json`. The dashboard no longer just asserts these are
fresh — it re-checks both on every page load (file-timestamp comparison for figures,
value comparison for ablations) and shows a live warning if either goes stale again
in the future.

### Self-test code paths — checked, not an issue

`models/baselines.py` (`"Real data not found — generating synthetic data for
testing..."`) and `models/evolvegcn.py` (`make_fake_snapshot`, `fake_loss`) both
generate synthetic/fake data — but only inside `if __name__ == "__main__":` blocks
used as standalone sanity checks (shape/gradient tests), never imported by the actual
training or evaluation pipeline. Confirmed by reading both files end to end and
checking what `models/train.py` and `scripts/evaluate.py` actually import from them.

## First-pass findings (for the historical record)

### 1. Fabricated "robustness checks" and bootstrap CI — RESOLVED

`scripts/phase4_results.py::compute_robustness_checks()` computed
`mse * noise_factor` from closed-form formulas instead of retraining anything —
confirmed by `window=30d`/`window=90d` producing byte-identical MSE in the old
`ryan_robustness_checks.csv` (now archived). `compute_bootstrap_ci()` resampled a
Gaussian parameterized by an already-reported CI — circular, not an independent
bootstrap (old `ryan_bootstrap_ci.csv`, now archived). Replaced by
`scripts/robustness_real.py` (real retrains → `results/robustness_checks_real.csv`)
and `scripts/evaluate.py::bootstrap_ci()` (real resampling of actual predictions,
already used in `experiment1_accuracy.csv`). See Finding 6 above for how the
generating script itself was neutralized in the second pass.

### 2. Untrained Static GCN baseline — RESOLVED

`scripts/evaluate.py` previously called `.eval()` on `StaticGCN` without ever
training it. Fixed by adding `train_static_gcn()` (mirrors `models/train.py`'s
training loop) and calling it before generating predictions. Confirmed by the
re-run: Static GCN's MSE dropped from ~0.024 (the ~40x-larger untrained-network
signature) to 0.00055/0.00028/0.00022 (t+1/t+5/t+10) — the same order of magnitude
as every other model. `diebold_mariano_results.csv` now shows a mixed, plausible
picture instead of "beats noise at p<0.0001 everywhere": CASCADE vs. Static GCN is
*not* significant at t+1 (p=0.78) or t+10 (p=0.17, nominally favoring Static GCN),
and *is* significant at t+5 (p≈1.0e-06) — itself evidence the fix worked.

### 3. `robustness_checks_real.csv` — verified real and internally consistent

24 rows (8 parameter variants × 3 horizons), verified against the 4 shard files it
was merged from with zero rows lost or duplicated. The three "baseline"
configurations (`threshold=0.3`, `window=60d`, `11-asset (full)`) report
byte-identical MSE/MAE — **expected, not a red flag**: those three jobs are
literally the same config trained with the same fixed seed (42), so identical
results are exactly what deterministic training should produce — the opposite of
the earlier symmetric-formula fingerprint (different configs producing identical
output).

### 4. Parameter count vs. dataset size — disclosed, not a bug

EvolveGCN-H's GRU evolves a *flattened* GCN weight matrix; at `hidden_dim=64` this
is ≈101M parameters trained on 152 weekly training snapshots (effective window
starting ~Nov 2017, since ETH-USD has no Yahoo Finance history before then and
`build_graphs.py` drops snapshots with missing assets). Extreme ratio, inherent to
the architecture. Test-set directional accuracy (48.2-49.0% across horizons) and
train/validation loss converging to similar values with no runaway divergence remain
consistent with "the model learned something modest," not an inflated, overfit-looking
result.

### 5. Things that checked out fine (both passes)

- Train/val/test splits are strictly chronological everywhere, never shuffled — no
  look-ahead leakage (`models/train.py`, `scripts/run_ablations.py`).
- Model selection uses validation loss only; the test set is touched once, for final
  reporting (`models/train.py`).
- Regime labels come from real, exogenous VIX levels, not the prediction target
  (terminology describing them was wrong — Finding 7 — but the labels themselves
  were always real).
- `scripts/identify_shocks.py` independently cross-checks hand-picked shock events
  against a systematic rolling z-score rule.
- `scripts/event_catalog.py` explicitly leaves the narrative "transmission channel"
  field blank for a human to fill in, rather than generating plausible-sounding prose.
- `scripts/edge_importance.py` correctly avoids claiming an attention mechanism
  EvolveGCN-H doesn't have.
- `scripts/generate_figures.py` and `scripts/propagation_tables.py` read only from
  already-saved CSVs/JSON — no synthetic data injection.
- `fix_regime_labels.py` is a legitimate bugfix, not a result-shaping change.
- The 55%-single-day-return data quality flag turned out to be real historical
  events (Black Thursday 2020-03-12, negative-oil-price April 2020, the May 2021
  and June 2022 crypto crashes, etc.), not a data glitch — checked against known
  event dates before proceeding.
- `training_history.json` shows healthy convergence on the re-run: 80 epochs before
  early stopping, train loss ~0.00065, val loss ~0.0008, no divergence or instability.

## Results/scripts folder cleanup

- `robustness_checks_real_shard{1..4}of4.csv` and `robustness_checks_real_quick.csv`
  → moved to `results/_to_delete/` (fully superseded by the merged
  `robustness_checks_real.csv`; safe to permanently delete that folder yourself).
- `ryan_bootstrap_ci.csv`, `ryan_robustness_checks.csv`, `ryan_latex_tables.txt`
  → moved to `results/archive_fabricated_DO_NOT_USE/` rather than deleted outright,
  so there's still a paper trail. Nothing in the repo or the dashboard reads from
  that folder.
- `scripts/phase4_results.py` → moved to `scripts/archive_deprecated_DO_NOT_USE/`
  and hard-guarded against running (see Finding 6).

## Not part of this audit (flagged, not fixed)

- `README.md` was effectively empty (one line, no description) — replaced with an
  honest project summary pointing at this file and `DEPLOY.md`.
- `paper/main.tex` and `paper/references.bib` are both present but empty (0 bytes) —
  left as-is; not a data-integrity issue, just an unstarted stub. Worth deleting or
  filling in before a real paper submission, but out of scope for this audit.
