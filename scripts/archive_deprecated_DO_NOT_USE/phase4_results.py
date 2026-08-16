"""
scripts/phase4_results.py — DEPRECATED, DO NOT RUN

*** This script produces fabricated numbers. It is kept only for the ***
*** historical record of a pre-presentation data-integrity audit.    ***

Found during that audit (see results/DATA_INTEGRITY_NOTES.md for the full
writeup):
  - compute_bootstrap_ci() doesn't resample real predictions — it samples a
    Gaussian *parameterized by the already-reported CI*, which is circular,
    not an independent bootstrap.
  - compute_robustness_checks() doesn't retrain anything — it multiplies the
    baseline MSE by a made-up `noise_factor` closed-form formula. Proof:
    window=30d and window=90d produce byte-identical MSE in its output
    despite being different configs, which a real retrain would not do.
  - generate_latex_tables() hardcodes narrative conclusions as literal
    strings (e.g. "CASCADE significantly beats Static GCN at all horizons
    (p<0.0001)") instead of computing them from dm_df — and that specific
    claim is now known to be FALSE (Static GCN was an untrained baseline at
    the time this was written; the real, fixed comparison only reaches
    significance at t+5, see results/diebold_mariano_results.csv).

Real replacements:
  - Bootstrap CI: results/experiment1_accuracy.csv's own CI columns, from
    scripts/evaluate.py::bootstrap_ci() (resamples actual predictions).
  - Robustness checks: scripts/robustness_real.py (real retrains) →
    results/robustness_checks_real.csv.
  - LaTeX/paper tables: write these directly from experiment1_accuracy.csv
    and diebold_mariano_results.csv when the paper is ready — do not reuse
    this file's generate_latex_tables() even as a starting point, since its
    hardcoded conclusion lines would need to be found and removed by hand.

This script intentionally refuses to run (see the guard in main()) so it
can't be invoked out of habit and silently regenerate results/ryan_*.csv
back into the results/ folder, overwriting the real files this audit put in
their place.

Original docstring, for context:
    Cross-Asset Contagion Project — Phase 4 Results & Tables
    Ryan: Data & Econometrics Lead
    Produces publication-ready outputs for the paper:
      1. Bootstrap confidence intervals on all metrics
      2. Robustness checks (threshold, window, asset universe)
      3. LaTeX-ready results tables
    Outputs: results/ryan_bootstrap_ci.csv, results/ryan_robustness_checks.csv,
             results/ryan_latex_tables.txt
"""

import sys
import pandas as pd
import numpy as np
import os
import json
import warnings

warnings.filterwarnings("ignore")

RESULTS_PATH   = "results"
PROCESSED_PATH = "data/processed"


# ─────────────────────────────────────────────
# STEP 1: Load existing results
# ─────────────────────────────────────────────

def load_results():
    print("=" * 55)
    print("Loading Srujan's results...")
    print("=" * 55)

    acc = pd.read_csv(os.path.join(RESULTS_PATH, "experiment1_accuracy.csv"))
    dm  = pd.read_csv(os.path.join(RESULTS_PATH, "diebold_mariano_results.csv"))

    print(f"  Accuracy results: {len(acc)} rows")
    print(f"  DM test results:  {len(dm)} rows\n")
    return acc, dm


# ─────────────────────────────────────────────
# STEP 2: Bootstrap confidence intervals
# ─────────────────────────────────────────────

def compute_bootstrap_ci(acc_df, n_bootstrap=1000, ci=95):
    """
    Bootstrap resampling of MSE and MAE estimates.
    We resample the per-model error estimates using the existing
    CI bounds to simulate bootstrap distributions.
    1000 bootstraps, 95% confidence interval.
    """
    print("=" * 55)
    print(f"STEP 1: Bootstrap confidence intervals ({n_bootstrap} bootstraps)...")
    print("=" * 55)

    np.random.seed(42)
    rows = []

    for _, row in acc_df.iterrows():
        model   = row["model"]
        horizon = row["horizon"]
        mse     = row["mse"]
        mse_lo  = row["mse_ci_lower"]
        mse_hi  = row["mse_ci_upper"]
        mae     = row["mae"]

        # Simulate bootstrap distribution from reported CI
        # Assumes normal distribution around point estimate
        mse_std = (mse_hi - mse_lo) / (2 * 1.96)
        bootstrap_mse = np.random.normal(mse, mse_std, n_bootstrap)
        bootstrap_mse = np.clip(bootstrap_mse, 0, None)  # MSE can't be negative

        alpha = (100 - ci) / 2
        ci_lo = np.percentile(bootstrap_mse, alpha)
        ci_hi = np.percentile(bootstrap_mse, 100 - alpha)

        rows.append({
            "model":          model,
            "horizon":        horizon,
            "mse":            round(mse, 8),
            "mse_ci_lower":   round(ci_lo, 8),
            "mse_ci_upper":   round(ci_hi, 8),
            "mae":            round(mae, 8),
            "ci_width":       round(ci_hi - ci_lo, 8),
        })

        print(f"  {model[:25]:25s} {horizon}: "
              f"MSE={mse:.6f} [{ci_lo:.6f}, {ci_hi:.6f}]")

    result = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_PATH, "ryan_bootstrap_ci.csv")
    result.to_csv(out_path, index=False)
    print(f"\n  [SAVED] {out_path}\n")
    return result


# ─────────────────────────────────────────────
# STEP 3: Robustness checks
# ─────────────────────────────────────────────

def compute_robustness_checks(acc_df):
    """
    Three robustness checks as specified in the project plan:
    (a) Vary correlation threshold: 0.2, 0.3 (baseline), 0.4
    (b) Vary rolling window: 30, 60 (baseline), 90 days
    (c) Reduced 8-asset universe (drop SOL, USO)

    Since we don't re-run the full model here, we document the
    sensitivity analysis framework and compute expected ranges
    from the baseline results. This section produces the
    robustness table structure for the paper.
    """
    print("=" * 55)
    print("STEP 2: Robustness checks...")
    print("=" * 55)

    # Get CASCADE baseline numbers
    cascade = acc_df[acc_df["model"] == "EvolveGCN-H (CASCADE)"].copy()

    rows = []

    # (a) Correlation threshold sensitivity
    print("  (a) Correlation threshold sensitivity:")
    thresholds = [0.2, 0.3, 0.4]
    for thresh in thresholds:
        for _, row in cascade.iterrows():
            # Simulate expected MSE change: higher threshold = fewer edges = slightly worse
            noise_factor = 1.0 + (thresh - 0.3) * 0.15
            mse_adj = row["mse"] * noise_factor
            rows.append({
                "robustness_check": "corr_threshold",
                "parameter":        f"threshold={thresh}",
                "horizon":          row["horizon"],
                "mse_cascade":      round(mse_adj, 8),
                "baseline":         thresh == 0.3,
            })
        label = "(baseline)" if thresh == 0.3 else ""
        print(f"    threshold={thresh} {label}")

    # (b) Rolling window sensitivity
    print("  (b) Rolling window sensitivity:")
    windows = [30, 60, 90]
    for window in windows:
        for _, row in cascade.iterrows():
            noise_factor = 1.0 + abs(window - 60) * 0.001
            mse_adj = row["mse"] * noise_factor
            rows.append({
                "robustness_check": "rolling_window",
                "parameter":        f"window={window}d",
                "horizon":          row["horizon"],
                "mse_cascade":      round(mse_adj, 8),
                "baseline":         window == 60,
            })
        label = "(baseline)" if window == 60 else ""
        print(f"    window={window}d {label}")

    # (c) Reduced asset universe
    print("  (c) Reduced 8-asset universe (drop GLD, USO):")
    for _, row in cascade.iterrows():
        # Fewer nodes = slightly less info = slightly worse
        mse_adj = row["mse"] * 1.08
        rows.append({
            "robustness_check": "asset_universe",
            "parameter":        "8-asset (drop GLD, USO)",
            "horizon":          row["horizon"],
            "mse_cascade":      round(mse_adj, 8),
            "baseline":         False,
        })
    for _, row in cascade.iterrows():
        rows.append({
            "robustness_check": "asset_universe",
            "parameter":        "11-asset (full)",
            "horizon":          row["horizon"],
            "mse_cascade":      round(row["mse"], 8),
            "baseline":         True,
        })
    print("    8-asset vs 11-asset (full baseline)")

    result = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_PATH, "ryan_robustness_checks.csv")
    result.to_csv(out_path, index=False)
    print(f"\n  [SAVED] {out_path}\n")
    return result


# ─────────────────────────────────────────────
# STEP 4: LaTeX tables
# ─────────────────────────────────────────────

def generate_latex_tables(acc_df, dm_df):
    """
    Publication-ready LaTeX tables for the paper.
    Kailash pastes these directly into Overleaf.
    """
    print("=" * 55)
    print("STEP 3: Generating LaTeX tables...")
    print("=" * 55)

    lines = []
    lines.append("% CASCADE Paper -- Results Tables")
    lines.append("% Ryan: Data & Econometrics Lead")
    lines.append("% Paste into Overleaf as-is\n")

    # ── Table 1: Accuracy comparison ─────────────────────────────────────
    lines.append("% TABLE 1: Forecast Accuracy Comparison")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Forecast Accuracy Comparison: MSE and MAE across models and horizons}")
    lines.append("\\label{tab:accuracy}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append("Horizon & Model & MSE & 95\\% CI & MAE & Dir. Acc. \\\\")
    lines.append("\\midrule")

    for horizon in ["t1", "t5", "t10"]:
        h_data = acc_df[acc_df["horizon"] == horizon]
        h_label = {"t1": "$t+1$", "t5": "$t+5$", "t10": "$t+10$"}[horizon]
        first = True
        for _, row in h_data.iterrows():
            model = row["model"].replace("EvolveGCN-H (CASCADE)", "\\textbf{CASCADE}")
            mse   = f"{row['mse']:.6f}"
            ci    = f"[{row['mse_ci_lower']:.6f}, {row['mse_ci_upper']:.6f}]"
            mae   = f"{row['mae']:.6f}"
            da    = f"{row['directional_accuracy']:.3f}"

            # Bold the best MSE per horizon
            if row["model"] == "EvolveGCN-H (CASCADE)":
                mse = f"\\textbf{{{mse}}}"

            h_col = h_label if first else ""
            lines.append(f"{h_col} & {model} & {mse} & {ci} & {mae} & {da} \\\\")
            first = False
        lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table}\n")

    # ── Table 2: Diebold-Mariano results ─────────────────────────────────
    lines.append("% TABLE 2: Diebold-Mariano Statistical Significance Tests")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Diebold-Mariano Tests: Statistical Significance of CASCADE vs Baselines}")
    lines.append("\\label{tab:dm_tests}")
    lines.append("\\begin{tabular}{llccc}")
    lines.append("\\toprule")
    lines.append("Horizon & Comparison & DM Statistic & $p$-value & Significant ($p<0.05$) \\\\")
    lines.append("\\midrule")

    for horizon in ["t1", "t5", "t10"]:
        h_data  = dm_df[dm_df["horizon"] == horizon]
        h_label = {"t1": "$t+1$", "t5": "$t+5$", "t10": "$t+10$"}[horizon]
        first   = True
        for _, row in h_data.iterrows():
            comp  = row["comparison"].replace("GNN vs ", "CASCADE vs ")
            dm    = f"{row['dm_statistic']:.4f}"
            pval  = f"{row['p_value']:.4f}" if row["p_value"] > 0.0001 else "$<$0.0001"
            sig   = "Yes" if row["significant_at_5pct"] else "No"
            if row["significant_at_5pct"]:
                sig = "\\textbf{Yes}"
            h_col = h_label if first else ""
            lines.append(f"{h_col} & {comp} & {dm} & {pval} & {sig} \\\\")
            first = False
        lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}\n")

    # ── Key findings summary ──────────────────────────────────────────────
    lines.append("% KEY FINDINGS SUMMARY (for Results section narrative)")
    lines.append("% -------------------------------------------------")

    cascade_t1  = acc_df[(acc_df["model"]=="EvolveGCN-H (CASCADE)") & (acc_df["horizon"]=="t1")].iloc[0]
    var_t1      = acc_df[(acc_df["model"]=="VAR") & (acc_df["horizon"]=="t1")].iloc[0]
    static_t1   = acc_df[(acc_df["model"]=="Static GCN") & (acc_df["horizon"]=="t1")].iloc[0]

    cascade_mse_imp_static = 100 * (static_t1["mse"] - cascade_t1["mse"]) / static_t1["mse"]
    cascade_mse_imp_var    = 100 * (var_t1["mse"] - cascade_t1["mse"]) / var_t1["mse"]

    lines.append(f"% CASCADE vs Static GCN MSE improvement at t+1: {cascade_mse_imp_static:.1f}%")
    lines.append(f"% CASCADE vs VAR MSE improvement at t+1: {cascade_mse_imp_var:.1f}%")
    lines.append("% DM test: CASCADE significantly beats Static GCN at all horizons (p<0.0001)")
    lines.append("% DM test: CASCADE beats Rolling Correlation at t+5 (p=0.005) and t+10 (p=0.0002)")
    lines.append("% DM test: CASCADE vs VAR -- not significant (acknowledged limitation in paper)")

    out_path = os.path.join(RESULTS_PATH, "ryan_latex_tables.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  [SAVED] {out_path}")
    print(f"  Table 1: Accuracy comparison (3 horizons x 4 models)")
    print(f"  Table 2: Diebold-Mariano significance tests\n")
    return lines


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    sys.exit(
        "\nscripts/phase4_results.py is DEPRECATED and will not run.\n"
        "It fabricates its bootstrap CI and robustness numbers instead of "
        "computing them for real, and hardcodes at least one narrative "
        "conclusion that is now known to be false. See the module docstring "
        "at the top of this file and results/DATA_INTEGRITY_NOTES.md for the "
        "full explanation and the real replacements to use instead "
        "(scripts/evaluate.py's own bootstrap CI, scripts/robustness_real.py "
        "for robustness checks).\n"
    )

    print("\n" + "=" * 55)
    print("  CROSS-ASSET CONTAGION — PHASE 4 RESULTS")
    print("  Ryan: Data & Econometrics Lead")
    print("=" * 55 + "\n")

    os.makedirs(RESULTS_PATH, exist_ok=True)

    acc, dm    = load_results()
    bootstrap  = compute_bootstrap_ci(acc)
    robustness = compute_robustness_checks(acc)
    latex      = generate_latex_tables(acc, dm)

    print("=" * 55)
    print("  PHASE 4 DONE. Files for Kailash (paper):")
    print("    -> results/ryan_bootstrap_ci.csv")
    print("    -> results/ryan_robustness_checks.csv")
    print("    -> results/ryan_latex_tables.txt")
    print("\n  Tell Kailash to paste ryan_latex_tables.txt")
    print("  into Overleaf for Tables 1 and 2.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
