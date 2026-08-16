"""
streamlit_app.py

CASCADE — Cross-Asset Shock Contagion via Adaptive Dynamic Edges
Results dashboard.

This app presents ONLY results that survived a full data-integrity audit —
see results/DATA_INTEGRITY_NOTES.md for the complete history. Summary of what
was found and fixed, all reconfirmed against a real re-run on real data:

  - Fabricated results. scripts/phase4_results.py generated
    ryan_bootstrap_ci.csv (a circular "bootstrap" resampled from an
    already-reported CI, not real predictions), ryan_robustness_checks.csv
    (a closed-form mse * noise_factor formula, no retraining), and
    ryan_latex_tables.txt (built from both, plus a hardcoded narrative claim
    that turned out to be false). All three were archived to
    results/archive_fabricated_DO_NOT_USE/ — this app never reads that
    folder — and replaced with real numbers (experiment1_accuracy.csv's own
    bootstrap CI, scripts/robustness_real.py's real retrains). The script
    that produced them now refuses to run (see its docstring) so it can't
    silently regenerate fabricated files again.
  - Untrained baseline. Static GCN was previously compared against an
    untrained (randomly initialized) network. scripts/evaluate.py was
    patched to train it properly; the fix was confirmed by a ~40x MSE drop
    and a plausible, mixed significance picture instead of "beats noise
    everywhere." Shown normally below.
  - Misleading terminology (not a results bug, but misleading if read as
    code documentation): several comments/config entries called the regime
    labels "HMM output" when the actual method is a simple VIX-level
    threshold rule (scripts/fix_regime_labels.py) — corrected throughout.
    A separate, unrelated K-means regime detector in
    scripts/phase2_econometrics.py wrote to the same output filename as that
    threshold rule; it's now renamed to avoid the collision and marked as
    not used by the pipeline.
  - Staleness. ablation_results.csv and results/figures/* were briefly stale
    after a re-run; both have since been regenerated and the dashboard
    checks their freshness live on every load (Ablations, Figures, and Data
    Integrity tabs) rather than just asserting they're current in prose.

One methodological caveat that is NOT a bug, but is disclosed explicitly:
Experiments 2-4 (Shock Propagation, Regime Analysis, Structural Break) run
the trained model over all snapshots (train+val+test combined), not
held-out data only, because they're interpretability analyses of what the
trained model represents, not accuracy claims. Only Experiment 1
(Predictive Accuracy) and its Diebold-Mariano tests are pure held-out-test
numbers. See the captions on those tabs and the Data Integrity tab.

Run: streamlit run streamlit_app.py
"""

import json
import os

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Palette (validated categorical palette — see dataviz skill / results/DATA_INTEGRITY_NOTES.md
# for why these specific hues: fixed order, colorblind-safe, never cycled)
# ─────────────────────────────────────────────────────────────────────────────

INK_PRIMARY   = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED     = "#898781"
SURFACE       = "#fcfcfb"
GRIDLINE      = "#e1e0d9"

CAT = {
    "CASCADE (EvolveGCN-H)": "#2a78d6",   # slot 1 blue
    "Rolling Correlation":   "#eb6834",   # slot 2 orange
    "VAR":                   "#1baf7a",   # slot 3 aqua
    "Static GCN":            "#eda100",   # slot 4 yellow — now a real trained baseline
}

STATUS = {
    "calm":   "#0ca30c",
    "stress": "#fab219",
    "crisis": "#d03b3b",
}

ARCHIVED_FABRICATED_FILES = {
    "ryan_bootstrap_ci.csv":      "circular bootstrap — resamples a Gaussian parameterized by an already-reported CI, not real predictions",
    "ryan_robustness_checks.csv": "closed-form formula (mse * noise_factor), never retrains anything",
    "ryan_latex_tables.txt":      "built from the two files above",
}
ARCHIVE_DIR = "archive_fabricated_DO_NOT_USE"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")


def rpath(name):
    return os.path.join(RESULTS_DIR, name)


def figures_staleness():
    """
    Live check: are results/figures/* older than the CSVs/JSON they're
    supposed to visualize? Returns (is_stale, detail_str) or (None, reason)
    if the check can't be run (missing files).
    """
    fig_path = os.path.join(FIG_DIR, "fig1_training_curves.png")
    source_path = rpath("training_history.json")
    if not os.path.exists(fig_path) or not os.path.exists(source_path):
        return None, "figure or source file not found — skipping live check"
    fig_mtime = os.path.getmtime(fig_path)
    src_mtime = os.path.getmtime(source_path)
    is_stale = fig_mtime < src_mtime
    return is_stale, (fig_mtime, src_mtime)


@st.cache_data
def load_csv(name):
    path = rpath(name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_json(name):
    path = rpath(name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def base_layout(fig, title=None, height=420):
    fig.update_layout(
        title=title,
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(color=INK_PRIMARY, family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
        height=height,
        margin=dict(l=40, r=20, t=50 if title else 20, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRIDLINE, zeroline=False, linecolor=GRIDLINE)
    fig.update_yaxes(showgrid=True, gridcolor=GRIDLINE, zeroline=False, linecolor=GRIDLINE)
    return fig


st.set_page_config(
    page_title="CASCADE — Cross-Asset Contagion Dashboard",
    page_icon="\U0001F578️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("CASCADE — Cross-Asset Shock Contagion via Adaptive Dynamic Edges")
st.caption(
    "A temporal graph neural network (EvolveGCN-H) forecasting cross-asset returns "
    "and modeling shock propagation across equities, bonds, commodities, and crypto (2015–2024)."
)

st.success(
    "**Data-integrity audit: resolved.** Fabricated robustness/CI files were replaced with real "
    "retrains, the Static GCN baseline (previously untrained) was fixed and reconfirmed, and "
    "misleading code comments were corrected. See the **Data Integrity & Limitations** tab for "
    "the full history and live freshness checks on every load.",
    icon="✅",
)

tabs = st.tabs([
    "Overview",
    "Predictive Accuracy",
    "Shock Propagation",
    "Regime Analysis",
    "Structural Break",
    "Ablations",
    "Robustness",
    "Figures",
    "Data Integrity & Limitations",
])

# ─────────────────────────────────────────────────────────────────────────────
# Overview
# ─────────────────────────────────────────────────────────────────────────────

with tabs[0]:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("What this is")
        st.markdown(
            """
CASCADE trains **EvolveGCN-H** — a graph neural network whose weight matrices are
evolved over time by a GRU — on weekly correlation/DCC-GARCH graphs of 10 assets
(equities, bonds, commodities, crypto, plus VIX and DXY as macro features) to
forecast returns at t+1, t+5, and t+10 trading days, and to simulate how a shock
at one asset propagates to the others.

**Data:** SPY, EEM, LQD, HYG, TLT, GLD, USO, BTC-USD, ETH-USD, DXY, plus VIX as a
feature. 2015-01-01 to 2024-12-31. Chronological split: train through 2020-12-31,
validation through 2022-12-31, test on 2023–2024 — never shuffled, so there is
no look-ahead leakage in the split itself.

**Honest headline finding:** on the held-out 2023–2024 test period, CASCADE's
directional accuracy sits at **48.2–49.0%** across all three horizons —
statistically indistinguishable from a coin flip — and a Diebold-Mariano test
finds it is **not significantly more accurate than a linear VAR baseline** at
any horizon (VAR is nominally *better*, and nearly significant, at t+10:
p=0.06). CASCADE does significantly beat naive rolling-correlation at t+5 and
t+10, and beats the now-properly-trained Static GCN baseline at t+5 (p≈1e-06)
but not at t+1 or t+10. This is a defensible, market-efficiency-consistent
result — not a "beats everything" story — and that's the story this dashboard
tells.
            """
        )
    with col2:
        st.subheader("Model")
        st.markdown(
            """
| | |
|---|---|
| Architecture | EvolveGCN-H (Pareja et al., AAAI 2020) |
| Hidden dim | 64 |
| GCN layers | 2 |
| Trainable params | ≈101M |
| Train / val / test snapshots | 152 / 105 / 102 (weekly) |
| Dropout | 0.3 |
| Horizons | t+1, t+5, t+10 trading days |
            """
        )
        st.caption(
            "≈101M parameters on 152 training snapshots is an extreme "
            "parameters-to-data ratio — see Data Integrity tab for why this "
            "is disclosed rather than hidden, and why it doesn't appear to have "
            "produced a misleadingly inflated result. The effective training window "
            "starts ~Nov 2017, not 2015: build_graphs.py drops any weekly snapshot "
            "where an asset has no data yet, and ETH-USD has no Yahoo Finance "
            "history before Nov 2017."
        )

# ─────────────────────────────────────────────────────────────────────────────
# Predictive Accuracy
# ─────────────────────────────────────────────────────────────────────────────

with tabs[1]:
    st.subheader("Predictive accuracy vs. baselines")
    st.caption(
        "All four models below are real, independently trained — Static GCN was "
        "previously compared against an untrained network (see Data Integrity tab); "
        "that was fixed and this is the corrected, re-run comparison."
    )

    acc = load_csv("experiment1_accuracy.csv")
    dm = load_csv("diebold_mariano_results.csv")

    if acc is None:
        st.info("results/experiment1_accuracy.csv not found.")
    else:
        acc_clean = acc.copy()
        acc_clean["model"] = acc_clean["model"].replace(
            {"EvolveGCN-H (CASCADE)": "CASCADE (EvolveGCN-H)"}
        )
        horizon_labels = {"t1": "t+1", "t5": "t+5", "t10": "t+10"}
        acc_clean["horizon_label"] = acc_clean["horizon"].map(horizon_labels)

        horizon_pick = st.radio("Horizon", ["t+1", "t+5", "t+10"], horizontal=True)
        sub = acc_clean[acc_clean["horizon_label"] == horizon_pick]

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for _, row in sub.iterrows():
                color = CAT.get(row["model"], INK_MUTED)
                fig.add_trace(go.Bar(
                    x=[row["model"]], y=[row["mse"]],
                    error_y=dict(
                        type="data",
                        array=[row["mse_ci_upper"] - row["mse"]],
                        arrayminus=[row["mse"] - row["mse_ci_lower"]],
                        color=INK_SECONDARY,
                    ),
                    marker_color=color, name=row["model"], showlegend=False,
                ))
            fig = base_layout(fig, title=f"Test MSE with 95% bootstrap CI — {horizon_pick}")
            fig.update_yaxes(title="MSE (lower is better)")
            st.plotly_chart(fig, width='stretch')
        with c2:
            fig2 = go.Figure()
            for _, row in sub.iterrows():
                color = CAT.get(row["model"], INK_MUTED)
                fig2.add_trace(go.Bar(
                    x=[row["model"]], y=[row["directional_accuracy"] * 100],
                    marker_color=color, name=row["model"], showlegend=False,
                ))
            fig2.add_hline(y=50, line_dash="dash", line_color=INK_MUTED,
                            annotation_text="coin flip", annotation_position="top left")
            fig2 = base_layout(fig2, title=f"Directional accuracy — {horizon_pick}")
            fig2.update_yaxes(title="% correct sign", range=[40, 60])
            st.plotly_chart(fig2, width='stretch')

        st.dataframe(
            sub[["model", "mse", "mse_ci_lower", "mse_ci_upper", "mae", "directional_accuracy"]]
            .style.format({
                "mse": "{:.6f}", "mse_ci_lower": "{:.6f}", "mse_ci_upper": "{:.6f}",
                "mae": "{:.6f}", "directional_accuracy": "{:.1%}",
            }),
            width='stretch',
        )

    st.markdown("#### Diebold-Mariano significance tests (CASCADE vs. each baseline)")
    if dm is None:
        st.info("results/diebold_mariano_results.csv not found.")
    else:
        dm_clean = dm.copy()
        dm_clean["horizon"] = dm_clean["horizon"].map(horizon_labels)
        dm_clean["significant_at_5pct"] = dm_clean["significant_at_5pct"].map(
            {True: "Yes", False: "No"}
        )
        st.dataframe(
            dm_clean.rename(columns={
                "dm_statistic": "DM statistic", "p_value": "p-value",
                "better_model": "Favored model", "significant_at_5pct": "p < 0.05?",
            }).style.format({"DM statistic": "{:+.3f}", "p-value": "{:.4f}"}),
            width='stretch',
        )
        st.caption(
            "Negative DM statistic favors CASCADE. CASCADE significantly beats Rolling "
            "Correlation at t+5/t+10, and significantly beats Static GCN at t+5 "
            "(p≈1.0e-06) but not at t+1 (p=0.78) or t+10 (p=0.17, nominally favoring "
            "Static GCN). CASCADE vs. VAR is not significant at any horizon (VAR is "
            "nominally better, and nearly significant, at t+10: p=0.06)."
        )

# ─────────────────────────────────────────────────────────────────────────────
# Shock Propagation
# ─────────────────────────────────────────────────────────────────────────────

with tabs[2]:
    st.subheader("Shock propagation — event injection")
    st.caption(
        "For each historical shock, the model replays the pre-shock snapshot with the "
        "realized shock return injected at the source asset and reports the predicted "
        "change (“spillover”) at every other node, t+1 ahead. **Interpretability "
        "analysis, not an accuracy claim:** computed over all snapshots (train+val+test "
        "combined), including snapshots the model was trained on — unlike the "
        "Predictive Accuracy tab, this is not held-out-only. See Data Integrity tab."
    )
    exp2 = load_csv("experiment2_shock_propagation.csv")
    if exp2 is None:
        st.info("results/experiment2_shock_propagation.csv not found.")
    else:
        events = exp2["event"].unique().tolist()
        event_pick = st.selectbox("Event", events)
        sub = exp2[exp2["event"] == event_pick].sort_values("spillover")
        meta = sub.iloc[0]
        st.markdown(
            f"**{event_pick}** — {meta['date']} — shock at **{meta['source']}** "
            f"of {meta['shock_val']*100:+.1f}%"
        )
        fig = go.Figure(go.Bar(
            x=sub["spillover"], y=sub["target"], orientation="h",
            marker_color=["#e34948" if v < 0 else "#2a78d6" for v in sub["spillover"]],
        ))
        fig = base_layout(fig, title="Predicted t+1 spillover by target asset", height=380)
        fig.update_xaxes(title="Predicted change in t+1 return")
        st.plotly_chart(fig, width='stretch')
        st.dataframe(sub[["target", "spillover", "shocked_pred", "baseline_pred"]]
                     .style.format({"spillover": "{:.6f}", "shocked_pred": "{:.6f}", "baseline_pred": "{:.6f}"}),
                     width='stretch')

    prop = load_csv("contagion_propagation_table.csv")
    if prop is not None:
        st.markdown("#### Predicted vs. realized (multi-horizon)")
        st.caption(
            "“Actual” is the realized market return on the real historical event date, "
            "not a backtest of a trading signal — see propagation_tables.py for the exact framing."
        )
        st.dataframe(prop, width='stretch', height=300)

# ─────────────────────────────────────────────────────────────────────────────
# Regime Analysis
# ─────────────────────────────────────────────────────────────────────────────

with tabs[3]:
    st.subheader("Regime-conditioned behavior")
    exp3 = load_csv("experiment3_regime_analysis.csv")
    if exp3 is None or exp3.empty:
        st.info("results/experiment3_regime_analysis.csv not found or empty.")
    else:
        st.caption(
            "VIX-threshold regimes (calm < 20, 20 ≤ stress ≤ 30, crisis > 30), from "
            "scripts/fix_regime_labels.py — not an HMM, despite older comments "
            "elsewhere in this codebase saying so (see Data Integrity tab). Note the "
            "crisis bucket has very few snapshots in this sample — treat that row as "
            "directional, not statistically robust. **Interpretability analysis, not "
            "an accuracy claim:** computed over all snapshots (train+val+test "
            "combined), not held-out-only — see Data Integrity tab."
        )
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(
                x=exp3["regime"], y=exp3["n_snapshots"],
                marker_color=[STATUS.get(r, INK_MUTED) for r in exp3["regime"]],
            ))
            fig = base_layout(fig, title="Snapshots per regime (sample size)")
            st.plotly_chart(fig, width='stretch')
        with c2:
            fig2 = go.Figure(go.Bar(
                x=exp3["regime"], y=exp3["mean_pred_magnitude"],
                marker_color=[STATUS.get(r, INK_MUTED) for r in exp3["regime"]],
            ))
            fig2 = base_layout(fig2, title="Mean |predicted return| by regime")
            st.plotly_chart(fig2, width='stretch')
        st.dataframe(exp3, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# Structural Break
# ─────────────────────────────────────────────────────────────────────────────

with tabs[4]:
    st.subheader("Crypto structural break — pre vs. post institutional adoption (Oct 2020)")
    st.caption(
        "**Interpretability analysis, not an accuracy claim:** computed over all "
        "snapshots (train+val+test combined), not held-out-only — see Data Integrity tab."
    )
    exp4 = load_csv("experiment4_structural_break.csv")
    if exp4 is None:
        st.info("results/experiment4_structural_break.csv not found.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Pre-2020", x=exp4["asset"], y=exp4["pre_2020_mean_spillover"],
                              marker_color="#3987e5"))
        fig.add_trace(go.Bar(name="Post-2020", x=exp4["asset"], y=exp4["post_2020_mean_spillover"],
                              marker_color="#e34948"))
        fig = base_layout(fig, title="Mean predicted spillover from a standardized BTC shock")
        fig.update_layout(barmode="group")
        st.plotly_chart(fig, width='stretch')
        st.dataframe(exp4, width='stretch')
        st.caption(
            f"Pre-2020 window: n={int(exp4['n_pre'].iloc[0])} snapshots — "
            f"Post-2020 window: n={int(exp4['n_post'].iloc[0])} snapshots."
        )

# ─────────────────────────────────────────────────────────────────────────────
# Ablations
# ─────────────────────────────────────────────────────────────────────────────

with tabs[5]:
    st.subheader("Ablation study")
    st.caption(
        "Both ablations below are real, independently retrained models (see "
        "scripts/run_ablations.py) — not formula-derived."
    )
    abl = load_csv("ablation_results.csv")
    if abl is None:
        st.info("results/ablation_results.csv not found.")
    else:
        acc_for_staleness = load_csv("experiment1_accuracy.csv")
        if acc_for_staleness is not None and "CASCADE (full)" in abl["model"].values:
            abl_full_t1 = abl.loc[
                (abl["model"] == "CASCADE (full)") & (abl["horizon"] == abl["horizon"].unique()[0]),
                "mse",
            ]
            cascade_row = acc_for_staleness[
                acc_for_staleness["model"].isin(["EvolveGCN-H (CASCADE)", "CASCADE (EvolveGCN-H)"])
            ]
            if not abl_full_t1.empty and not cascade_row.empty:
                t1_mask = cascade_row["horizon"] == "t1"
                if t1_mask.any():
                    live_mse = cascade_row.loc[t1_mask, "mse"].iloc[0]
                    abl_mse = abl_full_t1.iloc[0]
                    if abs(live_mse - abl_mse) > 5e-8:  # ablation_results.csv rounds MSE to 8dp
                        st.warning(
                            f"**Stale ablation data.** `ablation_results.csv`'s CASCADE (full) "
                            f"t+1 MSE ({abl_mse:.8f}) does not match the current "
                            f"`experiment1_accuracy.csv` CASCADE t+1 MSE ({live_mse:.8f}) — "
                            f"the ablation table was generated from an older checkpoint. "
                            f"Re-run `python scripts/run_ablations.py` before citing this "
                            f"table alongside the Predictive Accuracy tab. See Data Integrity tab.",
                            icon="⚠️",
                        )
        horizon_pick2 = st.radio("Horizon", abl["horizon"].unique().tolist(), horizontal=True, key="abl_h")
        sub = abl[abl["horizon"] == horizon_pick2]
        fig = go.Figure(go.Bar(
            x=sub["model"], y=sub["mse"],
            marker_color=["#2a78d6", "#eb6834", "#1baf7a"][:len(sub)],
        ))
        fig = base_layout(fig, title=f"Test MSE by configuration — {horizon_pick2}")
        st.plotly_chart(fig, width='stretch')
        st.dataframe(sub, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# Robustness
# ─────────────────────────────────────────────────────────────────────────────

with tabs[6]:
    st.subheader("Robustness checks")
    st.caption(
        "Real retrains (scripts/robustness_real.py) varying the correlation threshold, "
        "rolling-correlation window, and asset universe — not the closed-form "
        "perturbation this replaced (see Data Integrity tab). Three rows "
        "(threshold=0.3, window=60d, 11-asset (full)) are all the same baseline "
        "config with the same fixed seed, so byte-identical MSE across them is "
        "expected, not a red flag."
    )
    rob = load_csv("robustness_checks_real.csv")
    if rob is None:
        st.info("results/robustness_checks_real.csv not found.")
    else:
        checks = rob["robustness_check"].unique().tolist()
        check_pick = st.selectbox("Check", checks)
        sub = rob[rob["robustness_check"] == check_pick].copy()
        sub["horizon"] = sub["horizon"].map(horizon_labels).fillna(sub["horizon"])
        fig = px.bar(
            sub, x="parameter", y="mse", color="horizon", barmode="group",
            color_discrete_sequence=["#2a78d6", "#eb6834", "#1baf7a"],
        )
        fig = base_layout(fig, title=f"Test MSE by parameter — {check_pick}")
        st.plotly_chart(fig, width='stretch')
        st.dataframe(
            sub.rename(columns={"baseline": "is_baseline_config"})
            .style.format({"mse": "{:.6f}", "mae": "{:.6f}", "directional_accuracy": "{:.1%}"}),
            width='stretch',
        )

# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

with tabs[7]:
    st.subheader("Generated figures")
    is_stale, detail = figures_staleness()
    if is_stale is True:
        st.warning(
            "**Stale figures.** `results/figures/fig1_training_curves.png` is older "
            "than `results/training_history.json` — the figures were generated before "
            "the most recent training run. Re-run `python scripts/generate_figures.py` "
            "before presenting these.",
            icon="⚠️",
        )
    elif is_stale is False:
        st.success("Checked live: figures are newer than the current results files.", icon="✅")
    if not os.path.isdir(FIG_DIR):
        st.info("results/figures/ not found.")
    else:
        pngs = sorted(f for f in os.listdir(FIG_DIR) if f.endswith(".png"))
        if not pngs:
            st.info("No PNG figures found in results/figures/.")
        cols = st.columns(2)
        for i, fname in enumerate(pngs):
            with cols[i % 2]:
                st.image(os.path.join(FIG_DIR, fname), caption=fname, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# Data Integrity & Limitations
# ─────────────────────────────────────────────────────────────────────────────

with tabs[8]:
    st.subheader("Data Integrity & Limitations")
    st.success(
        "Every issue found in this audit is resolved and reconfirmed against a real "
        "re-run — including two staleness items that are checked live below (against "
        "the actual results files on every load) rather than just asserted in prose. "
        "One methodological caveat is disclosed further down that isn't a bug, but is "
        "worth reading before presenting this externally.",
        icon="✅",
    )

    st.markdown("#### Archived fabricated files — resolved, replaced with real data")
    for fname, reason in ARCHIVED_FABRICATED_FILES.items():
        archived_path = os.path.join(ARCHIVE_DIR, fname)
        exists = os.path.exists(rpath(archived_path))
        st.markdown(
            f"- `{fname}` {'(moved to `results/' + ARCHIVE_DIR + '/`, never loaded by this app)' if exists else '(not found)'} "
            f"— {reason}"
        )
    st.markdown(
        "Replaced by: `experiment1_accuracy.csv`'s own bootstrap CI (real resampling of "
        "actual predictions, in `scripts/evaluate.py::bootstrap_ci()`) and "
        "`robustness_checks_real.csv` (real retrains, `scripts/robustness_real.py`) — see "
        "the Robustness tab."
    )

    st.markdown("#### Static GCN baseline — fixed and reconfirmed")
    st.markdown(
        """
`scripts/evaluate.py` previously instantiated the `StaticGCN` baseline and called
`.eval()` on it **without ever training it** — every historical "Static GCN" number
was a randomly-initialized network's output, not a trained baseline. This was fixed
in `scripts/evaluate.py` (it now trains Static GCN the same way EvolveGCN-H is
trained, with early stopping on validation loss) and confirmed by the real re-run:
Static GCN's MSE dropped from ~0.024 (the ~40x-larger untrained-network signature) to
0.00055 / 0.00028 / 0.00022 across t+1/t+5/t+10 — the same order of magnitude as every
other model. The Diebold-Mariano results against it are now a mixed, plausible
picture (significant at t+5, not at t+1 or t+10) rather than "beats noise
everywhere," which is itself evidence the fix worked. See the Predictive Accuracy tab.
        """
    )

    st.markdown("#### Robustness checks — real retrains, verified")
    st.markdown(
        """
`scripts/robustness_real.py` varies the correlation threshold, rolling-correlation
window, and asset universe and retrains from scratch for each of 8 variants — replacing
the archived `ryan_robustness_checks.csv`, which computed `mse * noise_factor` from a
closed-form formula and never retrained anything (proven by identical MSE for
`window=30d` vs. `window=90d` in that file, despite being different configs). The
merged `robustness_checks_real.csv` (24 rows = 8 variants × 3 horizons) was verified
against the 4 shard files it was built from with zero rows lost or duplicated. Its
three "baseline" rows (`threshold=0.3`, `window=60d`, `11-asset (full)`) report
byte-identical MSE — expected, not a red flag, since those three jobs are the same
config trained with the same fixed seed (42); identical inputs producing identical
outputs is a correctness signal, the opposite of the earlier fabrication fingerprint.
        """
    )

    st.markdown("#### `ablation_results.csv` freshness — checked live, not asserted")
    st.markdown(
        "This briefly went stale after a re-run (the ablation table was generated from "
        "an older checkpoint than the current accuracy numbers) and has since been "
        "regenerated. Rather than just claim that here, the check below re-runs on "
        "every page load against whatever files are actually in `results/` right now:"
    )
    abl_check = load_csv("ablation_results.csv")
    acc_check = load_csv("experiment1_accuracy.csv")
    if abl_check is not None and acc_check is not None and "CASCADE (full)" in abl_check["model"].values:
        abl_t1 = abl_check.loc[
            (abl_check["model"] == "CASCADE (full)") & (abl_check["horizon"] == abl_check["horizon"].unique()[0]),
            "mse",
        ]
        cascade_row = acc_check[acc_check["model"].isin(["EvolveGCN-H (CASCADE)", "CASCADE (EvolveGCN-H)"])]
        t1_mask = cascade_row["horizon"] == "t1" if not cascade_row.empty else pd.Series([], dtype=bool)
        if not abl_t1.empty and t1_mask.any():
            live_mse = cascade_row.loc[t1_mask, "mse"].iloc[0]
            abl_mse = abl_t1.iloc[0]
            if abs(live_mse - abl_mse) > 5e-8:  # ablation_results.csv rounds MSE to 8dp
                st.warning(
                    f"**Stale right now:** ablation CASCADE(full) t+1 MSE = "
                    f"{abl_mse:.8f}, current experiment1_accuracy.csv CASCADE t+1 MSE = "
                    f"{live_mse:.8f}. These describe two different trained models — "
                    f"re-run `python scripts/run_ablations.py` before citing this table.",
                    icon="⚠️",
                )
            else:
                st.success(
                    "Checked live: ablation and current accuracy numbers match "
                    "(within 8-decimal rounding). Not stale as of this page load.",
                    icon="✅",
                )
    else:
        st.info("Can't run the live check — one of the two source files is missing.")

    st.markdown("#### `results/figures/*` freshness — checked live, not asserted")
    st.markdown(
        "Same situation: figures generated before a re-run would silently show stale "
        "curves. Checked on the Figures tab (and summarized here) by comparing file "
        "timestamps — `fig1_training_curves.png` vs. `training_history.json` — on "
        "every page load, not just asserted."
    )
    is_stale, detail = figures_staleness()
    if is_stale is True:
        st.warning(
            "**Stale right now:** figures/ predates the current training_history.json. "
            "Re-run `python scripts/generate_figures.py`.",
            icon="⚠️",
        )
    elif is_stale is False:
        st.success("Checked live: figures are newer than the current results files. Not stale.", icon="✅")
    else:
        st.info(f"Can't run the live check — {detail}")

    st.markdown("#### Fabrication script neutralized, not just avoided")
    st.markdown(
        """
`scripts/phase4_results.py` (the source of the three fabricated files above) still
existed in the repo and, if run out of habit, would have silently regenerated
`results/ryan_*.csv` — overwriting the real replacements this audit put in their
place. It also hardcoded a narrative conclusion directly into its LaTeX output
regardless of what the data said ("CASCADE significantly beats Static GCN at all
horizons (p<0.0001)"), which is now known to be false. Rather than rely on nobody
running it again, the script itself now refuses to run — `python
scripts/phase4_results.py` exits immediately with an explanation — and it was moved
to `scripts/archive_deprecated_DO_NOT_USE/` alongside an expanded docstring
documenting exactly what was wrong with it, for the historical record.
        """
    )

    st.markdown("#### Regime-label terminology and a filename collision — fixed")
    st.markdown(
        """
Several comments and one `config.yaml` entry described the regime labels feeding the
model (calm/stress/crisis) as "HMM output." The actual method
(`scripts/fix_regime_labels.py`) is a simple VIX-level threshold rule — not a hidden
Markov model. This was purely a documentation/comment issue (the labels themselves
were always real, exogenous VIX levels, never the prediction target), but it's the
kind of thing a technical reviewer reading the code would reasonably flag, so all
references were corrected.

Separately: an unrelated K-means-based regime detector in
`scripts/phase2_econometrics.py` (clustering on VIX level + credit spread + equity
vol — a real, legitimate alternative method) wrote its output to the exact same
filename (`data/processed/regime_labels.csv`) as the threshold rule above. Running
`phase2_econometrics.py` for its legitimate DCC-GARCH/Granger-causality outputs would
have silently overwritten the regime labels the model actually uses with a different
regime definition. Fixed by renaming its output to
`regime_labels_kmeans_ALTERNATIVE.csv` so the two can no longer collide.
        """
    )

    st.markdown("#### Experiments 2-4 use all snapshots, not held-out test data only")
    st.markdown(
        """
Shock Propagation, Regime Analysis, and Structural Break all run the trained model
over every snapshot (train + validation + test combined) — confirmed by reading
`scripts/evaluate.py`'s `all_snaps` construction, which feeds all three. This is a
defensible choice for what they're measuring (how does the trained model represent
different regimes/shocks across its full input, an interpretability question) but it
means those three tabs are **not** held-out-accuracy claims the way the Predictive
Accuracy tab is — a meaningful fraction of what they show reflects snapshots the
model saw during training. This wasn't previously disclosed and is now called out on
each of those three tabs directly, so a reviewer doesn't mistake them for
out-of-sample confirmation with the same rigor as Experiment 1.
        """
    )

    st.markdown("#### Parameter count vs. dataset size")
    st.markdown(
        """
EvolveGCN-H's GRU evolves a *flattened* GCN weight matrix, so parameter count scales
roughly with `hidden_dim`². At `hidden_dim=64` the model has **≈101M trainable
parameters**, trained on **152 training snapshots** (weekly, effective window starting
~Nov 2017) via full-batch gradient descent on a single non-shuffled sequence — an
extreme parameters-to-data ratio. Disclosed explicitly rather than left for a technical
reviewer to discover. The saving grace: test-set directional accuracy sits at
48.2–49.0% (chance level) and train/validation loss converge to similar values with no
runaway divergence — consistent with a model that learned something modest and
stopped early, not one that memorized noise into an inflated result.
        """
    )

    st.markdown("#### What checked out fine")
    st.markdown(
        """
- Train/val/test splits are strictly chronological, never shuffled — no look-ahead leakage,
  confirmed by reading `models/train.py` and `scripts/run_ablations.py`'s split logic directly.
- Model selection (checkpointing) uses validation loss only; the test set is touched
  once, for final reporting — confirmed by reading `models/train.py`.
- Regime labels come from VIX (exogenous, contemporaneously observed), not the prediction target.
- The four hand-picked shock events are independently cross-checked against a systematic
  rolling z-score rule (`scripts/identify_shocks.py`), which honestly flags which events are
  statistical outliers vs. narrative/regime picks.
- `scripts/event_catalog.py` explicitly leaves the narrative "transmission channel"
  field blank for a human to fill in rather than generating plausible-sounding prose —
  it only pre-populates fields traceable to real upstream data.
- Edge-importance uses gradient saliency and explicitly avoids claiming an attention
  mechanism EvolveGCN-H doesn't have.
- `fix_regime_labels.py` is a legitimate bugfix, not a result-shaping change.
- `scripts/generate_figures.py` and `scripts/propagation_tables.py` both read only from
  already-saved CSVs/JSON — no synthetic data injection, no cherry-picked axis ranges.
- The synthetic-data code paths in `models/baselines.py` and `models/evolvegcn.py`
  (`make_fake_snapshot`, "generating synthetic data for testing") are self-test
  scaffolding behind `if __name__ == "__main__":` guards, never imported by the real
  training/evaluation pipeline — confirmed by reading both files end to end.
- The 55%-single-day-return data quality flag turned out to be real historical events
  (Black Thursday 2020-03-12, negative-oil-price April 2020, the May 2021 and June 2022
  crypto crashes, etc.), checked against known event dates, not a data glitch.
- `training_history.json` shows healthy convergence on the re-run: 80 epochs before
  early stopping, train/val loss around 0.00065/0.0008, no divergence or instability.
        """
    )

    st.caption(
        "Full writeup with exact numbers and code references: "
        "results/DATA_INTEGRITY_NOTES.md"
    )

    integrity_path = rpath("DATA_INTEGRITY_NOTES.md")
    if os.path.exists(integrity_path):
        with open(integrity_path) as f:
            with st.expander("Show full audit notes (results/DATA_INTEGRITY_NOTES.md)"):
                st.markdown(f.read())
