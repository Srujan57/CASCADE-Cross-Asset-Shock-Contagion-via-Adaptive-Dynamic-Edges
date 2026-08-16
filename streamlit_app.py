"""
streamlit_app.py

CASCADE — Cross-Asset Shock Contagion via Adaptive Dynamic Edges
Public results dashboard.

Presents results from a temporal graph neural network (EvolveGCN-H) trained
to forecast cross-asset returns and simulate shock propagation across
equities, bonds, commodities, and crypto (2015-2024). This app only reads
pre-computed result files from results/ — it does not train or re-run the
model.

See the "Methodology & Limitations" tab for how the model was trained and
evaluated, what each analysis does and doesn't claim, and the caveats worth
knowing before drawing conclusions from any single number here.

Run: streamlit run streamlit_app.py
"""

import json
import os

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Palette (validated categorical palette — see dataviz skill for why these
# specific hues: fixed order, colorblind-safe, never cycled)
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
    "Static GCN":            "#eda100",   # slot 4 yellow
}

STATUS = {
    "calm":   "#0ca30c",
    "stress": "#fab219",
    "crisis": "#d03b3b",
}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")


def rpath(name):
    return os.path.join(RESULTS_DIR, name)


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

tabs = st.tabs([
    "Overview",
    "Predictive Accuracy",
    "Shock Propagation",
    "Regime Analysis",
    "Structural Break",
    "Ablations",
    "Robustness",
    "Figures",
    "Methodology & Limitations",
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
            "parameters-to-data ratio — see the Methodology & Limitations tab for "
            "why that doesn't appear to have produced an inflated result. The "
            "effective training window starts ~Nov 2017, not 2015: build_graphs.py "
            "drops any weekly snapshot where an asset has no data yet, and ETH-USD "
            "has no Yahoo Finance history before Nov 2017."
        )

# ─────────────────────────────────────────────────────────────────────────────
# Predictive Accuracy
# ─────────────────────────────────────────────────────────────────────────────

with tabs[1]:
    st.subheader("Predictive accuracy vs. baselines")
    st.caption(
        "All four models are independently trained and evaluated on the same "
        "held-out test period (2023–2024): CASCADE (EvolveGCN-H), a naive rolling-"
        "correlation baseline, a linear VAR model, and a Static GCN (a graph "
        "network without EvolveGCN-H's temporal weight evolution)."
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
        "Predictive Accuracy tab, this is not held-out-only. See the Methodology & "
        "Limitations tab for what that means."
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
            "scripts/fix_regime_labels.py. Note the crisis bucket has very few "
            "snapshots in this sample — treat that row as directional, not "
            "statistically robust. **Interpretability analysis, not an accuracy "
            "claim:** computed over all snapshots (train+val+test combined), not "
            "held-out-only — see the Methodology & Limitations tab for what that means."
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
        "snapshots (train+val+test combined), not held-out-only — see the "
        "Methodology & Limitations tab for what that means."
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
        "Each configuration below is an independently retrained model (see "
        "scripts/run_ablations.py), removing one component of the pipeline at a "
        "time to see how much it contributes to accuracy."
    )
    abl = load_csv("ablation_results.csv")
    if abl is None:
        st.info("results/ablation_results.csv not found.")
    else:
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
        "Each row is an independently retrained model (scripts/robustness_real.py) "
        "varying the correlation threshold, rolling-correlation window, or asset "
        "universe. Three rows (threshold=0.3, window=60d, 11-asset (full)) are the "
        "same baseline configuration trained with the same fixed seed, so they "
        "report identical MSE by design."
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
# Methodology & Limitations
# ─────────────────────────────────────────────────────────────────────────────

with tabs[8]:
    st.subheader("Methodology & Limitations")
    st.caption(
        "How the model was trained and evaluated, what each tab does and doesn't "
        "claim, and the caveats worth knowing before drawing conclusions from any "
        "single number in this dashboard."
    )

    st.markdown("#### How the model was trained and evaluated")
    st.markdown(
        """
Data runs 2015-01-01 through 2024-12-31 and is split **chronologically, never
shuffled**: training through 2020-12-31, validation through 2022-12-31, and a
final test period of 2023–2024 that the model only sees once, for final
reporting. Model selection (checkpointing) is based on validation loss alone,
so the test set never influences which checkpoint gets reported. Confidence
intervals in the Predictive Accuracy tab come from bootstrap resampling of
the model's actual held-out predictions, not from a formula.
        """
    )

    st.markdown("#### What each tab claims — and what it doesn't")
    st.markdown(
        """
**Predictive Accuracy** (and its Diebold-Mariano significance tests) is the
only tab measuring out-of-sample accuracy: every number there comes from the
2023–2024 test period the model never trained or was tuned on.

**Shock Propagation, Regime Analysis, and Structural Break** are
interpretability analyses, not accuracy claims. They run the trained model
over every available snapshot — training, validation, and test combined — to
show how the model represents different market regimes and shock events
across its full input, not to measure generalization to unseen data. Treat
findings on these three tabs as "here's what the trained model represents,"
not "here's how accurate the model is."

**Ablations and Robustness** each show independently retrained models —
removing one pipeline component, or varying one hyperparameter, at a time —
evaluated the same way as the main model.
        """
    )

    st.markdown("#### Model size relative to data")
    st.markdown(
        """
EvolveGCN-H's GRU evolves a *flattened* GCN weight matrix, so parameter count
scales roughly with `hidden_dim`². At `hidden_dim=64` the model has **≈101M
trainable parameters**, trained on **152 weekly training snapshots**
(effective window starting ~Nov 2017, since ETH-USD has no exchange history
before then) via full-batch gradient descent on a single non-shuffled
sequence — an extreme parameters-to-data ratio for this kind of model. Two
things are consistent with that not having produced an inflated result:
test-set directional accuracy sits at 48.2–49.0% across all horizons
(chance level, not an inflated-looking number), and train/validation loss
converge to similar values with no runaway divergence.
        """
    )

    st.markdown("#### Regime labels")
    st.markdown(
        """
The calm/stress/crisis regime feature comes from a VIX-level threshold rule
(calm < 20, 20 ≤ stress ≤ 30, crisis > 30), applied to real, exogenous VIX
levels — not derived from the prediction target. In this sample the crisis
bucket has very few snapshots, so results conditioned on it (Regime Analysis
tab) should be read as directional rather than statistically robust.
        """
    )

    st.markdown("#### Event selection")
    st.markdown(
        """
The hand-picked shock events used in the Shock Propagation tab are
cross-checked against a systematic rolling z-score rule
(`scripts/identify_shocks.py`) that independently flags statistical
outliers, so the event list isn't purely a narrative pick. The event
catalog's "transmission channel" field is intentionally left blank for a
human to fill in with domain judgment rather than auto-generated.
        """
    )

    st.markdown("#### Scope")
    st.markdown(
        """
This dashboard only reads pre-computed CSV/JSON/PNG files from `results/` —
it never imports the model code, touches raw data, or re-runs training.
Every chart and table above reflects whatever is currently in `results/`.
        """
    )
