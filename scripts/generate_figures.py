"""
scripts/generate_figures.py

Generates all summary figures for CASCADE.

Figures produced (saved to results/figures/):
    fig1_training_curves.pdf/png       — train vs val loss over epochs
    fig2_accuracy_comparison.pdf/png   — MSE across all models and horizons
    fig3_shock_propagation.pdf/png     — spillover heatmaps for 4 shock events
    fig4_regime_analysis.pdf/png       — regime-conditioned contagion magnitude
    fig5_structural_break.pdf/png      — BTC spillover pre vs post 2020
    fig6_contagion_network.pdf/png     — DCC correlation network graph

Run: python scripts/generate_figures.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("networkx not found — fig6 will be skipped. pip install networkx")

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

# Consistent color palette across all figures
COLORS = {
    "CASCADE":      "#1B4F8A",   # deep blue
    "Rolling Corr": "#7F8C8D",   # gray
    "VAR":          "#E67E22",   # orange
    "Static GCN":   "#C0392B",   # red
    "calm":         "#2ECC71",   # green
    "stress":       "#F39C12",   # amber
    "crisis":       "#E74C3C",   # red
    "pre2020":      "#3498DB",   # blue
    "post2020":     "#E74C3C",   # red
}

FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def save_fig(fig, name):
    """Save figure as both PDF (print-quality) and PNG (for preview/dashboard)."""
    for ext in ["pdf", "png"]:
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {name}.pdf + .png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Training Curves
# ─────────────────────────────────────────────────────────────────────────────

def fig1_training_curves(history_path="results/training_history.json"):
    print("\nFig 1: Training curves...")

    with open(history_path) as f:
        history = json.load(f)

    train_loss = history["train_loss"]
    val_loss   = history["val_loss"]

    # Train loss is per-epoch; val loss is every 5 epochs
    train_epochs = list(range(1, len(train_loss) + 1))
    val_epochs   = list(range(5, 5 * len(val_loss) + 1, 5))

    # Best checkpoint epoch (lowest val loss)
    best_val_idx   = int(np.argmin(val_loss))
    best_val_epoch = val_epochs[best_val_idx]
    best_val_value = val_loss[best_val_idx]

    fig, ax = plt.subplots(figsize=(6, 3.5))

    ax.plot(train_epochs, train_loss,
            color=COLORS["CASCADE"], linewidth=1.5,
            label="Train loss", alpha=0.9)
    ax.plot(val_epochs, val_loss,
            color=COLORS["VAR"], linewidth=2.0,
            linestyle="--", marker="o", markersize=3,
            label="Validation loss")

    # Mark best checkpoint
    ax.axvline(best_val_epoch, color="#27AE60", linestyle=":",
               linewidth=1.5, alpha=0.8)
    ax.annotate(f"Best checkpoint\nepoch {best_val_epoch}\nval={best_val_value:.5f}",
                xy=(best_val_epoch, best_val_value),
                xytext=(best_val_epoch + 3, best_val_value + 0.0003),
                fontsize=8, color="#27AE60",
                arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.2))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("CASCADE EvolveGCN-H — Training Convergence")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    save_fig(fig, "fig1_training_curves")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Predictive Accuracy Comparison
# ─────────────────────────────────────────────────────────────────────────────

def fig2_accuracy_comparison(accuracy_path="results/experiment1_accuracy.csv",
                              dm_path="results/diebold_mariano_results.csv"):
    print("Fig 2: Accuracy comparison...")

    acc_df = pd.read_csv(accuracy_path)
    dm_df  = pd.read_csv(dm_path)

    horizons    = ["t1", "t5", "t10"]
    h_labels    = ["t+1 day", "t+5 days", "t+10 days"]
    model_order = ["EvolveGCN-H (CASCADE)", "VAR",
                   "Rolling Correlation", "Static GCN"]
    model_short = ["CASCADE", "VAR", "Rolling Corr", "Static GCN"]
    model_colors = [COLORS["CASCADE"], COLORS["VAR"],
                    COLORS["Rolling Corr"], COLORS["Static GCN"]]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=False)

    for col, (h, h_label) in enumerate(zip(horizons, h_labels)):
        ax   = axes[col]
        data = acc_df[acc_df["horizon"] == h]

        mse_vals  = []
        ci_lowers = []
        ci_uppers = []

        for model in model_order:
            row = data[data["model"] == model]
            if row.empty:
                mse_vals.append(0); ci_lowers.append(0); ci_uppers.append(0)
                continue
            mse = float(row["mse"].values[0])
            ci_l = float(row["mse_ci_lower"].values[0])
            ci_u = float(row["mse_ci_upper"].values[0])
            mse_vals.append(mse)
            ci_lowers.append(mse - ci_l)
            ci_uppers.append(ci_u - mse)

        x = np.arange(len(model_order))
        bars = ax.bar(x, mse_vals, color=model_colors, width=0.6,
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.errorbar(x, mse_vals,
                    yerr=[ci_lowers, ci_uppers],
                    fmt="none", color="#2C3E50", capsize=4,
                    linewidth=1.2, capthick=1.2)

        # Significance stars above CASCADE bar
        cascade_mse = mse_vals[0]
        for i, model in enumerate(model_order[1:], start=1):
            # Look up DM result
            dm_row = dm_df[
                (dm_df["comparison"].str.contains(
                    model.split("(")[0].strip(), regex=False)) &
                (dm_df["horizon"] == h)
            ]
            if not dm_row.empty:
                pval = float(dm_row["p_value"].values[0])
                star = "***" if pval < 0.001 else \
                       "**"  if pval < 0.01  else \
                       "*"   if pval < 0.05  else ""
                if star:
                    y_pos = mse_vals[i] + ci_uppers[i] + 0.00002
                    ax.text(i, y_pos, star, ha="center",
                            fontsize=9, color="#2C3E50", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(model_short, rotation=25, ha="right", fontsize=8)
        ax.set_title(h_label)
        if col == 0:
            ax.set_ylabel("MSE (test set)")
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FormatStrFormatter("%.4f"))

    # Legend
    patches = [mpatches.Patch(color=c, label=l, alpha=0.85)
               for c, l in zip(model_colors, model_short)]
    fig.legend(handles=patches, loc="upper center",
               ncol=4, bbox_to_anchor=(0.5, 1.02),
               frameon=False, fontsize=8.5)

    note = "* p<0.05   ** p<0.01   *** p<0.001  (Diebold-Mariano test, GNN vs baseline)"
    fig.text(0.5, -0.04, note, ha="center", fontsize=7.5,
             color="#7F8C8D", style="italic")

    fig.suptitle("Predictive Accuracy — Test Set", y=1.06, fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig2_accuracy_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Shock Propagation Heatmaps
# ─────────────────────────────────────────────────────────────────────────────

def fig3_shock_propagation(shock_path="results/experiment2_shock_propagation.csv"):
    print("Fig 3: Shock propagation heatmaps...")

    df = pd.read_csv(shock_path)
    events = df["event"].unique()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()

    # Symmetric colormap centered at 0
    vmax = df["spillover"].abs().max() * 1.1
    vmax = max(vmax, 1e-5)   # avoid zero range
    cmap = plt.cm.RdBu_r

    for idx, event in enumerate(events):
        ax      = axes[idx]
        ev_data = df[df["event"] == event].copy()
        assets  = ev_data["target"].tolist()
        spills  = ev_data["spillover"].values

        # Horizontal bar chart — cleaner than heatmap for 1D data
        colors = [cmap(0.5 + s / (2 * vmax)) for s in spills]
        bars   = ax.barh(assets, spills, color=colors,
                         edgecolor="white", linewidth=0.5)

        # Value labels
        for bar, val in zip(bars, spills):
            x_pos = val + (vmax * 0.03 if val >= 0 else -vmax * 0.03)
            ha    = "left" if val >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.5f}", va="center", ha=ha, fontsize=7.5)

        ax.axvline(0, color="#2C3E50", linewidth=0.8, alpha=0.5)
        ax.set_xlim(-vmax * 1.4, vmax * 1.4)

        # Title with shock info
        row0    = ev_data.iloc[0]
        source  = row0["source"]
        shock_v = row0["shock_val"] * 100
        date    = row0["date"]
        ax.set_title(f"{event}\n{date}  |  {source} = {shock_v:+.1f}%",
                     fontsize=9.5, fontweight="bold")

        ax.set_xlabel("Predicted spillover (return change)", fontsize=8.5)
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Experiment 2: Shock Propagation\n"
                 "Predicted spillover at each asset given shock at source",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5, wspace=0.35)
    save_fig(fig, "fig3_shock_propagation")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Regime Conditioning
# ─────────────────────────────────────────────────────────────────────────────

def fig4_regime_analysis(regime_path="results/experiment3_regime_analysis.csv"):
    print("Fig 4: Regime analysis...")

    df = pd.read_csv(regime_path)

    regimes     = df["regime"].tolist()
    w_norm_l1   = df["mean_w_norm_l1"].tolist()
    pred_mag    = df["mean_pred_magnitude"].tolist()
    n_snaps     = df["n_snapshots"].tolist()
    reg_colors  = [COLORS[r] for r in regimes]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))

    # ── Left: Prediction magnitude by regime ──────────────────────────────
    ax = axes[0]
    bars = ax.bar(regimes, pred_mag, color=reg_colors,
                  width=0.5, alpha=0.85, edgecolor="white")
    for bar, val, n in zip(bars, pred_mag, n_snaps):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(pred_mag) * 0.02,
                f"{val:.5f}\n(n={n})",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("Mean |predicted return|")
    ax.set_title("Contagion Magnitude by Regime")
    ax.set_ylim(0, max(pred_mag) * 1.35)

    # Annotate the ratio
    if len(pred_mag) >= 3:
        ratio = pred_mag[2] / pred_mag[0]
        ax.text(0.98, 0.95,
                f"Crisis / Calm ratio: {ratio:.2f}×",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color=COLORS["crisis"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=COLORS["crisis"], alpha=0.8))

    # ── Right: W norm (layer 1) by regime ─────────────────────────────────
    ax = axes[1]
    bars = ax.bar(regimes, w_norm_l1, color=reg_colors,
                  width=0.5, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, w_norm_l1):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(w_norm_l1) * 0.01,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("Mean Frobenius norm of W (layer 1)")
    ax.set_title("Weight Matrix Activation by Regime")
    ax.set_ylim(0, max(w_norm_l1) * 1.2)

    # Regime color legend
    patches = [mpatches.Patch(color=COLORS[r],
                              label=f"{r.capitalize()} (n={n})", alpha=0.85)
               for r, n in zip(regimes, n_snaps)]
    fig.legend(handles=patches, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8.5)

    fig.suptitle("Experiment 3: Regime-Conditioned Contagion Structure",
                 fontsize=11, fontweight="bold", y=1.06)
    fig.tight_layout()
    save_fig(fig, "fig4_regime_analysis")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Crypto Structural Break (Pre vs Post 2020)
# ─────────────────────────────────────────────────────────────────────────────

def fig5_structural_break(break_path="results/experiment4_structural_break.csv"):
    print("Fig 5: Structural break...")

    df = pd.read_csv(break_path).sort_values("change", ascending=True)

    assets   = df["asset"].tolist()
    pre_vals = df["pre_2020_mean_spillover"].tolist()
    post_vals= df["post_2020_mean_spillover"].tolist()
    changes  = df["change"].tolist()

    y = np.arange(len(assets))
    height = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Left: Pre vs Post grouped bars ────────────────────────────────────
    ax = axes[0]
    ax.barh(y + height/2, pre_vals,  height=height,
            color=COLORS["pre2020"],  alpha=0.8, label="Pre-2020")
    ax.barh(y - height/2, post_vals, height=height,
            color=COLORS["post2020"], alpha=0.8, label="Post-2020")

    ax.set_yticks(y)
    ax.set_yticklabels(assets)
    ax.axvline(0, color="#2C3E50", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Mean BTC shock spillover (predicted return change)")
    ax.set_title("BTC Shock Spillover:\nPre vs Post 2020 Institutional Adoption")
    ax.legend(loc="lower right", fontsize=8.5)

    # ── Right: Change (post - pre) ────────────────────────────────────────
    ax = axes[1]
    change_colors = [COLORS["post2020"] if c > 0 else COLORS["pre2020"]
                     for c in changes]
    bars = ax.barh(y, changes, color=change_colors, alpha=0.85,
                   edgecolor="white")

    for bar, val, asset in zip(bars, changes, assets):
        x_pos = val + (max(abs(c) for c in changes) * 0.03
                       if val >= 0
                       else -max(abs(c) for c in changes) * 0.03)
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.6f}", va="center", ha=ha, fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels(assets)
    ax.axvline(0, color="#2C3E50", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Change in spillover (post − pre 2020)")
    ax.set_title("Spillover Change Post-2020\n(positive = increased contagion)")

    # Annotation
    ax.text(0.02, 0.97,
            "Structural break: Oct 2020\n"
            "(institutional BTC adoption)",
            transform=ax.transAxes, va="top", fontsize=8,
            color="#2C3E50", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#EBF5FB",
                      edgecolor="#3498DB", alpha=0.8))

    fig.suptitle("Experiment 4: Crypto Structural Break — "
                 "BTC Contagion Pre vs Post Institutional Adoption",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig5_structural_break")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Contagion Network (DCC average correlations)
# ─────────────────────────────────────────────────────────────────────────────

def fig6_contagion_network(dcc_avg_path="data/processed/dcc_avg_correlation.csv",
                            threshold=0.3):
    if not HAS_NX:
        print("Fig 6: Skipped (networkx not available)")
        return
    if not os.path.exists(dcc_avg_path):
        print("Fig 6: Skipped (dcc_avg_correlation.csv not found)")
        return

    print("Fig 6: Contagion network...")

    corr = pd.read_csv(dcc_avg_path, index_col=0)
    assets = corr.columns.tolist()

    # Node classification for color coding
    asset_class = {
        "SPY": "equity",  "EEM": "equity",
        "LQD": "bond",    "HYG": "bond",    "TLT": "bond",
        "GLD": "commodity","USO": "commodity",
        "BTC": "crypto",  "ETH": "crypto",
        "DXY": "macro",
    }
    class_colors = {
        "equity":    "#3498DB",
        "bond":      "#27AE60",
        "commodity": "#F39C12",
        "crypto":    "#9B59B6",
        "macro":     "#7F8C8D",
    }

    G = nx.Graph()
    for asset in assets:
        G.add_node(asset, asset_class=asset_class.get(asset, "other"))

    for i, a1 in enumerate(assets):
        for j, a2 in enumerate(assets):
            if i >= j:
                continue
            r = float(corr.loc[a1, a2])
            if abs(r) > threshold:
                G.add_edge(a1, a2, weight=abs(r), sign=np.sign(r))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Layout
    pos = nx.spring_layout(G, seed=42, k=2.5)

    # Node colors
    node_colors = [class_colors.get(asset_class.get(n, "other"), "#95A5A6")
                   for n in G.nodes()]

    # Edge widths and colors by correlation strength and sign
    edges      = list(G.edges(data=True))
    widths     = [d["weight"] * 4 for _, _, d in edges]
    edge_colors= ["#2980B9" if d["sign"] > 0 else "#E74C3C"
                  for _, _, d in edges]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=900, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9,
                            font_color="white", font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths,
                           edge_color=edge_colors, alpha=0.6)

    # Edge weight labels on thick edges
    edge_labels = {(u, v): f"{d['weight']:.2f}"
                   for u, v, d in edges if d["weight"] > 0.5}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax,
                                 font_size=7, alpha=0.8)

    # Legend — asset classes
    class_patches = [mpatches.Patch(color=c, label=cls.capitalize(), alpha=0.9)
                     for cls, c in class_colors.items()]
    edge_patches  = [mpatches.Patch(color="#2980B9", alpha=0.6,
                                     label="Positive correlation"),
                     mpatches.Patch(color="#E74C3C", alpha=0.6,
                                     label="Negative correlation")]
    ax.legend(handles=class_patches + edge_patches,
              loc="lower left", fontsize=8, framealpha=0.9)

    n_edges = G.number_of_edges()
    ax.set_title(f"Cross-Asset Contagion Network\n"
                 f"DCC-GARCH average correlations  |  "
                 f"threshold |r| > {threshold}  |  "
                 f"{n_edges} edges",
                 fontsize=10.5, fontweight="bold")
    ax.axis("off")

    fig.tight_layout()
    save_fig(fig, "fig6_contagion_network")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  CASCADE — Generating Figures")
    print("=" * 55)

    fig1_training_curves()
    fig2_accuracy_comparison()
    fig3_shock_propagation()
    fig4_regime_analysis()
    fig5_structural_break()
    fig6_contagion_network()

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("\nFiles:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        size = os.path.getsize(os.path.join(FIGURES_DIR, f))
        print(f"  {f:45s} {size:>10,} bytes")
