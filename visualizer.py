"""
Visualizer — Generates and saves all trust score charts.
Outputs PNG files to the reports/ directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from core.trust_engine import TrustReport

PALETTE = {
    "bg":       "#0d1117",
    "card":     "#161b22",
    "border":   "#21262d",
    "safe":     "#00e676",
    "moderate": "#ffeb3b",
    "risk":     "#ff5252",
    "accent":   "#028090",
    "text":     "#c9d1d9",
    "muted":    "#8b949e",
}


def _setup_dark_axes(ax):
    ax.set_facecolor(PALETTE["card"])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["border"])
    ax.tick_params(colors=PALETTE["text"])
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    ax.grid(color=PALETTE["border"], linewidth=0.6, alpha=0.6)


def score_color(score: float) -> str:
    if score >= 75:
        return PALETTE["safe"]
    elif score >= 50:
        return PALETTE["moderate"]
    return PALETTE["risk"]


def plot_full_dashboard(report: TrustReport, output_dir: str = "reports") -> str:
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(18, 14), facecolor=PALETTE["bg"])
    fig.suptitle(
        f"Trust Score Dashboard — {report.dataset_name}",
        fontsize=18, fontweight="bold", color=PALETTE["text"], y=0.98
    )

    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Gauge / Score display
    ax_gauge = fig.add_subplot(gs[0, 0])
    ax_gauge.set_facecolor(PALETTE["card"])
    ax_gauge.set_xlim(0, 10)
    ax_gauge.set_ylim(0, 6)
    ax_gauge.axis("off")
    score = report.trust_score
    color = score_color(score)
    ax_gauge.add_patch(mpatches.FancyBboxPatch((0.3, 0.3), 9.4, 5.4,
        boxstyle="round,pad=0.1", facecolor=PALETTE["bg"], edgecolor=color, linewidth=2))
    ax_gauge.text(5, 3.8, f"{score:.1f}", ha="center", va="center",
                  fontsize=44, fontweight="bold", color=color)
    ax_gauge.text(5, 2.5, "/ 100", ha="center", va="center", fontsize=18, color=PALETTE["muted"])
    ax_gauge.text(5, 1.5, report.risk_label, ha="center", va="center",
                  fontsize=13, fontweight="bold", color=color)
    ax_gauge.set_title("Trust Score", color=PALETTE["text"], fontsize=11)

    # ── 2. Radar chart
    ax_radar = fig.add_subplot(gs[0, 1], polar=True)
    ax_radar.set_facecolor(PALETTE["card"])
    names   = list(report.metrics.keys())
    scores  = [report.metrics[n].score for n in names]
    angles  = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
    values  = scores + [scores[0]]
    angles += [angles[0]]
    ax_radar.plot(angles, values, "o-", linewidth=2, color=PALETTE["accent"])
    ax_radar.fill(angles, values, alpha=0.2, color=PALETTE["accent"])
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(names, color=PALETTE["text"], fontsize=9)
    ax_radar.set_yticklabels([])
    ax_radar.set_ylim(0, 100)
    ax_radar.grid(color=PALETTE["border"])
    ax_radar.tick_params(colors=PALETTE["text"])
    ax_radar.set_title("Quality Radar", color=PALETTE["text"], fontsize=11, pad=15)

    # ── 3. Metric scores bar
    ax_bar = fig.add_subplot(gs[0, 2])
    _setup_dark_axes(ax_bar)
    bar_names   = list(report.metrics.keys())
    bar_scores  = [report.metrics[n].score for n in bar_names]
    bar_colors  = [score_color(s) for s in bar_scores]
    bars = ax_bar.barh(bar_names, bar_scores, color=bar_colors, edgecolor="none", height=0.55)
    for bar, val in zip(bars, bar_scores):
        ax_bar.text(min(val + 1, 105), bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", color=PALETTE["text"], fontsize=9)
    ax_bar.set_xlim(0, 115)
    ax_bar.axvline(75, color=PALETTE["safe"],    linestyle="--", linewidth=1, alpha=0.5, label="Safe (75)")
    ax_bar.axvline(50, color=PALETTE["moderate"], linestyle="--", linewidth=1, alpha=0.5, label="Moderate (50)")
    ax_bar.legend(fontsize=7, loc="lower right", facecolor=PALETTE["bg"],
                  labelcolor=PALETTE["text"], framealpha=0.5)
    ax_bar.set_xlabel("Score", color=PALETTE["text"])
    ax_bar.set_title("Metric Scores", color=PALETTE["text"], fontsize=11)

    # ── 4. Weighted contributions
    ax_contrib = fig.add_subplot(gs[1, 0])
    _setup_dark_axes(ax_contrib)
    contribs = [report.metrics[n].weighted_score for n in names]
    wedge_colors = [score_color(report.metrics[n].score) for n in names]
    wedges, texts, autotexts = ax_contrib.pie(
        contribs, labels=names, colors=wedge_colors, autopct="%1.1f%%",
        startangle=140, textprops={"color": PALETTE["text"], "fontsize": 8},
        wedgeprops={"edgecolor": PALETTE["bg"], "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_color(PALETTE["bg"])
        at.set_fontsize(8)
    ax_contrib.set_title("Score Contributions", color=PALETTE["text"], fontsize=11)

    # ── 5. Missing values heatmap
    ax_miss = fig.add_subplot(gs[1, 1])
    _setup_dark_axes(ax_miss)
    miss_data = {col: p["null_pct"] for col, p in report.column_profiles.items()}
    cols_sorted = sorted(miss_data, key=miss_data.get, reverse=True)[:12]
    values_miss = [miss_data[c] for c in cols_sorted]
    col_colors  = [score_color(100 - v) for v in values_miss]
    ax_miss.barh(cols_sorted, values_miss, color=col_colors, edgecolor="none", height=0.6)
    ax_miss.set_xlabel("Missing %", color=PALETTE["text"])
    ax_miss.set_title("Missing Values by Column", color=PALETTE["text"], fontsize=11)
    ax_miss.axvline(20, color=PALETTE["moderate"], linestyle="--", linewidth=1, alpha=0.7)

    # ── 6. Severity breakdown of recommendations
    ax_rec = fig.add_subplot(gs[1, 2])
    _setup_dark_axes(ax_rec)
    sev_count = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    for rec in report.recommendations:
        sev_count[rec.severity] = sev_count.get(rec.severity, 0) + 1
    sev_colors = {
        "Critical": PALETTE["risk"],
        "High":     "#ff9800",
        "Medium":   PALETTE["moderate"],
        "Low":      PALETTE["safe"],
    }
    labels  = [k for k in sev_count if sev_count[k] > 0]
    sizes   = [sev_count[k] for k in labels]
    colors  = [sev_colors[k] for k in labels]
    if sizes:
        ax_rec.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%",
                   startangle=90,
                   textprops={"color": PALETTE["text"], "fontsize": 9},
                   wedgeprops={"edgecolor": PALETTE["bg"], "linewidth": 1.5})
    else:
        ax_rec.text(0.5, 0.5, "No Issues!", ha="center", va="center",
                    color=PALETTE["safe"], fontsize=14)
        ax_rec.axis("off")
    ax_rec.set_title("Recommendations by Severity", color=PALETTE["text"], fontsize=11)

    # ── 7. Numeric column distribution (first 3)
    numeric_cols = [c for c, p in report.column_profiles.items() if "mean" in p][:3]
    for idx, col in enumerate(numeric_cols):
        ax_dist = fig.add_subplot(gs[2, idx])
        _setup_dark_axes(ax_dist)
        # Note: we don't have the raw data here; show profile stats as a mini bar
        stats = report.column_profiles[col]
        labels_d = ["Min", "Q1*", "Median", "Mean", "Q3*", "Max"]
        q1   = stats["mean"] - stats["std"]
        q3   = stats["mean"] + stats["std"]
        vals = [stats["min"], q1, stats["median"], stats["mean"], q3, stats["max"]]
        bar_colors_d = [PALETTE["accent"]] * 6
        bar_colors_d[3] = PALETTE["moderate"]  # highlight mean
        ax_dist.bar(labels_d, vals, color=bar_colors_d, edgecolor="none", width=0.6)
        ax_dist.set_title(f"'{col}' Stats", color=PALETTE["text"], fontsize=10)
        ax_dist.set_xlabel("", color=PALETTE["text"])
        ax_dist.set_ylabel("Value", color=PALETTE["text"])

    # ── Save
    safe_name = report.dataset_name.replace(" ", "_").replace("/", "_")
    filepath  = os.path.join(output_dir, f"dashboard_{safe_name}.png")
    plt.savefig(filepath, dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  🖼️  Dashboard chart saved → {filepath}")
    return filepath


def plot_comparison(comparison_df, output_dir: str = "reports") -> str:
    """Bar chart comparing trust scores across multiple datasets."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=PALETTE["bg"])
    _setup_dark_axes(ax)

    datasets = comparison_df["Dataset"].tolist()
    scores   = comparison_df["Trust Score"].tolist()
    colors   = [score_color(s) for s in scores]

    bars = ax.bar(datasets, scores, color=colors, edgecolor=PALETTE["bg"], width=0.5)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{score:.1f}", ha="center", va="bottom",
                color=PALETTE["text"], fontsize=11, fontweight="bold")

    ax.axhline(75, color=PALETTE["safe"],     linestyle="--", linewidth=1.2, alpha=0.7, label="Safe threshold (75)")
    ax.axhline(50, color=PALETTE["moderate"], linestyle="--", linewidth=1.2, alpha=0.7, label="Moderate threshold (50)")
    ax.set_ylabel("Trust Score", color=PALETTE["text"], fontsize=12)
    ax.set_title("Dataset Trust Score Comparison", color=PALETTE["text"], fontsize=14, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(facecolor=PALETTE["bg"], labelcolor=PALETTE["text"], fontsize=9)

    safe_patch     = mpatches.Patch(color=PALETTE["safe"],     label="Safe (≥75)")
    moderate_patch = mpatches.Patch(color=PALETTE["moderate"], label="Moderate (50–74)")
    risk_patch     = mpatches.Patch(color=PALETTE["risk"],     label="High Risk (<50)")
    ax.legend(handles=[safe_patch, moderate_patch, risk_patch],
              loc="upper right", facecolor=PALETTE["card"],
              labelcolor=PALETTE["text"], fontsize=9, framealpha=0.8)

    filepath = os.path.join(output_dir, "comparison_chart.png")
    plt.savefig(filepath, dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  🖼️  Comparison chart saved → {filepath}")
    return filepath
